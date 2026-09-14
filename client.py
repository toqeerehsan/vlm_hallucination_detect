#!/usr/bin/env python3
"""
client.py — one interface over HuggingFace-transformers (local), Azure, and mock.

Messages use a small NEUTRAL schema so the pipeline stays backend-agnostic:
    [{"role": "system|user|assistant", "text": str, "images": [image_path, ...]}, ...]
Each backend converts this to its own format:
  - hf    : transformers multimodal chat (PIL images embedded), in-process generate()
  - azure : OpenAI chat with base64 data-URLs (no local server)
  - mock  : offline echo for testing plumbing
"""
import base64
import json
import mimetypes
import os
import time


def load_models(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)['models']


def encode_image(path):
    mime = mimetypes.guess_type(path)[0] or 'image/jpeg'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def _fold_system(messages):
    if not messages or messages[0]['role'] != 'system':
        return messages
    sys_text = messages[0].get('text', '')
    out = []
    folded = False
    for m in messages[1:]:
        if not folded and m['role'] == 'user':
            m = dict(m)
            m['text'] = (sys_text + '\n\n' + m.get('text', '')).strip()
            folded = True
        out.append(m)
    return out


class VLMClient:
    def __init__(self, model_key, models):
        if model_key not in models:
            raise KeyError(f'model "{model_key}" not in models.json (have: {list(models)})')
        self.key = model_key
        self.cfg = models[model_key]
        self.provider = self.cfg['provider']
        self._backend = None
        self._init_backend()

    def _env(self, name, required=True):
        val = os.environ.get(self.cfg.get(name, ''))
        if required and not val:
            raise EnvironmentError(f'env var {self.cfg.get(name)} (for {name}) is not set')
        return val

    def _init_backend(self):
        if self.provider == 'mock':
            return
        if self.provider == 'azure':
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=self._env('api_key_env'),
                azure_endpoint=self._env('endpoint_env'),
                api_version=self.cfg['api_version'])
            self._model_name = self.cfg['deployment']
        elif self.provider == 'hf':
            self._backend = _HFBackend(self.cfg)
        else:
            raise ValueError(f'unknown provider: {self.provider}')

    def chat(self, messages, retries=3, backoff=2.0):
        max_new = self.cfg.get('max_new_tokens', self.cfg.get('max_tokens', 3072))
        temp = self.cfg.get('temperature', 0.0)

        if self.provider == 'mock':
            return _mock_annotate(messages)

        if self.provider == 'hf':
            if self.cfg.get('merge_system_into_user'):
                messages = _fold_system(messages)
            return self._backend.generate(messages, max_new, temp)

        # azure
        oai = _to_openai(messages)
        params = {'model': self._model_name, 'messages': oai}
        # Reasoning models (GPT-5.x, o-series) reject max_tokens and a
        # non-default temperature. Both are configurable per model in models.json:
        #   "max_tokens_param": "max_completion_tokens"
        #   "omit_temperature": true
        token_param = self.cfg.get('max_tokens_param', 'max_tokens')
        params[token_param] = max_new
        if not self.cfg.get('omit_temperature'):
            params['temperature'] = temp
        last = None
        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(**params)
                return resp.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
        raise RuntimeError(f'azure chat failed after {retries} tries: {last}')


def _to_openai(messages):
    """Neutral -> OpenAI chat format (images as base64 data-URLs)."""
    out = []
    for m in messages:
        if m['role'] == 'system':
            out.append({'role': 'system', 'content': m.get('text', '')})
            continue
        parts = []
        for p in m.get('images', []) or []:
            parts.append({'type': 'image_url', 'image_url': {'url': encode_image(p)}})
        if m.get('text'):
            parts.append({'type': 'text', 'text': m['text']})
        out.append({'role': m['role'], 'content': parts})
    return out


class _HFBackend:
    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except Exception:  # older transformers
            from transformers import AutoModelForVision2Seq as _AutoVLM

        model_id = cfg.get('model_path') or cfg['model']
        dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
                 'auto': 'auto'}.get(cfg.get('dtype', 'bfloat16'), torch.bfloat16)

        #self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except OSError:
            # Some repos (e.g. Mistral) ship the processor via mistral-common or need explicit classes
            from transformers import AutoImageProcessor, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            img = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
            from transformers import AutoProcessor as _AP
            self.processor = _AP.from_pretrained(model_id, image_processor=img,
                                                 tokenizer=tok, trust_remote_code=True)
        #self.model = _AutoVLM.from_pretrained(
        #    model_id, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
        #self.model.eval()
        self.model = _AutoVLM.from_pretrained(
            model_id, dtype=dtype, device_map='auto', trust_remote_code=True)
        adapter = cfg.get('adapter')
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()   # fold LoRA in for fast inference
        self.model.eval()
        
        self._torch = torch
        self.ct_kwargs = cfg.get('chat_template_kwargs', {}) or {}

    def _load_image(self, path):
        from PIL import Image
        return Image.open(path).convert('RGB')

    def _to_transformers(self, messages):
        conv = []
        for m in messages:
            content = []
            for p in m.get('images', []) or []:
                content.append({'type': 'image', 'image': self._load_image(p)})
            if m.get('text'):
                content.append({'type': 'text', 'text': m['text']})
            conv.append({'role': m['role'], 'content': content})
        return conv

    def generate(self, messages, max_new_tokens, temperature):
        torch = self._torch
        conv = self._to_transformers(messages)
        try:
            inputs = self.processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors='pt', **self.ct_kwargs)
        except TypeError:
            inputs = self.processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors='pt')
        
        inputs = {k: (v.to(self.model.device) if hasattr(v, 'to') else v)
                  for k, v in inputs.items()}
        gen = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
        if temperature > 0:
            gen['temperature'] = temperature
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen)
        in_len = inputs['input_ids'].shape[1]
        trimmed = out[0][in_len:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


def _mock_annotate(messages):
    text = None
    for m in reversed(messages):
        if m.get('role') == 'user' and 'RESPONSE:' in (m.get('text') or ''):
            text = m['text'].split('RESPONSE:', 1)[1].strip()
            break
    if not text:
        return ''
    marker = ' is '
    i = text.find(marker)
    if i != -1:
        j = i + len(marker)
        k = text.find(' ', j)
        if k == -1:
            k = len(text)
        return text[:j] + '<hall label="mischaracterization" prob="prob2">' + text[j:k] + '</hall>' + text[k:]
    return text
