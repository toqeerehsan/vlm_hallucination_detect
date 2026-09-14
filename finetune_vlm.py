#!/usr/bin/env python3
"""
finetune_vlm.py — bf16 LoRA (+rsLoRA +LoRA+) SFT for inline hallucination tagging.

ONE recipe for all three models. No quantization, so the resulting adapter merges
cleanly into the bf16 base via client.py's existing merge_and_unload() path.

Key properties:
  * Prompts are built by calling prompt_builder.build_messages() with ZERO shots,
    so training format == inference format at NUM_SHOTS=0 (exact parity).
  * Loss is computed on the assistant turn ONLY (prompt/system/image tokens are
    masked to -100). Without this the model learns to regurgitate the prompt.
  * LoRA targets LLM linear layers only; the vision tower is excluded and frozen.
  * Adapter is pushed to the Hub as a PRIVATE repo, loadable by models.json
    exactly like the existing *-ft entries.

Usage: see run_finetune_slurm.sh
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

import torch
from torch.utils.data import Dataset

# Reuse the EXACT inference-time prompt construction.
from prompt_builder import build_messages


# --------------------------------------------------------------------------
# inline-annotation inspection (read-only: targets are used VERBATIM)
# --------------------------------------------------------------------------
_TAG_RE = re.compile(r'<hall\s+label="([^"]*)"\s+prob="([^"]*)">')
_ANY_TAG_RE = re.compile(r'</?hall[^>]*>')

# Declared by SYSTEM_WITH_LABELS; the data already matches, so nothing is
# rewritten. These sets exist only to surface surprises in the log.
_EXPECTED_LABELS = {'invention', 'mischaracterization', 'OCR', 'miscounting', 'other'}
_EXPECTED_PROBS = {'prob1', 'prob2', 'prob3'}


def strip_tags(s):
    """Remove all <hall> markup, leaving the plain response text."""
    return _ANY_TAG_RE.sub('', s)


def scan_inline(inline, counters):
    """Count labels/probs. Does NOT modify the inline text."""
    for label, prob in _TAG_RE.findall(inline):
        counters['label'][label] += 1
        counters['prob'][prob] += 1
        if label not in _EXPECTED_LABELS:
            counters['bad_label'][label] += 1
        if prob not in _EXPECTED_PROBS:
            counters['bad_prob'][prob] += 1


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _fold_system(messages):
    """Same behaviour as client.py::_fold_system (needed for Gemma templates)."""
    if not messages or messages[0]['role'] != 'system':
        return messages
    sys_text = messages[0].get('text', '')
    out, folded = [], False
    for m in messages[1:]:
        if not folded and m['role'] == 'user':
            m = dict(m)
            m['text'] = (sys_text + '\n\n' + m.get('text', '')).strip()
            folded = True
        out.append(m)
    return out


def resolve_image(row, image_dir):
    """train rows carry `image_path` (already repo-relative) or `image_name`."""
    p = row.get('image_path')
    if p:
        if os.path.exists(p):
            return p
        cand = os.path.join(image_dir, os.path.basename(p))
        return cand if os.path.exists(cand) else None
    name = row.get('image_name')
    if name:
        cand = os.path.join(image_dir, name)
        return cand if os.path.exists(cand) else None
    return None


class InlineSFTDataset(Dataset):
    def __init__(self, path, image_dir, args, counters):
        self.rows = []
        missing_img = 0
        bad_target = 0
        negatives = 0
        n_spans = 0
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                inline = o.get('response_inline')
                resp = o.get('response')
                if not inline or not resp:
                    continue
                # HARD REQUIREMENT: the target must reduce to the response exactly,
                # otherwise the model is taught to alter text and every downstream
                # character offset breaks.
                if strip_tags(inline) != resp:
                    bad_target += 1
                    continue
                scan_inline(inline, counters)
                if '<hall' not in inline:
                    negatives += 1
                else:
                    n_spans += inline.count('<hall')
                img = resolve_image(o, image_dir)
                if img is None:
                    missing_img += 1
                self.rows.append({
                    'id': o.get('id'),
                    'prompt': o.get('prompt', ''),
                    'response': resp,
                    'inline': inline,
                    'image_path': img,
                    'image_name': o.get('image_name') or (
                        os.path.basename(o['image_path']) if o.get('image_path') else None),
                })
        if args.limit:
            self.rows = self.rows[:args.limit]
        n = max(len(self.rows), 1)
        print(f'[data] {path}\n'
              f'       kept={len(self.rows)}  dropped_bad_target={bad_target}  '
              f'missing_images={missing_img}\n'
              f'       negatives(no tags)={negatives} ({100.0 * negatives / n:.1f}%)  '
              f'total_spans={n_spans}', flush=True)
        if bad_target:
            print(f'       !! {bad_target} rows where strip(inline) != response '
                  f'— check prepare_ft_splits.py', flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Builds prompt+completion, masks loss to the assistant turn only."""

    #def __init__(self, processor, args):
    #    self.processor = processor
    #    self.args = args
    #    self.ct_kwargs = args.chat_template_kwargs or {}
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args
        self.ct_kwargs = args.chat_template_kwargs or {}
        self.n_overlong = 0
        self.n_mask_clamped = 0

    def _to_conv(self, neutral, image_path):
        from PIL import Image
        conv = []
        for m in neutral:
            content = []
            for p in (m.get('images') or []):
                content.append({'type': 'image', 'image': Image.open(p).convert('RGB')})
            if m.get('text'):
                content.append({'type': 'text', 'text': m['text']})
            conv.append({'role': m['role'], 'content': content})
        return conv

    def _apply(self, conv, add_generation_prompt):
        try:
            return self.processor.apply_chat_template(
                conv, add_generation_prompt=add_generation_prompt, tokenize=True,
                return_dict=True, return_tensors='pt', **self.ct_kwargs)
        except TypeError:
            return self.processor.apply_chat_template(
                conv, add_generation_prompt=add_generation_prompt, tokenize=True,
                return_dict=True, return_tensors='pt')

    def __call__(self, batch):
        feats = []
        for row in batch:
            neutral = build_messages(
                row['prompt'], row['response'], row['image_path'], [],
                response_field='response', inline_field='response_inline',
                fewshot_image_paths=None,
                include_imagename=self.args.include_imagename,
                eval_image_name=row.get('image_name'),
            )
            if self.args.merge_system_into_user:
                neutral = _fold_system(neutral)

            prompt_conv = self._to_conv(neutral, row['image_path'])
            full_conv = prompt_conv + [
                {'role': 'assistant', 'content': [{'type': 'text', 'text': row['inline']}]}
            ]

            p_enc = self._apply(prompt_conv, True)
            f_enc = self._apply(full_conv, False)

            plen = p_enc['input_ids'].shape[1]
            ids = f_enc['input_ids'][0]
            if ids.shape[0] > self.args.max_seq_len:
                # Do NOT drop: with batch_size=1 an empty batch becomes None and
                # Trainer._prepare_inputs() raises TypeError. Keep it and warn.
                # Truncating instead would cut the target and teach early stopping.
                self.n_overlong += 1
                if self.n_overlong <= 5 or self.n_overlong % 100 == 0:
                    print(f'[warn] over max_seq_len: {ids.shape[0]} > '
                          f'{self.args.max_seq_len} (n={self.n_overlong}). '
                          f'Lower --max-pixels or raise --max-seq-len.', flush=True)
            # Chat templates can make the generation prompt tokenize LONGER than
            # the same prefix inside the full conversation -> every label masked
            # -> CE over an empty set -> nan loss. Clamp to keep >=1 supervised.
            if plen >= ids.shape[0]:
                self.n_mask_clamped += 1
                if self.n_mask_clamped <= 3:
                    print(f'[warn] mask clamp: plen={plen} >= len={ids.shape[0]}',
                          flush=True)
                plen = max(int(ids.shape[0]) - 1, 0)
            labels = ids.clone()
            labels[:plen] = -100
            feats.append((f_enc, ids, labels))

        if not feats:
            raise RuntimeError('collator produced an empty batch')

        # batch-size 1 is the supported path (multimodal padding is model-specific)
        f_enc, ids, labels = feats[0]
        out = {k: (v if torch.is_tensor(v) else v) for k, v in f_enc.items()}
        out['input_ids'] = ids.unsqueeze(0)
        out['labels'] = labels.unsqueeze(0)
        if 'attention_mask' not in out:
            out['attention_mask'] = torch.ones_like(out['input_ids'])
        return out


# --------------------------------------------------------------------------
# LoRA targeting: LLM linears only, vision tower excluded
# --------------------------------------------------------------------------
_SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
             'gate_proj', 'up_proj', 'down_proj')
_VISION_HINTS = ('vision', 'visual', 'image_tower', 'vision_tower',
                 'multi_modal_projector', 'mm_projector', 'patch_embed', 'siglip')


def collect_target_modules(model, include_mlp=True):
    suffixes = _SUFFIXES if include_mlp else _SUFFIXES[:4]
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        low = name.lower()
        if any(h in low for h in _VISION_HINTS):
            continue
        if 'lm_head' in low:
            continue
        if name.split('.')[-1] in suffixes:
            names.append(name)
    return names


def freeze_vision(model):
    n = 0
    for name, p in model.named_parameters():
        if any(h in name.lower() for h in _VISION_HINTS):
            p.requires_grad = False
            n += 1
    print(f'[freeze] vision/projector params frozen: {n}', flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--method', default='lora', choices=['lora'],
                    help='only bf16 lora is supported (merges cleanly at inference)')
    ap.add_argument('--train-file', required=True)
    ap.add_argument('--eval-file', default=None)
    ap.add_argument('--image-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--hub-id', default=None)
    ap.add_argument('--private', action='store_true', default=True)
    ap.add_argument('--merge-system-into-user', action='store_true')
    ap.add_argument('--include_imagename', dest='include_imagename', action='store_true')
    ap.add_argument('--chat-template-kwargs', default=None)
    ap.add_argument('--lora-r', type=int, default=64)
    ap.add_argument('--lora-alpha', type=int, default=64)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--loraplus-ratio', type=float, default=16.0)
    ap.add_argument('--no-mlp', action='store_true',
                    help='attention-only targeting (use for MoE bases)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epochs', type=float, default=2)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--max-seq-len', type=int, default=6192)
    ap.add_argument('--max-pixels', type=int, default=None,
                    help='cap image resolution (Qwen-style processors). '
                         '802816 = 1024*28*28 -> ~1024 vision tokens. Use for Qwen; '
                         'ignored safely by Gemma/Mistral processors.')
    ap.add_argument('--warmup-ratio', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    if args.chat_template_kwargs:
        args.chat_template_kwargs = json.loads(args.chat_template_kwargs)

    from transformers import AutoProcessor, Trainer, TrainingArguments
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoVLM
    from peft import LoraConfig, get_peft_model

    counters = {'label': Counter(), 'prob': Counter(),
                'bad_label': Counter(), 'bad_prob': Counter()}

    print(f'[load] {args.model} (bf16)', flush=True)
    #processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    proc_kwargs = {'trust_remote_code': True}
    if args.max_pixels:
        proc_kwargs['max_pixels'] = args.max_pixels
    try:
        processor = AutoProcessor.from_pretrained(args.model, **proc_kwargs)
    except (TypeError, ValueError):
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        if args.max_pixels:
            print('[warn] this processor ignored max_pixels', flush=True)

    model = AutoVLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.config.use_cache = False

    freeze_vision(model)
    targets = collect_target_modules(model, include_mlp=not args.no_mlp)
    print(f'[lora] targeting {len(targets)} linear modules '
          f'(sample: {targets[:3]})', flush=True)

    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=True,                 # stabilises the higher rank
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=targets,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    train_ds = InlineSFTDataset(args.train_file, args.image_dir, args, counters)
    eval_ds = (InlineSFTDataset(args.eval_file, args.image_dir, args, counters)
               if args.eval_file and os.path.exists(args.eval_file) else None)
    print(f'[data] label dist: {dict(counters["label"])}', flush=True)
    print(f'[data] prob  dist: {dict(counters["prob"])}', flush=True)
    if counters['bad_label'] or counters['bad_prob']:
        print(f'[data] !! UNEXPECTED labels={dict(counters["bad_label"])} '
              f'probs={dict(counters["bad_prob"])} — these are passed through '
              f'unchanged; check the prompt/data contract', flush=True)

    collator = Collator(processor, args)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=3,
        eval_strategy='epoch' if eval_ds else 'no',
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        per_device_eval_batch_size=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        remove_unused_columns=False,
        report_to='none',
        seed=args.seed,
        dataloader_num_workers=2,
    )

    # LoRA+ : higher LR on the B matrices
    optimizers = (None, None)
    try:
        from peft.optimizers import create_loraplus_optimizer
        import bitsandbytes  # noqa: F401
        opt = create_loraplus_optimizer(
            model=model, optimizer_cls=torch.optim.AdamW,
            lr=args.lr, loraplus_lr_ratio=args.loraplus_ratio)
        optimizers = (opt, None)
        print(f'[loraplus] enabled, ratio={args.loraplus_ratio}', flush=True)
    except Exception as e:  # noqa: BLE001
        try:
            from peft.optimizers import create_loraplus_optimizer
            opt = create_loraplus_optimizer(
                model=model, optimizer_cls=torch.optim.AdamW,
                lr=args.lr, loraplus_lr_ratio=args.loraplus_ratio)
            optimizers = (opt, None)
            print(f'[loraplus] enabled, ratio={args.loraplus_ratio}', flush=True)
        except Exception as e2:  # noqa: BLE001
            print(f'[loraplus] unavailable ({e2}); plain AdamW', flush=True)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        optimizers=optimizers,
    )

    trainer.train()

    print(f'[save] adapter -> {args.output_dir}', flush=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.hub_id:
        token = os.environ.get('HF_TOKEN')
        print(f'[push] {args.hub_id} (private={args.private})', flush=True)
        model.push_to_hub(args.hub_id, private=args.private, token=token)
        try:
            processor.push_to_hub(args.hub_id, private=args.private, token=token)
        except Exception as e:  # noqa: BLE001
            print(f'[push] processor push skipped: {e}', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
