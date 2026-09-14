#!/usr/bin/env python3
"""
prompt_builder.py — system spec + a FIXED, constraint-selected 5-shot set.

Few-shot is the same 5 examples for every target sample:
  - 4 examples, ONE per language (en, fr, it, zh), each with >=2 annotated spans,
    every span 15..40 characters, and no None/garbage tags.
  - the 4 are chosen greedily so that together they cover ALL 5 hallucination
    labels (invention, mischaracterization, OCR, miscounting, other) AND ALL 6
    prob classes (prob1..prob6).
  - a 5th example with NO labels (a fully-supported response), so the model also
    sees when to output nothing.
Selection is deterministic (seed) and cached, so it is computed once per run.
There is no manual few-shot file anymore.
"""
import json
import random
import re
import sys

MIN_SPAN = 1
MAX_SPAN = 60
MIN_HALL_SPANS = 4
LANGS = ('en', 'fr', 'it', 'zh')
LABELS = {'invention', 'mischaracterization', 'OCR', 'miscounting', 'other'}
PROBS = {'prob1', 'prob2', 'prob3'}

_OPEN = re.compile(r'<hall\b([^>]*?)>')
_LAB = re.compile(r'label\s*=\s*"([^"]*)"')
_PRB = re.compile(r'prob\s*=\s*"([^"]*)"')
_CLOSE = '</hall>'

_FIXED_CACHE = None

SYSTEM_WITH_LABELS = """You are a hallucination-span annotator for vision-language model outputs.

You are given: an IMAGE, a PROMPT that was asked about it, and a RESPONSE that a vision-language model produced. Parts of the RESPONSE are often hallucinated — that is, not supported by the image.

Your job: MARK hallucinations with <hall> tags inserted around every hallucinated span and return the RESPONSE text EXACTLY as given, character for character. Change NOTHING else — do not fix typos, spacing, markdown, or line breaks. Do not add words. Do not add any
preamble, reasoning, or explanation. Output ONLY the annotated response text. 

Tag format:
  <hall label="LABEL" prob="PROBCLASS">hallucinated text</hall>

LABEL is exactly one of (never "None", never empty):
invention - entity/object/property/event not present in the image
mischaracterization - content is visible but described incorrectly
OCR - text visible in the image is misread
miscounting - a quantity of visible items is reported incorrectly
other - a hallucination that fits none of the above

PROBCLASS is exactly one of prob1..prob3 (never "None"): how likely a panel of three human annotators would agree the span is hallucinated (higher = clearer):
prob1 - ~0.3333 (weak) , prob2 - ~0.6667 , prob3 - ~1.0 (blatant).

How to annotate:
- Be thorough: check every claim against the image (colors, counts, object/species identity, brands, text, spatial relations, materials, details) and MARK anything not clearly supported.
- Mark a doubtful span with low-confidence (prob1) class rather than leaving unmarked.
- Mark the SMALLEST text carrying the error (a word or short phrase), not the whole sentence.
- Tags may nest (in rare cases), but never partially cross, never repeat the same tag on the same text, and never wrap markdown like **.
- Leave the response untagged ONLY if every claim is supported.
- Keep the response in its original language.
"""
#- Mark the SMALLEST text carrying the error (a word or short phrase), not the whole sentence.
#- Mark a doubtful span with low-confidence (prob1) class rather than leaving unmarked.


def get_system(no_labels=False):
    return SYSTEM_NO_LABELS if no_labels else SYSTEM_WITH_LABELS

def strip_label_attrs(inline):
    """
    Convert:
      <hall label="invention" prob="prob1">text</hall>
    to:
      <hall prob="prob1">text</hall>
    """
    if not inline:
        return inline

    inline = re.sub(r'(<hall\b[^>]*?)\s+label="[^"]*"', r'\1', inline)
    inline = re.sub(r'<hall\s+', '<hall ', inline)
    return inline

def parse_spans(inline):
    """Nesting-aware -> list of (label, prob, length_in_chars)."""
    clean, stack, spans = [], [], []
    i, L = 0, len(inline)
    while i < L:
        if inline.startswith('<hall', i):
            m = _OPEN.match(inline, i)
            if m:
                lm, pm = _LAB.search(m.group(1)), _PRB.search(m.group(1))
                stack.append((len(clean), lm.group(1) if lm else None, pm.group(1) if pm else None))
                i = m.end(); continue
        if inline.startswith(_CLOSE, i):
            if stack:
                st, lab, pr = stack.pop()
                spans.append((lab, pr, len(clean) - st))
            i += len(_CLOSE); continue
        clean.append(inline[i]); i += 1
    return spans


def _has_spans(o, inline_field):
    return '<hall' in (o.get(inline_field) or '')


def _candidate(o, inline_field, lo, hi, strict=True):
    """Return (o, labelset, probset) if the example qualifies, else None.
    strict: >=2 spans, all spans lo..hi, all tags valid. relaxed: >=2 spans lo..hi
    but tolerate some invalid tags (coverage from valid ones only)."""
    sp = parse_spans(o.get(inline_field, ''))
    if len(sp) < MIN_HALL_SPANS:
        return None
    if not all(lo <= l <= hi for (_, _, l) in sp):
        return None
    valid = [(lab, pr) for (lab, pr, _) in sp if lab in LABELS and pr in PROBS]
    if strict and len(valid) != len(sp):
        return None
    if not valid:
        return None
    return (o, {lab for lab, pr in valid}, {pr for lab, pr in valid})


def _lang_candidates(by_lang, lang, inline_field, lo, hi, rng):
    cands = [c for o in by_lang.get(lang, []) if (c := _candidate(o, inline_field, lo, hi, True))]
    if not cands:
        cands = [c for o in by_lang.get(lang, []) if (c := _candidate(o, inline_field, lo, hi, False))]
    rng.shuffle(cands)
    return cands


def select_fewshot(by_lang, langs=LANGS, lo=MIN_SPAN, hi=MAX_SPAN, seed=0,
                   inline_field='response_inline'):
    """Compute the fixed 5-shot set once (cached)."""
    global _FIXED_CACHE
    if _FIXED_CACHE is not None:
        return _FIXED_CACHE

    rng = random.Random(seed)
    cand = {lang: _lang_candidates(by_lang, lang, inline_field, lo, hi, rng) for lang in langs}

    covered_l, covered_p, selected, taken_lang = set(), set(), [], set()
    # greedy: repeatedly pick the (language, example) with max new coverage
    while len(taken_lang) < len(langs):
        best, best_gain, best_lang = None, -1, None
        for lang in langs:
            if lang in taken_lang:
                continue
            for (o, ls, ps) in cand[lang]:
                gain = len(ls - covered_l)
                if gain > best_gain:
                    best, best_gain, best_lang = (o, ls, ps), gain, lang
        if best is None:                       # some languages have no candidate at all
            for lang in langs:
                if lang not in taken_lang:
                    sys.stderr.write(f'[fewshot] WARNING: no >=2-span example in [{lo},{hi}] for "{lang}"\n')
                    taken_lang.add(lang)
            break
        selected.append(best[0]); covered_l |= best[1]; covered_p |= best[2]; taken_lang.add(best_lang)

    # 5th: an example with NO labels (fully-supported response)
    empty = None
    for lang in langs:
        for o in by_lang.get(lang, []):
            if not _has_spans(o, inline_field):
                empty = o; break
        if empty:
            break
    if empty is not None:
        selected.append(empty)
    else:
        sys.stderr.write('[fewshot] WARNING: no zero-label example found for the 5th shot\n')

    miss_l = LABELS - covered_l
    if miss_l:
        sys.stderr.write(f'[fewshot] WARNING: uncovered labels={sorted(miss_l)} '
                         f'(one-per-language constraint prevented full label coverage)\n')
    else:
        sys.stderr.write('[fewshot] selected 5 shots covering all labels\n')

    _FIXED_CACHE = selected
    return selected


def reset_fewshot_cache():
    global _FIXED_CACHE
    _FIXED_CACHE = None


def load_fewshot(paths, inline_field='response_inline', include_imagename=False, no_labels=False):
    """
    Load retrieved few-shot JSONL.

    Expected row format:
      {
        "id": eval_id,
        "eval_language": "...",
        "shots": [
          {
            "id": train_id,
            "language": "...",
            "prompt": "...",
            "image_name": "...",
            "response": "...",
            "response_inline": "...",
            "labels": [...]
          }
        ]
      }

    Returns:
      {
        "__mode__": "retrieved_by_id",
        "by_id": {
          eval_id: [shot1, shot2, ...]
        }
      }
    """
    if isinstance(paths, str):
        paths = [paths]

    by_id = {}

    for path in paths:
        if not path:
            continue

        with open(path, encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                o = json.loads(line)

                # This loader is for retrieved few-shot files.
                if not isinstance(o.get('shots'), list):
                    sys.stderr.write(
                        f'[fewshot] WARNING: skipping non-retrieved row '
                        f'in {path}:{line_no}\n'
                    )
                    continue

                eval_id = o.get('id')
                if eval_id is None:
                    sys.stderr.write(
                        f'[fewshot] WARNING: retrieved row without id '
                        f'in {path}:{line_no}; skipping\n'
                    )
                    continue

                shots = []
                for s in o.get('shots', []):
                    s = dict(s)

                    # Ensure fields expected by build_messages().
                    s.setdefault('response', '')
                    s.setdefault(inline_field, s.get('response_inline', ''))
                    s.setdefault('response_inline', s.get(inline_field, ''))
                    s.setdefault('labels', [])

                    if no_labels:
                        s[inline_field] = strip_label_attrs(s.get(inline_field, ''))
                        s['response_inline'] = strip_label_attrs(s.get('response_inline', ''))

                    shots.append(s)

                by_id[str(eval_id)] = shots

    sys.stderr.write(f'[fewshot] loaded retrieved shots for {len(by_id)} eval ids\n')

    return {
        '__mode__': 'retrieved_by_id',
        'by_id': by_id,
    }


def pick_shots(by_lang, language=None, k=5, seed=0,
               inline_field='response_inline', eval_id=None):
    """
    Pick retrieved shots by eval_id.

    language and seed are kept only for compatibility with run_inference.py.
    """
    if not isinstance(by_lang, dict) or by_lang.get('__mode__') != 'retrieved_by_id':
        sys.stderr.write(
            '[fewshot] WARNING: expected retrieved few-shot dictionary. '
            'Returning no shots.\n'
        )
        return []

    if eval_id is None:
        sys.stderr.write(
            '[fewshot] WARNING: eval_id was not passed to pick_shots(). '
            'Returning no shots.\n'
        )
        return []

    shots = by_lang.get('by_id', {}).get(str(eval_id), [])

    if not shots:
        sys.stderr.write(
            f'[fewshot] WARNING: no retrieved shots found for eval_id={eval_id}\n'
        )
        return []

    def _shot_sort_key(s):
        rank = s.get('retrieval_rank')
        score = s.get('retrieval_score')

        if rank is not None:
            return (0, int(rank))

        if score is not None:
            return (1, -float(score))

        return (2, 999999)

    shots = sorted(shots, key=_shot_sort_key)

    if k is not None and k > 0:
        shots = shots[:k]

    # The untagged (no-hallucination) demo sorts last by retrieval rank, and a
    # fixed final position teaches "the last example is untagged" as a positional
    # rule — recency bias then pushes the model toward marking nothing. Move it
    # to a middle slot instead. Deterministic, so runs stay reproducible.
    if shots:
        def _is_empty(s):
            return '<hall' not in (s.get(inline_field) or '')
        empties = [s for s in shots if _is_empty(s)]
        tagged = [s for s in shots if not _is_empty(s)]
        if empties and tagged:
            pos = len(shots) // 2                      # slot 3 of 6
            shots = tagged[:pos] + empties + tagged[pos:]

    sys.stderr.write(
        f'[fewshot] eval_id={eval_id} using {len(shots)} retrieved shots\n'
    )

    return shots


def _user_turn(prompt, response, image_path=None):
    return {'role': 'user',
            'text': f'PROMPT: {prompt}\nRESPONSE: {response}',
            'images': [image_path] if image_path else []}


def build_messages(prompt, response, image_path, shots,
                   response_field='response', inline_field='response_inline',
                   fewshot_image_paths=None,
                   include_imagename=False,
                   eval_image_name=None,
                   no_labels=False):
    msgs = [{'role': 'system', 'text': get_system(no_labels), 'images': []}]
    for ex in shots:
        demo_img = (fewshot_image_paths or {}).get(ex.get('id'))
        if demo_img is None and fewshot_image_paths:
            demo_img = fewshot_image_paths.get(ex.get('image_name'))
        msgs.append(_user_turn(ex['prompt'], ex[response_field], demo_img))
        #msgs.append({'role': 'assistant', 'text': ex[inline_field], 'images': []})
        assistant_inline = ex[inline_field]
        if no_labels:
            assistant_inline = strip_label_attrs(assistant_inline)

        msgs.append({'role': 'assistant', 'text': assistant_inline, 'images': []})
    msgs.append(_user_turn(prompt, response, image_path))
    return msgs

################################### REVIEW SYSTEM ###################################
REVIEW_SYSTEM = """You are reviewing a hallucination-span annotation.

You are given an IMAGE, the original PROMPT, the original RESPONSE, and a FIRST ANNOTATION. The IMAGE is the primary evidence.

Your job is to improve recall by adding missed hallucinated spans. Keep existing correct tags. Return the original RESPONSE text exactly, character for character, with <hall> tags. Do not rewrite, translate, explain, or add comments. Output only the final annotated response.

Use prob1 for weak unsupported spans, prob2 for likely hallucinations, and prob3 for very clear hallucinations.
"""

def build_review_messages(prompt, response, image_path, first_inline,
                          include_imagename=False, image_name=None,
                          language="en", no_labels=False):
    if no_labels:
        first_inline = strip_label_attrs(first_inline)
    return [
        {'role': 'system', 'text': REVIEW_SYSTEM, 'images': []},
        {
            'role': 'user',
            'text': (
                f'PROMPT: {prompt}\n'
                f'ORIGINAL RESPONSE: {response}\n\n'
                f'FIRST ANNOTATION:\n{first_inline}\n\n'
                'Review the first annotation and add any missed hallucinated spans. '
                'Return only the final annotated original response.'
            ),
            'images': [image_path] if image_path else []
        }
    ]