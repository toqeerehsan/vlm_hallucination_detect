#!/usr/bin/env python3
"""
prepare_ft_splits.py — split ONE inline train file into three model-specific SFT
sets that REDUCE each model's weakness while keeping the committee diverse.

Diversity levers (from the individual score files):
  * gemma-4-31b        : over-marks (low precision) + weak OCR
        -> MORE negatives (learn restraint) + OCR-rich positives
  * qwen3.6-27b        : strong OCR/balance, but 'other'=0 and lower inv/mischar recall
        -> push 'other' + invention/mischaracterization ; DOWNWEIGHT ocr (diverge from gemma)
        -> moderate negatives
  * mistral-small-3.1-24b : high precision, severely under-marks (low recall)
        -> hallucination-DENSE positives ; FEW negatives (learn to mark)

Also: filter to samples whose spans are all MIN..MAX chars, and collapse the rare
4-annotator prob classes into the 3 common ones for cleaner training targets.

Output: <out-dir>/train_<modelkey>.jsonl  (fields: id, language, image_path,
prompt, response, response_inline) + split_stats.json
"""
import argparse
import json
import os
import re
from collections import defaultdict, Counter

_OPEN = re.compile(r'<hall\b([^>]*?)>')
_LAB = re.compile(r'label\s*=\s*"([^"]*)"')
_PRB = re.compile(r'prob\s*=\s*"([^"]*)"')
_CLOSE = '</hall>'

# collapse 6 prob classes -> 3 (0.25->0.333, 0.5->0.667, 0.75->0.667)
#PROB_COLLAPSE = {'prob1': 'prob2', 'prob3': 'prob4', 'prob5': 'prob4'}
PROB_COLLAPSE = {}

# per-model category weights (higher = feed more of this category) + negative share
MODEL_PROFILES = {
    'gemma-4-31b': {
        'cat_w': {'ocr': 2.5, 'miscounting': 1.0, 'invention': 1.0,
                  'mischaracterization': 1.0, 'other': 0.8},
        'neg_weight': 0.40},
    'qwen3.6-27b': {
        'cat_w': {'other': 2.8, 'ocr': 1.8, 'invention': 1.6,
                  'mischaracterization': 1.4, 'miscounting': 1.0},
        'neg_weight': 0.32},
    'mistral-small-3.1-24b': {
        'cat_w': {'ocr': 2.2, 'mischaracterization': 2.0, 'invention': 2.0,
                  'miscounting': 1.2, 'other': 1.2},
        'neg_weight': 0.28},
}
MODEL_PROFILES = {
    'gemma-4-31b': {
        'cat_w': {'ocr': 1.0, 'miscounting': 1.0, 'invention': 1.0,
                  'mischaracterization': 1.0, 'other': 1.0},
        'neg_weight': 0.3333},
    'qwen3.6-27b': {
        'cat_w': {'ocr': 1.0, 'miscounting': 1.0, 'invention': 1.0,
                  'mischaracterization': 1.0, 'other': 1.0},
        'neg_weight': 0.3333},
    'mistral-small-3.1-24b': {
        'cat_w': {'ocr': 1.0, 'miscounting': 1.0, 'invention': 1.0,
                  'mischaracterization': 1.0, 'other': 1.0},
        'neg_weight': 0.3333},
}
MODELS = list(MODEL_PROFILES)

def parse_spans(inline):
    """-> list of (label, prob_class, length)."""
    clean, stack, out = [], [], []
    i, L = 0, len(inline or '')
    while i < L:
        if inline.startswith('<hall', i):
            m = _OPEN.match(inline, i)
            if m:
                lm, pm = _LAB.search(m.group(1)), _PRB.search(m.group(1))
                stack.append((len(clean), lm.group(1) if lm else None, pm.group(1) if pm else None))
                i = m.end(); continue
        if inline.startswith(_CLOSE, i):
            if stack:
                s, lab, pc = stack.pop(); out.append((lab, pc, len(clean) - s))
            i += len(_CLOSE); continue
        clean.append(inline[i]); i += 1
    return out


def strip_tags(inline):
    return re.sub(r'</?hall\b[^>]*>', '', inline or '')


def collapse_probs(inline):
    def repl(m):
        pc = m.group(1)
        return f'prob="{PROB_COLLAPSE.get(pc, pc)}"'
    return re.sub(r'prob\s*=\s*"([^"]*)"', repl, inline)


def sample_score(cats, cat_w):
    return sum(cat_w.get(c, 1.0) for c in cats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-inline', required=True)
    ap.add_argument('--image-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--min-span', type=int, default=1)
    ap.add_argument('--max-span', type=int, default=128)
    ap.add_argument('--collapse-probs', action='store_true', default=True)
    ap.add_argument('--no-collapse-probs', dest='collapse_probs', action='store_false')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--inline-field', default='response_inline')
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    positives, negatives = [], []
    n_total = n_droplen = 0
    with open(args.train_inline, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            n_total += 1
            inline = o.get(args.inline_field, '')
            spans = parse_spans(inline)
            if spans and any(not (args.min_span <= L <= args.max_span) for (_, _, L) in spans):
                n_droplen += 1
                continue
            if args.collapse_probs:
                inline = collapse_probs(inline)
            rec = {
                'id': o.get('id'), 'language': o.get('language', 'en'),
                'image_path': os.path.join(args.image_dir, o.get('image_name', '')),
                'prompt': o.get('prompt', ''),
                'response': o.get('response') or strip_tags(inline),
                'response_inline': inline,
                '_cats': [lab for (lab, _, _) in spans if lab],
            }
            (negatives if not spans else positives).append(rec)

    # ---- assign POSITIVES per language (capacity-balanced greedy) so each model
    #      gets a proportional mix of en/fr/it/zh ----
    assign = {m: [] for m in MODELS}
    counts = {m: 0 for m in MODELS}
    by_lang_pos = defaultdict(list)
    for rec in positives:
        by_lang_pos[rec['language']].append(rec)
    for lang, recs in by_lang_pos.items():
        cap_l = len(recs) // len(MODELS) + 1        # equal share of THIS language per model
        counts_l = {m: 0 for m in MODELS}
        scored = []
        for rec in recs:
            cats = rec['_cats']
            s = {m: sample_score(cats, MODEL_PROFILES[m]['cat_w']) for m in MODELS}
            best = max(s.values()); second = sorted(s.values())[-2] if len(s) > 1 else 0
            scored.append((best - second, s, rec))
        scored.sort(key=lambda x: -x[0])
        for _, s, rec in scored:
            order = sorted(MODELS, key=lambda m: -s[m])
            for m in order:
                if counts_l[m] < cap_l:
                    assign[m].append(rec); counts_l[m] += 1; counts[m] += 1
                    break
            else:
                m = min(MODELS, key=lambda m: counts_l[m])
                assign[m].append(rec); counts_l[m] += 1; counts[m] += 1

    # ---- assign NEGATIVES per language by neg_weight share (gemma most, mistral least) ----
    wsum = sum(MODEL_PROFILES[m]['neg_weight'] for m in MODELS)
    by_lang_neg = defaultdict(list)
    for rec in negatives:
        by_lang_neg[rec['language']].append(rec)
    for lang, recs in by_lang_neg.items():
        rng.shuffle(recs)
        targets = {m: int(len(recs) * MODEL_PROFILES[m]['neg_weight'] / wsum) for m in MODELS}
        idx = 0
        for m in MODELS:
            for _ in range(targets[m]):
                if idx < len(recs):
                    assign[m].append(recs[idx]); idx += 1
        rr = 0
        while idx < len(recs):
            assign[MODELS[rr % len(MODELS)]].append(recs[idx]); idx += 1; rr += 1

    # ---- write + stats ----
    stats = {'source_total': n_total, 'dropped_span_len': n_droplen,
             'positives': len(positives), 'negatives': len(negatives), 'per_model': {}}
    for m in MODELS:
        recs = assign[m]
        rng.shuffle(recs)
        cat_c = Counter()
        lang_c = Counter()
        n_neg = 0
        for r in recs:
            lang_c[r['language']] += 1
            if r['_cats']:
                cat_c.update(r['_cats'])
            else:
                n_neg += 1
        with open(os.path.join(args.out_dir, f'train_{m}.jsonl'), 'w', encoding='utf-8') as fo:
            for r in recs:
                r2 = {k: r[k] for k in ('id', 'language', 'image_path', 'prompt', 'response', 'response_inline')}
                fo.write(json.dumps(r2, ensure_ascii=False) + '\n')
        stats['per_model'][m] = {
            'size': len(recs), 'negatives': n_neg,
            'neg_frac': round(n_neg / max(len(recs), 1), 3),
            'languages': dict(lang_c),
            'span_categories': dict(cat_c)}
    with open(os.path.join(args.out_dir, 'split_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""USAGE
python prepare_ft_splits.py \
  --train-inline converted/train/train.all.labeled.inline_3probs.jsonl \
  --image-dir shroom_data/all-images/shroom-vis-images \
  --out-dir fine-tuning/ft-splits \
  --min-span 1 --max-span 90
"""