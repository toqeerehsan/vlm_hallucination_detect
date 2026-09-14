#!/usr/bin/env python3
"""
aggregate.py — build the committee prediction from N per-model offset files.

For every character of each response:
  votes[c]      = how many models flagged c
  committee_prob[c] = mean over models of (that model's prob for c, else 0)
                      -> blends AGREEMENT (how many) and CONFIDENCE (each model's prob)
  category[c]   = label with the largest summed prob among flagging models
  mask[c]       = votes[c] >= --min-votes           (IoU track: inclusive by default)

Optionally a calibration map (from fit_calibration.py) is applied to committee_prob
so the emitted probabilities match dev-empirical agreement (correlation track).

Writes, per input basename, into <output-dir>:
  <base>.inline.jsonl    committee tags on the exact response
  <base>.offsets.jsonl   {start,end,prob,label} runs  (this is what you score)
"""
import argparse
import json
import os
from collections import defaultdict

from verify_repair import emit_nested


def load_value_to_class(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    d = d.get('prob_to_class', d)
    pairs = sorted((float(k), v) for k, v in d.items())

    def to_class(p):
        return min(pairs, key=lambda kv: abs(kv[0] - p))[1]
    return to_class


def apply_calibration(p, calib):
    if not calib:
        return p
    xs, ys = calib['x'], calib['y']
    if p <= xs[0]:
        return ys[0]
    if p >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if p <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            t = 0 if x1 == x0 else (p - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def per_model_char_arrays(labels, n):
    """Return (char_prob, char_label): for each char, the max-prob covering span."""
    cp = [0.0] * n
    cl = [None] * n
    for sp in labels:
        s, e = int(sp['start']), int(sp['end'])
        p = float(sp.get('prob') or 0.0)
        lab = sp.get('label')
        for c in range(max(0, s), min(e, n)):
            if p > cp[c]:
                cp[c] = p
                cl[c] = lab
    return cp, cl


def runs_from_chars(mask, prob, label, to_class):
    """RLE over (label, rounded prob) on masked chars -> offset spans + inline span dicts."""
    n = len(mask)
    offsets, inline_spans = [], []
    i = 0
    idx = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        key = (label[i], round(prob[i], 6))
        while j < n and mask[j] and (label[j], round(prob[j], 6)) == key:
            j += 1
        offsets.append({'start': i, 'end': j, 'prob': round(prob[i], 6), 'label': label[i]})
        inline_spans.append({'start': i, 'end': j, 'label': label[i] or 'other',
                             'pc': to_class(prob[i]), 'idx': idx})
        idx += 1
        i = j
    return offsets, inline_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dir containing <model>/<base>.offsets.jsonl')
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--inputs', nargs='+', required=True, help='original input files (for basenames)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--prob-value-to-class', required=True)
    ap.add_argument('--min-votes', type=int, default=1)
    ap.add_argument('--calibration', default=None, help='optional calibration json from fit_calibration.py')
    args = ap.parse_args()

    to_class = load_value_to_class(args.prob_value_to_class)
    calib = None
    if args.calibration and os.path.exists(args.calibration):
        with open(args.calibration, encoding='utf-8') as f:
            calib = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    n_models = len(args.models)
    grand = {'models': args.models, 'min_votes': args.min_votes, 'inputs': {}}

    for data_path in args.inputs:
        base = os.path.splitext(os.path.basename(data_path))[0]
        # load each model's offsets for this base
        per_model = {}
        for m in args.models:
            fp = os.path.join(args.root, m, f'{base}.offsets.jsonl')
            if not os.path.exists(fp):
                continue
            d = {}
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        o = json.loads(line)
                        d[o['id']] = o
            per_model[m] = d
        if not per_model:
            grand['inputs'][base] = {'error': 'no per-model offsets found'}
            continue

        # union of ids, in the order of the first available model
        first = next(iter(per_model.values()))
        ids = list(first.keys())
        seen = set(ids)
        for d in per_model.values():
            for i in d:
                if i not in seen:
                    ids.append(i); seen.add(i)

        out_inline = os.path.join(args.output_dir, f'{base}.inline.jsonl')
        out_off = os.path.join(args.output_dir, f'{base}.offsets.jsonl')
        counts = {'samples': 0, 'spans': 0, 'chars_flagged': 0}

        with open(out_inline, 'w', encoding='utf-8') as fi, open(out_off, 'w', encoding='utf-8') as fo:
            for _id in ids:
                # response text: take from any model that has this id
                resp = None
                meta = None
                for m in args.models:
                    if m in per_model and _id in per_model[m]:
                        resp = per_model[m][_id]['response']
                        meta = per_model[m][_id]
                        break
                if resp is None:
                    continue
                n = len(resp)
                votes = [0] * n
                probsum = [0.0] * n
                lab_w = [defaultdict(float) for _ in range(n)]

                for m in args.models:
                    d = per_model.get(m)
                    if not d or _id not in d:
                        continue
                    cp, cl = per_model_char_arrays(d[_id]['labels'], n)
                    for c in range(n):
                        if cp[c] > 0:
                            votes[c] += 1
                            probsum[c] += cp[c]
                            lab_w[c][cl[c]] += cp[c]

                mask = [votes[c] >= args.min_votes for c in range(n)]
                prob = [apply_calibration(probsum[c] / n_models, calib) if mask[c] else 0.0
                        for c in range(n)]
                label = [max(lab_w[c].items(), key=lambda kv: kv[1])[0] if lab_w[c] else None
                         for c in range(n)]

                offsets, inline_spans = runs_from_chars(mask, prob, label, to_class)
                inline = emit_nested(resp, inline_spans)

                counts['samples'] += 1
                counts['spans'] += len(offsets)
                counts['chars_flagged'] += sum(1 for x in mask if x)

                base_row = {k: meta[k] for k in ('id', 'language', 'prompt', 'image_name') if k in meta}
                fi.write(json.dumps({**base_row, 'response': resp,
                                     'response_inline': inline}, ensure_ascii=False) + '\n')
                fo.write(json.dumps({**base_row, 'response': resp,
                                     'labels': offsets}, ensure_ascii=False) + '\n')
        grand['inputs'][base] = counts
        print(f'[committee] {base}: {counts}')

    with open(os.path.join(args.output_dir, 'committee_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(grand, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
