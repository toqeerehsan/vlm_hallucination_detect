#!/usr/bin/env python3
"""
inline_to_offsets.py

Convert inline <hall> annotations back into offset labels for evaluation.

    <hall label="mischaracterization" prob="prob1">cables</hall>
        ->  {"start": .., "end": .., "prob": 0.25, "label": "mischaracterization"}

Offsets are computed on the TAG-STRIPPED (clean) text: strip every tag, and the
start/end fall out of the clean-text cursor.  `end` is exclusive.

This script is deliberately lenient because it also runs on raw model output:
malformed / unclosed / unknown tags are repaired-or-skipped and counted, never
crash.  If a gold `response` field is present it is used to assert the clean
text matches exactly (a mismatch means the model altered the text).

Usage:
    python inline_to_offsets.py \
        --prob-map prob_class_to_value.json \
        --output-dir eval/dev \
        --inline-field response_inline \
        -- file1.jsonl file2.jsonl
"""
import argparse
import json
import os
import re
from collections import Counter

OPEN_RE = re.compile(r'<hall\b([^>]*?)>')
ATTR_LABEL = re.compile(r'label\s*=\s*"([^"]*)"')
ATTR_PROB = re.compile(r'prob\s*=\s*"([^"]*)"')
CLOSE = '</hall>'


def parse_inline(inline, class_to_value, agg):
    """Return (clean_text, labels[]). labels are {start,end,prob,label}."""
    clean, labels = [], []
    stack = []  # (start_offset, label, prob_class); depth should stay <=1
    i, L = 0, len(inline)
    while i < L:
        if inline.startswith('<hall', i):
            m = OPEN_RE.match(inline, i)
            if m:
                attrs = m.group(1)
                lm, pm = ATTR_LABEL.search(attrs), ATTR_PROB.search(attrs)
                label = lm.group(1) if lm else None
                pc = pm.group(1) if pm else None
                if label is None or pc is None:
                    agg['malformed_open'] += 1
                stack.append((len(clean), label, pc))  # nesting supported (LIFO)
                i = m.end()
                continue
        if inline.startswith(CLOSE, i):
            if stack:
                _emit(stack.pop(), len(clean), labels, class_to_value, agg)
            else:
                agg['stray_close'] += 1
            i += len(CLOSE)
            continue
        clean.append(inline[i])
        i += 1
    while stack:  # unclosed at EOF -> close at end
        agg['unclosed_eof'] += 1
        _emit(stack.pop(), len(clean), labels, class_to_value, agg)
    labels.sort(key=lambda d: (d['start'], d['end']))
    return ''.join(clean), labels


def _emit(open_rec, end, labels, class_to_value, agg):
    start, label, pc = open_rec
    if end <= start:
        agg['empty_span'] += 1
        return
    val = class_to_value.get(pc)
    if val is None:
        agg['unknown_prob_class'] += 1
        agg['unknown_prob_class_examples'].append(pc)
        val = None
    if label is not None:
        agg['label_counts'][label] += 1
    labels.append({'start': start, 'end': end, 'prob': val, 'label': label})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob-map', required=True, help='class->value JSON (prob_class_to_value.json)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--inline-field', default='response_inline')
    ap.add_argument('--response-field', default='response', help='if present, verify clean text matches')
    ap.add_argument('--labels-field', default='labels')
    ap.add_argument('--stats-out', default=None)
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()

    with open(args.prob_map, encoding='utf-8') as f:
        pm = json.load(f)
    class_to_value = pm.get('class_to_prob', pm)
    class_to_value = {k: float(v) for k, v in class_to_value.items()}

    os.makedirs(args.output_dir, exist_ok=True)
    agg = {
        'files': 0, 'samples': 0, 'tags_parsed': 0,
        'malformed_open': 0, 'stray_close': 0, 'auto_closed': 0,
        'unclosed_eof': 0, 'empty_span': 0, 'unknown_prob_class': 0,
        'text_mismatch': 0, 'label_counts': Counter(),
        'unknown_prob_class_examples': [], 'mismatch_examples': [],
    }

    for path in args.files:
        agg['files'] += 1
        base = os.path.basename(path)
        out_path = os.path.join(args.output_dir, base)
        with open(path, encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                agg['samples'] += 1
                clean, labels = parse_inline(obj[args.inline_field], class_to_value, agg)
                agg['tags_parsed'] += len(labels)
                gold = obj.get(args.response_field)
                if gold is not None and clean != gold:
                    agg['text_mismatch'] += 1
                    if len(agg['mismatch_examples']) < 25:
                        agg['mismatch_examples'].append(obj.get('id'))
                out = dict(obj)
                out[args.labels_field] = labels
                out['response'] = clean  # reconstructed clean text
                fout.write(json.dumps(out, ensure_ascii=False) + '\n')

    agg['label_counts'] = dict(agg['label_counts'])
    stats_path = args.stats_out or os.path.join(args.output_dir, 'reverse_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
