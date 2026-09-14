#!/usr/bin/env python3
"""
run_inference.py — few-shot inline annotation for ONE model over MANY input files.

Inputs may already contain inline annotation (gold `response_inline`); if so, the
plain response is recovered by stripping tags, so the same files work as input.
Few-shot demos are pooled from several train files and sampled per language so
all four languages are covered.

Writes:  <output-dir>/<model>/<inputbase>.inline.jsonl   (+ .stats.json)
"""
import argparse
import json
import os
import sys
import re

from contextlib import nullcontext
from client import VLMClient, load_models
from prompt_builder import build_messages, build_review_messages, load_fewshot, pick_shots
from verify_repair import verify_and_repair, strip_tags


def plain_response(o, response_field, inline_field):
    if o.get(response_field):
        return o[response_field]
    if o.get(inline_field):
        return strip_tags(o[inline_field])
    raise KeyError(f'row {o.get("id")} has neither {response_field} nor {inline_field}')

_TAG_OPEN_RE = re.compile(r'<hall\b([^>]*)>')
_ATTR_LABEL_RE = re.compile(r'label\s*=\s*"([^"]*)"')
_ATTR_PROB_RE = re.compile(r'prob\s*=\s*"([^"]*)"')
_TAG_CLOSE = '</hall>'


def extract_hall_spans(inline):
    """
    Extract nested-aware hall spans as plain-text offsets.

    Returns:
      [
        {
          "start": int,
          "end": int,
          "label": str,
          "prob": str,
          "text": str
        }
      ]
    """
    spans = []
    stack = []
    plain_chars = []

    i = 0
    while i < len(inline):
        m = _TAG_OPEN_RE.match(inline, i)
        if m:
            attrs = m.group(1)
            lm = _ATTR_LABEL_RE.search(attrs)
            pm = _ATTR_PROB_RE.search(attrs)

            stack.append({
                "start": len(plain_chars),
                "label": lm.group(1) if lm else None,
                "prob": pm.group(1) if pm else None,
            })
            i = m.end()
            continue

        if inline.startswith(_TAG_CLOSE, i):
            if stack:
                st = stack.pop()
                start = st["start"]
                end = len(plain_chars)
                spans.append({
                    "start": start,
                    "end": end,
                    "label": st["label"],
                    "prob": st["prob"],
                    "text": ''.join(plain_chars[start:end]),
                })
            i += len(_TAG_CLOSE)
            continue

        plain_chars.append(inline[i])
        i += 1

    return spans


def span_key(span):
    return (
        span["start"],
        span["end"],
        span["label"],
        span["prob"],
        span["text"],
    )


def review_change_counts(first_inline, final_inline):
    """
    corrected = number of first-pass spans changed/removed by review.
    total = number of final spans after correction.

    Example:
      first: one broad span
      final: three shorter spans
      -> corrected=1, total=3
    """
    first_spans = extract_hall_spans(first_inline)
    final_spans = extract_hall_spans(final_inline)

    final_keys = {span_key(s) for s in final_spans}

    corrected = sum(
        1 for s in first_spans
        if span_key(s) not in final_keys
    )

    return {
        "corrected": corrected,
        "total": len(final_spans),          # final labels after correction
        "first_tags": len(first_spans),
        "final_tags": len(final_spans),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--models-json', default='models.json')
    ap.add_argument('--inputs', nargs='+', required=True, help='one or more data JSONL files')
    ap.add_argument('--image-dir', required=True)
    ap.add_argument('--train-files', nargs='*', default=[], help='inline train files for few-shot')
    ap.add_argument('--num-shots', type=int, default=4)
    ap.add_argument('--fewshot-images', choices=['none', 'include'], default='none')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--response-field', default='response')
    ap.add_argument('--inline-field', default='response_inline')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--start', type=int, default=0, help='0-based input sample index to start from. Example: --start 250 skips samples 0..249.')
    ap.add_argument('--review-pass', action='store_true', help='Run a second pass to add missed hallucination spans.')
    ap.add_argument('--include_imagename', action='store_true', dest='include_imagename', help='Prefix PROMPT with [IMAGE: ...] using meaningful text extracted from image_name.')
    ap.add_argument('--no-labels', action='store_true', help='Use <hall prob="..."> tags without label attributes.')
    args = ap.parse_args()

    models = load_models(args.models_json)
    client = VLMClient(args.model, models)
    supports_images = models[args.model].get('supports_images', True)
    #by_lang = load_fewshot(args.train_files, args.inline_field, include_imagename=args.include_imagename) if args.train_files else {}
    by_lang = load_fewshot(
        args.train_files,
        args.inline_field,
        include_imagename=args.include_imagename,
        no_labels=args.no_labels
    ) if args.train_files else {}

    mdir = os.path.join(args.output_dir, args.model)
    os.makedirs(mdir, exist_ok=True)

    for data_path in args.inputs:
        base = os.path.splitext(os.path.basename(data_path))[0]
        #total_rows = sum(1 for _ in open(data_path, encoding='utf-8'))   # <-- ADD THIS
        total_rows = sum(1 for line in open(data_path, encoding='utf-8') if line.strip())

        if args.start >= total_rows:
            sys.stderr.write(
                f'[skip] {data_path}: start={args.start} >= total_rows={total_rows}\n'
            )
            continue

        end_index = total_rows if args.limit is None else min(total_rows, args.start + args.limit)
        target_rows = end_index - args.start
        out_path = os.path.join(mdir, f'{base}.inline.jsonl')
        corrected_path = os.path.join(mdir, f'{base}.inline.corrected.jsonl') if args.review_pass else None
        stats = {
            'model': args.model,
            'input': data_path,
            'start': args.start,
            'limit': args.limit,
            'total': 0,
            'image_missing': 0,
            'api_errors': 0,
            'ok': 0,
            'repaired': 0,
            'failed': 0,
            'review_corrected_spans': 0,
            'review_final_spans': 0,
            'review_first_spans': 0,
            'review_changed_rows': 0,
        }
        corrected_cm = open(corrected_path, 'w', encoding='utf-8') if args.review_pass else nullcontext(None)
        with open(data_path, encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout, corrected_cm as fcorr:
            sample_idx = -1

            for line in fin:
                line = line.strip()
                if not line:
                    continue

                sample_idx += 1

                if sample_idx < args.start:
                    continue

                if args.limit is not None and stats['total'] >= args.limit:
                    break

                o = json.loads(line)
                stats['total'] += 1
                lang = o.get('language', 'en')
                resp = plain_response(o, args.response_field, args.inline_field)

                img_path = os.path.join(args.image_dir, o['image_name'])
                image_path = None
                if supports_images and os.path.exists(img_path):
                    image_path = img_path
                elif supports_images:
                    stats['image_missing'] += 1

                #shots = pick_shots(by_lang, lang, args.num_shots, seed=args.seed) if by_lang else []
                shots = pick_shots(
                    by_lang,
                    lang,
                    args.num_shots,
                    seed=args.seed,
                    inline_field=args.inline_field,
                    eval_id=o.get('id')
                ) if by_lang else []
                fewshot_paths = None
                if args.fewshot_images == 'include' and shots:
                    fewshot_paths = {}
                    for ex in shots:
                        p = os.path.join(args.image_dir, ex.get('image_name', ''))
                        if os.path.exists(p):
                            fewshot_paths[ex.get('id')] = p

                messages = build_messages(
                    o['prompt'],
                    resp,
                    image_path,
                    shots,
                    response_field=args.response_field,
                    inline_field=args.inline_field,
                    fewshot_image_paths=fewshot_paths,
                    include_imagename=args.include_imagename,
                    eval_image_name=o.get('image_name'),
                    no_labels=args.no_labels
                )

                try:
                    raw = client.chat(messages)
                except Exception as e:  # noqa: BLE001
                    stats['api_errors'] += 1
                    raw = resp
                    sys.stderr.write(f'[api-error] id={o.get("id")}: {e}\n')

                first_inline, first_pass_status = verify_and_repair(raw, resp, no_labels=args.no_labels)

                final_inline = first_inline
                final_status = first_pass_status

                raw_review = None
                status_review = None
                review_counts = None

                if args.review_pass:
                    review_messages = build_review_messages(
                        o['prompt'],
                        resp,
                        image_path,
                        first_inline,
                        include_imagename=args.include_imagename,
                        image_name=o.get('image_name'),
                        language=lang,
                        no_labels=args.no_labels,
                    )

                    try:
                        raw_review = client.chat(review_messages)
                        final_inline, final_status = verify_and_repair(raw_review, resp, no_labels=args.no_labels)
                        status_review = final_status

                    except Exception as e:  # noqa: BLE001
                        stats['api_errors'] += 1
                        sys.stderr.write(f'[review-api-error] id={o.get("id")}: {e}\n')

                    review_counts = review_change_counts(first_inline, final_inline)
                    
                    stats['review_corrected_spans'] += review_counts['corrected']
                    stats['review_final_spans'] += review_counts['final_tags']
                    stats['review_first_spans'] += review_counts['first_tags']
                    if review_counts['corrected'] > 0:
                        stats['review_changed_rows'] += 1

                status_for_stats = final_status if args.review_pass else first_pass_status
                stats['ok' if status_for_stats in ('ok', 'ok_empty') else status_for_stats] = \
                    stats.get('ok' if status_for_stats in ('ok', 'ok_empty') else status_for_stats, 0) + 1

                out_first = {k: o[k] for k in ('id', 'language', 'prompt', 'image_name') if k in o}
                out_first['response'] = resp
                out_first[args.inline_field] = first_inline
                out_first['annotation_status'] = first_pass_status
                out_first['first_pass_status'] = first_pass_status

                fout.write(json.dumps(out_first, ensure_ascii=False) + '\n')
                fout.flush()

                if args.review_pass:
                    out_corr = dict(out_first)
                    out_corr[args.inline_field] = final_inline
                    out_corr['annotation_status'] = final_status

                    fcorr.write(json.dumps(out_corr, ensure_ascii=False) + '\n')
                    fcorr.flush()
                                                              # <-- ADD
                if args.review_pass:
                    print(
                        f'[{args.model}] {base} #{stats["total"]}/{target_rows} '
                        f'idx={sample_idx} id={o.get("id")} status={final_status} '
                        f'tags={review_counts["first_tags"]} '
                        f'corrected={review_counts["corrected"]} '
                        f'total={review_counts["total"]}',
                        flush=True
                    )
                else:
                    print(
                        f'[{args.model}] {base} #{stats["total"]}/{target_rows} '
                        f'idx={sample_idx} id={o.get("id")} status={first_pass_status} '
                        f'tags={first_inline.count("<hall")}',
                        flush=True
                    )

        with open(os.path.splitext(out_path)[0] + '.stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f'[{args.model}] {base}: {stats}')


if __name__ == '__main__':
    main()
