#!/usr/bin/env python3
"""
retrieve_fewshots_unique_filtered.py

Retrieval-based few-shot selection for SHROOM-style JSONL files.

Key behavior:
  - Reads multiple train JSONL files and multiple eval JSONL files.
  - Eval labels are never used.
  - Same-language retrieval.
  - Filters train candidates so selected shots have hallucination spans whose
    character lengths are between --min-span-chars and --max-span-chars.
  - Retrieves unique (cleaned prompt + image_name) train examples.
  - Computes normalized image/response/prompt similarities per eval item.
  - Selects 5 diverse examples from top-k unique candidates.
  - Adds 1 random same-language eligible example.
  - Saves one JSONL output file.
  - Prints per-eval shot-count and label-class statistics after writing output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer


PLEASE_ELAB_RE = re.compile(r"\s*Please elaborate\.\s*$", flags=re.IGNORECASE)


def read_jsonl(path: str, require_train_labels: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_source_file"] = path
            obj["_source_line"] = line_no
            if require_train_labels and "labels" not in obj:
                obj["labels"] = []
            rows.append(obj)
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_prompt(prompt: str) -> str:
    return PLEASE_ELAB_RE.sub("", prompt or "").strip()


def normalized_prompt_key(row: Dict[str, Any]) -> str:
    return clean_prompt(str(row.get("prompt", ""))).lower().strip()


def prompt_image_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (normalized_prompt_key(row), str(row.get("image_name", "")).strip())


def strip_inline_tags(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<hall\b[^>]*>", "", text)
    text = text.replace("</hall>", "")
    return text

PROB6_TO_PROB3 = {
    "prob1": "prob1",
    "prob2": "prob1",
    "prob3": "prob2",
    "prob4": "prob2",
    "prob5": "prob2",
    "prob6": "prob3",
}

PROB_VALUE_TO_PROB3 = {
    0.25: "prob1",
    0.3333333333: "prob1",
    0.5: "prob2",
    0.6666666667: "prob2",
    0.75: "prob2",
    1.0: "prob3",
}

def map_prob_class_to_3(prob_class: Any) -> str:
    """
    Map prob1..prob6 to prob1..prob3:
      prob1, prob2 -> prob1
      prob3, prob4, prob5 -> prob2
      prob6 -> prob3
    """
    if prob_class is None:
        return "prob1"

    p = str(prob_class).strip()
    return PROB6_TO_PROB3.get(p, p if p in {"prob1", "prob2", "prob3"} else "prob1")


def map_prob_value_to_3(prob_value: Any) -> float:
    """
    Map numeric probability values to 3-class numeric values:
      0.25, 0.3333333333 -> 0.3333333333
      0.5, 0.6666666667, 0.75 -> 0.6666666667
      1.0 -> 1.0
    """
    try:
        v = float(prob_value)
    except Exception:
        return 0.3333333333

    if abs(v - 1.0) < 1e-6:
        return 1.0

    if v <= 0.3333333333 + 1e-6:
        return 0.3333333333

    # 0.5, 0.6667, 0.75 all become prob2 numeric value
    return 0.6666666667


def map_response_inline_probs_to_3(text: str) -> str:
    """
    Rewrite prob="prob1"..prob="prob6" inside response_inline to prob1..prob3.
    """
    if not text:
        return ""

    def repl(m):
        old_prob = m.group(1)
        new_prob = map_prob_class_to_3(old_prob)
        return f'prob="{new_prob}"'

    return re.sub(r'prob\s*=\s*"([^"]+)"', repl, text)


def map_labels_probs_to_3(labels: Any) -> List[Dict[str, Any]]:
    """
    Copy labels and map numeric prob values to 3-class numeric values.
    """
    if not isinstance(labels, list):
        return []

    out = []
    for lab in labels:
        if not isinstance(lab, dict):
            continue

        lab = dict(lab)
        lab["prob"] = map_prob_value_to_3(lab.get("prob"))
        out.append(lab)

    return out

def get_response(row: Dict[str, Any]) -> str:
    if row.get("response") is not None:
        return str(row.get("response", ""))
    if row.get("response_inline") is not None:
        return strip_inline_tags(str(row.get("response_inline", "")))
    return ""


def _as_int_or_none(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def valid_label_span_lengths(row: Dict[str, Any], min_span_chars: int, max_span_chars: int) -> List[int]:
    """Return lengths of hallucination spans whose character length is in range."""
    labels = row.get("labels") or []
    if not isinstance(labels, list):
        return []

    lengths: List[int] = []
    for lab in labels:
        if not isinstance(lab, dict) or not lab.get("label"):
            continue
        start = _as_int_or_none(lab.get("start"))
        end = _as_int_or_none(lab.get("end"))
        if start is None or end is None:
            continue
        span_len = end - start
        if min_span_chars <= span_len <= max_span_chars:
            lengths.append(span_len)
    return lengths


def all_label_spans_in_range(row: Dict[str, Any], min_span_chars: int, max_span_chars: int) -> bool:
    """Strict filter: every label span in the example must be in range."""
    labels = row.get("labels") or []
    if not isinstance(labels, list) or not labels:
        return False

    for lab in labels:
        if not isinstance(lab, dict) or not lab.get("label"):
            return False
        start = _as_int_or_none(lab.get("start"))
        end = _as_int_or_none(lab.get("end"))
        if start is None or end is None:
            return False
        span_len = end - start
        if span_len < min_span_chars or span_len > max_span_chars:
            return False
    return True


def valid_label_count(
    row: Dict[str, Any],
    min_span_chars: int,
    max_span_chars: int,
    span_filter_mode: str,
) -> int:
    """
    Count valid hallucination spans.

    span_filter_mode:
      - all: row is valid only if all labels are within the length range.
      - any: row is valid if it has enough in-range spans; other labels may exist.
    """
    labels = row.get("labels") or []
    if not isinstance(labels, list):
        return 0

    if span_filter_mode == "all":
        if not all_label_spans_in_range(row, min_span_chars, max_span_chars):
            return 0
        return len(labels)

    return len(valid_label_span_lengths(row, min_span_chars, max_span_chars))


def eligible_train_example(
    row: Dict[str, Any],
    min_label_spans: int,
    min_span_chars: int,
    max_span_chars: int,
    span_filter_mode: str,
) -> bool:
    return valid_label_count(row, min_span_chars, max_span_chars, span_filter_mode) >= min_label_spans

def is_empty_train_example(row: Dict[str, Any]) -> bool:
    """
    Empty / fully-supported few-shot example:
      - no offset labels
      - no inline <hall> tags
    """
    labels = row.get("labels") or []
    inline = str(row.get("response_inline", "") or "")

    return (
        isinstance(labels, list)
        and len(labels) == 0
        and "<hall" not in inline
    )

def label_names(row: Dict[str, Any]) -> List[str]:
    labels = row.get("labels") or []
    if not isinstance(labels, list):
        return []
    out: List[str] = []
    for lab in labels:
        if isinstance(lab, dict) and lab.get("label"):
            out.append(str(lab["label"]))
    return out


def stable_int(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def minmax_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = values.astype(np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    denom = vmax - vmin
    if denom < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / denom).astype(np.float32)


class TextEmbedder:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 1024, cache_dir: Optional[str] = None) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        all_vecs: List[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            batch = list(texts[start:start + batch_size])
            toks = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks)
            last_hidden = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_vecs.append(emb.detach().cpu().float().numpy())
        if not all_vecs:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(all_vecs).astype(np.float32)


class ImageEmbedder:
    def __init__(self, model_name: str, device: str = "auto", cache_dir: Optional[str] = None) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def _load_image(self, path: str) -> Optional[Image.Image]:
        try:
            img = Image.open(path)
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            return img.convert("RGB")
        except Exception as e:
            sys.stderr.write(f"[image-warning] Could not load {path}: {e}\n")
            return None

    def _to_feature_tensor(self, feats: Any) -> torch.Tensor:
        if isinstance(feats, torch.Tensor):
            return feats
        if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
            return feats.image_embeds
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            return feats.pooler_output
        if hasattr(feats, "last_hidden_state") and feats.last_hidden_state is not None:
            return feats.last_hidden_state[:, 0]
        if isinstance(feats, (tuple, list)) and feats and isinstance(feats[0], torch.Tensor):
            return feats[0]
        if isinstance(feats, dict):
            for key in ("image_embeds", "pooler_output", "last_hidden_state"):
                if key in feats and feats[key] is not None:
                    val = feats[key]
                    return val[:, 0] if key == "last_hidden_state" else val
        raise TypeError(f"Could not convert image model output to tensor: {type(feats)}")

    @torch.inference_mode()
    def encode_paths(self, paths: Sequence[Optional[str]], batch_size: int = 16) -> np.ndarray:
        valid_positions = [i for i, p in enumerate(paths) if p and os.path.exists(p)]
        if not valid_positions:
            return np.zeros((len(paths), 1), dtype=np.float32)
        result: Optional[np.ndarray] = None
        for start in tqdm(range(0, len(valid_positions), batch_size), desc="Encoding images"):
            pos_batch = valid_positions[start:start + batch_size]
            pil_images: List[Image.Image] = []
            kept_positions: List[int] = []
            for pos in pos_batch:
                img = self._load_image(paths[pos])  # type: ignore[arg-type]
                if img is not None:
                    pil_images.append(img)
                    kept_positions.append(pos)
            if not pil_images:
                continue
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
            if hasattr(self.model, "get_image_features"):
                feats = self.model.get_image_features(**inputs)
            elif hasattr(self.model, "vision_model"):
                feats = self.model.vision_model(**inputs)
            else:
                feats = self.model(**inputs)
            feats = self._to_feature_tensor(feats)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            arr = feats.detach().cpu().float().numpy()
            if result is None:
                result = np.zeros((len(paths), arr.shape[1]), dtype=np.float32)
            for local_i, pos in enumerate(kept_positions):
                result[pos] = arr[local_i]
        if result is None:
            return np.zeros((len(paths), 1), dtype=np.float32)
        return result.astype(np.float32)


def image_paths_for_rows(rows: Sequence[Dict[str, Any]], image_dir: str) -> List[Optional[str]]:
    return [str(Path(image_dir) / str(r.get("image_name"))) if r.get("image_name") else None for r in rows]


def unique_ranked_indices(candidate_indices: np.ndarray, scores: np.ndarray, train_rows: Sequence[Dict[str, Any]], top_k: int) -> List[int]:
    """Return top-k candidates after collapsing duplicate (cleaned prompt, image_name) pairs."""
    order = np.argsort(-scores)
    seen_pairs = set()
    out: List[int] = []
    for local_j in order:
        idx = int(candidate_indices[local_j])
        key = prompt_image_key(train_rows[idx])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        out.append(idx)
        if len(out) >= top_k:
            break
    return out


def select_diverse_from_topk(top_indices: List[int], train_rows: Sequence[Dict[str, Any]], score_by_idx: Dict[int, float], select_k: int) -> List[int]:
    """Select label-diverse examples from already eligible, unique top-k candidates."""
    selected: List[int] = []
    covered_labels = set()
    while len(selected) < select_k:
        remaining = [i for i in top_indices if i not in selected]
        if not remaining:
            break
        best_i = None
        best_key = None
        for i in remaining:
            labs = set(label_names(train_rows[i]))
            new_labs = labs - covered_labels
            key = (len(new_labs), len(labs), len(train_rows[i].get("labels") or []), score_by_idx.get(i, -999.0))
            if best_key is None or key > best_key:
                best_key = key
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        covered_labels.update(label_names(train_rows[best_i]))
    return selected[:select_k]


def pick_random_eligible_unique(
    train_indices_same_lang: Sequence[int],
    train_rows: Sequence[Dict[str, Any]],
    already_selected: set,
    already_pairs: set,
    min_label_spans: int,
    min_span_chars: int,
    max_span_chars: int,
    span_filter_mode: str,
    seed: int,
    eval_id: str,
    random_k: int,
) -> List[int]:
    rng = random.Random(seed + stable_int(eval_id))
    pool = [
        i for i in train_indices_same_lang
        if i not in already_selected
        and prompt_image_key(train_rows[i]) not in already_pairs
        and eligible_train_example(train_rows[i], min_label_spans, min_span_chars, max_span_chars, span_filter_mode)
    ]
    rng.shuffle(pool)
    return pool[:random_k]


def compact_train_shot(
    row: Dict[str, Any],
    selection_reason: str,
    rank: Optional[int],
    score: Optional[float],
    image_sim: Optional[float],
    response_sim: Optional[float],
    prompt_sim: Optional[float],
    raw_image_sim: Optional[float],
    raw_response_sim: Optional[float],
    raw_prompt_sim: Optional[float],
    save_raw_similarities: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": row.get("id"),
        "language": row.get("language"),
        "prompt": row.get("prompt"),
        "image_name": row.get("image_name"),
        "response": get_response(row),
        #"response_inline": row.get("response_inline", ""),
        #"labels": row.get("labels", []),
        "response_inline": map_response_inline_probs_to_3(row.get("response_inline", "")),
        "labels": map_labels_probs_to_3(row.get("labels", [])),
        "selection_reason": selection_reason,
        "retrieval_rank": rank,
        "retrieval_score": score,
        "image_similarity": image_sim,
        "response_similarity": response_sim,
        "prompt_similarity": prompt_sim,
    }
    if save_raw_similarities:
        out["raw_image_similarity"] = raw_image_sim
        out["raw_response_similarity"] = raw_response_sim
        out["raw_prompt_similarity"] = raw_prompt_sim
    return out



def collect_shot_label_stats(shots: Sequence[Dict[str, Any]]) -> Tuple[int, Counter]:
    """Return total label-span count and class counts for a list of selected shots."""
    class_counts: Counter = Counter()
    total_spans = 0

    for shot in shots:
        labels = shot.get("labels") or []
        if not isinstance(labels, list):
            continue

        for lab in labels:
            if not isinstance(lab, dict):
                continue
            label = lab.get("label")
            if not label:
                continue
            total_spans += 1
            class_counts[str(label)] += 1

    return total_spans, class_counts


def print_output_statistics(outputs: Sequence[Dict[str, Any]], expected_shots: int) -> None:
    """
    Print per-eval and global statistics after the output file is written.

    Per eval ID:
      - number of selected train shots
      - total hallucination label spans across selected shots
      - label-class distribution across selected shots
    """
    print("\n=== Few-shot output statistics ===")

    total_eval = len(outputs)
    shot_counts: List[int] = []
    global_class_counts: Counter = Counter()
    global_reason_counts: Counter = Counter()
    incomplete = 0

    for row in outputs:
        eval_id = row.get("id")
        shots = row.get("shots") or []
        if not isinstance(shots, list):
            shots = []

        n_shots = len(shots)
        shot_counts.append(n_shots)
        if n_shots < expected_shots:
            incomplete += 1

        label_span_total, class_counts = collect_shot_label_stats(shots)
        global_class_counts.update(class_counts)

        for shot in shots:
            global_reason_counts[str(shot.get("selection_reason", "unknown"))] += 1

        class_counts_dict = dict(sorted(class_counts.items()))
        print(
            f"[eval-stats] id={eval_id} "
            f"shots={n_shots}/{expected_shots} "
            f"label_spans={label_span_total} "
            f"classes={class_counts_dict}"
        )

    if shot_counts:
        avg_shots = sum(shot_counts) / len(shot_counts)
        min_shots = min(shot_counts)
        max_shots = max(shot_counts)
    else:
        avg_shots = 0.0
        min_shots = 0
        max_shots = 0

    print("\n=== Few-shot global summary ===")
    print(f"eval_rows={total_eval}")
    print(f"expected_shots_per_eval={expected_shots}")
    print(f"avg_shots_per_eval={avg_shots:.2f}")
    print(f"min_shots_per_eval={min_shots}")
    print(f"max_shots_per_eval={max_shots}")
    print(f"eval_rows_with_fewer_than_expected_shots={incomplete}")
    print(f"global_label_classes={dict(sorted(global_class_counts.items()))}")
    print(f"selection_reasons={dict(sorted(global_reason_counts.items()))}")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--eval-files", nargs="+", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--text-model", default="BAAI/bge-m3")
    ap.add_argument("--image-model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--image-weight", type=float, default=0.50)
    ap.add_argument("--response-weight", type=float, default=0.30)
    ap.add_argument("--prompt-weight", type=float, default=0.20)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--empty-top-k", type=int, default=10)
    ap.add_argument("--select-k", type=int, default=5)
    ap.add_argument("--random-k", type=int, default=1)
    ap.add_argument("--empty-k", type=int, default=1, help="Number of similar same-language empty/no-hallucination examples to add.")
    ap.add_argument("--min-label-spans", type=int, default=2)
    ap.add_argument("--min-span-chars", type=int, default=2)
    ap.add_argument("--max-span-chars", type=int, default=64)
    ap.add_argument(
        "--span-filter-mode",
        choices=["all", "any"],
        default="all",
        help="all = every label span in a selected example must be within range; any = require enough in-range spans only.",
    )
    ap.add_argument("--text-batch-size", type=int, default=16)
    ap.add_argument("--image-batch-size", type=int, default=32)
    ap.add_argument("--text-max-length", type=int, default=1024)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalization", choices=["minmax", "none"], default="minmax")
    ap.add_argument("--save-raw-similarities", action="store_true")
    ap.add_argument("--allow-cross-language-fallback", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    weight_sum = args.image_weight + args.response_weight + args.prompt_weight
    if abs(weight_sum - 1.0) > 1e-6:
        sys.stderr.write(f"[warning] weights sum to {weight_sum:.4f}, not 1.0.\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows: List[Dict[str, Any]] = []
    for p in args.train_files:
        train_rows.extend(read_jsonl(p, require_train_labels=True))

    eval_rows: List[Dict[str, Any]] = []
    for p in args.eval_files:
        eval_rows.extend(read_jsonl(p, require_train_labels=False))

    if not train_rows:
        raise ValueError("No train rows loaded.")
    if not eval_rows:
        raise ValueError("No eval rows loaded.")

    print(f"Loaded train rows: {len(train_rows)}")
    print(f"Loaded eval rows:  {len(eval_rows)}")

    train_by_lang: Dict[str, List[int]] = defaultdict(list)
    eligible_by_lang: Dict[str, List[int]] = defaultdict(list)
    empty_by_lang: Dict[str, List[int]] = defaultdict(list)

    for idx, row in enumerate(train_rows):
        lang = str(row.get("language", ""))
        train_by_lang[lang].append(idx)

        if eligible_train_example(
            row,
            args.min_label_spans,
            args.min_span_chars,
            args.max_span_chars,
            args.span_filter_mode,
        ):
            eligible_by_lang[lang].append(idx)

        if is_empty_train_example(row):
            empty_by_lang[lang].append(idx)

    print("Train rows by language:")
    for lang, idxs in sorted(train_by_lang.items()):
        print(
            f"  {lang}: {len(idxs)} total, "
            f"{len(eligible_by_lang.get(lang, []))} eligible, "
            f"{len(empty_by_lang.get(lang, []))} empty"
        )

    label_counter = Counter()
    for r in train_rows:
        label_counter.update(label_names(r))
    print("Train label types:", dict(sorted(label_counter.items())))
    print(
        f"Eligibility: min_label_spans={args.min_label_spans}, "
        f"span length={args.min_span_chars}..{args.max_span_chars}, "
        f"span_filter_mode={args.span_filter_mode}"
    )

    text_encoder = TextEmbedder(args.text_model, device=args.device, max_length=args.text_max_length, cache_dir=args.hf_cache_dir)
    train_responses = [get_response(r) for r in train_rows]
    eval_responses = [get_response(r) for r in eval_rows]
    train_prompts = [clean_prompt(str(r.get("prompt", ""))) for r in train_rows]
    eval_prompts = [clean_prompt(str(r.get("prompt", ""))) for r in eval_rows]

    print("Encoding train responses")
    train_resp_emb = text_encoder.encode(train_responses, batch_size=args.text_batch_size)
    print("Encoding eval responses")
    eval_resp_emb = text_encoder.encode(eval_responses, batch_size=args.text_batch_size)
    print("Encoding train prompts")
    train_prompt_emb = text_encoder.encode(train_prompts, batch_size=args.text_batch_size)
    print("Encoding eval prompts")
    eval_prompt_emb = text_encoder.encode(eval_prompts, batch_size=args.text_batch_size)

    image_encoder = ImageEmbedder(args.image_model, device=args.device, cache_dir=args.hf_cache_dir)
    train_image_paths = image_paths_for_rows(train_rows, args.image_dir)
    eval_image_paths = image_paths_for_rows(eval_rows, args.image_dir)
    print(f"Missing train images: {sum(1 for p in train_image_paths if not p or not os.path.exists(p))}")
    print(f"Missing eval images:  {sum(1 for p in eval_image_paths if not p or not os.path.exists(p))}")
    print("Encoding train images")
    train_img_emb = image_encoder.encode_paths(train_image_paths, batch_size=args.image_batch_size)
    print("Encoding eval images")
    eval_img_emb = image_encoder.encode_paths(eval_image_paths, batch_size=args.image_batch_size)

    outputs: List[Dict[str, Any]] = []
    for e_idx, e_row in enumerate(tqdm(eval_rows, desc="Retrieving shots")):
        eval_id = str(e_row.get("id", f"eval-{e_idx}"))
        lang = str(e_row.get("language", ""))

        candidate_indices = list(eligible_by_lang.get(lang, []))
        if not candidate_indices:
            if args.allow_cross_language_fallback:
                candidate_indices = [i for idxs in eligible_by_lang.values() for i in idxs]
            else:
                raise ValueError(
                    f"No eligible train examples for language={lang!r}. "
                    "Try --span-filter-mode any, reduce --min-label-spans, or use --allow-cross-language-fallback."
                )

        cand = np.array(candidate_indices, dtype=np.int64)
        raw_image_sims = train_img_emb[cand] @ eval_img_emb[e_idx]
        raw_response_sims = train_resp_emb[cand] @ eval_resp_emb[e_idx]
        raw_prompt_sims = train_prompt_emb[cand] @ eval_prompt_emb[e_idx]

        if args.normalization == "minmax":
            image_sims = minmax_normalize(raw_image_sims)
            response_sims = minmax_normalize(raw_response_sims)
            prompt_sims = minmax_normalize(raw_prompt_sims)
        else:
            image_sims = raw_image_sims.astype(np.float32)
            response_sims = raw_response_sims.astype(np.float32)
            prompt_sims = raw_prompt_sims.astype(np.float32)

        scores = args.image_weight * image_sims + args.response_weight * response_sims + args.prompt_weight * prompt_sims

        top_indices = unique_ranked_indices(cand, scores, train_rows, args.top_k)

        score_by_idx: Dict[int, float] = {}
        img_sim_by_idx: Dict[int, float] = {}
        resp_sim_by_idx: Dict[int, float] = {}
        prompt_sim_by_idx: Dict[int, float] = {}
        raw_img_sim_by_idx: Dict[int, float] = {}
        raw_resp_sim_by_idx: Dict[int, float] = {}
        raw_prompt_sim_by_idx: Dict[int, float] = {}
        for local_j, idx_np in enumerate(cand):
            idx = int(idx_np)
            score_by_idx[idx] = round(float(scores[local_j]), 6)
            img_sim_by_idx[idx] = round(float(image_sims[local_j]), 6)
            resp_sim_by_idx[idx] = round(float(response_sims[local_j]), 6)
            prompt_sim_by_idx[idx] = round(float(prompt_sims[local_j]), 6)
            raw_img_sim_by_idx[idx] = round(float(raw_image_sims[local_j]), 6)
            raw_resp_sim_by_idx[idx] = round(float(raw_response_sims[local_j]), 6)
            raw_prompt_sim_by_idx[idx] = round(float(raw_prompt_sims[local_j]), 6)

        # Unique ranks after duplicate prompt+image collapse.
        rank_by_idx: Dict[int, int] = {}
        seen_pairs = set()
        unique_rank = 0
        for local_j in np.argsort(-scores):
            idx = int(cand[local_j])
            key = prompt_image_key(train_rows[idx])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            unique_rank += 1
            rank_by_idx[idx] = unique_rank

        selected = select_diverse_from_topk(top_indices, train_rows, score_by_idx, args.select_k)
        selected_set = set(selected)
        selected_pairs = {prompt_image_key(train_rows[i]) for i in selected}

        random_selected = pick_random_eligible_unique(
            candidate_indices,
            train_rows,
            already_selected=selected_set,
            already_pairs=selected_pairs,
            min_label_spans=args.min_label_spans,
            min_span_chars=args.min_span_chars,
            max_span_chars=args.max_span_chars,
            span_filter_mode=args.span_filter_mode,
            seed=args.seed,
            eval_id=eval_id,
            random_k=args.random_k,
        )

        # Select similar same-language empty / fully-supported examples.
        empty_selected: List[int] = []
        empty_score_by_idx: Dict[int, float] = {}
        empty_img_sim_by_idx: Dict[int, float] = {}
        empty_resp_sim_by_idx: Dict[int, float] = {}
        empty_prompt_sim_by_idx: Dict[int, float] = {}
        empty_raw_img_sim_by_idx: Dict[int, float] = {}
        empty_raw_resp_sim_by_idx: Dict[int, float] = {}
        empty_raw_prompt_sim_by_idx: Dict[int, float] = {}

        if args.empty_k > 0:
            empty_candidate_indices = list(empty_by_lang.get(lang, []))

            if not empty_candidate_indices and args.allow_cross_language_fallback:
                empty_candidate_indices = [i for idxs in empty_by_lang.values() for i in idxs]

            if empty_candidate_indices:
                empty_cand = np.array(empty_candidate_indices, dtype=np.int64)

                empty_raw_image_sims = train_img_emb[empty_cand] @ eval_img_emb[e_idx]
                empty_raw_response_sims = train_resp_emb[empty_cand] @ eval_resp_emb[e_idx]
                empty_raw_prompt_sims = train_prompt_emb[empty_cand] @ eval_prompt_emb[e_idx]

                if args.normalization == "minmax":
                    empty_image_sims = minmax_normalize(empty_raw_image_sims)
                    empty_response_sims = minmax_normalize(empty_raw_response_sims)
                    empty_prompt_sims = minmax_normalize(empty_raw_prompt_sims)
                else:
                    empty_image_sims = empty_raw_image_sims.astype(np.float32)
                    empty_response_sims = empty_raw_response_sims.astype(np.float32)
                    empty_prompt_sims = empty_raw_prompt_sims.astype(np.float32)

                empty_scores = (
                    args.image_weight * empty_image_sims
                    + args.response_weight * empty_response_sims
                    + args.prompt_weight * empty_prompt_sims
                )

                for local_j, idx_np in enumerate(empty_cand):
                    idx = int(idx_np)
                    empty_score_by_idx[idx] = round(float(empty_scores[local_j]), 6)
                    empty_img_sim_by_idx[idx] = round(float(empty_image_sims[local_j]), 6)
                    empty_resp_sim_by_idx[idx] = round(float(empty_response_sims[local_j]), 6)
                    empty_prompt_sim_by_idx[idx] = round(float(empty_prompt_sims[local_j]), 6)
                    empty_raw_img_sim_by_idx[idx] = round(float(empty_raw_image_sims[local_j]), 6)
                    empty_raw_resp_sim_by_idx[idx] = round(float(empty_raw_response_sims[local_j]), 6)
                    empty_raw_prompt_sim_by_idx[idx] = round(float(empty_raw_prompt_sims[local_j]), 6)

                empty_ranked = unique_ranked_indices(
                    empty_cand,
                    empty_scores,
                    train_rows,
                    #top_k=max(args.empty_k * 5, args.empty_k),
                    top_k=args.empty_top_k,
                )

                already_all = set(selected) | set(random_selected)
                already_pairs_all = {prompt_image_key(train_rows[i]) for i in already_all}

                for idx in empty_ranked:
                    if idx in already_all:
                        continue
                    if prompt_image_key(train_rows[idx]) in already_pairs_all:
                        continue

                    empty_selected.append(idx)
                    already_pairs_all.add(prompt_image_key(train_rows[idx]))

                    if len(empty_selected) >= args.empty_k:
                        break
            else:
                sys.stderr.write(
                    f"[warning] eval id={eval_id}: no empty same-language examples for language={lang!r}\n"
                )

        shots: List[Dict[str, Any]] = []
        for idx in selected:
            shots.append(compact_train_shot(
                train_rows[idx], "topk_diverse", rank_by_idx.get(idx), score_by_idx.get(idx),
                img_sim_by_idx.get(idx), resp_sim_by_idx.get(idx), prompt_sim_by_idx.get(idx),
                raw_img_sim_by_idx.get(idx), raw_resp_sim_by_idx.get(idx), raw_prompt_sim_by_idx.get(idx),
                args.save_raw_similarities,
            ))
        for idx in random_selected:
            shots.append(compact_train_shot(
                train_rows[idx], "random_same_language_eligible_unique", rank_by_idx.get(idx), score_by_idx.get(idx),
                img_sim_by_idx.get(idx), resp_sim_by_idx.get(idx), prompt_sim_by_idx.get(idx),
                raw_img_sim_by_idx.get(idx), raw_resp_sim_by_idx.get(idx), raw_prompt_sim_by_idx.get(idx),
                args.save_raw_similarities,
            ))

        for local_rank, idx in enumerate(empty_selected, start=1):
            shots.append(compact_train_shot(
                train_rows[idx],
                "similar_empty_same_language",
                local_rank,
                empty_score_by_idx.get(idx),
                empty_img_sim_by_idx.get(idx),
                empty_resp_sim_by_idx.get(idx),
                empty_prompt_sim_by_idx.get(idx),
                empty_raw_img_sim_by_idx.get(idx),
                empty_raw_resp_sim_by_idx.get(idx),
                empty_raw_prompt_sim_by_idx.get(idx),
                args.save_raw_similarities,
            ))

        expected = args.select_k + args.random_k + args.empty_k
        if len(shots) < expected:
            sys.stderr.write(
                f"[warning] eval id={eval_id}: only selected {len(shots)}/{expected} shots after "
                "unique prompt+image and span-length filtering.\n"
            )

        outputs.append({
            "id": e_row.get("id"),
            "eval_language": lang,
            "eval_prompt": e_row.get("prompt"),
            "eval_image_name": e_row.get("image_name"),
            "eval_response": get_response(e_row),
            "shots": shots,
        })

    write_jsonl(args.output_file, outputs)
    print(f"Saved: {args.output_file}")
    print(f"Rows written: {len(outputs)}")

    expected_shots = args.select_k + args.random_k + args.empty_k
    print_output_statistics(outputs, expected_shots=expected_shots)


if __name__ == "__main__":
    main()
