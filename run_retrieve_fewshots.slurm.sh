#!/usr/bin/env bash
#SBATCH --job-name=retrieve_fewshots
#SBATCH --partition=nvidia_h100
#SBATCH --nodelist=g0005
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err

set -euo pipefail

# -----------------------------
# Environment
# -----------------------------

mkdir -p logs

echo "Python: $(which python)"
python --version
echo "CUDA:"


# -----------------------------
# Paths: edit these
# -----------------------------
PROJECT_DIR="/home/tetoqeer/TE/shroom"
cd "${PROJECT_DIR}"

IMAGE_DIR="shroom_data/all-images/shroom-vis-images"
#OUTPUT_FILE="${PROJECT_DIR}/converted/train/retrieved_fewshots_5-2shots_top10_minspans3_503020.jsonl"
OUTPUT_FILE="${PROJECT_DIR}/converted/train/retrieved_fewshots_5-1empty_top10_minspans3_503020_new1-64.jsonl"

TRAIN_FILES=(
  "converted/train/train.all.labeled.inline.jsonl"
)

EVAL_FILES=(
  "converted/test/test.all.labeled.inline.jsonl"
  "converted/dev/dev.all.labeled.inline.jsonl"
  "shroom_data/test/shroom-vision.test.en.unlabeled.jsonl"
  "shroom_data/test/shroom-vision.test.fr.unlabeled.jsonl"
  "shroom_data/test/shroom-vision.test.it.unlabeled.jsonl"
  "shroom_data/test/shroom-vision.test.zh.unlabeled.jsonl"
)

# -----------------------------
# Similarity models
# -----------------------------
# Text default: multilingual retrieval model.
TEXT_MODEL="BAAI/bge-m3"

# Image default: newer SigLIP2 image encoder.
IMAGE_MODEL="openai/clip-vit-large-patch14"

# Optional Hugging Face cache directory. Leave empty to use default.
HF_CACHE_DIR=""

# -----------------------------
# Retrieval weights
# -----------------------------
IMAGE_WEIGHT="0.50"
RESPONSE_WEIGHT="0.30"
PROMPT_WEIGHT="0.20"

# Similarity normalization:
#   minmax = normalize each channel across same-language train candidates per eval item
#   none   = use raw cosine similarities
NORMALIZATION="minmax"

TOP_K="10"
EMPTY_TOP_K="10"
# 5 retrieval-diverse shots + 1 random eligible shot = 6 total shots.
SELECT_K="5"
RANDOM_K="0"
EMPTY_K="1"

MIN_LABEL_SPANS="2"

MIN_SPAN_CHARS="1"
MAX_SPAN_CHARS="64"
SPAN_FILTER_MODE="all"   # options: all, any

TEXT_BATCH_SIZE="16"
IMAGE_BATCH_SIZE="32"
TEXT_MAX_LENGTH="1024"
SEED="42"
DEVICE="auto"

# Set to 1 if you want raw cosine similarities saved too.
SAVE_RAW_SIMILARITIES="0"

CMD=(
  python retrieve_fewshots.py
  --train-files "${TRAIN_FILES[@]}"
  --eval-files "${EVAL_FILES[@]}"
  --image-dir "${IMAGE_DIR}"
  --output-file "${OUTPUT_FILE}"
  --text-model "${TEXT_MODEL}"
  --image-model "${IMAGE_MODEL}"
  --image-weight "${IMAGE_WEIGHT}"
  --response-weight "${RESPONSE_WEIGHT}"
  --prompt-weight "${PROMPT_WEIGHT}"
  --normalization "${NORMALIZATION}"
  --top-k "${TOP_K}"
  --empty-top-k "${EMPTY_TOP_K}"
  --select-k "${SELECT_K}"
  --random-k "${RANDOM_K}"
  --empty-k "${EMPTY_K}"
  --min-label-spans "${MIN_LABEL_SPANS}"
  --min-span-chars "${MIN_SPAN_CHARS}"
  --max-span-chars "${MAX_SPAN_CHARS}"
  --span-filter-mode "${SPAN_FILTER_MODE}"
  --text-batch-size "${TEXT_BATCH_SIZE}"
  --image-batch-size "${IMAGE_BATCH_SIZE}"
  --text-max-length "${TEXT_MAX_LENGTH}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ -n "${HF_CACHE_DIR}" ]]; then
  CMD+=(--hf-cache-dir "${HF_CACHE_DIR}")
fi

if [[ "${SAVE_RAW_SIMILARITIES}" == "1" ]]; then
  CMD+=(--save-raw-similarities)
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
