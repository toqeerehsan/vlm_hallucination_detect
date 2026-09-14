#!/usr/bin/env bash
#SBATCH --job-name=vlm_ft
#SBATCH --partition=
#SBATCH --nodelist=
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err
#
# ONE recipe, all three models, run SEQUENTIALLY in a single job:
#   bf16 LoRA + rsLoRA + LoRA+ , r=64, all LLM linears, vision tower frozen,
#   loss masked to the assistant turn, zero-shot prompt (matches NUM_SHOTS=0).
# No quantization -> adapters merge cleanly via client.py's merge_and_unload().
#
# Submit: sbatch run_finetune_slurm.sh
set -euo pipefail
source .env

module purge
module load cuda/13.1

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Qwen3.6 Gated-DeltaNet: avoid tilelang JIT (CUDA header mismatch) -> native path
export FLA_USE_FAST_OPS=0
export FLA_NO_TILELANG=1

########################## COMMON SETTINGS ##########################
HF_USER="QSTS-VTT"
SUFFIX="shroom-ft-v3"                 # new suffix: keeps the old adapters intact
export HF_TOKEN="$QSTS_HF_TOKEN"

IMAGE_DIR="shroom_data/all-images/shroom-vis-images/"
FT_DATA="fine-tuning/ft-splits"
OUT_ROOT="ckpts"

# dev file may be dev.jsonl or dev.json depending on how prepare_ft_splits ran
DEV_FILE=""
for c in "${FT_DATA}/dev.jsonl" "${FT_DATA}/dev.json"; do
    [ -f "$c" ] && { DEV_FILE="$c"; break; }
done
[ -z "${DEV_FILE}" ] && echo "WARN: no dev file found in ${FT_DATA} — training without eval"
echo "dev file: ${DEV_FILE:-<none>}"

# per-model train splits (one file per model, as prepared)
TRAIN_GEMMA="${FT_DATA}/train_gemma-4-31b.jsonl"
TRAIN_QWEN="${FT_DATA}/train_qwen3.6-27b.jsonl"
TRAIN_MISTRAL="${FT_DATA}/train_mistral-small-3.1-24b.jsonl"

for f in "${TRAIN_GEMMA}" "${TRAIN_QWEN}" "${TRAIN_MISTRAL}"; do
    [ -f "$f" ] || { echo "MISSING train split: $f"; exit 1; }
    echo "train split: $f  ($(wc -l < "$f") rows)"
done

CU13_LIB="$(python -c "import os,nvidia; print(os.path.join(os.path.dirname(nvidia.__file__),'cu13','lib'))" 2>/dev/null || true)"
export LD_LIBRARY_PATH="${CU13_LIB}:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p logs "${OUT_ROOT}"

# Shared recipe (identical for all three models on purpose)
EVAL_ARG=()
[ -n "${DEV_FILE}" ] && EVAL_ARG=(--eval-file "${DEV_FILE}")

#COMMON=(--method lora
#        --image-dir "${IMAGE_DIR}"
#        --lora-r 64 --lora-alpha 64 --lora-dropout 0.05
#        --loraplus-ratio 16
#        --lr 1e-4 --epochs 3
#        --batch-size 1 --grad-accum 8
#        --max-seq-len 6192)
COMMON=(--method lora
        --image-dir "${IMAGE_DIR}"
        --lora-r 64 --lora-alpha 16 --lora-dropout 0.05
        --loraplus-ratio 4
        --lr 5e-5 --epochs 2
        --warmup-ratio 0.05
        --batch-size 1 --grad-accum 8
        --max-seq-len 6192)


# Smoke test first:  SMOKE=1 sbatch run_finetune_slurm.sh
LIMIT_ARG=""
[ "${SMOKE:-0}" = "1" ] && LIMIT_ARG="--limit 32"
#####################################################################

# ------------------------- 1/4  GEMMA-4-31B -------------------------
#KEY="gemma-4-31b"
#echo "### FT ${KEY}"
#python -u finetune_vlm.py \
#  --model "google/gemma-4-31b-it" \
#  --train-file "${TRAIN_GEMMA}" \
#  --output-dir "${OUT_ROOT}/${KEY}-v3" \
#  --hub-id "${HF_USER}/${KEY}-${SUFFIX}" --private \
#  --merge-system-into-user \
#  "${COMMON[@]}" "${EVAL_ARG[@]}" ${LIMIT_ARG}

# ------------------------- 2/4  QWEN3.6-27B -------------------------
#KEY="qwen3.6-27b"
#echo "### FT ${KEY}"
#python -u finetune_vlm.py \
#  --model "Qwen/Qwen3.6-27B" \
#  --train-file "${TRAIN_QWEN}" \
#  --output-dir "${OUT_ROOT}/${KEY}-v2" \
#  --hub-id "${HF_USER}/${KEY}-${SUFFIX}" --private \
#  --chat-template-kwargs '{"enable_thinking": false}' \
#  "${COMMON[@]}" "${EVAL_ARG[@]}" ${LIMIT_ARG}
#KEY="qwen3.6-27b"
#echo "### FT ${KEY}"
#python -u finetune_vlm.py \
#  --model "Qwen/Qwen3.6-27B" \
#  --train-file "${TRAIN_QWEN}" \
#  --output-dir "${OUT_ROOT}/${KEY}-v3" \
#  --hub-id "${HF_USER}/${KEY}-${SUFFIX}" --private \
#  --chat-template-kwargs '{"enable_thinking": false}' \
#  "${COMMON[@]}" "${EVAL_ARG[@]}" ${LIMIT_ARG} \
#  --max-pixels 802816 --max-seq-len 8192
# -------------------- 3/4  MISTRAL-SMALL-3.1-24B --------------------
KEY="qwen3-vl-30b-a3b"
echo "### FT ${KEY}"
python -u finetune_vlm.py \
  --model "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --train-file "${TRAIN_MISTRAL}" \
  --output-dir "${OUT_ROOT}/${KEY}-v3" \
  --hub-id "${HF_USER}/${KEY}-${SUFFIX}" --private \
  --no-mlp \
  --max-pixels 802816 --max-seq-len 8192 \
  "${COMMON[@]}" "${EVAL_ARG[@]}" ${LIMIT_ARG}
# -------------------- 4/4  MISTRAL-SMALL-3.1-24B --------------------
#KEY="mistral-small-3.1-24b"
#echo "### FT ${KEY}"
#python -u finetune_vlm.py \
#  --model "mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
#  --train-file "${TRAIN_MISTRAL}" \
#  --output-dir "${OUT_ROOT}/${KEY}-v3" \
#  --hub-id "${HF_USER}/${KEY}-${SUFFIX}" --private \
#  "${COMMON[@]}" "${EVAL_ARG[@]}" ${LIMIT_ARG}

echo "Done. Adapters in ${OUT_ROOT}/  and pushed PRIVATE to ${HF_USER}/<key>-${SUFFIX}"
echo "Next: add the -ft entries to models.json, set NUM_SHOTS=0, run the pipeline."
