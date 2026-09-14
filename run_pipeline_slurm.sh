#!/usr/bin/env bash
#SBATCH --job-name=vlm-class-check
#SBATCH --partition=
#SBATCH --nodelist=g0003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err
#
# Open models run via HuggingFace transformers IN-PROCESS (no vLLM/server).
# Submit:  sbatch run_pipeline_slurm.sh
set -euo pipefail

set -a; source .env; set +a # for azure

CU13_LIB="$(python -c "import os,nvidia; print(os.path.join(os.path.dirname(nvidia.__file__),'cu13','lib'))" 2>/dev/null || true)"
export LD_LIBRARY_PATH="${CU13_LIB}:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

#export TORCHINDUCTOR_DISABLE=1
#export TORCH_COMPILE_DISABLE=1

##############################  EDIT THESE  ##############################
MODELS="gemma-4-31b qwen3.6-27b"                 # 1 to 3 KEYS from models.json (space-separated)
INPUTS="converted/test/test.en.labeled.jsonl converted/test/test.fr.labeled.jsonl converted/test/test.it.labeled.jsonl converted/test/test.zh.labeled.jsonl"
#INPUTS="converted/test-shroom/shroom-vision.test.en.unlabeled.jsonl converted/test-shroom/shroom-vision.test.fr.unlabeled.jsonl converted/test-shroom/shroom-vision.test.it.unlabeled.jsonl converted/test-shroom/shroom-vision.test.zh.unlabeled.jsonl"
#TRAIN_FILES=""
TRAIN_FILES="converted/train/retrieved_fewshots_8shots_top15_minspans3_503020.jsonl"
OUTPUT_DIR="outputs-6shot3-noimg-check/test"
IMAGE_DIR="shroom_data/all-images/shroom-vis-images"
NUM_SHOTS=6
FEWSHOT_IMAGES="include"                  # none | include
INCLUDE_IMAGENAME=0                         # 1 = add [IMAGE: ...] to PROMPT, 0 = off
START=0                                    # 0-based start index: 0 for first chunk, 250 for second chunk
LIMIT=700                                  # number of rows to process from START; empty = all after START

INLINE_TO_OFFSETS="converted/inline_to_offsets.py"
PROB_CLASS_TO_VALUE="converted/prob_class_to_value.json"
PROB_VALUE_TO_CLASS="converted/prob_value_to_class.json"
MIN_VOTES=1
CALIBRATION=""

# Environment bootstrap (EDIT for your cluster):
# source ~/anaconda3/etc/profile.d/conda.sh && conda activate llm-ft2
# Fix for the PIL/GLIBCXX error you hit — make conda's libstdc++ win over /lib64:
# export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
# Optional: keep HF weights on a big disk / load offline
# export HF_HOME=/scratch/$USER/hf
# export HF_HUB_OFFLINE=1
#########################################################################

mkdir -p logs "${OUTPUT_DIR}"
echo "Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}  |  models: ${MODELS}"

START_ARG=""
[ -n "${START}" ] && START_ARG="--start ${START}"
LIMIT_ARG=""
[ -n "${LIMIT}" ] && LIMIT_ARG="--limit ${LIMIT}"
IMAGENAME_ARG=""
[ "${INCLUDE_IMAGENAME}" = "1" ] && IMAGENAME_ARG="--include_imagename"
# ------------------------- per-model inference -------------------------
for MODEL in ${MODELS}; do
    echo "### MODEL=${MODEL} (transformers, in-process)"
    python -u run_inference.py \
        --model "${MODEL}" --models-json models.json \
        --inputs ${INPUTS} --image-dir "${IMAGE_DIR}" \
        --train-files ${TRAIN_FILES} \
        --num-shots "${NUM_SHOTS}" \
        --fewshot-images "${FEWSHOT_IMAGES}" ${START_ARG} ${LIMIT_ARG} ${IMAGENAME_ARG} \
        --output-dir "${OUTPUT_DIR}"

    # alignment: inline -> offsets for every input (temp dir avoids name collision)
    for DATA in ${INPUTS}; do
        BASE=$(basename "${DATA}" .jsonl)
        python "${INLINE_TO_OFFSETS}" \
            --prob-map "${PROB_CLASS_TO_VALUE}" \
            --output-dir "${OUTPUT_DIR}/${MODEL}/_off" \
            --inline-field response_inline \
            -- "${OUTPUT_DIR}/${MODEL}/${BASE}.inline.jsonl"
        mv "${OUTPUT_DIR}/${MODEL}/_off/${BASE}.inline.jsonl" "${OUTPUT_DIR}/${MODEL}/${BASE}.offsets.jsonl"
    done
    #for DATA in ${INPUTS}; do
    #    BASE=$(basename "${DATA}" .jsonl)
    #    python "${INLINE_TO_OFFSETS}" \
    #        --prob-map "${PROB_CLASS_TO_VALUE}" \
    #        --output-dir "${OUTPUT_DIR}/${MODEL}/_off" \
    #        --inline-field response_inline \
    #        -- "${OUTPUT_DIR}/${MODEL}/${BASE}.inline.corrected.jsonl"
    #    mv "${OUTPUT_DIR}/${MODEL}/_off/${BASE}.inline.corrected.jsonl" "${OUTPUT_DIR}/${MODEL}/${BASE}.offsets.corrected.jsonl"
    #done

    rm -rf "${OUTPUT_DIR}/${MODEL}/_off"
    ### Combine files
    #python combine_offsets.py --output ${OUTPUT_DIR}/${MODEL}/test.ALL.labeled.offsets.jsonl -- \
    #    ${OUTPUT_DIR}/${MODEL}/test.en.labeled.offsets.jsonl \
    #    ${OUTPUT_DIR}/${MODEL}/test.it.labeled.offsets.jsonl \
    #    ${OUTPUT_DIR}/${MODEL}/test.fr.labeled.offsets.jsonl \
    #    ${OUTPUT_DIR}/${MODEL}/test.zh.labeled.offsets.jsonl
    #python participant_kit/scorer.py converted/test/test.all.labeled.scorer.jsonl ${OUTPUT_DIR}/${MODEL}/test.ALL.labeled.offsets.jsonl test_scores.txt
done

# ----------------------------- committee -------------------------------
#CAL_ARG=""
#[ -n "${CALIBRATION}" ] && CAL_ARG="--calibration ${CALIBRATION}"
#python aggregate.py \
#    --root "${OUTPUT_DIR}" --models ${MODELS} --inputs ${INPUTS} \
#    --output-dir "${OUTPUT_DIR}/committee" \
#    --prob-value-to-class "${PROB_VALUE_TO_CLASS}" \
#    --min-votes "${MIN_VOTES}" ${CAL_ARG}

echo "Done. Per-model: ${OUTPUT_DIR}/<model>/  |  Committee: ${OUTPUT_DIR}/committee/"
