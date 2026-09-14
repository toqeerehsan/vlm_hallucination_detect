#!/usr/bin/env bash
# =============================================================================
# run_committee.sh — majority-vote committee from explicit per-model dirs.
#
# You give it one directory per judge. Each must contain, per language, either
#   <base>.offsets.jsonl   (preferred, used directly)
#   <base>.inline.jsonl    (converted automatically)
# where <base> is e.g. test.en.labeled
#
# The dirs are symlinked into a staging root laid out the way aggregate.py
# expects (<root>/<model>/<base>.offsets.jsonl), so aggregate.py is UNCHANGED.
#
# No calibration. Sweeps MIN_VOTES and scores each setting if gold is given.
# CPU-only.
# =============================================================================
set -euo pipefail

############################  EDIT THESE  ############################
# One directory per judge. Order does not matter. Name is taken from basename.
MODEL_DIRS=(
"outputs-shroom-unlabeled/ft-test-all-0shot-img/gemma-4-31b-ft"
"outputs-shroom-unlabeled/test-all-6shot3-img/gemma-4-31b"
"outputs-shroom-unlabeled/ft-test-all-0shot-img/mistral-small-3.1-24b-ft"
"outputs-shroom-unlabeled/ft-test-all-0shot-img/qwen3.6-27b-ft"
"outputs-shroom-unlabeled/ft-test-all-3shot-img/qwen3-vl-30b-a3b-ft"

"outputs-shroom-unlabeled/_multi_layer_probe_8b"
"outputs-shroom-unlabeled/_multi_stage_probe_8b"

#"outputs-ft-v3/test-all-0shot-img-g05/gemma-4-31b-ft"
#"outputs-6shot3-img/test/gemma-4-31b"
#"outputs-ft-v3/test-all-0shot-img-g05/mistral-small-3.1-24b-ft"
#"outputs-ft-v3/test-all-0shot-img-g05/qwen3.6-27b-ft"
#"outputs-ft-v3/test-all-3shot-img-g05/qwen3-vl-30b-a3b-ft"

#"outputs-others_models/test/multi_layer_probe_8b_test_labelled"
#"outputs-others_models/test/multi_stage_probe_8b_test_labelled"
)

UNLABELED_SET=1
SPLIT="test"                                  # test | dev
LABELED_TPL="${SPLIT}.{lang}.labeled"
UNLABELED_TPL="shroom-vision.${SPLIT}.{lang}.unlabeled"
LANGS="en fr it zh"

if [ "${UNLABELED_SET}" = "1" ]; then
    BASE_TPL="${UNLABELED_TPL}"
    INPUT_DIR="converted/shroom-unlabeled"
    OUTPUT_DIR="_committee_test_unseen_final_X"
else
    BASE_TPL="${LABELED_TPL}"
    INPUT_DIR="converted/test"
    OUTPUT_DIR="_committee_test_seen_final_X"
fi
#INPUT_DIR="converted/test"                    # holds <split>.<lang>.labeled.jsonl
#OUTPUT_DIR="outputs_committee_test_unseen" # results go here          # results go here

# Vote thresholds to try. "1 2 3" sweeps; "2" runs a single setting.
MIN_VOTES_LIST="1 2 3 4"

# Optional: gold file for scoring each setting. Empty = build only, no scoring.
GOLD="converted/test/test.all.labeled.scorer.jsonl"
SCORER="participant_kit/scorer.py"
COMBINER="combine_offsets.py"                 # builds the ALL (cumulative) file

# maps + helpers
C2V="converted/prob_class_to_value.json"      # class -> value (inline -> offsets)
V2C="converted/prob_value_to_class.json"      # value -> class (committee inline)
I2O="converted/inline_to_offsets.py"
#####################################################################
resolve_off() {  # $1 dir  $2 base  -> path or empty
    for ext in offsets offset; do
        [ -f "$1/$2.${ext}.jsonl" ] && { echo "$1/$2.${ext}.jsonl"; return; }
    done
    echo ""
}

command -v realpath >/dev/null || realpath() { python -c "import os,sys;print(os.path.abspath(sys.argv[1]))" "$1"; }

STAGE="${OUTPUT_DIR}/_stage"
rm -rf "${STAGE}"
mkdir -p "${STAGE}" "${OUTPUT_DIR}"

MODEL_NAMES=()

echo "=== STEP 1: collect per-model offsets ==="
for DIR in "${MODEL_DIRS[@]}"; do
    [ -d "${DIR}" ] || { echo "MISSING model dir: ${DIR}"; exit 1; }
    NAME="$(basename "${DIR}")"
    MODEL_NAMES+=("${NAME}")
    mkdir -p "${STAGE}/${NAME}"

    for L in ${LANGS}; do
        #BASE="${SPLIT}.${L}.labeled"
        BASE="${BASE_TPL/\{lang\}/$L}"
        #OFF="${DIR}/${BASE}.offsets.jsonl"
        OFF="$(resolve_off "${DIR}" "${BASE}")"
        INL="${DIR}/${BASE}.inline.jsonl"
        if [ ! -f "${OFF}" ] && [ -f "${INL}" ]; then
            echo "  [convert] ${NAME}/${BASE}: inline -> offsets"
            TMPD="${DIR}/_off"; mkdir -p "${TMPD}"
            if [ -f repair_jsonl.py ]; then
                python repair_jsonl.py "${INL}" "${INL}.tmp" && mv "${INL}.tmp" "${INL}"
            fi
            python "${I2O}" --prob-map "${C2V}" --output-dir "${TMPD}" \
                --inline-field response_inline -- "${INL}"
            mv "${TMPD}/${BASE}.inline.jsonl" "${OFF}"
            rm -rf "${TMPD}"
        fi

        if [ -f "${OFF}" ]; then
            ln -sf "$(realpath "${OFF}")" "${STAGE}/${NAME}/${BASE}.offsets.jsonl"
            echo "  [ok] ${NAME}/${BASE}  ($(wc -l < "${OFF}") rows)"
        else
            echo "  [WARN] no offsets or inline for ${NAME}/${BASE}"
        fi
    done
done

echo "judges: ${MODEL_NAMES[*]}"

INPUTS=""
for L in ${LANGS}; do INPUTS="${INPUTS} ${INPUT_DIR}/${BASE_TPL/\{lang\}/$L}.jsonl"; done

echo
echo "=== STEP 2: committee, sweeping MIN_VOTES=${MIN_VOTES_LIST} ==="
for MV in ${MIN_VOTES_LIST}; do
    OUT="${OUTPUT_DIR}/minvotes${MV}"
    mkdir -p "${OUT}"
    echo
    echo "--- MIN_VOTES=${MV} -> ${OUT}"
    python aggregate.py \
        --root "${STAGE}" --models "${MODEL_NAMES[@]}" \
        --inputs ${INPUTS} \
        --output-dir "${OUT}" \
        --prob-value-to-class "${V2C}" \
        --min-votes "${MV}"

    # ---- cumulative (ALL languages) file ----
    ALL_PRED="${OUT}/${SPLIT}.ALL.labeled.offsets.jsonl"
    PARTS=""
    for L in ${LANGS}; do
        P="${OUT}/${BASE_TPL/\{lang\}/$L}.offsets.jsonl"
        [ -f "${P}" ] && PARTS="${PARTS} ${P}"
    done
    if [ -n "${PARTS}" ]; then
        if [ -f "${COMBINER}" ]; then
            python "${COMBINER}" --output "${ALL_PRED}" -- ${PARTS}
        else
            cat ${PARTS} > "${ALL_PRED}"          # fallback: plain concatenation
        fi
        echo "  [ALL] $(wc -l < "${ALL_PRED}") rows -> ${ALL_PRED}"
    fi

    if [ "${UNLABELED_SET}" = "0" ] && [ -n "${GOLD}" ] && [ -f "${GOLD}" ] && [ -f "${SCORER}" ]; then
        for L in ${LANGS} ALL; do
            PRED="${OUT}/${BASE_TPL/\{lang\}/$L}.offsets.jsonl"
            [ -f "${PRED}" ] || continue
            python "${SCORER}" "${GOLD}" "${PRED}" "${OUT}/scores.${L}.txt" \
                > "${OUT}/scorer_stdout.${L}.txt" 2>&1 || true
            echo "  --- ${L} (min_votes=${MV})"
            grep -E "^(Cor|IoU|acc|f1|prec|rec)" "${OUT}/scorer_stdout.${L}.txt" \
                | sed 's/^/      /' || true
        done
    fi
done

# ---- SUMMARY TABLE across all MIN_VOTES settings ----
python - "${OUTPUT_DIR}" "${SPLIT}" "${MIN_VOTES_LIST}" "${LANGS}" <<'PYEOF'
import os, re, sys
out_root, split, mv_list, langs = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4].split()
cols = langs + ['ALL']
def grab(path):
    if not os.path.exists(path):
        return None
    txt = open(path, encoding='utf-8', errors='ignore').read()
    def f(k):
        m = re.search(rf'^{k}\s*:\s*([0-9.]+)', txt, re.M)
        return float(m.group(1)) if m else None
    return {'IoU': f('IoU'), 'Cor': f('Cor'), 'rec': f('rec'), 'prec': f('prec')}

for metric in ('IoU', 'Cor'):
    print(f'\n=== {metric} ===')
    print(f'{"min_votes":<10}' + ''.join(f'{c:>10}' for c in cols))
    for mv in mv_list:
        row = f'{mv:<10}'
        for c in cols:
            d = grab(os.path.join(out_root, f'minvotes{mv}', f'scorer_stdout.{c}.txt'))
            v = d[metric] if d else None
            row += f'{v:>10.4f}' if v is not None else f'{"-":>10}'
        print(row)

print('\n=== ALL: precision / recall ===')
print(f'{"min_votes":<10}{"prec":>10}{"rec":>10}')
for mv in mv_list:
    d = grab(os.path.join(out_root, f'minvotes{mv}', 'scorer_stdout.ALL.txt'))
    if d and d['prec'] is not None:
        print(f'{mv:<10}{d["prec"]:>10.4f}{d["rec"]:>10.4f}')
PYEOF

echo
echo "Done. Per-setting outputs: ${OUTPUT_DIR}/minvotes<N>/${SPLIT}.<lang>.labeled.offsets.jsonl"
echo "Pick the best MIN_VOTES PER LANGUAGE (submissions are ranked separately per language)."