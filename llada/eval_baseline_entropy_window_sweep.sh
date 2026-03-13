#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:2}"
TASK="${TASK:-humaneval}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
START_ID="${START_ID:-0}"
GEN_LENGTH="${GEN_LENGTH:-256}"
TOP_K_PER_TYPE="${TOP_K_PER_TYPE:-10}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ROLLOUT_MODE="${ROLLOUT_MODE:-sigmoid}"
OUT_DIR="${OUT_DIR:-results_baseline_entropy_events_humaneval}"
WINDOW_LIST="${WINDOW_LIST:-1,2,4,8,10}"
METRIC="${METRIC:-local_mean_delta}"
VALUE_COL="${VALUE_COL:-local_entropy_mean_delta}"
MEAN_MODE="${MEAN_MODE:-event_mean}"

for arg in "$@"; do
    case "$arg" in
        --quick)
            NUM_SAMPLES=3
            TOP_K_PER_TYPE=3
            ;;
        --full)
            NUM_SAMPLES=50
            ;;
    esac
done

echo "════════════════════════════════════════════════════════════════════"
echo "  Baseline Entropy Local Window Sweep"
echo "════════════════════════════════════════════════════════════════════"
echo "  Device:          ${DEVICE}"
echo "  Task:            ${TASK}"
echo "  Samples:         ${NUM_SAMPLES} (start=${START_ID})"
echo "  Gen length:      ${GEN_LENGTH}"
echo "  Top-k/type:      ${TOP_K_PER_TYPE}"
echo "  Temperature:     ${TEMPERATURE}"
echo "  Rollout mode:    ${ROLLOUT_MODE}"
echo "  Window list:     ${WINDOW_LIST}"
echo "  Metric:          ${METRIC}"
echo "  Value column:    ${VALUE_COL}"
echo "  Mean mode:       ${MEAN_MODE}"
echo "  Output dir:      ${OUT_DIR}"
echo "════════════════════════════════════════════════════════════════════"
echo ""

IFS=',' read -r -a WINDOWS <<< "${WINDOW_LIST}"
for W in "${WINDOWS[@]}"; do
    W_TRIMMED="$(echo "${W}" | xargs)"
    if [[ -z "${W_TRIMMED}" ]]; then
        continue
    fi
    echo "[Run] window_radius=${W_TRIMMED}"
    python3 gsm8k_baseline_entropy_event_eval.py \
        --device "${DEVICE}" \
        --task "${TASK}" \
        --num-samples "${NUM_SAMPLES}" \
        --start-id "${START_ID}" \
        --gen-length "${GEN_LENGTH}" \
        --window-radius "${W_TRIMMED}" \
        --top-k-per-type "${TOP_K_PER_TYPE}" \
        --temperature "${TEMPERATURE}" \
        --rollout-mode "${ROLLOUT_MODE}" \
        --out-dir "${OUT_DIR}"
done

echo ""
echo "[Plot] local entropy mean delta vs W"
python3 plot_baseline_local_entropy_window_sweep.py \
    --results-dir "${OUT_DIR}" \
    --window-list "${WINDOW_LIST}" \
    --task "${TASK}" \
    --metric "${METRIC}" \
    --value-col "${VALUE_COL}" \
    --mean-mode "${MEAN_MODE}" \
    --out-prefix "baseline_local_entropy_window_sweep_task-${TASK}"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Done. Plots and summaries saved to: ${OUT_DIR}/"
echo "════════════════════════════════════════════════════════════════════"
