#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:2}"
TASK="${TASK:-gsm8k}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
START_ID="${START_ID:-0}"
GEN_LENGTH="${GEN_LENGTH:-256}"
WINDOW_RADIUS="${WINDOW_RADIUS:-4}"
TOP_K_PER_TYPE="${TOP_K_PER_TYPE:-10}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ROLLOUT_MODE="${ROLLOUT_MODE:-sigmoid}"
OUT_DIR="${OUT_DIR:-results_baseline_entropy_events}"

for arg in "$@"; do
    case "$arg" in
        --quick)
            NUM_SAMPLES=3
            ;;
        --full)
            NUM_SAMPLES=50
            ;;
    esac
done

echo "════════════════════════════════════════════════════════════════════"
echo "  Baseline Entropy Event Experiment (High vs Low rollout)"
echo "════════════════════════════════════════════════════════════════════"
echo "  Device:          ${DEVICE}"
echo "  Task:            ${TASK}"
echo "  Samples:         ${NUM_SAMPLES} (start=${START_ID})"
echo "  Gen length:      ${GEN_LENGTH}"
echo "  Window radius:   ${WINDOW_RADIUS}"
echo "  Top-k/type:      ${TOP_K_PER_TYPE}"
echo "  Temperature:     ${TEMPERATURE}"
echo "  Rollout mode:    ${ROLLOUT_MODE}"
echo "  Output dir:      ${OUT_DIR}"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python3 gsm8k_baseline_entropy_event_eval.py \
    --device "${DEVICE}" \
    --task "${TASK}" \
    --num-samples "${NUM_SAMPLES}" \
    --start-id "${START_ID}" \
    --gen-length "${GEN_LENGTH}" \
    --window-radius "${WINDOW_RADIUS}" \
    --top-k-per-type "${TOP_K_PER_TYPE}" \
    --temperature "${TEMPERATURE}" \
    --rollout-mode "${ROLLOUT_MODE}" \
    --out-dir "${OUT_DIR}"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Done. Results saved to: ${OUT_DIR}/"
echo "════════════════════════════════════════════════════════════════════"
