#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:2}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
START_ID="${START_ID:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-8}"
STEPS_PER_BLOCK="${STEPS_PER_BLOCK:-32}"
WINDOW_RADIUS="${WINDOW_RADIUS:-4}"
TOP_K_PER_TYPE="${TOP_K_PER_TYPE:-3}"
THRESHOLD="${THRESHOLD:-0.9}"
TAU="${TAU:-0.8}"
ROLLOUT_MODE="${ROLLOUT_MODE:-sigmoid}"
OUT_DIR="${OUT_DIR:-results_window_fragmentation_keep_split}"

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
echo "  Window Fragmentation (keep-whole vs force-split)"
echo "════════════════════════════════════════════════════════════════════"
echo "  Device:          ${DEVICE}"
echo "  Samples:         ${NUM_SAMPLES} (start=${START_ID})"
echo "  Blocks:          ${NUM_BLOCKS}"
echo "  Steps/block:     ${STEPS_PER_BLOCK}"
echo "  Window radius:   ${WINDOW_RADIUS}"
echo "  Top-k/type:      ${TOP_K_PER_TYPE}"
echo "  Threshold:       ${THRESHOLD}"
echo "  PMR tau:         ${TAU}"
echo "  Rollout mode:    ${ROLLOUT_MODE}"
echo "  Output dir:      ${OUT_DIR}"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python gsm8k_window_fragmentation_keep_split_eval.py \
    --device "${DEVICE}" \
    --num-samples "${NUM_SAMPLES}" \
    --start-id "${START_ID}" \
    --num-blocks "${NUM_BLOCKS}" \
    --steps-per-block "${STEPS_PER_BLOCK}" \
    --window-radius "${WINDOW_RADIUS}" \
    --top-k-per-type "${TOP_K_PER_TYPE}" \
    --threshold "${THRESHOLD}" \
    --tau "${TAU}" \
    --rollout-mode "${ROLLOUT_MODE}" \
    --out-dir "${OUT_DIR}"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Done. Results saved to: ${OUT_DIR}/"
echo "════════════════════════════════════════════════════════════════════"
