#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# High-Mass Neighborhood Unmask Tracking: equal-mass vs inverse-CDF
# ═══════════════════════════════════════════════════════════════════════════
#
# 핵심 질문:
#   1. High-mass region 안에는 어떤 category 토큰이 enrichment 되는가?
#   2. 늦게 unmask되는 토큰은 어떤 category인가?
#   3. Inverse-CDF가 semantic 토큰을 더 coherent하게 같이 unmask하는가?
#
# Usage:
#   bash eval_unmask_tracking.sh                    # 기본 (10 samples)
#   bash eval_unmask_tracking.sh --quick            # 빠른 테스트 (3 samples)
#   bash eval_unmask_tracking.sh --full             # 전체 (50 samples)
#   NUM_SAMPLES=20 DEVICE=cuda:1 bash eval_unmask_tracking.sh
# ═══════════════════════════════════════════════════════════════════════════

# ── Configurable parameters ──
DEVICE="${DEVICE:-cuda:2}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
START_ID="${START_ID:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-8}"
STEPS_PER_BLOCK="${STEPS_PER_BLOCK:-32}"
LAM="${LAM:-1.0}"
THRESHOLD="${THRESHOLD:-0.9}"
ROLLOUT_MODE="${ROLLOUT_MODE:-sigmoid}"
TOP_K="${TOP_K:-3}"
WINDOW_RADIUS="${WINDOW_RADIUS:-4}"
LATE_RATIO="${LATE_RATIO:-0.25}"
LATE_TOP_N="${LATE_TOP_N:-3}"
OUT_DIR="${OUT_DIR:-results_high_mass_unmask_tracking}"

# ── Parse quick/full mode ──
for arg in "$@"; do
    case $arg in
        --quick)
            NUM_SAMPLES=3
            ;;
        --full)
            NUM_SAMPLES=50
            ;;
    esac
done

echo "════════════════════════════════════════════════════════════════════"
echo "  High-Mass Neighborhood Unmask Tracking"
echo "════════════════════════════════════════════════════════════════════"
echo "  Device:          ${DEVICE}"
echo "  Samples:         ${NUM_SAMPLES} (start=${START_ID})"
echo "  Blocks:          ${NUM_BLOCKS}"
echo "  Steps/block:     ${STEPS_PER_BLOCK}"
echo "  Lambda:          ${LAM}"
echo "  Threshold:       ${THRESHOLD}"
echo "  Rollout mode:    ${ROLLOUT_MODE}"
echo "  Top-k:           ${TOP_K}"
echo "  Window radius:   ${WINDOW_RADIUS} (window size = $((WINDOW_RADIUS * 2 + 1)))"
echo "  Late ratio:      ${LATE_RATIO}"
echo "  Late top-n:      ${LATE_TOP_N}"
echo "  Output dir:      ${OUT_DIR}"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python gsm8k_high_mass_unmask_tracking_eval.py \
    --device "${DEVICE}" \
    --num-samples "${NUM_SAMPLES}" \
    --start-id "${START_ID}" \
    --num-blocks "${NUM_BLOCKS}" \
    --steps-per-block "${STEPS_PER_BLOCK}" \
    --lam "${LAM}" \
    --threshold "${THRESHOLD}" \
    --rollout-mode "${ROLLOUT_MODE}" \
    --top-k-high-mass "${TOP_K}" \
    --window-radius "${WINDOW_RADIUS}" \
    --late-ratio "${LATE_RATIO}" \
    --late-top-n "${LATE_TOP_N}" \
    --out-dir "${OUT_DIR}"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Done. Results saved to: ${OUT_DIR}/"
echo "════════════════════════════════════════════════════════════════════"
