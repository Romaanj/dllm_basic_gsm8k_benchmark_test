#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# Block-Free Confidence Maturation Probe
# ═══════════════════════════════════════════════════════════════════════════
#
# Block partition 영향을 제거하고 diffusion 모델 자체가
# Step-0 high-mass region을 어떻게 다루는지 분석.
#
# 실험 1: Confidence Maturation (0.5/0.8/0.9 도달 step, stabilization)
# 실험 2: Token Category Enrichment (semantic-bearing token 비율)
# 실험 3: Coupling Probe (trajectory correlation, convergence sync)
#
# Usage:
#   bash eval_block_free_probe.sh                    # 기본 (10 samples, tps=1)
#   bash eval_block_free_probe.sh --quick            # 빠른 테스트 (3 samples, tps=4)
#   bash eval_block_free_probe.sh --full             # 전체 (30 samples, tps=1)
#   bash eval_block_free_probe.sh --fast             # 빠른 근사 (10 samples, tps=4)
#   DEVICE=cuda:1 NUM_SAMPLES=20 bash eval_block_free_probe.sh
#
# 예상 실행 시간 (8B model, single GPU):
#   tps=1, 10 samples: ~30-50 min  (256 fwd passes/sample)
#   tps=4, 10 samples: ~8-15 min   (64 fwd passes/sample)
#   tps=1, 30 samples: ~1.5-2.5 hr
# ═══════════════════════════════════════════════════════════════════════════

# ── Configurable parameters ──
DEVICE="${DEVICE:-cuda:2}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
START_ID="${START_ID:-0}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-1}"
ROLLOUT_MODE="${ROLLOUT_MODE:-sigmoid}"
TOP_K="${TOP_K:-3}"
WINDOW_RADIUS="${WINDOW_RADIUS:-4}"
OUT_DIR="${OUT_DIR:-results_block_free_probe}"

# ── Parse mode flags ──
for arg in "$@"; do
    case $arg in
        --quick)
            NUM_SAMPLES=3
            TOKENS_PER_STEP=4
            ;;
        --fast)
            TOKENS_PER_STEP=4
            ;;
        --full)
            NUM_SAMPLES=30
            TOKENS_PER_STEP=1
            ;;
    esac
done

EST_FWD=$((256 / TOKENS_PER_STEP))
EST_TOTAL=$((EST_FWD * NUM_SAMPLES))

echo "════════════════════════════════════════════════════════════════════"
echo "  Block-Free Confidence Maturation Probe"
echo "════════════════════════════════════════════════════════════════════"
echo "  Device:              ${DEVICE}"
echo "  Samples:             ${NUM_SAMPLES} (start=${START_ID})"
echo "  Tokens per step:     ${TOKENS_PER_STEP}"
echo "  Rollout mode:        ${ROLLOUT_MODE}"
echo "  Top-k:               ${TOP_K}"
echo "  Window radius:       ${WINDOW_RADIUS} (window size = $((WINDOW_RADIUS * 2 + 1)))"
echo "  Output dir:          ${OUT_DIR}"
echo "  Est. fwd passes:     ~${EST_FWD}/sample × ${NUM_SAMPLES} = ~${EST_TOTAL} total"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python gsm8k_block_free_confidence_probe.py \
    --device "${DEVICE}" \
    --num-samples "${NUM_SAMPLES}" \
    --start-id "${START_ID}" \
    --tokens-per-step "${TOKENS_PER_STEP}" \
    --rollout-mode "${ROLLOUT_MODE}" \
    --top-k-high-mass "${TOP_K}" \
    --window-radius "${WINDOW_RADIUS}" \
    --out-dir "${OUT_DIR}"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Done. Results saved to: ${OUT_DIR}/"
echo "════════════════════════════════════════════════════════════════════"
