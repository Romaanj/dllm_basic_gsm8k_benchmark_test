#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# Anchor Intervention Probe: Oracle Reveal & Delay Unmask
# ═══════════════════════════════════════════════════════════════════════════
#
# 실험 A: Oracle Reveal — window 토큰을 즉시 공개, 주변 confidence lift 측정
# 실험 B: Delay Unmask  — window 토큰 unmask를 d step 지연, 주변 손해 측정
#
# Usage:
#   bash eval_anchor_intervention.sh                    # 기본 (10 samples, tps=1)
#   bash eval_anchor_intervention.sh --quick            # 빠른 테스트 (3 samples, tps=4)
#   bash eval_anchor_intervention.sh --fast             # 빠른 근사 (10 samples, tps=4)
#   bash eval_anchor_intervention.sh --full             # 전체 (20 samples, tps=1)
#   DEVICE=cuda:1 bash eval_anchor_intervention.sh
#
# 예상 실행 시간 (8B model, single GPU, top_k=3):
#   tps=1, 10 samples: ~2-3 hr  (10 runs/sample × 256 fwd × 10 samples)
#   tps=4, 10 samples: ~30-45 min
#   tps=1,  3 samples: ~40-60 min
# ═══════════════════════════════════════════════════════════════════════════

DEVICE="${DEVICE:-cuda:2}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
START_ID="${START_ID:-0}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-1}"
ROLLOUT_MODE="${ROLLOUT_MODE:-sigmoid}"
TOP_K="${TOP_K:-3}"
WINDOW_RADIUS="${WINDOW_RADIUS:-4}"
NEIGHBOR_RADIUS="${NEIGHBOR_RADIUS:-8}"
DELAY_STEPS="${DELAY_STEPS:-20}"
K_VALUES="${K_VALUES:-1,2,4,8,16,32,64}"
OUT_DIR="${OUT_DIR:-results_anchor_intervention}"
EXTRA_ARGS=""

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
            NUM_SAMPLES=20
            ;;
        --skip-low-delay)
            EXTRA_ARGS="${EXTRA_ARGS} --skip-low-mass-delay"
            ;;
    esac
done

N_RUNS=$((1 + TOP_K * 4))
FWD_PER_RUN=$((256 / TOKENS_PER_STEP))
EST_TOTAL=$((N_RUNS * FWD_PER_RUN * NUM_SAMPLES))

echo "════════════════════════════════════════════════════════════════════"
echo "  Anchor Intervention Probe"
echo "════════════════════════════════════════════════════════════════════"
echo "  Device:              ${DEVICE}"
echo "  Samples:             ${NUM_SAMPLES} (start=${START_ID})"
echo "  Tokens per step:     ${TOKENS_PER_STEP}"
echo "  Top-k:               ${TOP_K}"
echo "  Window radius:       ${WINDOW_RADIUS} (size=$((WINDOW_RADIUS * 2 + 1)))"
echo "  Neighbor radius:     ${NEIGHBOR_RADIUS}"
echo "  Delay steps:         ${DELAY_STEPS}"
echo "  k values:            ${K_VALUES}"
echo "  Output dir:          ${OUT_DIR}"
echo "  Est. runs/sample:    ~${N_RUNS}  (1 base + ${TOP_K}×4 interventions)"
echo "  Est. total fwd:      ~${EST_TOTAL}"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python gsm8k_anchor_intervention_probe.py \
    --device "${DEVICE}" \
    --num-samples "${NUM_SAMPLES}" \
    --start-id "${START_ID}" \
    --tokens-per-step "${TOKENS_PER_STEP}" \
    --rollout-mode "${ROLLOUT_MODE}" \
    --top-k-high-mass "${TOP_K}" \
    --window-radius "${WINDOW_RADIUS}" \
    --neighbor-radius "${NEIGHBOR_RADIUS}" \
    --delay-steps "${DELAY_STEPS}" \
    --k-values "${K_VALUES}" \
    --out-dir "${OUT_DIR}" \
    ${EXTRA_ARGS}

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Done. Results saved to: ${OUT_DIR}/"
echo "════════════════════════════════════════════════════════════════════"

