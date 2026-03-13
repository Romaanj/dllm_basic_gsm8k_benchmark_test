#!/bin/bash
# High-Mass Neighborhood Fragmentation: equal-mass vs inverse-CDF 비교
#
# 사용법:
#   bash eval_neighborhood_fragmentation.sh
#
# Docker 안에서 실행:
#   docker exec -it dllm_ljw bash -c "cd /workspace/dllm_ljw/Fast-dLLM/llada && bash eval_neighborhood_fragmentation.sh"

set -euo pipefail

DEVICE="${DEVICE:-cuda:3}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
START_ID="${START_ID:-0}"
NUM_BLOCKS="${NUM_BLOCKS:-8}"
LAM="${LAM:-1.0}"
ROLLOUT_MODE="${ROLLOUT_MODE:-sigmoid}"
TOP_K="${TOP_K:-3}"
WINDOW_RADIUS="${WINDOW_RADIUS:-8}"
OUT_DIR="${OUT_DIR:-results_equal_mass/high_mass_probe}"

echo "=================================================="
echo " High-Mass Neighborhood Fragmentation Experiment"
echo "=================================================="
echo " device:        ${DEVICE}"
echo " num_samples:   ${NUM_SAMPLES}"
echo " start_id:      ${START_ID}"
echo " num_blocks:    ${NUM_BLOCKS}"
echo " lam:           ${LAM}"
echo " rollout_mode:  ${ROLLOUT_MODE}"
echo " top_k:         ${TOP_K}"
echo " window_radius: ${WINDOW_RADIUS}"
echo " out_dir:       ${OUT_DIR}"
echo "=================================================="

python gsm8k_high_mass_neighborhood_fragmentation_eval.py \
    --device "${DEVICE}" \
    --num-samples "${NUM_SAMPLES}" \
    --start-id "${START_ID}" \
    --num-blocks "${NUM_BLOCKS}" \
    --lam "${LAM}" \
    --rollout-mode "${ROLLOUT_MODE}" \
    --top-k-high-mass "${TOP_K}" \
    --window-radius "${WINDOW_RADIUS}" \
    --out-dir "${OUT_DIR}"

echo ""
echo "[Done] Results saved to ${OUT_DIR}"
