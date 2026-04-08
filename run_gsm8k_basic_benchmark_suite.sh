#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID="${GPU_ID:-2}"
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
GEN_LENGTH="${GEN_LENGTH:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-final_results/gsm8k_basic_benchmark_len${GEN_LENGTH}_b${BLOCK_LENGTH}}"

if (( GEN_LENGTH % BLOCK_LENGTH != 0 )); then
    echo "[ERROR] GEN_LENGTH=${GEN_LENGTH} must be divisible by BLOCK_LENGTH=${BLOCK_LENGTH}."
    exit 1
fi

NUM_BLOCKS=$((GEN_LENGTH / BLOCK_LENGTH))
mkdir -p "${OUTPUT_ROOT}"

run_step() {
    local name="$1"
    shift
    echo "============================================================"
    echo "[START] ${name}"
    echo "============================================================"
    "$@"
    echo "[DONE ] ${name}"
    echo ""
}

run_step "basic_inverse_cdf" env     GPU_ID="${GPU_ID}"     MODEL_PATH="${MODEL_PATH}"     GEN_LENGTH="${GEN_LENGTH}"     NUM_BLOCKS="${NUM_BLOCKS}"     SEED="${SEED}"     OUTPUT_DIR="${OUTPUT_ROOT}/inverse_cdf"     bash "${SCRIPT_DIR}/eval_gsm8k_basic_inverse_cdf.sh"

run_step "basic_fixed" env     GPU_ID="${GPU_ID}"     MODEL_PATH="${MODEL_PATH}"     GEN_LENGTH="${GEN_LENGTH}"     BLOCK_LENGTH="${BLOCK_LENGTH}"     SEED="${SEED}"     OUTPUT_DIR="${OUTPUT_ROOT}/fixed"     bash "${SCRIPT_DIR}/eval_gsm8k_basic_fixed.sh"

run_step "basic_argmax1" env     GPU_ID="${GPU_ID}"     MODEL_PATH="${MODEL_PATH}"     GEN_LENGTH="${GEN_LENGTH}"     BLOCK_LENGTH="${BLOCK_LENGTH}"     SEED="${SEED}"     OUTPUT_DIR="${OUTPUT_ROOT}/argmax1"     bash "${SCRIPT_DIR}/eval_gsm8k_basic_argmax1.sh"

echo "All GSM8K basic benchmark runs completed."
