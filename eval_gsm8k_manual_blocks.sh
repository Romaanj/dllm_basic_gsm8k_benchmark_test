#!/bin/bash
# Manual-block GSM8K lm-evaluation-harness script

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
num_fewshot=5
model_path="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
gen_length="${GEN_LENGTH:-256}"
manual_block_sizes="${MANUAL_BLOCK_SIZES:-64|28|28|28|27|27|27|27}"
output_dir="${OUTPUT_DIR:-results_gsm8k_manual_${manual_block_sizes//|/_}_seed${SEED}_len${gen_length}}"
mkdir -p "${output_dir}"

steps_per_block=32
threshold=0.9

echo "=== Running GSM8K manual_blocks (sizes=${manual_block_sizes}, gen_length=${gen_length}) ==="
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --num_processes 1 eval_hybrid_cdf.py \
    --tasks ${task} \
    --model llada_hybrid_cdf \
    --model_args "model_path=${model_path},gen_length=${gen_length},strategy=manual_blocks,manual_block_sizes=${manual_block_sizes},steps_per_block=${steps_per_block},threshold=${threshold},seed=${SEED},save_dir=${output_dir}/inference" \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --log_samples \
    --confirm_run_unsafe_code \
    --output_path ${output_dir}/gsm8k_manual_results.json

echo "=== Done ==="
