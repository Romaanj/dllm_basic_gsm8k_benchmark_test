#!/bin/bash
# Basic block-local argmax1 lm-evaluation-harness GSM8K script

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
num_fewshot=5
model_path="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
GPU_ID="${GPU_ID:-2}"
SEED="${SEED:-42}"
gen_length="${GEN_LENGTH:-512}"
block_length="${BLOCK_LENGTH:-32}"
output_dir="${OUTPUT_DIR:-final_results/gsm8k_basic_argmax1_len${gen_length}_b${block_length}}"
mkdir -p "${output_dir}"

echo "=== Running GSM8K basic_argmax1 (B=${block_length}, gen_length=${gen_length}) ==="
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --num_processes 1 eval_argmax1_basic.py     --tasks ${task} --num_fewshot ${num_fewshot}     --confirm_run_unsafe_code --model llada_argmax1_basic     --model_args model_path=${model_path},gen_length=${gen_length},block_length=${block_length},seed=${SEED},show_speed=True,save_dir=${output_dir}/gsm8k_basic_argmax1_B${block_length}     --output_path ${output_dir}/gsm8k_basic_argmax1_B${block_length}.json --log_samples

echo "=== Done ==="
