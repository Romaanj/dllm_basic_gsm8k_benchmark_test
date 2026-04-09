#!/bin/bash
# Smoothed Inverse-CDF lm-evaluation-harness GSM8K script
#
# Moving average smoothing → inverse → CDF equal-mass partition
# peak의 영향을 주변으로 퍼뜨린 뒤 inverse CDF로 블록 분할

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
num_fewshot=5
model_path="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
GPU_ID="${GPU_ID:-2}"
SEED="${SEED:-42}"
gen_length="${GEN_LENGTH:-512}"
num_blocks="${NUM_BLOCKS:-24}"
smoothing_window="${SMOOTHING_WINDOW:-32}"
output_dir="${OUTPUT_DIR:-final_results/gsm8k_smoothed_inverse_cdf_len${gen_length}_b${num_blocks}_w${smoothing_window}}"
mkdir -p "${output_dir}"

steps_per_block=32
threshold=0.9

echo "=== Running GSM8K smoothed_inverse_cdf (N=${num_blocks}, gen_length=${gen_length}, window=${smoothing_window}) ==="
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --num_processes 1 eval_smoothed_inverse_cdf.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_smoothed_inverse_cdf \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},num_blocks=${num_blocks},threshold=${threshold},smoothing_window=${smoothing_window},seed=${SEED},show_speed=True,save_dir=${output_dir}/gsm8k_smoothed_inverse_cdf_N${num_blocks}_w${smoothing_window} \
    --output_path ${output_dir}/gsm8k_smoothed_inverse_cdf_N${num_blocks}_w${smoothing_window}.json --log_samples

echo "=== Done ==="
