#!/bin/bash
# Equal-Mass Chunking lm-evaluation-harness 기반 GSM8K 평가 스크립트

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
num_fewshot=5
model_path='GSAI-ML/LLaDA-8B-Instruct'
output_dir=results_original_cdf_lmeval
mkdir -p ${output_dir}

gen_length=256
steps_per_block=32
num_blocks=8
min_block_size=1
max_block_size=100
threshold=0.9
block_length=32

# # ─── 1. Fixed-block baseline (비교용) ───
# echo "=== Running fixed_block baseline ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_equal_mass.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_equal_mass \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},block_length=${block_length},strategy=fixed_block,threshold=${threshold},show_speed=True \
#     --output_path ${output_dir}/gsm8k_fixed_block.json

# ─── 2. Equal-mass sigmoid ───
echo "=== Running equal_mass_sigmoid ==="
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 eval_equal_mass.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_equal_mass \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=equal_mass_sigmoid,num_blocks=${num_blocks},min_block_size=${min_block_size},max_block_size=${max_block_size},threshold=${threshold},show_speed=True \
    --output_path ${output_dir}/gsm8k_equal_mass_sigmoid_N${num_blocks}.json

# num_blocks=8
# # ─── 2. Equal-mass sigmoid ───
# echo "=== Running equal_mass_sigmoid ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_equal_mass.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_equal_mass \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=equal_mass_sigmoid,num_blocks=${num_blocks},min_block_size=${min_block_size},max_block_size=${max_block_size},threshold=${threshold},show_speed=True \
#     --output_path ${output_dir}/gsm8k_equal_mass_sigmoid_N${num_blocks}.json

# ─── 3. Equal-mass baseline rollout (optional) ───
# echo "=== Running equal_mass_baseline ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_equal_mass.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_equal_mass \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=equal_mass_baseline,num_blocks=${num_blocks},min_block_size=${min_block_size},max_block_size=${max_block_size},threshold=${threshold},show_speed=True \
#     --output_path ${output_dir}/gsm8k_equal_mass_baseline_N${num_blocks}.json

echo "=== Done ==="
