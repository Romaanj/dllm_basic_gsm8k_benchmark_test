#!/bin/bash
# Equal-Mass Chunking lm-evaluation-harness 기반 MBPP 평가 스크립트 (3-shot)

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=mbpp
num_fewshot=3
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

# # ─── 1. Fixed-block baseline (비교용, 옵션) ───
# echo "=== Running MBPP fixed_block baseline (3-shot) ==="
# CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 eval_equal_mass.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_equal_mass \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},block_length=${block_length},strategy=fixed_block,threshold=${threshold},show_speed=True,save_dir=${output_dir}/mbpp_fixed_block_new \
#     --output_path ${output_dir}/mbpp_fixed_block_new --log_samples

# # ─── 2. Equal-mass sigmoid (N=6) ───
# echo "=== Running MBPP equal_mass_sigmoid (3-shot, N=6) ==="
# CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 eval_equal_mass.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_equal_mass \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=equal_mass_sigmoid,num_blocks=${num_blocks},min_block_size=${min_block_size},max_block_size=${max_block_size},threshold=${threshold},show_speed=True,save_dir=${output_dir}/mbpp_equal_mass_sigmoid_N${num_blocks} \
#     --output_path ${output_dir}/mbpp_equal_mass_sigmoid_N${num_blocks}.json

# ─── 3. Equal-mass sigmoid (N=8) ───
num_blocks=8
echo "=== Running MBPP equal_mass_sigmoid (3-shot, N=8) ==="
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 eval_equal_mass.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_equal_mass \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=equal_mass_sigmoid,num_blocks=${num_blocks},min_block_size=${min_block_size},max_block_size=${max_block_size},threshold=${threshold},show_speed=True,save_dir=${output_dir}/mbpp_equal_mass_sigmoid_N_new${num_blocks} \
    --output_path ${output_dir}/mbpp_equal_mass_sigmoid_N_new${num_blocks} --log_samples

echo "=== Done ==="

