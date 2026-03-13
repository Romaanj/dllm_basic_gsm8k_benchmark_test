#!/bin/bash
# Hybrid-CDF Equal-Mass lm-evaluation-harness 기반 GSM8K 평가 스크립트
#
# hybrid_cdf = λ * attention_cdf + (1-λ) * uniform_cdf
# λ=1.0 → 순수 attention,  λ=0.0 → 균등 분할

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
num_fewshot=5
model_path='GSAI-ML/LLaDA-8B-Instruct'
output_dir=results_hybrid_cdf_lmeval
mkdir -p ${output_dir}

gen_length=256
steps_per_block=32
num_blocks=8
threshold=0.9
block_length=32

# ─── 1. Hybrid-CDF sigmoid (λ=0.7) ───
lam=0.7
echo "=== Running hybrid_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_hybrid_cdf \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},show_speed=True,save_dir=${output_dir}/gsm8k_hybrid_cdf_lam${lam}_N${num_blocks} \
    --output_path ${output_dir}/gsm8k_hybrid_cdf_sigmoid_lam${lam}_N${num_blocks}.json

# ─── 2. Hybrid-CDF sigmoid (λ=0.5) ───
lam=0.5
echo "=== Running hybrid_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_hybrid_cdf \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},show_speed=True,save_dir=${output_dir}/gsm8k_hybrid_cdf_lam${lam}_N${num_blocks} \
    --output_path ${output_dir}/gsm8k_hybrid_cdf_sigmoid_lam${lam}_N${num_blocks}.json

# ─── 3. Hybrid-CDF sigmoid (λ=0.3) ───
lam=0.3
echo "=== Running hybrid_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_hybrid_cdf \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},show_speed=True,save_dir=${output_dir}/gsm8k_hybrid_cdf_lam${lam}_N${num_blocks} \
    --output_path ${output_dir}/gsm8k_hybrid_cdf_sigmoid_lam${lam}_N${num_blocks}.json

# # ─── 4. Fixed-block baseline (비교용, 주석 해제 시 사용) ───
# echo "=== Running fixed_block baseline ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_hybrid_cdf \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},block_length=${block_length},strategy=fixed_block,threshold=${threshold},show_speed=True,save_dir=${output_dir}/gsm8k_fixed_block \
#     --output_path ${output_dir}/gsm8k_fixed_block.json

echo "=== Done ==="
