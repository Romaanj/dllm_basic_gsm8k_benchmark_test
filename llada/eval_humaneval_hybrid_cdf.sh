#!/bin/bash
# Hybrid-CDF Equal-Mass lm-evaluation-harness 기반 HumanEval 평가 스크립트
#
# hybrid_cdf = λ * attention_cdf + (1-λ) * uniform_cdf

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
model_path='GSAI-ML/LLaDA-8B-Instruct'
output_dir=results_hybrid_cdf_lmeval
mkdir -p ${output_dir}

gen_length=256
steps_per_block=32
num_blocks=16
threshold=0.9
block_length=32

# ─── 1. Hybrid-CDF sigmoid (λ=0.7) ───
lam=1.0
echo "=== Running HumanEval hybrid_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
    --tasks ${task} \
    --confirm_run_unsafe_code --model llada_hybrid_cdf \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},show_speed=True,save_dir=${output_dir}/humaneval_hybrid_cdf_lam${lam}_N${num_blocks} \
    --output_path ${output_dir}/humaneval_hybrid_cdf_sigmoid_lam${lam}_N${num_blocks} --log_samples


# ─── 1. Hybrid-CDF sigmoid (λ=0.7) ───
lam=0.7
echo "=== Running HumanEval hybrid_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
    --tasks ${task} \
    --confirm_run_unsafe_code --model llada_hybrid_cdf \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},show_speed=True,save_dir=${output_dir}/humaneval_hybrid_cdf_lam${lam}_N${num_blocks} \
    --output_path ${output_dir}/humaneval_hybrid_cdf_sigmoid_lam${lam}_N${num_blocks} --log_samples

# ─── 2. Hybrid-CDF sigmoid (λ=0.5) ───
lam=0.5
echo "=== Running HumanEval hybrid_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
    --tasks ${task} \
    --confirm_run_unsafe_code --model llada_hybrid_cdf \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},show_speed=True,save_dir=${output_dir}/humaneval_hybrid_cdf_lam${lam}_N${num_blocks} \
    --output_path ${output_dir}/humaneval_hybrid_cdf_sigmoid_lam${lam}_N${num_blocks} --log_samples

# # ─── 3. Hybrid-CDF sigmoid (λ=0.3) ───
# lam=0.3
# echo "=== Running HumanEval hybrid_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
# CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
#     --tasks ${task} \
#     --confirm_run_unsafe_code --model llada_hybrid_cdf \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},show_speed=True,save_dir=${output_dir}/humaneval_hybrid_cdf_lam${lam}_N${num_blocks} \
#     --output_path ${output_dir}/humaneval_hybrid_cdf_sigmoid_lam${lam}_N${num_blocks} --log_samples

# # ─── 4. Fixed-block baseline (비교용, 주석 해제 시 사용) ───
# echo "=== Running HumanEval fixed_block baseline ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
#     --tasks ${task} \
#     --confirm_run_unsafe_code --model llada_hybrid_cdf \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},block_length=${block_length},strategy=fixed_block,threshold=${threshold},show_speed=True,save_dir=${output_dir}/humaneval_fixed_block \
#     --output_path ${output_dir}/humaneval_fixed_block --log_samples

echo "=== Done ==="
echo ""
echo "NOTICE: HumanEval 결과에 대해 postprocess를 수행하세요:"
echo "  python postprocess_code.py {output_dir 내 samples_xxx.jsonl 파일}"
