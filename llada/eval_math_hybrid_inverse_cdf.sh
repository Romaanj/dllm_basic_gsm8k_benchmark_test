#!/bin/bash
# Hybrid-Inverse-CDF lm-evaluation-harness 기반 minerva_math (MATH) 평가 스크립트
#
# 가설: semantic complexity가 높은 구간(attention 집중)에 더 큰 블록을 배정하면
#       contextual thinking budget이 늘어나 정보 손실이 줄어든다.
#
# 핵심: inverse=True → scores의 역수로 CDF를 구성
#   inv_scores = 1 / (scores + ε)  →  정규화  →  cumsum  →  inverse_attn_cdf
#   hybrid_cdf = λ * inverse_attn_cdf + (1-λ) * uniform_cdf
#
# 의존성: pip install lm-eval[math]

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=minerva_math
num_fewshot=4
model_path='GSAI-ML/LLaDA-8B-Instruct'
output_dir=results_hybrid_inverse_cdf_lmeval
mkdir -p ${output_dir}

gen_length=256
steps_per_block=32
threshold=0.9
lam=1.0

# ─── 1. Inverse-CDF sigmoid (N=8) ───
num_blocks=8
echo "=== Running minerva_math hybrid_inverse_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_hybrid_cdf \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},inverse=True,show_speed=True,save_dir=${output_dir}/minerva_math_hybrid_inverse_cdf_lam${lam}_N${num_blocks} \
    --output_path ${output_dir}/minerva_math_hybrid_inverse_cdf_sigmoid_lam${lam}_N${num_blocks} --log_samples

# ----2. Fixed-block baseline ----
echo "=== Running minerva_math fixed_block baseline ==="
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_hybrid_cdf \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=fixed_block,threshold=${threshold},show_speed=True,save_dir=${output_dir}/minerva_math_fixed_block \
    --output_path ${output_dir}/minerva_math_fixed_block.json --log_samples



# # ─── 2. Inverse-CDF sigmoid (N=16) ───
# num_blocks=16
# echo "=== Running minerva_math hybrid_inverse_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_hybrid_cdf \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},inverse=True,show_speed=True,save_dir=${output_dir}/minerva_math_hybrid_inverse_cdf_lam${lam}_N${num_blocks} \
#     --output_path ${output_dir}/minerva_math_hybrid_inverse_cdf_sigmoid_lam${lam}_N${num_blocks}.json

# # ─── 3. Inverse-CDF sigmoid (N=12) ───
# num_blocks=12
# echo "=== Running minerva_math hybrid_inverse_cdf_sigmoid (λ=${lam}, N=${num_blocks}) ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_hybrid_cdf.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_hybrid_cdf \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=hybrid_cdf_sigmoid,num_blocks=${num_blocks},lam=${lam},threshold=${threshold},inverse=True,show_speed=True,save_dir=${output_dir}/minerva_math_hybrid_inverse_cdf_lam${lam}_N${num_blocks} \
#     --output_path ${output_dir}/minerva_math_hybrid_inverse_cdf_sigmoid_lam${lam}_N${num_blocks}.json

echo "=== Done ==="
echo ""
echo "NOTE: minerva_math uses SymPy for answer checking. Install with: pip install lm-eval[math]"
