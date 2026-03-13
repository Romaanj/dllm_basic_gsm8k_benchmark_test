#!/bin/bash
# Equal-Mass Chunking lm-evaluation-harness 기반 minerva_math (MATH) 평가 스크립트
# minerva_math: Minerva 논문 4-shot 프롬프트 + SymPy 기반 답 검증 (hendrycks_math 4-shot 대체)
# 의존성: pip install lm-eval[math]

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=minerva_math
num_fewshot=4
model_path='GSAI-ML/LLaDA-8B-Instruct'
output_dir=results_equal_mass_lmeval

gen_length=512
steps_per_block=32
num_blocks=16
min_block_size=4
max_block_size=48
threshold=0.9
block_length=32

mkdir -p ${output_dir}

# # ─── 1. Fixed-block parallel baseline ───
# echo "=== Running fixed_block baseline (minerva_math, 4-shot) ==="
# CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 eval_equal_mass.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_equal_mass \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},block_length=${block_length},strategy=fixed_block,threshold=${threshold},show_speed=True \
#     --output_path ${output_dir}/minerva_math_fixed_block.json

# ─── 2. Equal-mass sigmoid ───
echo "=== Running equal_mass_sigmoid (minerva_math, 4-shot) ==="
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 eval_equal_mass.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_equal_mass \
    --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=equal_mass_sigmoid,num_blocks=${num_blocks},min_block_size=${min_block_size},max_block_size=${max_block_size},threshold=${threshold},show_speed=True \
    --output_path ${output_dir}/minerva_math_equal_mass_sigmoid_N${num_blocks}.json

# ─── 3. Equal-mass baseline rollout (optional) ───
# echo "=== Running equal_mass_baseline ==="
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_equal_mass.py \
#     --tasks ${task} --num_fewshot ${num_fewshot} \
#     --confirm_run_unsafe_code --model llada_equal_mass \
#     --model_args model_path=${model_path},gen_length=${gen_length},steps_per_block=${steps_per_block},strategy=equal_mass_baseline,num_blocks=${num_blocks},min_block_size=${min_block_size},max_block_size=${max_block_size},threshold=${threshold},show_speed=True \
#     --output_path ${output_dir}/minerva_math_equal_mass_baseline_N${num_blocks}.json

echo "=== Done ==="
echo ""
echo "NOTE: minerva_math uses SymPy for answer checking. Install with: pip install lm-eval[math]"
