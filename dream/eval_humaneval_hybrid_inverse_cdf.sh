#!/bin/bash
# Dream Hybrid-Inverse-CDF HumanEval evaluation

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
model="Dream-org/Dream-v0-Base-7B"
output_dir=evals_results/hybrid_inverse_cdf_lmeval
mkdir -p ${output_dir}

length=256
steps_per_block=32
threshold=0.9
lam=1.0

# N = 8
num_blocks=8
echo "=== Running HumanEval dream hybrid_inverse_cdf (lam=${lam}, N=${num_blocks}) ==="
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=hybrid_inverse_cdf,use_cache=false,block_use_kv_cache=false,escape_until=true,threshold=${threshold},hybrid_num_blocks=${num_blocks},hybrid_lam=${lam},hybrid_inverse_cdf=true,hybrid_rollout_mode=sigmoid,hybrid_steps_per_block=${steps_per_block} \
    --tasks ${task} \
    --batch_size 1 \
    --output_path ${output_dir}/humaneval_hybrid_inverse_cdf_sigmoid_lam${lam}_N${num_blocks} \
    --log_samples \
    --confirm_run_unsafe_code

block_length=$((length / num_blocks))
echo "=== Running HumanEval dream fixed_block_parallel (block_length=${block_length}, N=${num_blocks}) ==="
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=confidence_threshold,use_cache=true,block_use_kv_cache=false,escape_until=true,threshold=${threshold},block_length=${block_length} \
    --tasks ${task} \
    --batch_size 1 \
    --output_path ${output_dir}/humaneval_fixed_block_parallel_threshold${threshold}_N${num_blocks} \
    --log_samples \
    --confirm_run_unsafe_code

# N = 16
num_blocks=16
echo "=== Running HumanEval dream hybrid_inverse_cdf (lam=${lam}, N=${num_blocks}) ==="
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=hybrid_inverse_cdf,use_cache=false,block_use_kv_cache=false,escape_until=true,threshold=${threshold},hybrid_num_blocks=${num_blocks},hybrid_lam=${lam},hybrid_inverse_cdf=true,hybrid_rollout_mode=sigmoid,hybrid_steps_per_block=${steps_per_block} \
    --tasks ${task} \
    --batch_size 1 \
    --output_path ${output_dir}/humaneval_hybrid_inverse_cdf_sigmoid_lam${lam}_N${num_blocks} \
    --log_samples \
    --confirm_run_unsafe_code

block_length=$((length / num_blocks))
echo "=== Running HumanEval dream fixed_block_parallel (block_length=${block_length}, N=${num_blocks}) ==="
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=confidence_threshold,use_cache=true,block_use_kv_cache=false,escape_until=true,threshold=${threshold},block_length=${block_length} \
    --tasks ${task} \
    --batch_size 1 \
    --output_path ${output_dir}/humaneval_fixed_block_parallel_threshold${threshold}_N${num_blocks} \
    --log_samples \
    --confirm_run_unsafe_code

# N = 12
num_blocks=12
echo "=== Running HumanEval dream hybrid_inverse_cdf (lam=${lam}, N=${num_blocks}) ==="
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=hybrid_inverse_cdf,use_cache=false,block_use_kv_cache=false,escape_until=true,threshold=${threshold},hybrid_num_blocks=${num_blocks},hybrid_lam=${lam},hybrid_inverse_cdf=true,hybrid_rollout_mode=sigmoid,hybrid_steps_per_block=${steps_per_block} \
    --tasks ${task} \
    --batch_size 1 \
    --output_path ${output_dir}/humaneval_hybrid_inverse_cdf_sigmoid_lam${lam}_N${num_blocks} \
    --log_samples \
    --confirm_run_unsafe_code

block_length=$((length / num_blocks))
echo "=== Running HumanEval dream fixed_block_parallel (block_length=${block_length}, N=${num_blocks}) ==="
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=confidence_threshold,use_cache=true,block_use_kv_cache=false,escape_until=true,threshold=${threshold},block_length=${block_length} \
    --tasks ${task} \
    --batch_size 1 \
    --output_path ${output_dir}/humaneval_fixed_block_parallel_threshold${threshold}_N${num_blocks} \
    --log_samples \
    --confirm_run_unsafe_code

echo "=== Done ==="
echo ""
echo "NOTICE: HumanEval 결과에 대해 postprocess를 수행하세요:"
echo "  python postprocess_code.py {output_dir 내 samples_xxx.jsonl 파일}"
