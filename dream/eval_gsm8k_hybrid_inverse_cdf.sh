#!/bin/bash
# Dream Hybrid-Inverse-CDF GSM8K evaluation

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
num_fewshot=5
model="Dream-org/Dream-v0-Base-7B"
output_dir=evals_results/hybrid_inverse_cdf_lmeval
mkdir -p ${output_dir}

length=256
steps_per_block=32
threshold=0.9
lam=1.0

# N = 8
num_blocks=8
echo "=== Running GSM8K dream hybrid_inverse_cdf (lam=${lam}, N=${num_blocks}) ==="
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=hybrid_inverse_cdf,use_cache=false,block_use_kv_cache=false,threshold=${threshold},hybrid_num_blocks=${num_blocks},hybrid_lam=${lam},hybrid_inverse_cdf=true,hybrid_rollout_mode=sigmoid,hybrid_steps_per_block=${steps_per_block} \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path ${output_dir}/gsm8k_hybrid_inverse_cdf_sigmoid_lam${lam}_N${num_blocks}.json

# Fixed-block front-to-back parallel decoding (no KV cache)
block_length=$((length / num_blocks))
echo "=== Running GSM8K dream fixed_block_parallel (block_length=${block_length}, N=${num_blocks}) ==="
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=confidence_threshold,use_cache=true,block_use_kv_cache=false,threshold=${threshold},block_length=${block_length} \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path ${output_dir}/gsm8k_fixed_block_parallel_threshold${threshold}_N${num_blocks}.json



# # N = 16
# num_blocks=16
# echo "=== Running GSM8K dream hybrid_inverse_cdf (lam=${lam}, N=${num_blocks}) ==="
# accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=hybrid_inverse_cdf,use_cache=false,block_use_kv_cache=false,threshold=${threshold},hybrid_num_blocks=${num_blocks},hybrid_lam=${lam},hybrid_inverse_cdf=true,hybrid_rollout_mode=sigmoid,hybrid_steps_per_block=${steps_per_block} \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path ${output_dir}/gsm8k_hybrid_inverse_cdf_sigmoid_lam${lam}_N${num_blocks}.json
#
# block_length=$((length / num_blocks))
# echo "=== Running GSM8K dream fixed_block_parallel (block_length=${block_length}, N=${num_blocks}) ==="
# accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=confidence_threshold,use_cache=true,block_use_kv_cache=false,threshold=${threshold},block_length=${block_length} \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path ${output_dir}/gsm8k_fixed_block_parallel_threshold${threshold}_N${num_blocks}.json

# # N = 12
# num_blocks=12
# echo "=== Running GSM8K dream hybrid_inverse_cdf (lam=${lam}, N=${num_blocks}) ==="
# accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=hybrid_inverse_cdf,use_cache=false,block_use_kv_cache=false,threshold=${threshold},hybrid_num_blocks=${num_blocks},hybrid_lam=${lam},hybrid_inverse_cdf=true,hybrid_rollout_mode=sigmoid,hybrid_steps_per_block=${steps_per_block} \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path ${output_dir}/gsm8k_hybrid_inverse_cdf_sigmoid_lam${lam}_N${num_blocks}.json
#
# block_length=$((length / num_blocks))
# echo "=== Running GSM8K dream fixed_block_parallel (block_length=${block_length}, N=${num_blocks}) ==="
# accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=confidence_threshold,use_cache=true,block_use_kv_cache=false,threshold=${threshold},block_length=${block_length} \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path ${output_dir}/gsm8k_fixed_block_parallel_threshold${threshold}_N${num_blocks}.json

echo "=== Done ==="
