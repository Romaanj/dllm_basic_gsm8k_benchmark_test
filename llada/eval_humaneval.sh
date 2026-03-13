# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=256
block_length=32
steps=$((length / block_length))

output_dir=evals_results
mkdir -p ${output_dir}
# baseline
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
--output_path ${output_dir}/baseline/humaneval-ns0-${length} --log_samples

# # prefix cache
# accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True \
# --output_path evals_results/prefix_cache/humaneval-ns0-${length} --log_samples

# parallel
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True \
--output_path ${output_dir}/parallel/humaneval-ns0-${length} --log_samples

# # prefix cache+parallel
# accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True \
# --output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples

# # dual cache+parallel
# accelerate launch eval_llada.py --tasks ${task} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
# --output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

## NOTICE: use postprocess for humaneval
python postprocess_code.py {the samples_xxx.jsonl file under output_path}
