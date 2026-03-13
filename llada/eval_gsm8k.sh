# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
output_dir=results_gsm8k
# You can change the model path to LLaDA-1.5 by setting model_path='GSAI-ML/LLaDA-1.5'
mkdir -p ${output_dir}

# baseline
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
--output_path ${output_dir}/gsm8k_baseline.json

# # prefix cache
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,show_speed=True 


# parallel
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True \
--output_path ${output_dir}/gsm8k_parallel.json

# # parallel factor
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},factor=${factor},show_speed=True


# # # # prefix cache+parallel
# # # accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# # # --confirm_run_unsafe_code --model llada_dist \
# # # --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True

# # # dual cache+parallel
# # accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# # --confirm_run_unsafe_code --model llada_dist \
# # --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True

# # prefix cache+parallel factor
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,factor=${factor},show_speed=True

# # dual cache+parallel factor
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,factor=${factor},show_speed=True
