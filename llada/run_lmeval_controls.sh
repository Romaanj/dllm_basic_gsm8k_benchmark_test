#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_lmeval_controls.sh
#   bash run_lmeval_controls.sh --tasks gsm8k,humaneval --model-path GSAI-ML/LLaDA-8B-Instruct
#
# Note:
#   - HumanEval task name can differ by lm-eval version.
#     If "humaneval" fails, try "openai_humaneval".

TASKS="gsm8k,humaneval"
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
GEN_LENGTH=256
NUM_BLOCKS=8
STEPS_PER_BLOCK=32
THRESHOLD=0.9
LAM=1.0
SCHEDULER_SEED=42
CONTROL_MIN_SIZE=28
CONTROL_MAX_SIZE=32
OUT_ROOT="results_controls"
NUM_PROCESSES=1
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      TASKS="$2"; shift 2 ;;
    --model-path)
      MODEL_PATH="$2"; shift 2 ;;
    --gen-length)
      GEN_LENGTH="$2"; shift 2 ;;
    --num-blocks)
      NUM_BLOCKS="$2"; shift 2 ;;
    --steps-per-block)
      STEPS_PER_BLOCK="$2"; shift 2 ;;
    --threshold)
      THRESHOLD="$2"; shift 2 ;;
    --lam)
      LAM="$2"; shift 2 ;;
    --scheduler-seed)
      SCHEDULER_SEED="$2"; shift 2 ;;
    --control-min-size)
      CONTROL_MIN_SIZE="$2"; shift 2 ;;
    --control-max-size)
      CONTROL_MAX_SIZE="$2"; shift 2 ;;
    --out-root)
      OUT_ROOT="$2"; shift 2 ;;
    --num-processes)
      NUM_PROCESSES="$2"; shift 2 ;;
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

mkdir -p "$OUT_ROOT"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

common_args="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps_per_block=${STEPS_PER_BLOCK},strategy=hybrid_cdf_sigmoid,num_blocks=${NUM_BLOCKS},lam=${LAM},threshold=${THRESHOLD},scheduler_seed=${SCHEDULER_SEED}"

run_case_task () {
  local tag="$1"
  local mode_args="$2"
  local task="$3"
  local fewshot="$4"
  local need_log_samples="$5"
  local save_dir="${OUT_ROOT}/${tag}/${task}"
  local output_path="${OUT_ROOT}/${tag}/${task}_results.json"
  local extra_args=""
  if [[ "$need_log_samples" == "true" ]]; then
    extra_args="--log_samples"
  fi
  mkdir -p "$save_dir"

  echo "============================================================"
  echo "[RUN] ${tag} | task=${task} | fewshot=${fewshot}"
  echo " save_dir   : ${save_dir}"
  echo " output     : ${output_path}"
  echo " model_args : ${common_args},${mode_args},show_speed=True,save_dir=${save_dir}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" accelerate launch --num_processes "${NUM_PROCESSES}" eval_hybrid_cdf.py \
    --model llada_hybrid_cdf \
    --tasks "${task}" \
    --num_fewshot "${fewshot}" \
    --confirm_run_unsafe_code \
    --model_args "${common_args},${mode_args},show_speed=True,save_dir=${save_dir}" \
    --output_path "${output_path}" \
    ${extra_args}
}

run_case () {
  local tag="$1"
  local mode_args="$2"

  IFS=',' read -ra TASK_ARR <<< "${TASKS}"
  for raw_task in "${TASK_ARR[@]}"; do
    task="$(echo "$raw_task" | xargs)"
    if [[ -z "$task" ]]; then
      continue
    fi
    if [[ "$task" == "gsm8k" ]]; then
      run_case_task "$tag" "$mode_args" "$task" 5 false
    elif [[ "$task" == "humaneval" || "$task" == "openai_humaneval" ]]; then
      run_case_task "$tag" "$mode_args" "$task" 0 true
    else
      echo "[WARN] Unknown task '$task': defaulting to 0-shot without log_samples."
      run_case_task "$tag" "$mode_args" "$task" 0 false
    fi
  done
}

# 0) Inverse-CDF baseline
run_case \
  "inverse_base" \
  "inverse=true,control_mode=none"

# 1) Variance-matched control
run_case \
  "balanced_random" \
  "inverse=true,control_mode=balanced_random,control_min_size=${CONTROL_MIN_SIZE},control_max_size=${CONTROL_MAX_SIZE}"

# 2) Same-variance permuted-location control
run_case \
  "inverse_permuted" \
  "inverse=true,control_mode=inverse_permuted"

echo
echo "Done. Results under: ${OUT_ROOT}"
echo "For HumanEval, run postprocess on samples_*.jsonl:"
echo "  python postprocess_code.py <path-to-samples_jsonl>"
