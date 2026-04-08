# LLaDA GSM8K Basic Benchmark

Minimal reproducible subset for running the GSM8K basic benchmark suite with three decoding modes:

- `inverse_cdf`
- `fixed`
- `argmax1`

This package intentionally excludes `adablock` and past experiment outputs.

## Included scripts

- `run_gsm8k_basic_benchmark_suite.sh`
- `eval_gsm8k_basic_inverse_cdf.sh`
- `eval_gsm8k_basic_fixed.sh`
- `eval_gsm8k_basic_argmax1.sh`
- `eval_inverse_cdf_basic.py`
- `eval_fixed_basic.py`
- `eval_argmax1_basic.py`
- `eval_hybrid_cdf.py`
- `gsm8k_hybrid_cdf_eval.py`
- `cap_partition.py`
- `model/`

## Requirements

- Python 3.10+
- CUDA-capable GPU
- Access to the Hugging Face model you want to evaluate

Install dependencies:

```bash
pip install -r requirements.txt
```

## Recommended setup

Run commands from the repository root:

```bash
cd dllm_basic_gsm8k_benchmark_test
```

Environment variables commonly used:

```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

## Run the full basic suite

```bash
GPU_ID=0 MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct GEN_LENGTH=256 BLOCK_LENGTH=32 SEED=42 bash run_gsm8k_basic_benchmark_suite.sh
```

This runs:

- inverse-CDF basic benchmark
- fixed-block basic benchmark
- block-local argmax1 benchmark

Outputs are written under:

```text
final_results/gsm8k_basic_benchmark_len<GEN_LENGTH>_b<BLOCK_LENGTH>/
```

## Run individual modes

Inverse-CDF:

```bash
GPU_ID=0 MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct GEN_LENGTH=256 NUM_BLOCKS=8 SEED=42 bash eval_gsm8k_basic_inverse_cdf.sh
```

Fixed-block:

```bash
GPU_ID=0 MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct GEN_LENGTH=256 BLOCK_LENGTH=32 SEED=42 bash eval_gsm8k_basic_fixed.sh
```

Argmax1:

```bash
GPU_ID=0 MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct GEN_LENGTH=256 BLOCK_LENGTH=32 SEED=42 bash eval_gsm8k_basic_argmax1.sh
```

Manual blocks:

```bash
GPU_ID=0 \
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct \
GEN_LENGTH=256 \
MANUAL_BLOCK_SIZES='64|28|28|28|27|27|27|27' \
SEED=42 \
bash eval_gsm8k_manual_blocks.sh
```

## Notes

- `GEN_LENGTH` must be divisible by `BLOCK_LENGTH` for the suite script.
- `NUM_BLOCKS` for inverse-CDF is computed as `GEN_LENGTH / BLOCK_LENGTH` in the suite.
- `MANUAL_BLOCK_SIZES` must sum to `GEN_LENGTH`.
- These scripts register custom `lm_eval` model names through local Python entrypoints, so run the provided wrappers rather than calling `python -m lm_eval` directly.
