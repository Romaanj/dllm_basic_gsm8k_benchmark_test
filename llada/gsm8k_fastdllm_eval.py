"""
GSM8K Evaluation for Fast-dLLM (Original Generate)
====================================================
Evaluates the original Fast-dLLM generate function on GSM8K,
so you can compare it against the Dynamic Block Scouting results.

Metrics reported:
  - Accuracy (exact match on final numeric answer)
  - Average latency per sample (seconds)
  - Average NFE (Number of Forward Evaluations) per sample
  - Average generated token count

Usage examples:
  # Default: steps=128, block_length=128 (full parallel)
  python gsm8k_fastdllm_eval.py --num-samples 200

  # Semi-autoregressive: block_length=32
  python gsm8k_fastdllm_eval.py --steps 128 --block-length 32

  # Threshold-based unmasking
  python gsm8k_fastdllm_eval.py --threshold 0.9

  # Multiple configs in one run
  python gsm8k_fastdllm_eval.py --configs "128:128:None" "128:32:None" "128:128:0.9"
"""

import argparse
import csv
import re
import time
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from generate import generate


# ---------------------------------------------------------------------------
# Answer extraction (same as dynamic_block eval)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    """Extract the final numeric answer from a GSM8K-style response."""
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_gsm8k(
    ds,
    model,
    tokenizer,
    use_chat_template: bool,
    gen_length: int,
    steps: int,
    block_length: int,
    temperature: float,
    mask_id: int,
    threshold: float = None,
    factor: float = None,
    remasking: str = "low_confidence",
    out_csv: str = "gsm8k_fastdllm.csv",
    desc: str = "GSM8K (Fast-dLLM)",
) -> Dict[str, float]:
    """
    Run GSM8K evaluation using the original Fast-dLLM generate function.
    """
    results: List[Dict[str, str]] = []
    correct = 0
    total = 0
    total_time = 0.0
    total_tokens = 0
    total_nfe = 0

    pbar = tqdm(ds, desc=desc, total=len(ds))
    for idx, sample in enumerate(pbar):
        question = sample["question"]
        gold = extract_answer(sample["answer"])

        if use_chat_template:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt_str = question

        input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(
            model.device
        )
        prompt_len = input_ids.shape[1]

        if model.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        out_ids, nfe = generate(
            model=model,
            prompt=input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            mask_id=mask_id,
            threshold=threshold,
            factor=factor,
        )

        if model.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        gen_text = tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=True)
        pred = extract_answer(gen_text)
        is_correct = int(pred == gold)
        token_count = len(tokenizer.encode(gen_text))

        correct += is_correct
        total += 1
        total_time += elapsed
        total_tokens += token_count
        total_nfe += nfe

        results.append(
            {
                "id": str(idx),
                "mode": f"fastdllm_s{steps}_b{block_length}"
                        + (f"_t{threshold}" if threshold is not None else ""),
                "gold": gold,
                "pred": pred,
                "correct": str(is_correct),
                "latency": f"{elapsed:.4f}",
                "nfe": str(nfe),
                "tokens": str(token_count),
                "gen_text": gen_text[:200],
            }
        )
        acc_so_far = correct / max(1, total)
        avg_nfe_so_far = total_nfe / max(1, total)
        pbar.set_postfix({"acc": f"{acc_so_far:.3f}", "nfe": f"{avg_nfe_so_far:.1f}"})

    # Write CSV
    if results:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  Per-sample results saved to {out_csv}")

    return {
        "accuracy": correct / max(1, total),
        "avg_time": total_time / max(1, total),
        "avg_nfe": total_nfe / max(1, total),
        "avg_tokens": total_tokens / max(1, total),
        "total": total,
    }


def _print_summary(label: str, summary: Dict[str, float]) -> None:
    print(
        f"[{label}] "
        f"Accuracy: {summary['accuracy']:.4f} | "
        f"Avg NFE: {summary['avg_nfe']:.1f} | "
        f"Avg time: {summary['avg_time']:.4f}s | "
        f"Avg tokens: {summary['avg_tokens']:.1f} | "
        f"Total: {summary['total']}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GSM8K benchmark for Fast-dLLM (original generate)."
    )
    # General
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--out-csv", type=str, default="gsm8k_fastdllm.csv")

    # Fast-dLLM generate params
    p.add_argument(
        "--steps", type=int, default=128,
        help="Total denoising steps.",
    )
    p.add_argument(
        "--block-length", type=int, default=128,
        help="Block length for semi-autoregressive generation. "
             "Set equal to gen-length for full parallel.",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold for unmasking (None = schedule-based).",
    )
    p.add_argument(
        "--factor", type=float, default=None,
        help="Dynamic transfer factor (None = standard).",
    )
    p.add_argument(
        "--remasking", type=str, default="low_confidence",
        choices=["low_confidence", "random"],
        help="Remasking strategy.",
    )

    # Multi-config mode
    p.add_argument(
        "--configs", nargs="*", default=None,
        help='Run multiple configs in one run. Format: "steps:block_length:threshold" '
             '(use "None" for no threshold). '
             'Example: --configs "128:128:None" "128:32:None" "128:128:0.9"',
    )

    return p.parse_args()


def _parse_config_str(config_str: str) -> Dict[str, Any]:
    """Parse a config string like '128:128:None' into a dict."""
    parts = config_str.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid config format: '{config_str}'. "
            "Expected 'steps:block_length:threshold'."
        )
    steps = int(parts[0])
    block_length = int(parts[1])
    threshold = None if parts[2].strip().lower() == "none" else float(parts[2])
    return {"steps": steps, "block_length": block_length, "threshold": threshold}


def main() -> None:
    args = parse_args()

    if args.device is None:
        args.device = "cuda:2" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # ------------------------------------------------------------------
    # Load model & tokenizer
    # ------------------------------------------------------------------
    print("Loading model...")
    model = (
        LLaDAModelLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch_dtype
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Load GSM8K dataset
    # ------------------------------------------------------------------
    print(f"Loading GSM8K (seed={args.seed}, n={args.num_samples})...")
    ds = (
        load_dataset("gsm8k", "main", split="test")
        .shuffle(seed=args.seed)
        .select(range(min(args.num_samples, 1319)))
    )

    use_chat_template = not args.no_chat_template

    # ------------------------------------------------------------------
    # Build list of configs to evaluate
    # ------------------------------------------------------------------
    if args.configs:
        configs = [_parse_config_str(c) for c in args.configs]
    else:
        configs = [
            {
                "steps": args.steps,
                "block_length": args.block_length,
                "threshold": args.threshold,
            }
        ]

    all_summaries: List[Tuple[str, Dict[str, float]]] = []

    for ci, cfg in enumerate(configs):
        steps = cfg["steps"]
        block_length = cfg["block_length"]
        threshold = cfg["threshold"]

        label = (
            f"Fast-dLLM (steps={steps}, block={block_length}"
            + (f", threshold={threshold}" if threshold is not None else "")
            + ")"
        )

        # CSV filename per config
        if len(configs) == 1:
            csv_path = args.out_csv
        else:
            base = args.out_csv.replace(".csv", "")
            csv_path = (
                f"{base}_s{steps}_b{block_length}"
                + (f"_t{threshold}" if threshold is not None else "")
                + ".csv"
            )

        print()
        print("=" * 60)
        print(f"Config {ci + 1}/{len(configs)}: {label}")
        print(f"  gen_length   = {args.gen_length}")
        print(f"  steps        = {steps}")
        print(f"  block_length = {block_length}")
        print(f"  threshold    = {threshold}")
        print(f"  factor       = {args.factor}")
        print(f"  remasking    = {args.remasking}")
        print(f"  temperature  = {args.temperature}")
        print("=" * 60)

        summary = evaluate_gsm8k(
            ds=ds,
            model=model,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
            gen_length=args.gen_length,
            steps=steps,
            block_length=block_length,
            temperature=args.temperature,
            mask_id=args.mask_id,
            threshold=threshold,
            factor=args.factor,
            remasking=args.remasking,
            out_csv=csv_path,
            desc=f"GSM8K ({label})",
        )

        all_summaries.append((label, summary))
        print()
        _print_summary(label, summary)

    # ------------------------------------------------------------------
    # Print comparison table if multiple configs
    # ------------------------------------------------------------------
    if len(all_summaries) > 1:
        print()
        print("=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Config':<45} {'Accuracy':>10} {'Avg NFE':>10} {'Avg Time':>10}")
        print("-" * 80)
        for label, summary in all_summaries:
            short_label = label if len(label) <= 44 else label[:41] + "..."
            print(
                f"{short_label:<45} "
                f"{summary['accuracy']:>9.4f} "
                f"{summary['avg_nfe']:>9.1f} "
                f"{summary['avg_time']:>9.4f}s"
            )
        print("=" * 80)


if __name__ == "__main__":
    main()
