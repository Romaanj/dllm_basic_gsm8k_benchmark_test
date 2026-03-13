import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_equal_mass_high_mass_probe import (
    build_topk_high_mass_report,
    trace_equal_mass_generation,
)


def summarize_values(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
        }

    t = torch.tensor(values, dtype=torch.float64)
    q = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    qs = torch.quantile(t, q)
    return {
        "count": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "p25": float(qs[0].item()),
        "p50": float(qs[1].item()),
        "p75": float(qs[2].item()),
        "max": float(t.max().item()),
    }


def build_histogram(values: List[float], num_bins: int) -> Dict[str, Any]:
    if not values:
        return {"num_bins": num_bins, "bin_edges": [], "counts": []}
    t = torch.tensor(values, dtype=torch.float64)
    hist = torch.histc(t.float(), bins=num_bins, min=0.0, max=1.0)
    # 0~1 균등 bin edge 생성
    edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, dtype=torch.float64)
    return {
        "num_bins": num_bins,
        "bin_edges": [float(x.item()) for x in edges],
        "counts": [int(x.item()) for x in hist],
    }


def parse_sample_ids(args: argparse.Namespace) -> List[int]:
    if args.sample_ids:
        ids = []
        for x in args.sample_ids.split(","):
            x = x.strip()
            if x:
                ids.append(int(x))
        return sorted(set(ids))

    start = args.start_id
    end = start + args.num_samples
    return list(range(start, end))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch aggregate high-mass token unmask confidence stats on GSM8K."
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default=None)

    p.add_argument(
        "--sample-ids",
        type=str,
        default="",
        help="쉼표 구분 sample id 목록. 지정 시 start-id/num-samples보다 우선.",
    )
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-chat-template", action="store_true")

    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--steps-per-block", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--min-block-size", type=int, default=1)
    p.add_argument("--max-block-size", type=int, default=80)
    p.add_argument(
        "--rollout-mode",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "sigmoid_inverted", "baseline"],
    )
    p.add_argument(
        "--chunking-mode",
        type=str,
        default="inverse_cdf",
        choices=["equal_mass", "inverse_cdf", "hybrid_cdf"],
        help="블록 분할 방식. 기본값은 inverse_cdf.",
    )
    p.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="hybrid/inverse CDF 혼합계수 λ (1.0=순수 CDF, 0.0=uniform).",
    )
    p.add_argument("--top-k-high-mass", type=int, default=3)
    p.add_argument("--hist-bins", type=int, default=20)

    p.add_argument("--out-dir", type=str, default="results_equal_mass/high_mass_probe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is None:
        args.device = "cuda:3" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}")
    model = (
        LLaDAModelLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=args.seed)

    sample_ids = parse_sample_ids(args)
    valid_ids = [i for i in sample_ids if 0 <= i < len(ds)]
    skipped_ids = [i for i in sample_ids if i < 0 or i >= len(ds)]
    if not valid_ids:
        raise ValueError("유효한 sample id가 없습니다.")

    all_top_records: List[Dict[str, Any]] = []
    all_block_token_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sid in tqdm(valid_ids, desc="High-mass batch probe"):
        try:
            sample = ds[int(sid)]
            question = sample["question"]
            if args.no_chat_template:
                prompt_str = question
            else:
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    add_generation_prompt=True,
                    tokenize=False,
                )

            input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)
            trace = trace_equal_mass_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=input_ids,
                gen_length=args.gen_length,
                mask_id=args.mask_id,
                num_blocks=args.num_blocks,
                steps_per_block=args.steps_per_block,
                temperature=args.temperature,
                threshold=args.threshold,
                min_block_size=args.min_block_size,
                max_block_size=args.max_block_size,
                rollout_mode=args.rollout_mode,
                chunking_mode=args.chunking_mode,
                lam=args.lam,
            )
            top_report = build_topk_high_mass_report(trace, top_k=args.top_k_high_mass)

            for item in top_report:
                top_tok = item["top_token"]
                blk = item["block_info"]
                rank = int(item["rank_by_rollout_score"])

                top_record = {
                    "sample_id": int(sid),
                    "rank": rank,
                    "rel_index": int(top_tok["rel_index"]),
                    "rollout_score": float(top_tok["rollout_score"]),
                    "cdf_value": float(top_tok["cdf_value"]),
                    "unmask_confidence": top_tok["unmask_confidence"],
                    "unmask_decode_iter": top_tok["unmask_decode_iter"],
                    "unmask_block_idx": top_tok["unmask_block_idx"],
                    "unmask_block_step": top_tok["unmask_block_step"],
                    "block_idx": int(blk["block_idx"]),
                    "block_size": int(blk["block_size"]),
                    "token_id": int(top_tok["token_id"]),
                    "token_str": str(top_tok["token_str"]),
                    "token_text_decoded": str(top_tok["token_text_decoded"]),
                }
                all_top_records.append(top_record)

                for btr in item["block_tokens_unmask_confidence"]:
                    all_block_token_records.append(
                        {
                            "sample_id": int(sid),
                            "rank": rank,
                            "top_rel_index": int(top_tok["rel_index"]),
                            "block_idx": int(blk["block_idx"]),
                            "block_size": int(blk["block_size"]),
                            "rel_index": int(btr["rel_index"]),
                            "rollout_score": float(btr["rollout_score"]),
                            "cdf_value": float(btr["cdf_value"]),
                            "unmask_confidence": btr["unmask_confidence"],
                            "unmask_decode_iter": btr["unmask_decode_iter"],
                            "unmask_block_step": btr["unmask_block_step"],
                            "token_id": int(btr["token_id"]),
                            "token_str": str(btr["token_str"]),
                            "token_text_decoded": str(btr["token_text_decoded"]),
                        }
                    )
        except Exception as e:
            failures.append({"sample_id": int(sid), "error": str(e)})

    # 집계
    top_conf = [
        float(r["unmask_confidence"])
        for r in all_top_records
        if r["unmask_confidence"] is not None
    ]
    top_block_sizes = [float(r["block_size"]) for r in all_top_records]
    block_token_conf = [
        float(r["unmask_confidence"])
        for r in all_block_token_records
        if r["unmask_confidence"] is not None
    ]

    by_rank: Dict[int, List[float]] = {}
    for r in all_top_records:
        if r["unmask_confidence"] is None:
            continue
        rk = int(r["rank"])
        by_rank.setdefault(rk, []).append(float(r["unmask_confidence"]))

    rank_summary = {str(rk): summarize_values(vals) for rk, vals in sorted(by_rank.items())}

    aggregate = {
        "top_token_unmask_confidence": summarize_values(top_conf),
        "top_token_unmask_confidence_hist": build_histogram(top_conf, args.hist_bins),
        "top_token_block_size": summarize_values(top_block_sizes),
        "block_internal_unmask_confidence": summarize_values(block_token_conf),
        "block_internal_unmask_confidence_hist": build_histogram(block_token_conf, args.hist_bins),
        "top_token_unmask_confidence_by_rank": rank_summary,
    }

    # 저장
    os.makedirs(args.out_dir, exist_ok=True)
    tag = (
        f"batch_{len(valid_ids)}samples_start{min(valid_ids)}"
        f"_N{args.num_blocks}_spb{args.steps_per_block}_th{args.threshold}"
        f"_{args.rollout_mode}_{args.chunking_mode}_lam{args.lam}"
    )
    out_json = os.path.join(args.out_dir, f"high_mass_batch_stats_{tag}.json")
    out_csv_top = os.path.join(args.out_dir, f"high_mass_batch_top_tokens_{tag}.csv")
    out_csv_block = os.path.join(args.out_dir, f"high_mass_batch_block_tokens_{tag}.csv")

    payload = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "sample_ids_requested": sample_ids,
            "sample_ids_used": valid_ids,
            "sample_ids_skipped": skipped_ids,
            "gen_length": args.gen_length,
            "num_blocks": args.num_blocks,
            "steps_per_block": args.steps_per_block,
            "temperature": args.temperature,
            "threshold": args.threshold,
            "min_block_size": args.min_block_size,
            "max_block_size": args.max_block_size,
            "rollout_mode": args.rollout_mode,
            "chunking_mode": args.chunking_mode,
            "lam": args.lam,
            "top_k_high_mass": args.top_k_high_mass,
            "hist_bins": args.hist_bins,
            "use_chat_template": not args.no_chat_template,
        },
        "aggregate": aggregate,
        "counts": {
            "top_records": len(all_top_records),
            "block_token_records": len(all_block_token_records),
            "failures": len(failures),
        },
        "failures": failures,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if all_top_records:
        with open(out_csv_top, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_top_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_top_records)

    if all_block_token_records:
        with open(out_csv_block, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_block_token_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_block_token_records)

    print(f"[Saved] {out_json}")
    if all_top_records:
        print(f"[Saved] {out_csv_top}")
    if all_block_token_records:
        print(f"[Saved] {out_csv_block}")
    print("\nAggregate top-token confidence:", aggregate["top_token_unmask_confidence"])


if __name__ == "__main__":
    main()
