"""
High-Mass Neighborhood Fragmentation: equal-mass vs inverse-CDF 비교 실험
=========================================================================

가설:
  equal-mass(inverse=False)는 high-mass region 근처를 더 잘게 찢고,
  inverse-CDF(inverse=True)는 그 주변을 더 덩어리로 유지한다.

방법:
  1) 각 샘플에서 Step 0 rollout score 기반 top-k high-mass token을 고른다.
  2) 각 token 주변 ±w window를 잡는다.
  3) equal-mass / inverse-CDF 두 가지로 block 분할을 만든다 (같은 rollout score 사용).
  4) window 내 지표 4가지를 측정하여 paired 비교한다.

지표:
  - mean_block_size_in_window
  - min_block_size_in_window
  - boundary_count
  - fragmentation_count
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_hybrid_cdf_eval import (
    get_baseline_rollout,
    get_depth_adaptive_rollout,
    hybrid_cdf_chunking,
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


def parse_sample_ids(args: argparse.Namespace) -> List[int]:
    if args.sample_ids:
        ids: List[int] = []
        for x in args.sample_ids.split(","):
            x = x.strip()
            if x:
                ids.append(int(x))
        return sorted(set(ids))
    return list(range(args.start_id, args.start_id + args.num_samples))


def compute_window_metrics(
    blocks: List[Tuple[int, int]],
    window_start: int,
    window_end_exclusive: int,
) -> Dict[str, Any]:
    """window [window_start, window_end_exclusive) 안에서 block partition 지표를 계산."""
    pieces: List[int] = []
    boundaries_inside = 0

    for bi, (s, e) in enumerate(blocks):
        inter_s = max(s, window_start)
        inter_e = min(e, window_end_exclusive)
        if inter_s < inter_e:
            pieces.append(inter_e - inter_s)

        if bi < len(blocks) - 1:
            boundary = e
            if window_start < boundary < window_end_exclusive:
                boundaries_inside += 1

    if not pieces:
        return {
            "mean_block_size_in_window": None,
            "min_block_size_in_window": None,
            "boundary_count": boundaries_inside,
            "fragmentation_count": 0,
            "piece_sizes": [],
        }

    return {
        "mean_block_size_in_window": float(sum(pieces) / len(pieces)),
        "min_block_size_in_window": int(min(pieces)),
        "boundary_count": int(boundaries_inside),
        "fragmentation_count": int(len(pieces)),
        "piece_sizes": [int(x) for x in pieces],
    }


@torch.no_grad()
def get_step0_gen_scores(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    rollout_mode: str,
) -> torch.Tensor:
    """Step 0 forward pass로 generation 영역의 rollout score를 계산."""
    prompt_len = prompt.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_len] = prompt.clone()

    outputs = model(x, output_attentions=True)
    if rollout_mode == "sigmoid":
        rollout_scores = get_depth_adaptive_rollout(outputs.attentions).to(torch.float64)
    elif rollout_mode == "sigmoid_inverted":
        rollout_scores = get_depth_adaptive_rollout(
            outputs.attentions, invert_depth=True
        ).to(torch.float64)
    else:
        rollout_scores = get_baseline_rollout(outputs.attentions).to(torch.float64)

    return rollout_scores[prompt_len : prompt_len + gen_length]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare high-mass neighborhood fragmentation between "
            "equal-mass(inverse=False) and inverse-CDF(inverse=True)."
        )
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--sample-ids", type=str, default="")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument(
        "--rollout-mode",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "sigmoid_inverted", "baseline"],
    )
    p.add_argument("--top-k-high-mass", type=int, default=3)
    p.add_argument(
        "--window-radius",
        type=int,
        default=8,
        help="Top high-mass token 주변 +-w window 반경.",
    )
    p.add_argument("--out-dir", type=str, default="results_equal_mass/high_mass_probe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

    detailed_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sid in tqdm(valid_ids, desc="Neighborhood fragmentation probe"):
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
            gen_scores = get_step0_gen_scores(
                model=model,
                prompt=input_ids,
                gen_length=args.gen_length,
                mask_id=args.mask_id,
                rollout_mode=args.rollout_mode,
            )

            blocks_equal = hybrid_cdf_chunking(
                gen_scores=gen_scores,
                num_blocks=args.num_blocks,
                lam=args.lam,
                inverse=False,
            )
            blocks_inverse = hybrid_cdf_chunking(
                gen_scores=gen_scores,
                num_blocks=args.num_blocks,
                lam=args.lam,
                inverse=True,
            )

            k = min(max(args.top_k_high_mass, 1), int(gen_scores.numel()))
            top_indices = torch.argsort(gen_scores, descending=True)[:k].tolist()
            top_indices = list(dict.fromkeys(int(i) for i in top_indices))

            for rank, center_idx in enumerate(top_indices, start=1):
                ws = max(0, int(center_idx) - int(args.window_radius))
                we = min(args.gen_length, int(center_idx) + int(args.window_radius) + 1)

                eq_metrics = compute_window_metrics(blocks_equal, ws, we)
                inv_metrics = compute_window_metrics(blocks_inverse, ws, we)

                row: Dict[str, Any] = {
                    "sample_id": int(sid),
                    "rank_by_rollout_score": int(rank),
                    "center_rel_index": int(center_idx),
                    "center_rollout_score": float(gen_scores[int(center_idx)].item()),
                    "window_start_rel": int(ws),
                    "window_end_rel_exclusive": int(we),
                    "window_length": int(we - ws),
                    "equal_mean_block_size": eq_metrics["mean_block_size_in_window"],
                    "equal_min_block_size": eq_metrics["min_block_size_in_window"],
                    "equal_boundary_count": eq_metrics["boundary_count"],
                    "equal_fragmentation_count": eq_metrics["fragmentation_count"],
                    "equal_piece_sizes": eq_metrics["piece_sizes"],
                    "inverse_mean_block_size": inv_metrics["mean_block_size_in_window"],
                    "inverse_min_block_size": inv_metrics["min_block_size_in_window"],
                    "inverse_boundary_count": inv_metrics["boundary_count"],
                    "inverse_fragmentation_count": inv_metrics["fragmentation_count"],
                    "inverse_piece_sizes": inv_metrics["piece_sizes"],
                }

                if row["equal_mean_block_size"] is not None and row["inverse_mean_block_size"] is not None:
                    row["delta_mean_block_size"] = float(
                        row["inverse_mean_block_size"] - row["equal_mean_block_size"]
                    )
                else:
                    row["delta_mean_block_size"] = None

                if row["equal_min_block_size"] is not None and row["inverse_min_block_size"] is not None:
                    row["delta_min_block_size"] = int(
                        row["inverse_min_block_size"] - row["equal_min_block_size"]
                    )
                else:
                    row["delta_min_block_size"] = None

                row["delta_boundary_count"] = int(
                    row["inverse_boundary_count"] - row["equal_boundary_count"]
                )
                row["delta_fragmentation_count"] = int(
                    row["inverse_fragmentation_count"] - row["equal_fragmentation_count"]
                )
                detailed_rows.append(row)

        except Exception as e:
            failures.append({"sample_id": int(sid), "error": str(e)})

    # ── Aggregate ──
    def _collect(key: str) -> List[float]:
        return [float(r[key]) for r in detailed_rows if r[key] is not None]

    aggregate = {
        "equal_mass": {
            "mean_block_size": summarize_values(_collect("equal_mean_block_size")),
            "min_block_size": summarize_values(_collect("equal_min_block_size")),
            "boundary_count": summarize_values(_collect("equal_boundary_count")),
            "fragmentation_count": summarize_values(_collect("equal_fragmentation_count")),
        },
        "inverse_cdf": {
            "mean_block_size": summarize_values(_collect("inverse_mean_block_size")),
            "min_block_size": summarize_values(_collect("inverse_min_block_size")),
            "boundary_count": summarize_values(_collect("inverse_boundary_count")),
            "fragmentation_count": summarize_values(_collect("inverse_fragmentation_count")),
        },
        "paired_delta_inverse_minus_equal": {
            "delta_mean_block_size": summarize_values(_collect("delta_mean_block_size")),
            "delta_min_block_size": summarize_values(_collect("delta_min_block_size")),
            "delta_boundary_count": summarize_values(_collect("delta_boundary_count")),
            "delta_fragmentation_count": summarize_values(_collect("delta_fragmentation_count")),
        },
    }

    # ── Save ──
    os.makedirs(args.out_dir, exist_ok=True)
    tag = (
        f"neighborhood_batch{len(valid_ids)}_start{min(valid_ids)}"
        f"_N{args.num_blocks}_lam{args.lam}_{args.rollout_mode}"
        f"_topk{args.top_k_high_mass}_w{args.window_radius}"
    )
    out_json = os.path.join(args.out_dir, f"high_mass_neighborhood_fragmentation_{tag}.json")
    out_csv = os.path.join(args.out_dir, f"high_mass_neighborhood_fragmentation_windows_{tag}.csv")

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
            "mask_id": args.mask_id,
            "num_blocks": args.num_blocks,
            "lam": args.lam,
            "rollout_mode": args.rollout_mode,
            "top_k_high_mass": args.top_k_high_mass,
            "window_radius": args.window_radius,
            "use_chat_template": not args.no_chat_template,
            "comparison_modes": {
                "equal_mass": "hybrid_cdf_chunking(inverse=False)",
                "inverse_cdf": "hybrid_cdf_chunking(inverse=True)",
            },
        },
        "counts": {
            "window_records": len(detailed_rows),
            "failures": len(failures),
        },
        "aggregate": aggregate,
        "failures": failures,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if detailed_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detailed_rows)

    print(f"\n[Saved] {out_json}")
    if detailed_rows:
        print(f"[Saved] {out_csv}")

    print("\n" + "=" * 60)
    print("[Aggregate] equal-mass vs inverse-CDF")
    print("=" * 60)
    for metric in ["mean_block_size", "min_block_size", "boundary_count", "fragmentation_count"]:
        eq_val = aggregate["equal_mass"][metric].get("mean")
        inv_val = aggregate["inverse_cdf"][metric].get("mean")
        delta_val = aggregate["paired_delta_inverse_minus_equal"][f"delta_{metric}"].get("mean")
        eq_str = f"{eq_val:.2f}" if eq_val is not None else "N/A"
        inv_str = f"{inv_val:.2f}" if inv_val is not None else "N/A"
        delta_str = f"{delta_val:+.2f}" if delta_val is not None else "N/A"
        print(f"  {metric:30s}  equal={eq_str:>8s}  inverse={inv_str:>8s}  delta={delta_str:>8s}")
    print("=" * 60)

    delta_agg = aggregate["paired_delta_inverse_minus_equal"]
    print("\n[Paired delta detail (inverse - equal)]:")
    for k, v in delta_agg.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
