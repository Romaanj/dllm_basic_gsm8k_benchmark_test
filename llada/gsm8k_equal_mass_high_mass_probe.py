import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_equal_mass_eval import (
    add_gumbel_noise,
    equal_mass_chunking,
    get_baseline_rollout,
    get_depth_adaptive_rollout,
    get_num_transfer_tokens,
    select_transfer_index_threshold,
    select_transfer_index_topk,
)
from gsm8k_hybrid_cdf_eval import hybrid_cdf_chunking


@torch.no_grad()
def trace_equal_mass_generation(
    model,
    tokenizer,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    num_blocks: int,
    steps_per_block: int,
    temperature: float,
    threshold: Optional[float],
    min_block_size: int,
    max_block_size: int,
    rollout_mode: str,
    chunking_mode: str,
    lam: float,
) -> Dict[str, Any]:
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    # Step 0: rollout + block boundaries
    outputs = model(x, output_attentions=True)
    if rollout_mode == "sigmoid":
        rollout_scores = get_depth_adaptive_rollout(outputs.attentions).to(torch.float64)
    elif rollout_mode == "sigmoid_inverted":
        rollout_scores = get_depth_adaptive_rollout(
            outputs.attentions, invert_depth=True
        ).to(torch.float64)
    else:
        rollout_scores = get_baseline_rollout(outputs.attentions).to(torch.float64)

    gen_scores = rollout_scores[prompt_len : prompt_len + gen_length]
    if chunking_mode == "equal_mass":
        blocks = equal_mass_chunking(
            gen_scores=gen_scores,
            num_blocks=num_blocks,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
        )
    elif chunking_mode == "inverse_cdf":
        blocks = hybrid_cdf_chunking(
            gen_scores=gen_scores,
            num_blocks=num_blocks,
            lam=lam,
            inverse=True,
        )
    else:
        blocks = hybrid_cdf_chunking(
            gen_scores=gen_scores,
            num_blocks=num_blocks,
            lam=lam,
            inverse=False,
        )

    # cdf_value는 "블록 분할에 사용된 CDF" 기준으로 기록
    scores = gen_scores.detach().cpu().to(torch.float64).clamp(min=0)
    total_mass = scores.sum().item()
    if total_mass <= 1e-12:
        cdf = torch.zeros_like(scores)
    else:
        attn_cdf = torch.cumsum(scores, dim=0) / total_mass
        uniform_cdf = torch.arange(
            1, gen_length + 1, dtype=torch.float64
        ) / max(gen_length, 1)

        if chunking_mode == "equal_mass":
            cdf = attn_cdf
        elif chunking_mode == "inverse_cdf":
            inv_scores = 1.0 / (scores + 1e-10)
            inv_cdf = torch.cumsum(inv_scores, dim=0) / inv_scores.sum()
            cdf = lam * inv_cdf + (1.0 - lam) * uniform_cdf
        else:
            cdf = lam * attn_cdf + (1.0 - lam) * uniform_cdf

    # Trace buffers (relative generation indices)
    unmask_confidence: List[Optional[float]] = [None] * gen_length
    unmask_decode_iter: List[Optional[int]] = [None] * gen_length
    unmask_block_idx: List[Optional[int]] = [None] * gen_length
    unmask_block_step: List[Optional[int]] = [None] * gen_length
    confidence_trajectories: List[List[Dict[str, float]]] = [[] for _ in range(gen_length)]

    # Step 1+: block diffusion decoding + confidence trace
    decode_iter = 0
    nfe = 1  # Step 0 already consumed

    for block_idx, (block_start_rel, block_end_rel) in enumerate(blocks):
        block_start = prompt_len + block_start_rel
        block_end = prompt_len + block_end_rel

        block_mask = (x[:, block_start:block_end] == mask_id)
        if block_mask.sum() == 0:
            continue

        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        block_step = 0

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            nfe += 1
            decode_iter += 1

            mask_idx = (x == mask_id)
            mask_idx[:, block_end:] = False

            outputs = model(x)
            logits = outputs.logits

            logits_noisy = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            probs = F.softmax(logits.to(torch.float64), dim=-1)
            score = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_idx, x0, x)
            neg_inf = torch.tensor(
                torch.finfo(score.dtype).min, device=device, dtype=score.dtype
            )
            confidence = torch.where(mask_idx, score, neg_inf)

            # 각 iteration에서 아직 masked인 토큰 confidence를 기록 (trajectory)
            masked_abs = torch.where(mask_idx[0])[0]
            for abs_pos in masked_abs.tolist():
                rel_pos = abs_pos - prompt_len
                if 0 <= rel_pos < gen_length:
                    confidence_trajectories[rel_pos].append(
                        {
                            "iter": float(decode_iter),
                            "conf": float(confidence[0, abs_pos].item()),
                        }
                    )

            if threshold is not None:
                transfer_index = select_transfer_index_threshold(
                    confidence, mask_idx, threshold
                )
            else:
                max_i = num_transfer.size(1) - 1
                si = min(block_step, max_i)
                per_step = num_transfer[:, si]
                transfer_index = select_transfer_index_topk(
                    confidence, mask_idx, per_step
                )

            newly_unmasked = transfer_index & (x == mask_id)
            newly_unmasked_abs = torch.where(newly_unmasked[0])[0]
            for abs_pos in newly_unmasked_abs.tolist():
                rel_pos = abs_pos - prompt_len
                if 0 <= rel_pos < gen_length and unmask_confidence[rel_pos] is None:
                    unmask_confidence[rel_pos] = float(confidence[0, abs_pos].item())
                    unmask_decode_iter[rel_pos] = decode_iter
                    unmask_block_idx[rel_pos] = block_idx
                    unmask_block_step[rel_pos] = block_step

            x[transfer_index] = x0[transfer_index]
            block_step += 1

    gen_token_ids = x[0, prompt_len : prompt_len + gen_length]
    gen_tokens = tokenizer.convert_ids_to_tokens(gen_token_ids.tolist())

    token_rows: List[Dict[str, Any]] = []
    for i in range(gen_length):
        tok_id = int(gen_token_ids[i].item())
        token_rows.append(
            {
                "rel_index": i,
                "abs_index": int(prompt_len + i),
                "token_id": tok_id,
                "token_str": gen_tokens[i],
                "token_text_decoded": tokenizer.decode([tok_id], skip_special_tokens=True),
                "rollout_score": float(gen_scores[i].item()),
                "cdf_value": float(cdf[i].item()) if total_mass > 1e-12 else 0.0,
                "unmask_confidence": unmask_confidence[i],
                "unmask_decode_iter": unmask_decode_iter[i],
                "unmask_block_idx": unmask_block_idx[i],
                "unmask_block_step": unmask_block_step[i],
                "confidence_trajectory": confidence_trajectories[i],
            }
        )

    return {
        "prompt_len": int(prompt_len),
        "gen_length": int(gen_length),
        "nfe": int(nfe),
        "rollout_mode": rollout_mode,
        "chunking_mode": chunking_mode,
        "lam": float(lam),
        "block_boundaries": blocks,
        "block_sizes": [int(e - s) for s, e in blocks],
        "token_rows": token_rows,
    }


def build_topk_high_mass_report(
    trace: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    token_rows = trace["token_rows"]
    blocks: List[Tuple[int, int]] = trace["block_boundaries"]
    k = min(max(top_k, 1), len(token_rows))

    sorted_rows = sorted(token_rows, key=lambda r: r["rollout_score"], reverse=True)
    top_rows = sorted_rows[:k]

    report: List[Dict[str, Any]] = []
    for rank, row in enumerate(top_rows, start=1):
        rel_idx = int(row["rel_index"])
        block_idx = None
        block_start_rel = None
        block_end_rel = None

        for bi, (s, e) in enumerate(blocks):
            if s <= rel_idx < e:
                block_idx = bi
                block_start_rel = s
                block_end_rel = e
                break

        if block_idx is None:
            continue

        block_token_rows = [
            r
            for r in token_rows
            if block_start_rel <= int(r["rel_index"]) < block_end_rel
        ]

        entry: Dict[str, Any] = {
            "rank_by_rollout_score": rank,
            "top_token": row,
            "block_info": {
                "block_idx": int(block_idx),
                "block_start_rel": int(block_start_rel),
                "block_end_rel": int(block_end_rel),
                "block_size": int(block_end_rel - block_start_rel),
            },
            "block_tokens_unmask_confidence": block_token_rows,
        }

        # 다음 블록(block_idx+1): top-k가 속한 블록 바로 뒤 블록의 크기/내부 confidence
        next_bi = int(block_idx) + 1
        if next_bi < len(blocks):
            ns, ne = blocks[next_bi]
            next_block_token_rows = [
                r
                for r in token_rows
                if ns <= int(r["rel_index"]) < ne
            ]
            entry["next_block_info"] = {
                "block_idx": next_bi,
                "block_start_rel": int(ns),
                "block_end_rel": int(ne),
                "block_size": int(ne - ns),
            }
            entry["next_block_tokens_unmask_confidence"] = next_block_token_rows
        else:
            entry["next_block_info"] = None
            entry["next_block_tokens_unmask_confidence"] = []

        report.append(entry)

    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Probe top high-mass token unmask confidence in GSM8K equal-mass decoding."
        )
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:2")

    p.add_argument("--sample-id", type=int, default=2)
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
    if args.sample_id < 0 or args.sample_id >= len(ds):
        raise ValueError(f"sample-id {args.sample_id} out of range (0 ~ {len(ds)-1})")

    sample = ds[int(args.sample_id)]
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

    os.makedirs(args.out_dir, exist_ok=True)
    tag = (
        f"id{args.sample_id}_N{args.num_blocks}_spb{args.steps_per_block}"
        f"_th{args.threshold}_{args.rollout_mode}_{args.chunking_mode}_lam{args.lam}"
    )
    out_json = os.path.join(args.out_dir, f"high_mass_probe_{tag}.json")

    output = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "sample_id": args.sample_id,
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
            "use_chat_template": not args.no_chat_template,
        },
        "question": question,
        "trace_summary": {
            "prompt_len": trace["prompt_len"],
            "gen_length": trace["gen_length"],
            "nfe": trace["nfe"],
            "chunking_mode": trace["chunking_mode"],
            "lam": trace["lam"],
            "block_boundaries": trace["block_boundaries"],
            "block_sizes": trace["block_sizes"],
        },
        "top_high_mass_report": top_report,
        "all_tokens_trace": trace["token_rows"],
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {out_json}")
    print("\nTop high-mass token summary:")
    for item in top_report:
        top_tok = item["top_token"]
        blk = item["block_info"]
        print(
            f"- rank={item['rank_by_rollout_score']}, rel_idx={top_tok['rel_index']}, "
            f"rollout={top_tok['rollout_score']:.6f}, conf={top_tok['unmask_confidence']}, "
            f"block_idx={blk['block_idx']}, block_size={blk['block_size']}"
        )


if __name__ == "__main__":
    main()
