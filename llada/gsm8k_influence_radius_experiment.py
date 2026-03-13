"""
GSM8K Influence Radius Experiment (Targeted A/B Block Diffusion)
=================================================================

목표:
  Step 0 rollout score가 높은 token(T)을 기준으로,
  주변 window(W)의 token 품질이 "분절 확정(Case A)" vs "공동 진화(Case B)"에서
  어떻게 달라지는지 정량화.

실험 개요:
  1) Targeting
     - Step 0 deep-layer rollout score 계산
     - 샘플 내 상위 high_pct(기본 10%) 토큰 후보에서 타겟 T 선택
     - T 뒤 k개 토큰을 W로 설정
  2) A/B 디코딩 (fixed-block 스케줄 기반)
     - Case A: [T] 먼저 확정 후 [W] 확정 (Fragmented Commitment)
     - Case B: [T+W]를 하나의 블록에서 공동 확정 (Joint Refinement)
  3) 측정
     - W 위치의 GT token log-prob 평균
     - W 모든 token 정확 일치 여부(success)
     - delta_logp = avg_logp_B - avg_logp_A
  4) 시각화
     - X: target mass(score), Y: delta_logp
"""

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_hybrid_cdf_eval import (
    StreamingRollout,
    add_gumbel_noise,
    get_num_transfer_tokens,
    select_transfer_index_threshold,
    select_transfer_index_topk,
)


def parse_optional_float(value: str) -> Optional[float]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"none", "null"}:
        return None
    return float(value)


@torch.no_grad()
def get_step0_rollout_scores(
    model,
    input_ids: torch.Tensor,
    gen_length: int,
    mask_id: int,
    rollout_mode: str = "sigmoid",
) -> Optional[torch.Tensor]:
    device = model.device
    prompt_len = input_ids.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = input_ids.clone()

    invert_depth = rollout_mode == "sigmoid_inverted"
    hook_mode = "baseline" if rollout_mode == "baseline" else "sigmoid"

    core_model = model.model if hasattr(model, "model") else model
    blocks_list = core_model.transformer.blocks
    num_layers = len(blocks_list)

    streaming = StreamingRollout(
        num_layers=num_layers, mode=hook_mode, invert_depth=invert_depth
    )
    streaming.register(blocks_list)

    try:
        _ = model(x, output_attentions=True)
    finally:
        streaming.remove()

    scores = streaming.get_scores()
    if scores is None:
        return None
    return scores.to(torch.float64)[prompt_len : prompt_len + gen_length].clone()


def build_target_and_window(
    gen_scores: torch.Tensor,
    gt_valid_len: int,
    window_k: int,
    high_pct: float,
) -> Optional[Tuple[int, int, int, float]]:
    gen_length = int(gen_scores.numel())
    if gen_length <= 1 or gt_valid_len <= 1:
        return None

    high_k = max(1, int(math.ceil(gen_length * high_pct)))
    sorted_idx = torch.argsort(gen_scores, descending=True).tolist()
    candidate_idx = sorted_idx[:high_k]

    # window가 최소 1 token 확보되는 T를 상위 후보에서 선택
    target = None
    for idx in candidate_idx:
        if idx + 1 < gt_valid_len and idx < gen_length - 1:
            target = int(idx)
            break
    if target is None:
        return None

    w_start = target + 1
    w_end = min(target + 1 + window_k, gen_length, gt_valid_len)
    if w_end <= w_start:
        return None

    target_mass = float(gen_scores[target].item())
    return target, w_start, w_end, target_mass


def build_blocks_for_case(
    gen_length: int,
    block_length: int,
    target: int,
    w_start: int,
    w_end: int,
    case_name: str,
) -> List[Tuple[int, int]]:
    assert 0 <= target < gen_length
    assert w_start == target + 1
    assert w_start < w_end <= gen_length
    assert case_name in {"A", "B"}

    blocks: List[Tuple[int, int]] = []
    pos = 0
    while pos < gen_length:
        if pos < target:
            next_boundary = min(((pos // block_length) + 1) * block_length, target)
            if next_boundary == pos:
                next_boundary = min(pos + block_length, target)
            if next_boundary > pos:
                blocks.append((pos, next_boundary))
            pos = next_boundary
            continue

        if pos == target:
            if case_name == "A":
                blocks.append((target, target + 1))  # [T]
                pos = target + 1
                if w_end > pos:
                    blocks.append((pos, w_end))  # [W]
                    pos = w_end
            else:
                blocks.append((target, w_end))  # [T+W]
                pos = w_end
            continue

        next_boundary = min(((pos // block_length) + 1) * block_length, gen_length)
        if next_boundary == pos:
            next_boundary = min(pos + block_length, gen_length)
        if next_boundary > pos:
            blocks.append((pos, next_boundary))
        pos = next_boundary

    # sanity: invalid 구간 제거 + 경계 정렬
    cleaned = [(int(s), int(e)) for s, e in blocks if s < e]
    if not cleaned:
        return [(0, gen_length)]
    cleaned = sorted(cleaned, key=lambda x: (x[0], x[1]))
    return cleaned


@torch.no_grad()
def decode_with_custom_blocks(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    blocks: Sequence[Tuple[int, int]],
    steps_per_block: int,
    temperature: float,
    threshold: Optional[float],
    gt_ids: torch.Tensor,
    window_positions: Sequence[int],
) -> Dict[str, Any]:
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = prompt.clone()

    nfe = 0
    logp_by_pos: Dict[int, float] = {}

    for block_start_rel, block_end_rel in blocks:
        block_start = prompt_len + block_start_rel
        block_end = prompt_len + block_end_rel

        block_mask = x[:, block_start:block_end] == mask_id
        if block_mask.sum() == 0:
            continue

        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        step_i = 0

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            nfe += 1
            mask_idx = x == mask_id
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

            if threshold is not None:
                transfer_index = select_transfer_index_threshold(
                    confidence, mask_idx, threshold
                )
            else:
                max_i = num_transfer.size(1) - 1
                si = min(step_i, max_i)
                per_step = num_transfer[:, si]
                transfer_index = select_transfer_index_topk(
                    confidence, mask_idx, per_step
                )

            log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
            for rel_pos in window_positions:
                if rel_pos in logp_by_pos:
                    continue
                gt_id = int(gt_ids[rel_pos].item())
                if gt_id < 0:
                    continue
                abs_pos = prompt_len + rel_pos
                if bool(transfer_index[0, abs_pos].item()):
                    logp_by_pos[rel_pos] = float(log_probs[0, abs_pos, gt_id].item())

            x[transfer_index] = x0[transfer_index]
            step_i += 1

    generated = x[0, prompt_len : prompt_len + gen_length].clone()
    window_match = []
    window_logps = []
    for rel_pos in window_positions:
        gt_id = int(gt_ids[rel_pos].item())
        if gt_id < 0:
            continue
        pred_id = int(generated[rel_pos].item())
        window_match.append(int(pred_id == gt_id))
        if rel_pos in logp_by_pos:
            window_logps.append(logp_by_pos[rel_pos])

    avg_logp = float(np.mean(window_logps)) if window_logps else float("nan")
    success = float(all(window_match)) if window_match else 0.0

    return {
        "generated_ids": generated,
        "nfe": nfe,
        "avg_logp_w": avg_logp,
        "success_w": success,
        "window_logp_count": len(window_logps),
    }


def plot_mass_vs_delta(records: List[Dict[str, Any]], out_path: str) -> None:
    if not records:
        return

    x = np.array([r["target_mass"] for r in records], dtype=np.float64)
    y = np.array([r["delta_logp"] for r in records], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=22, alpha=0.5, color="#1f77b4", edgecolors="none")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # quantile bin 평균 추세선
    try:
        q_edges = np.quantile(x, np.linspace(0.0, 1.0, 11))
        q_edges = np.unique(q_edges)
        if len(q_edges) >= 3:
            bx = []
            by = []
            for i in range(len(q_edges) - 1):
                lo = q_edges[i]
                hi = q_edges[i + 1]
                if i == len(q_edges) - 2:
                    mask = (x >= lo) & (x <= hi)
                else:
                    mask = (x >= lo) & (x < hi)
                if mask.sum() == 0:
                    continue
                bx.append(float(x[mask].mean()))
                by.append(float(y[mask].mean()))
            if bx:
                ax.plot(bx, by, color="#d62728", linewidth=2.2, marker="o", label="Binned mean")
                ax.legend(loc="best")
    except Exception:
        pass

    ax.set_xlabel("Target Semantic Mass (Step 0 rollout score)")
    ax.set_ylabel("Delta Log-Prob on W (Case B - Case A)")
    ax.set_title("Influence Radius Effect: Joint Refinement Gain")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GSM8K influence-radius experiment with targeted A/B block decoding."
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:2")

    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--no-chat-template", action="store_true")

    p.add_argument("--rollout-mode", type=str, default="sigmoid", choices=["sigmoid", "sigmoid_inverted", "baseline"])
    p.add_argument("--high-pct", type=float, default=0.05, help="High-mass token 후보 비율 (예: 0.10)")
    p.add_argument("--window-k", type=int, default=6, help="T 뒤의 dependent window 길이")

    p.add_argument("--num-blocks", type=int, default=8, help="fixed-block 베이스 블록 개수")
    p.add_argument("--steps-per-block", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--threshold",
        type=parse_optional_float,
        default=0.9,
        help="confidence threshold. 'none' 입력 시 top-k schedule 사용",
    )

    p.add_argument("--out-dir", type=str, default="results_influence_radius")
    p.add_argument("--save-per-sample", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    os.makedirs(args.out_dir, exist_ok=True)
    block_length = args.gen_length // args.num_blocks
    if args.gen_length % args.num_blocks != 0:
        raise ValueError("gen_length must be divisible by num_blocks for fixed-block base scheduling.")

    print("=" * 72)
    print("GSM8K Influence Radius A/B Experiment")
    print("=" * 72)
    print(f"Model:             {args.model}")
    print(f"Device:            {args.device} ({args.dtype})")
    print(f"Samples:           {args.num_samples}")
    print(f"gen_length:        {args.gen_length}")
    print(f"num_blocks:        {args.num_blocks} (block_length={block_length})")
    print(f"steps_per_block:   {args.steps_per_block}")
    print(f"high_pct:          {args.high_pct}")
    print(f"window_k:          {args.window_k}")
    print(f"threshold:         {args.threshold}")
    print(f"rollout_mode:      {args.rollout_mode}")
    print("=" * 72)

    print("\nLoading model/tokenizer...")
    model = (
        LLaDAModelLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch_dtype
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=args.seed)
    ds = ds.select(range(min(args.num_samples, len(ds))))
    print(f"Loaded {len(ds)} samples.\n")

    per_sample: List[Dict[str, Any]] = []
    valid_count = 0
    skipped_no_rollout = 0
    skipped_short_gt = 0
    skipped_no_target = 0

    sum_logp_a = 0.0
    sum_logp_b = 0.0
    sum_delta = 0.0
    succ_a = 0.0
    succ_b = 0.0
    nfe_a = 0.0
    nfe_b = 0.0

    start_all = time.perf_counter()
    pbar = tqdm(ds, desc="Influence radius eval", total=len(ds))

    for sid, sample in enumerate(pbar):
        question = sample["question"]
        answer = sample["answer"]

        if args.no_chat_template:
            prompt_str = question
        else:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                tokenize=False,
            )
        input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)

        # GT token은 answer 원문을 직접 token화
        gt_ids_raw = tokenizer(answer, add_special_tokens=False)["input_ids"]
        gt_valid_len = min(len(gt_ids_raw), args.gen_length)
        if gt_valid_len < 2:
            skipped_short_gt += 1
            continue
        gt_ids = torch.full((args.gen_length,), -1, dtype=torch.long, device=model.device)
        gt_ids[:gt_valid_len] = torch.tensor(gt_ids_raw[:gt_valid_len], dtype=torch.long, device=model.device)

        gen_scores = get_step0_rollout_scores(
            model=model,
            input_ids=input_ids,
            gen_length=args.gen_length,
            mask_id=args.mask_id,
            rollout_mode=args.rollout_mode,
        )
        if gen_scores is None:
            skipped_no_rollout += 1
            continue

        target_window = build_target_and_window(
            gen_scores=gen_scores,
            gt_valid_len=gt_valid_len,
            window_k=args.window_k,
            high_pct=args.high_pct,
        )
        if target_window is None:
            skipped_no_target += 1
            continue

        target, w_start, w_end, target_mass = target_window
        window_positions = list(range(w_start, w_end))

        blocks_a = build_blocks_for_case(
            gen_length=args.gen_length,
            block_length=block_length,
            target=target,
            w_start=w_start,
            w_end=w_end,
            case_name="A",
        )
        blocks_b = build_blocks_for_case(
            gen_length=args.gen_length,
            block_length=block_length,
            target=target,
            w_start=w_start,
            w_end=w_end,
            case_name="B",
        )

        out_a = decode_with_custom_blocks(
            model=model,
            prompt=input_ids,
            gen_length=args.gen_length,
            mask_id=args.mask_id,
            blocks=blocks_a,
            steps_per_block=args.steps_per_block,
            temperature=args.temperature,
            threshold=args.threshold,
            gt_ids=gt_ids,
            window_positions=window_positions,
        )
        out_b = decode_with_custom_blocks(
            model=model,
            prompt=input_ids,
            gen_length=args.gen_length,
            mask_id=args.mask_id,
            blocks=blocks_b,
            steps_per_block=args.steps_per_block,
            temperature=args.temperature,
            threshold=args.threshold,
            gt_ids=gt_ids,
            window_positions=window_positions,
        )

        avg_logp_a = out_a["avg_logp_w"]
        avg_logp_b = out_b["avg_logp_w"]
        if math.isnan(avg_logp_a) or math.isnan(avg_logp_b):
            continue

        delta_logp = avg_logp_b - avg_logp_a
        valid_count += 1

        sum_logp_a += avg_logp_a
        sum_logp_b += avg_logp_b
        sum_delta += delta_logp
        succ_a += out_a["success_w"]
        succ_b += out_b["success_w"]
        nfe_a += out_a["nfe"]
        nfe_b += out_b["nfe"]

        rec = {
            "sample_id": sid,
            "target_index": target,
            "window_start": w_start,
            "window_end": w_end,
            "window_len": len(window_positions),
            "target_mass": target_mass,
            "avg_logp_case_a": avg_logp_a,
            "avg_logp_case_b": avg_logp_b,
            "delta_logp": delta_logp,
            "success_case_a": out_a["success_w"],
            "success_case_b": out_b["success_w"],
            "nfe_case_a": out_a["nfe"],
            "nfe_case_b": out_b["nfe"],
            "blocks_case_a": blocks_a,
            "blocks_case_b": blocks_b,
        }
        if args.save_per_sample:
            rec["question"] = question
            rec["answer"] = answer
        per_sample.append(rec)

        if valid_count > 0:
            pbar.set_postfix(
                {
                    "valid": valid_count,
                    "dlogp": f"{sum_delta / valid_count:.4f}",
                    "succA": f"{succ_a / valid_count:.3f}",
                    "succB": f"{succ_b / valid_count:.3f}",
                }
            )

    elapsed = time.perf_counter() - start_all

    summary = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "gen_length": args.gen_length,
            "mask_id": args.mask_id,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "rollout_mode": args.rollout_mode,
            "high_pct": args.high_pct,
            "window_k": args.window_k,
            "num_blocks": args.num_blocks,
            "block_length": block_length,
            "steps_per_block": args.steps_per_block,
            "temperature": args.temperature,
            "threshold": args.threshold,
            "no_chat_template": args.no_chat_template,
        },
        "counts": {
            "valid_samples": valid_count,
            "skipped_no_rollout": skipped_no_rollout,
            "skipped_short_gt": skipped_short_gt,
            "skipped_no_target": skipped_no_target,
            "elapsed_sec": elapsed,
        },
        "metrics": {},
    }

    if valid_count > 0:
        summary["metrics"] = {
            "avg_logp_case_a": sum_logp_a / valid_count,
            "avg_logp_case_b": sum_logp_b / valid_count,
            "avg_delta_logp": sum_delta / valid_count,
            "success_rate_case_a": succ_a / valid_count,
            "success_rate_case_b": succ_b / valid_count,
            "avg_nfe_case_a": nfe_a / valid_count,
            "avg_nfe_case_b": nfe_b / valid_count,
        }

    out_json = os.path.join(args.out_dir, "influence_radius_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "samples": per_sample,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_path = os.path.join(args.out_dir, "mass_vs_delta_logprob.png")
    plot_mass_vs_delta(per_sample, plot_path)

    print("\n" + "=" * 72)
    print("Influence Radius Experiment Result")
    print("=" * 72)
    print(f"Valid samples: {valid_count}")
    if valid_count > 0:
        m = summary["metrics"]
        print(f"Avg logP(A): {m['avg_logp_case_a']:.6f}")
        print(f"Avg logP(B): {m['avg_logp_case_b']:.6f}")
        print(f"Avg delta  : {m['avg_delta_logp']:.6f} (B - A)")
        print(f"Success A  : {m['success_rate_case_a']:.4f}")
        print(f"Success B  : {m['success_rate_case_b']:.4f}")
        print(f"Avg NFE A  : {m['avg_nfe_case_a']:.2f}")
        print(f"Avg NFE B  : {m['avg_nfe_case_b']:.2f}")
    print(f"Skipped (no rollout): {skipped_no_rollout}")
    print(f"Skipped (short GT):   {skipped_short_gt}")
    print(f"Skipped (no target):  {skipped_no_target}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"[Saved] JSON: {out_json}")
    print(f"[Saved] Plot: {plot_path}")


if __name__ == "__main__":
    main()
