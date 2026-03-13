"""
GSM8K Equal-Mass Chunking Evaluation
======================================
Sigmoid Depth-Adaptive Rollout Score의 누적 질량(CDF)을 기반으로
블록 경계를 동적으로 결정하는 방식.

알고리즘:
  Step 0: Forward pass (output_attentions=True)
    → Sigmoid Depth-Adaptive Rollout Score 계산 (생성 구간만)
    → CDF를 N등분하여 블록 경계 결정 (Equal-Mass Chunking)
  Step 1+: 블록별 Confidence-based Parallel Decoding (앞→뒤)

비교 대상:
  - equal_mass: Sigmoid Rollout CDF 기반 동적 블록 분할
  - fixed_block: 고정 크기 블록 분할 (기존 방식)

측정 지표:
  - Accuracy: GSM8K 정답률
  - Avg NFE: 평균 모델 호출 횟수
  - Avg Time: 평균 소요 시간
  - TPS: 초당 생성 토큰 수
"""

import argparse
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def extract_answer(text: str) -> str:
    match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    device = block_mask_index.device
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(torch.long).clone()
    cols = torch.arange(steps, device=device).unsqueeze(0)
    num_transfer_tokens = num_transfer_tokens + (cols < rem.unsqueeze(1)).to(torch.long)
    return num_transfer_tokens


# ═══════════════════════════════════════════════════════════════════════════
# Sigmoid Depth-Adaptive Rollout
# ═══════════════════════════════════════════════════════════════════════════

def get_depth_adaptive_rollout(
    attentions: Tuple[torch.Tensor, ...],
    invert_depth: bool = False,
) -> torch.Tensor:
    """Sigmoid Depth-Adaptive Rollout → (T,) score vector.

    invert_depth=False (기본): 깊은 레이어에 attention 가중치 높임 (alpha 증가)
    invert_depth=True: 얕은 레이어에 attention 가중치 높임 (실험용)
    """
    with torch.no_grad():
        if attentions[0].dim() == 4:
            avg_attn = [a.mean(dim=1) for a in attentions]
        else:
            avg_attn = list(attentions)

        L = len(avg_attn)
        res_attn = []

        for i, a in enumerate(avg_attn):
            I = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)
            slope = 0.5
            mid = L / 2
            # invert_depth=True → (mid - i): 얕은 레이어(i 작음)일수록 alpha 큼
            # invert_depth=False → (i - mid): 깊은 레이어(i 큼)일수록 alpha 큼
            depth_arg = (mid - i) if invert_depth else (i - mid)
            alpha = 0.5 * torch.sigmoid(
                torch.tensor(slope * depth_arg, device=a.device, dtype=a.dtype)
            )
            res_attn.append((1.0 - alpha) * I + alpha * a)

        rollout = res_attn[0]
        for i in range(1, len(res_attn)):
            rollout = torch.matmul(res_attn[i], rollout)

        return rollout[0].sum(dim=0)  # (T,)


def get_baseline_rollout(
    attentions: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Baseline Rollout (0.5*I + 0.5*A) → (T,) score vector."""
    with torch.no_grad():
        if attentions[0].dim() == 4:
            avg_attn = [a.mean(dim=1) for a in attentions]
        else:
            avg_attn = list(attentions)

        res_attn = [
            0.5 * torch.eye(a.size(-1), device=a.device, dtype=a.dtype) + 0.5 * a
            for a in avg_attn
        ]
        rollout = res_attn[0]
        for i in range(1, len(res_attn)):
            rollout = torch.matmul(res_attn[i], rollout)
        return rollout[0].sum(dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# Equal-Mass Chunking
# ═══════════════════════════════════════════════════════════════════════════

def equal_mass_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
    min_block_size: int = 4,
    max_block_size: int = 32,  # 하드웨어 병목 방지용 최대 크기 추가!
) -> List[Tuple[int, int]]:
    
    gen_length = gen_scores.numel()
    if num_blocks <= 0 or num_blocks > gen_length:
        return [(0, gen_length)]

    scores = gen_scores.detach().cpu().to(torch.float64)
    scores = scores.clamp(min=0) 

    total_mass = scores.sum().item()
    if total_mass < 1e-12:
        block_size = gen_length // num_blocks
        return [(i * block_size, min((i + 1) * block_size, gen_length)) for i in range(num_blocks)]

    target_mass = total_mass / num_blocks
    cumsum = torch.cumsum(scores, dim=0)

    boundaries = [0]
    for k in range(1, num_blocks):
        threshold = target_mass * k
        candidates = torch.where(cumsum >= threshold)[0]
        
        if candidates.numel() > 0:
            boundary = candidates[0].item() + 1
        else:
            boundary = gen_length

        current_size = boundary - boundaries[-1]

        # 1. 최소 크기 보장 (잘 작성한 부분)
        if current_size < min_block_size:
            boundary = min(boundaries[-1] + min_block_size, gen_length)
            
        # 2. 최대 크기 보장 (새로 추가한 부분)
        elif current_size > max_block_size:
            boundary = min(boundaries[-1] + max_block_size, gen_length)

        if boundary >= gen_length:
            break  # 끝에 도달했으면 불필요한 연산 조기 종료

        boundaries.append(boundary)

    if boundaries[-1] != gen_length:
        boundaries.append(gen_length)

    boundaries = sorted(set(boundaries))

    blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # 맨 마지막 블록이 min_block_size보다 작은 경우, 바로 앞 블록과 병합
    # (요청된 num_blocks를 엄격히 맞추기보다, 지나치게 작은 꼬리 블록을 피하는 쪽을 선택)
    if len(blocks) >= 2:
        last_start, last_end = blocks[-1]
        last_size = last_end - last_start
        if last_size < min_block_size:
            prev_start, _ = blocks[-2]
            # 앞 블록의 시작은 유지하고, 끝을 마지막 블록의 끝으로 확장
            blocks[-2] = (prev_start, last_end)
            blocks.pop()

    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# Transfer Index Selection
# ═══════════════════════════════════════════════════════════════════════════

def select_transfer_index_threshold(
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    transfer_index = mask_index & (confidence >= threshold)
    max_conf_idx = torch.argmax(confidence, dim=1, keepdim=True)
    force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_idx, True)
    transfer_index = (transfer_index | force_mask) & mask_index
    return transfer_index


def select_transfer_index_topk(
    confidence: torch.Tensor,
    mask_index: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
) -> torch.Tensor:
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    _, idx = torch.sort(confidence, dim=1, descending=True)
    B, L = confidence.shape
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded

    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return transfer_index


# ═══════════════════════════════════════════════════════════════════════════
# Generation: Equal-Mass Chunking + Confidence-based Parallel Decoding
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_equal_mass(
    model,
    tokenizer,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    num_blocks: int,
    steps_per_block: int,
    temperature: float = 0.0,
    threshold: Optional[float] = 0.9,
    min_block_size: int = 4,
    max_block_size: int = 32,
    rollout_mode: str = "sigmoid",
    verbose: bool = False,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """
    Equal-Mass Chunking + Confidence-based Parallel Decoding.

    Step 0: Forward pass → Rollout Score → CDF → 블록 경계 결정
    Step 1+: 블록별 Confidence-based decoding (앞→뒤)

    Returns:
        x: 최종 시퀀스 (1, prompt_len + gen_length)
        nfe: 총 모델 호출 횟수
        info: 블록 경계 등 추가 정보
    """
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    nfe = 0

    # ── Step 0: Rollout Score 계산 + Equal-Mass Chunking ──
    outputs = model(x, output_attentions=True)
    nfe += 1

    if rollout_mode == "sigmoid":
        rollout_scores = get_depth_adaptive_rollout(outputs.attentions).to(torch.float64)
    elif rollout_mode == "sigmoid_inverted":
        rollout_scores = get_depth_adaptive_rollout(
            outputs.attentions, invert_depth=True
        ).to(torch.float64)
    else:
        rollout_scores = get_baseline_rollout(outputs.attentions).to(torch.float64)

    gen_scores = rollout_scores[prompt_len: prompt_len + gen_length]

    blocks = equal_mass_chunking(
        gen_scores=gen_scores,
        num_blocks=num_blocks,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
    )

    if verbose:
        block_sizes = [e - s for s, e in blocks]
        print(f"  [equal_mass/{rollout_mode}] {len(blocks)} blocks, sizes={block_sizes}")

    # ── Step 1+: 블록별 Confidence-based Parallel Decoding ──
    for block_idx, (block_start_rel, block_end_rel) in enumerate(blocks):
        block_start = prompt_len + block_start_rel
        block_end = prompt_len + block_end_rel

        block_mask = (x[:, block_start:block_end] == mask_id)
        if block_mask.sum() == 0:
            continue

        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        step_i = 0

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            nfe += 1
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

            x[transfer_index] = x0[transfer_index]
            step_i += 1

    info = {
        "rollout_mode": rollout_mode,
        "num_blocks_requested": num_blocks,
        "num_blocks_actual": len(blocks),
        "block_boundaries": blocks,
        "block_sizes": [e - s for s, e in blocks],
    }
    return x, nfe, info


# ═══════════════════════════════════════════════════════════════════════════
# Generation: Fixed-Block Baseline
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_fixed_block(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    block_length: int,
    steps_per_block: int,
    temperature: float = 0.0,
    threshold: Optional[float] = 0.9,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """고정 크기 블록 + Confidence-based Parallel Decoding (비교 대상)."""
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0, (
        f"gen_length({gen_length})는 block_length({block_length})의 배수여야 합니다."
    )
    num_blocks = gen_length // block_length
    nfe = 0

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        block_mask = (x[:, block_start:block_end] == mask_id)
        if block_mask.sum() == 0:
            continue

        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        step_i = 0

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            nfe += 1
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

            x[transfer_index] = x0[transfer_index]
            step_i += 1

    info = {
        "block_length": block_length,
        "num_blocks": num_blocks,
    }
    return x, nfe, info


# ═══════════════════════════════════════════════════════════════════════════
# GSM8K Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_gsm8k(
    model,
    tokenizer,
    ds,
    strategies: List[str],
    gen_length: int,
    num_blocks: int,
    steps_per_block: int,
    block_length: int,
    temperature: float,
    threshold: Optional[float],
    mask_id: int,
    min_block_size: int,
    max_block_size: int,
    use_chat_template: bool,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    results: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, float]] = {}

    for strategy in strategies:
        summary[strategy] = {
            "correct": 0, "total": 0, "total_time": 0.0,
            "total_tokens": 0, "nfe_sum": 0,
        }

    pbar = tqdm(ds, desc="GSM8K Equal-Mass Eval", total=len(ds))

    for idx, sample in enumerate(pbar):
        question = sample["question"]
        gold = extract_answer(sample["answer"])

        if use_chat_template:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True, tokenize=False,
            )
        else:
            prompt_str = question

        input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(
            model.device
        )

        for strategy in strategies:
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            if strategy == "fixed_block":
                out_ids, nfe, info = generate_fixed_block(
                    model=model,
                    prompt=input_ids,
                    gen_length=gen_length,
                    mask_id=mask_id,
                    block_length=block_length,
                    steps_per_block=steps_per_block,
                    temperature=temperature,
                    threshold=threshold,
                )
            else:
                rollout_mode = strategy.replace("equal_mass_", "") if strategy.startswith("equal_mass_") else "sigmoid"
                out_ids, nfe, info = generate_equal_mass(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=input_ids,
                    gen_length=gen_length,
                    mask_id=mask_id,
                    num_blocks=num_blocks,
                    steps_per_block=steps_per_block,
                    temperature=temperature,
                    threshold=threshold,
                    min_block_size=min_block_size,
                    max_block_size=max_block_size,
                    rollout_mode=rollout_mode,
                    verbose=verbose and idx < 3,
                )

            if model.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            prompt_len = input_ids.shape[1]
            gen_text = tokenizer.decode(
                out_ids[0, prompt_len:], skip_special_tokens=True
            )
            pred = extract_answer(gen_text)
            is_correct = int(pred == gold)
            token_count = len(tokenizer.encode(gen_text))

            summary[strategy]["correct"] += is_correct
            summary[strategy]["total"] += 1
            summary[strategy]["total_time"] += elapsed
            summary[strategy]["total_tokens"] += token_count
            summary[strategy]["nfe_sum"] += nfe

            result_entry = {
                "id": idx,
                "strategy": strategy,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "nfe": nfe,
                "elapsed_sec": round(elapsed, 4),
                "token_count": token_count,
                "gen_text_preview": gen_text[:300],
                "gen_text_full": gen_text,
                "question": question,
            }
            if "block_sizes" in info:
                result_entry["block_sizes"] = str(info["block_sizes"])
            results.append(result_entry)

        accs = {}
        for s in strategies:
            t = summary[s]["total"]
            if t > 0:
                accs[s] = f"{summary[s]['correct'] / t:.1%}"
        pbar.set_postfix(accs)

    for strategy in strategies:
        s = summary[strategy]
        total = max(s["total"], 1)
        total_time = max(s["total_time"], 1e-9)
        s["accuracy"] = s["correct"] / total
        s["avg_time"] = s["total_time"] / total
        s["avg_nfe"] = s["nfe_sum"] / total
        s["tps"] = s["total_tokens"] / total_time
        s["avg_tokens"] = s["total_tokens"] / total

    return results, summary


# ═══════════════════════════════════════════════════════════════════════════
# Result Printing & Saving
# ═══════════════════════════════════════════════════════════════════════════

STRATEGY_LABELS = {
    "equal_mass_sigmoid": "Equal-Mass (Sigmoid Rollout)",
    "equal_mass_sigmoid_inverted": "Equal-Mass (Sigmoid Inverted, shallow layers up)",
    "equal_mass_baseline": "Equal-Mass (Baseline Rollout)",
    "fixed_block": "Fixed Block (Baseline)",
}


def print_summary_table(
    summary: Dict[str, Dict[str, float]], strategies: List[str]
) -> str:
    header = (
        f"{'Strategy':<35s} | {'Accuracy':>10s} | {'Avg NFE':>10s} | "
        f"{'Avg Time':>10s} | {'TPS':>10s} | {'Avg Tokens':>10s}"
    )
    sep = "-" * len(header)
    lines = ["", "=" * len(header),
             "  GSM8K Equal-Mass Chunking Results",
             "=" * len(header), header, sep]

    for strategy in strategies:
        s = summary[strategy]
        label = STRATEGY_LABELS.get(strategy, strategy)
        lines.append(
            f"{label:<35s} | {s['accuracy']:>9.2%} | {s['avg_nfe']:>10.1f} | "
            f"{s['avg_time']:>9.3f}s | {s['tps']:>10.1f} | {s['avg_tokens']:>10.1f}"
        )
    lines.append(sep)
    lines.append("")

    table_str = "\n".join(lines)
    print(table_str)
    return table_str


def save_results(
    results: List[Dict[str, Any]],
    summary: Dict[str, Dict[str, float]],
    strategies: List[str],
    out_csv: str,
    out_json: str,
    args: argparse.Namespace,
    table_str: str,
) -> None:
    fieldnames = [
        "id", "strategy", "gold", "pred", "correct",
        "nfe", "elapsed_sec", "token_count", "block_sizes",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[Saved] Per-sample CSV: {out_csv}")

    json_data = {
        "config": {
            "model": args.model,
            "gen_length": args.gen_length,
            "num_blocks": args.num_blocks,
            "steps_per_block": args.steps_per_block,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "threshold": args.threshold,
            "min_block_size": args.min_block_size,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "strategies": strategies,
        },
        "summary": {},
        "table": table_str,
    }
    for strategy in strategies:
        s = summary[strategy]
        json_data["summary"][strategy] = {
            "label": STRATEGY_LABELS.get(strategy, strategy),
            "accuracy": round(s["accuracy"], 4),
            "avg_nfe": round(s["avg_nfe"], 2),
            "avg_time": round(s["avg_time"], 4),
            "tps": round(s["tps"], 2),
            "correct": s["correct"],
            "total": s["total"],
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Summary JSON: {out_json}")


def extract_equal_mass_wins(
    results: List[Dict[str, Any]],
    strategies: List[str],
) -> List[Dict[str, Any]]:
    """
    equal_mass는 맞고 fixed_block은 틀린 케이스 추출.
    분석용으로 prompt와 생성 답변을 따로 기록.
    """
    equal_mass_strategies = [s for s in strategies if s.startswith("equal_mass_")]
    has_fixed = "fixed_block" in strategies

    if not equal_mass_strategies or not has_fixed:
        return []

    # Group by sample id
    by_id: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in results:
        idx = r["id"]
        if idx not in by_id:
            by_id[idx] = {}
        by_id[idx][r["strategy"]] = r

    wins: List[Dict[str, Any]] = []
    for idx, strat_results in by_id.items():
        fixed = strat_results.get("fixed_block")
        if fixed is None:
            continue

        for em_strat in equal_mass_strategies:
            em = strat_results.get(em_strat)
            if em is None:
                continue
            if em["correct"] and not fixed["correct"]:
                wins.append({
                    "sample_id": idx,
                    "question": em["question"],
                    "gold": em["gold"],
                    "equal_mass_strategy": em_strat,
                    "equal_mass_pred": em["pred"],
                    "equal_mass_gen_text": em["gen_text_full"],
                    "equal_mass_nfe": em["nfe"],
                    "fixed_pred": fixed["pred"],
                    "fixed_gen_text": fixed["gen_text_full"],
                    "fixed_nfe": fixed["nfe"],
                })

    return wins


def extract_fixed_wins(
    results: List[Dict[str, Any]],
    strategies: List[str],
) -> List[Dict[str, Any]]:
    """
    fixed_block은 맞고 equal_mass는 틀린 케이스 추출.
    분석용으로 prompt와 생성 답변을 따로 기록.
    """
    equal_mass_strategies = [s for s in strategies if s.startswith("equal_mass_")]
    has_fixed = "fixed_block" in strategies

    if not equal_mass_strategies or not has_fixed:
        return []

    by_id: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in results:
        idx = r["id"]
        if idx not in by_id:
            by_id[idx] = {}
        by_id[idx][r["strategy"]] = r

    wins: List[Dict[str, Any]] = []
    for idx, strat_results in by_id.items():
        fixed = strat_results.get("fixed_block")
        if fixed is None:
            continue

        for em_strat in equal_mass_strategies:
            em = strat_results.get(em_strat)
            if em is None:
                continue
            if fixed["correct"] and not em["correct"]:
                wins.append({
                    "sample_id": idx,
                    "question": em["question"],
                    "gold": em["gold"],
                    "equal_mass_strategy": em_strat,
                    "equal_mass_pred": em["pred"],
                    "equal_mass_gen_text": em["gen_text_full"],
                    "equal_mass_nfe": em["nfe"],
                    "fixed_pred": fixed["pred"],
                    "fixed_gen_text": fixed["gen_text_full"],
                    "fixed_nfe": fixed["nfe"],
                })

    return wins


def save_equal_mass_wins(
    wins: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """equal_mass wins 케이스를 JSON으로 저장."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wins, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Equal-Mass wins (분석용): {out_path} ({len(wins)} cases)")


def save_fixed_wins(
    wins: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """fixed wins 케이스를 JSON으로 저장."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wins, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Fixed wins (분석용): {out_path} ({len(wins)} cases)")


# ═══════════════════════════════════════════════════════════════════════════
# CLI & Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GSM8K Equal-Mass Chunking Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sigmoid equal-mass vs fixed-block, 200 samples
  python gsm8k_equal_mass_eval.py --num-samples 200

  # Sigmoid only, 8 blocks
  python gsm8k_equal_mass_eval.py --strategies equal_mass_sigmoid --num-blocks 8

  # All three strategies
  python gsm8k_equal_mass_eval.py --strategies equal_mass_sigmoid,equal_mass_baseline,fixed_block

  # Threshold-based decoding with 4 blocks
  python gsm8k_equal_mass_eval.py --num-blocks 4 --threshold 0.9
""",
    )

    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default=None)

    p.add_argument(
        "--strategies", type=str,
        default="equal_mass_sigmoid,fixed_block",
        help="비교할 전략 (쉼표 구분). "
             "가능: equal_mass_sigmoid, equal_mass_sigmoid_inverted, equal_mass_baseline, fixed_block",
    )

    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument(
        "--num-blocks", type=int, default=8,
        help="Equal-mass chunking에서 나눌 블록 개수 N",
    )
    p.add_argument("--steps-per-block", type=int, default=32)
    p.add_argument(
        "--block-length", type=int, default=32,
        help="Fixed-block 전략의 블록 크기 (gen-length / block-length = 블록 수)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--threshold", type=float, default=0.9,
        help="Confidence threshold. None이면 top-K schedule.",
    )
    p.add_argument(
        "--min-block-size", type=int, default=4,
        help="Equal-mass chunking의 최소 블록 크기",
    )
    p.add_argument(
        "--max-block-size", type=int, default=48,
        help="Equal-mass chunking의 최대 블록 크기",
    )
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--no-chat-template", action="store_true")

    p.add_argument("--num-samples", type=int, default=1319)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out-dir", type=str, default="results_equal_mass")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        args.device = "cuda:3" if torch.cuda.is_available() else "cpu"

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    valid = {"equal_mass_sigmoid", "equal_mass_sigmoid_inverted", "equal_mass_baseline", "fixed_block"}
    for s in strategies:
        if s not in valid:
            raise ValueError(f"Unknown strategy '{s}'. Valid: {valid}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  GSM8K Equal-Mass Chunking Evaluation")
    print("=" * 65)
    print(f"  Model:            {args.model}")
    print(f"  Device:           {args.device} ({args.dtype})")
    print(f"  Strategies:       {strategies}")
    print(f"  Gen length:       {args.gen_length}")
    print(f"  Num blocks (N):   {args.num_blocks}")
    print(f"  Steps/block:      {args.steps_per_block}")
    print(f"  Fixed block size: {args.block_length}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  Threshold:        {args.threshold}")
    print(f"  Min block size:   {args.min_block_size}")
    print(f"  Max block size:   {args.max_block_size}")
    print(f"  Samples:          {args.num_samples}")
    print(f"  Seed:             {args.seed}")
    print("=" * 65)

    print("\nLoading model...")
    model = (
        LLaDAModelLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch_dtype,
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading GSM8K dataset...")
    ds = (
        load_dataset("openai/gsm8k", "main", split="test")
        .shuffle(seed=args.seed)
        .select(range(min(args.num_samples, 1319)))
    )
    print(f"  Loaded {len(ds)} samples\n")

    results, summary = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        ds=ds,
        strategies=strategies,
        gen_length=args.gen_length,
        num_blocks=args.num_blocks,
        steps_per_block=args.steps_per_block,
        block_length=args.block_length,
        temperature=args.temperature,
        threshold=args.threshold,
        mask_id=args.mask_id,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        use_chat_template=not args.no_chat_template,
        verbose=args.verbose
    )

    table_str = print_summary_table(summary, strategies)

    tag = f"N{args.num_blocks}_spb{args.steps_per_block}"
    if args.threshold is not None:
        tag += f"_th{args.threshold}"
    out_csv = os.path.join(args.out_dir, f"gsm8k_equal_mass_{tag}.csv")
    out_json = os.path.join(args.out_dir, f"gsm8k_equal_mass_{tag}.json")

    save_results(
        results=results,
        summary=summary,
        strategies=strategies,
        out_csv=out_csv,
        out_json=out_json,
        args=args,
        table_str=table_str,
    )

    # equal_mass 맞고 fixed 틀린 케이스 별도 저장 (분석용)
    em_wins = extract_equal_mass_wins(results, strategies)
    if em_wins:
        out_em = os.path.join(args.out_dir, f"equal_mass_wins_{tag}.json")
        save_equal_mass_wins(em_wins, out_em)

    # fixed 맞고 equal_mass 틀린 케이스 별도 저장 (분석용)
    fixed_wins = extract_fixed_wins(results, strategies)
    if fixed_wins:
        out_fixed = os.path.join(args.out_dir, f"fixed_wins_{tag}.json")
        save_fixed_wins(fixed_wins, out_fixed)

    print("\nDone!")


if __name__ == "__main__":
    main()
