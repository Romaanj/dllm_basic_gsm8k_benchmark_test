"""
GSM8K Hybrid-CDF Equal-Mass Chunking Evaluation
=================================================
Attention Rollout Score의 CDF를 직선(uniform CDF)과 혼합하여
블록 경계를 결정하는 Hybrid-CDF 방식.

핵심 수식:
  hybrid_cdf[i] = λ * attention_cdf[i] + (1-λ) * (i / gen_length)

λ=1.0 → 순수 attention-based equal-mass (기존 방식)
λ=0.0 → 완전 균등 분할 (fixed-block과 동일)
0<λ<1 → attention 분포에 uniform 보정을 주어서
         너무 크거나 작은 블록이 자연스럽게 억제됨.

장점:
  - min_block_size / max_block_size 하이퍼파라미터가 필요 없음
  - 마지막 블록 병합 같은 ad-hoc 로직이 불필요
  - λ 하나로 attention 의존도를 부드럽게 조절
"""

import argparse
import csv
import json
import os
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions (gsm8k_equal_mass_eval.py와 공유)
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
# Rollout Score (기존과 동일)
# ═══════════════════════════════════════════════════════════════════════════

def get_depth_adaptive_rollout(
    attentions: Tuple[torch.Tensor, ...],
    invert_depth: bool = False,
) -> torch.Tensor:
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
            depth_arg = (mid - i) if invert_depth else (i - mid)
            alpha = 0.5 * torch.sigmoid(
                torch.tensor(slope * depth_arg, device=a.device, dtype=a.dtype)
            )
            res_attn.append((1.0 - alpha) * I + alpha * a)

        rollout = res_attn[0]
        for i in range(1, len(res_attn)):
            rollout = torch.matmul(res_attn[i], rollout)

        return rollout[0].sum(dim=0)


def get_baseline_rollout(
    attentions: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
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
# Streaming Rollout — forward hook 기반 메모리 효율적 rollout 계산
# ═══════════════════════════════════════════════════════════════════════════

class StreamingRollout:
    """Forward hook으로 attention rollout을 레이어별 즉시 계산.

    기존 방식: 32 레이어 attention matrix를 전부 메모리에 올린 뒤 rollout 계산
    이 방식:   레이어 하나씩 hook에서 rollout에 누적 → 즉시 해제 (메모리 ~32x 절감)
    """

    def __init__(self, num_layers: int, mode: str = "sigmoid", invert_depth: bool = False):
        self.num_layers = num_layers
        self.mode = mode
        self.invert_depth = invert_depth
        self.rollout: Optional[torch.Tensor] = None
        self._layer_idx = 0
        self._hooks: list = []

    def _hook_fn(self, module, input, output):
        x, cache, attn_weights = output
        if attn_weights is None:
            self._layer_idx += 1
            return output

        with torch.no_grad():
            a = attn_weights.mean(dim=1) if attn_weights.dim() == 4 else attn_weights
            I = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)

            if self.mode in ("sigmoid", "sigmoid_inverted"):
                slope = 0.5
                mid = self.num_layers / 2
                do_invert = self.invert_depth or (self.mode == "sigmoid_inverted")
                depth_arg = (mid - self._layer_idx) if do_invert else (self._layer_idx - mid)
                alpha = 0.5 * torch.sigmoid(
                    torch.tensor(slope * depth_arg, device=a.device, dtype=a.dtype)
                )
                res_a = (1.0 - alpha) * I + alpha * a
            else:
                res_a = 0.5 * I + 0.5 * a

            if self.rollout is None:
                self.rollout = res_a
            else:
                self.rollout = torch.matmul(res_a, self.rollout)

        self._layer_idx += 1
        return (x, cache, None)

    def register(self, blocks) -> None:
        for block in blocks:
            h = block.register_forward_hook(self._hook_fn)
            self._hooks.append(h)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_scores(self) -> Optional[torch.Tensor]:
        if self.rollout is None:
            return None
        return self.rollout[0].sum(dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid-CDF Equal-Mass Chunking
# ═══════════════════════════════════════════════════════════════════════════

def hybrid_cdf_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
    lam: float = 0.5,
    inverse: bool = False,
) -> List[Tuple[int, int]]:
    """
    Hybrid-CDF 기반 Equal-Mass Chunking.

    hybrid_cdf[i] = λ * attention_cdf[i] + (1-λ) * (i+1)/gen_length

    λ=1 → 순수 attention CDF,  λ=0 → uniform (fixed-block과 동일)
    min/max block size, 마지막 블록 병합 로직이 필요 없음.

    inverse=True 일 때:
      scores의 역수를 사용하여 attention이 집중된 곳에 큰 블록을 배정.
      inv_scores = 1 / (scores + ε) → 정규화 → CDF
      high attention → low inv_score → CDF가 천천히 상승 → 큰 블록
    """
    gen_length = gen_scores.numel()
    if num_blocks <= 0 or num_blocks > gen_length:
        return [(0, gen_length)]

    scores = gen_scores.detach().cpu().to(torch.float64)
    scores = scores.clamp(min=0)

    total_mass = scores.sum().item()

    # attention score가 거의 0이면 uniform fallback
    if total_mass < 1e-12:
        block_size = gen_length // num_blocks
        return [
            (i * block_size, min((i + 1) * block_size, gen_length))
            for i in range(num_blocks)
        ]

    if inverse:
        inv_scores = 1.0 / (scores + 1e-10)
        attn_cdf = torch.cumsum(inv_scores, dim=0) / inv_scores.sum()
    else:
        attn_cdf = torch.cumsum(scores, dim=0) / total_mass

    # uniform CDF: (i+1) / gen_length → [1/T, 1]
    uniform_cdf = torch.arange(1, gen_length + 1, dtype=torch.float64) / gen_length

    # hybrid CDF
    hybrid_cdf = lam * attn_cdf + (1.0 - lam) * uniform_cdf

    # CDF를 N등분: threshold = k/N (k=1,...,N-1)
    boundaries = [0]
    for k in range(1, num_blocks):
        threshold = k / num_blocks
        candidates = torch.where(hybrid_cdf >= threshold)[0]
        if candidates.numel() > 0:
            boundary = candidates[0].item() + 1
        else:
            boundary = gen_length

        if boundary >= gen_length:
            break

        boundaries.append(boundary)

    if boundaries[-1] != gen_length:
        boundaries.append(gen_length)

    boundaries = sorted(set(boundaries))
    blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    return blocks


def _sizes_to_blocks(sizes: List[int], gen_length: int) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    cur = 0
    for s in sizes:
        nxt = min(cur + int(s), gen_length)
        if nxt > cur:
            blocks.append((cur, nxt))
        cur = nxt
        if cur >= gen_length:
            break
    if not blocks:
        return [(0, gen_length)]
    if blocks[-1][1] < gen_length:
        blocks[-1] = (blocks[-1][0], gen_length)
    return blocks


def _sample_balanced_sizes(
    gen_length: int,
    num_blocks: int,
    min_size: int,
    max_size: int,
    rng: random.Random,
) -> List[int]:
    if num_blocks <= 0:
        return [gen_length]
    min_size = max(1, int(min_size))
    max_size = max(min_size, int(max_size))
    if min_size * num_blocks > gen_length or max_size * num_blocks < gen_length:
        base = gen_length // num_blocks
        rem = gen_length - base * num_blocks
        sizes = [base] * num_blocks
        for i in range(rem):
            sizes[i] += 1
        return sizes

    sizes: List[int] = []
    remain = gen_length
    for i in range(num_blocks - 1):
        left_blocks = num_blocks - i - 1
        lo = max(min_size, remain - left_blocks * max_size)
        hi = min(max_size, remain - left_blocks * min_size)
        if lo > hi:
            lo = hi = max(1, remain - left_blocks * min_size)
        s = rng.randint(lo, hi)
        sizes.append(s)
        remain -= s
    sizes.append(remain)
    return sizes


def balanced_random_chunking(
    gen_length: int,
    num_blocks: int,
    min_size: int = 28,
    max_size: int = 32,
    scheduler_seed: int = 0,
    sample_index: int = 0,
) -> List[Tuple[int, int]]:
    rng = random.Random(int(scheduler_seed) + int(sample_index))
    sizes = _sample_balanced_sizes(
        gen_length=gen_length,
        num_blocks=num_blocks,
        min_size=min_size,
        max_size=max_size,
        rng=rng,
    )
    return _sizes_to_blocks(sizes, gen_length)


def inverse_permuted_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
    lam: float = 1.0,
    scheduler_seed: int = 0,
    sample_index: int = 0,
) -> List[Tuple[int, int]]:
    inv_blocks = hybrid_cdf_chunking(
        gen_scores=gen_scores,
        num_blocks=num_blocks,
        lam=lam,
        inverse=True,
    )
    sizes = [e - s for s, e in inv_blocks]
    rng = random.Random(int(scheduler_seed) + int(sample_index))
    rng.shuffle(sizes)
    return _sizes_to_blocks(sizes, gen_scores.numel())


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
# Generation: Hybrid-CDF + Confidence-based Parallel Decoding
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_hybrid_cdf(
    model,
    tokenizer,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    num_blocks: int,
    steps_per_block: int,
    lam: float = 0.5,
    temperature: float = 0.0,
    threshold: Optional[float] = 0.9,
    rollout_mode: str = "sigmoid",
    inverse: bool = False,
    control_mode: str = "none",
    control_min_size: int = 28,
    control_max_size: int = 32,
    scheduler_seed: int = 0,
    sample_index: int = 0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """
    Hybrid-CDF Equal-Mass Chunking + Confidence-based Parallel Decoding.

    Step 0: Forward pass → Rollout Score → Hybrid CDF → 블록 경계 결정
    Step 1+: 블록별 Confidence-based decoding (앞→뒤)

    inverse=True: attention 역수 기반 CDF (high attention → big block)
    """
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    nfe = 0

    # ── Step 0: Rollout Score 계산 + Hybrid-CDF Chunking ──
    # StreamingRollout: forward hook으로 레이어별 즉시 rollout 누적.
    # 전체 attention matrix를 동시에 메모리에 올리지 않아 OOM 방지.
    torch.cuda.empty_cache()

    invert_depth = rollout_mode == "sigmoid_inverted"
    hook_mode = "baseline" if rollout_mode == "baseline" else "sigmoid"

    core_model = model.model if hasattr(model, "model") else model
    blocks_list = core_model.transformer.blocks
    num_layers = len(blocks_list)

    streaming = StreamingRollout(
        num_layers=num_layers, mode=hook_mode, invert_depth=invert_depth,
    )
    streaming.register(blocks_list)

    try:
        outputs = model(x, output_attentions=True)
        nfe += 1
    finally:
        streaming.remove()

    rollout_scores = streaming.get_scores()
    del outputs, streaming
    torch.cuda.empty_cache()

    if rollout_scores is not None:
        gen_scores = rollout_scores.to(torch.float64)[prompt_len: prompt_len + gen_length]
        if control_mode == "balanced_random":
            blocks = balanced_random_chunking(
                gen_length=gen_length,
                num_blocks=num_blocks,
                min_size=control_min_size,
                max_size=control_max_size,
                scheduler_seed=scheduler_seed,
                sample_index=sample_index,
            )
        elif control_mode == "inverse_permuted":
            blocks = inverse_permuted_chunking(
                gen_scores=gen_scores,
                num_blocks=num_blocks,
                lam=lam,
                scheduler_seed=scheduler_seed,
                sample_index=sample_index,
            )
        else:
            blocks = hybrid_cdf_chunking(
                gen_scores=gen_scores,
                num_blocks=num_blocks,
                lam=lam,
                inverse=inverse,
            )
    else:
        if verbose:
            print(f"  [WARNING] rollout computation returned None, falling back to uniform blocks")
        block_size = gen_length // num_blocks
        blocks = [
            (i * block_size, min((i + 1) * block_size, gen_length))
            for i in range(num_blocks)
        ]

    if verbose:
        block_sizes = [e - s for s, e in blocks]
        inv_tag = "inverse_cdf" if inverse else "hybrid_cdf"
        print(f"  [{inv_tag}/{rollout_mode}/λ={lam}] {len(blocks)} blocks, sizes={block_sizes}")

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
        "lam": lam,
        "inverse": inverse,
        "control_mode": control_mode,
        "control_min_size": control_min_size,
        "control_max_size": control_max_size,
        "scheduler_seed": scheduler_seed,
        "sample_index": sample_index,
        "num_blocks_requested": num_blocks,
        "num_blocks_actual": len(blocks),
        "block_boundaries": blocks,
        "block_sizes": [e - s for s, e in blocks],
    }
    return x, nfe, info


# ═══════════════════════════════════════════════════════════════════════════
# Generation: Fixed-Block Baseline (비교용, 기존과 동일)
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
