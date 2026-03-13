"""
High-Mass Neighborhood Unmask Tracking: equal-mass vs inverse-CDF 비교
======================================================================

핵심 질문:
  1. High-mass region에 어떤 종류의 토큰이 모여 있는가?
  2. 그 중 늦게 unmask되는 토큰은 어떤 category인가?
  3. Inverse-CDF가 semantic-bearing 토큰을 더 coherent하게 (같은 시점에) unmask하는가?

실험 설계:
  Step 0: deep-layer attention rollout score 계산 (한 번, 양 method 공유)
  Step 1: top-k high-mass token 선택 + ±w window 정의
  Step 2: equal-mass / inverse-CDF 각각으로 generation + 전 토큰 unmask tracking
  Step 3: window 내 token-level 분석, late-unmask enrichment, unmask ordering 비교

저장:
  A. token-level raw data (CSV)
  B. window-level summary (CSV)
  C. aggregate + late-unmask enrichment (JSON)
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

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
    hybrid_cdf_chunking,
    select_transfer_index_threshold,
    select_transfer_index_topk,
)


# ═══════════════════════════════════════════════════════════════════════════
# GSM8K Token Category Classification
# ═══════════════════════════════════════════════════════════════════════════

RELATION_WORDS = {
    "more", "than", "less", "left", "remaining", "total", "half", "twice",
    "difference", "ratio", "each", "every", "per", "between",
    "plus", "minus", "times", "divided",
    "equals", "equal", "double", "triple", "quarter", "third",
    "added", "subtracted", "multiplied",
    "altogether", "combined", "together", "split",
    "gives", "gave", "gets", "got", "needs", "needed", "takes", "took",
    "costs", "cost", "pays", "paid", "earns", "earned", "saves", "saved",
    "spends", "spent", "bought", "sold", "made", "lost", "gained",
    "if", "then", "since", "because", "after", "before",
    "how", "many", "much",
}

ANSWER_PHRASES = {
    "therefore", "so", "thus", "hence", "answer", "result",
    "final", "conclude", "conclusion",
}

OPERATORS = {"+", "-", "*", "/", "=", "%", "×", "÷", "^"}

SEMANTIC_CATEGORIES = {"number", "operator", "relation_word", "answer_phrase", "entity"}


def classify_token_gsm8k(token_str: str) -> str:
    """GSM8K 도메인에 맞는 토큰 카테고리 분류.

    Categories:
      number, operator, relation_word, entity, answer_phrase, punctuation, other
    """
    t = token_str.strip()
    t_clean = t.lstrip("ĠĊ▁ ")

    if not t_clean:
        return "other"

    if "####" in t_clean:
        return "answer_phrase"

    if re.match(r"^-?\d[\d,]*\.?\d*$", t_clean):
        return "number"

    if t_clean in OPERATORS:
        return "operator"

    if t_clean.lower() in RELATION_WORDS:
        return "relation_word"

    if t_clean.lower() in ANSWER_PHRASES:
        return "answer_phrase"

    if t_clean[0].isupper() and t_clean.isalpha() and len(t_clean) > 1:
        return "entity"

    if all(c in "()[]{}:;,.<>!?@#$%^&*-+=/'\"\\|~`\n" for c in t_clean):
        return "punctuation"

    return "other"


# ═══════════════════════════════════════════════════════════════════════════
# Step 0 Rollout Score Computation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_step0_rollout(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    rollout_mode: str = "sigmoid",
) -> torch.Tensor:
    """Step 0 forward → StreamingRollout으로 generation 영역 rollout score 반환."""
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

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
    finally:
        streaming.remove()

    scores = streaming.get_scores()
    del outputs, streaming
    torch.cuda.empty_cache()

    if scores is None:
        return torch.ones(gen_length, dtype=torch.float64) / gen_length

    return scores.to(torch.float64)[prompt_len: prompt_len + gen_length]


# ═══════════════════════════════════════════════════════════════════════════
# Generation with Unmask Tracking
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_with_unmask_tracking(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    blocks: List[Tuple[int, int]],
    steps_per_block: int,
    temperature: float = 0.0,
    threshold: Optional[float] = 0.9,
) -> Dict[str, Any]:
    """블록별 confidence-based decoding + 토큰별 unmask 시점/confidence 추적.

    Returns dict with:
      final_x: (1, prompt_len+gen_length) 최종 생성 결과
      unmask_step: gen_length 크기 리스트, 각 위치의 unmask global step (None if never)
      unmask_confidence: 각 위치의 unmask 시 confidence
      unmask_block_id: 각 위치가 속한 block index
      unmask_block_size: 해당 block의 크기
    """
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    unmask_step = [None] * gen_length
    unmask_confidence = [None] * gen_length
    unmask_block_id = [None] * gen_length
    unmask_block_size = [None] * gen_length

    global_step = 0
    nfe = 0

    for block_idx, (bs_rel, be_rel) in enumerate(blocks):
        block_start = prompt_len + bs_rel
        block_end = prompt_len + be_rel
        block_sz = be_rel - bs_rel

        block_mask = (x[:, block_start:block_end] == mask_id)
        if block_mask.sum() == 0:
            continue

        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        step_i = 0

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            global_step += 1
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
                torch.finfo(score.dtype).min, device=device, dtype=score.dtype,
            )
            confidence = torch.where(mask_idx, score, neg_inf)

            if threshold is not None:
                transfer_index = select_transfer_index_threshold(
                    confidence, mask_idx, threshold,
                )
            else:
                max_i = num_transfer.size(1) - 1
                si = min(step_i, max_i)
                per_step = num_transfer[:, si]
                transfer_index = select_transfer_index_topk(
                    confidence, mask_idx, per_step,
                )

            newly = transfer_index[0].nonzero(as_tuple=False).squeeze(-1)
            for pos in newly.tolist():
                rel = pos - prompt_len
                if 0 <= rel < gen_length and unmask_step[rel] is None:
                    unmask_step[rel] = global_step
                    unmask_confidence[rel] = float(confidence[0, pos].item())
                    unmask_block_id[rel] = block_idx
                    unmask_block_size[rel] = block_sz

            x[transfer_index] = x0[transfer_index]
            step_i += 1

    return {
        "final_x": x,
        "nfe": nfe,
        "total_steps": global_step,
        "unmask_step": unmask_step,
        "unmask_confidence": unmask_confidence,
        "unmask_block_id": unmask_block_id,
        "unmask_block_size": unmask_block_size,
        "blocks": blocks,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Window Extraction & Metrics
# ═══════════════════════════════════════════════════════════════════════════

def extract_window_tokens(
    center_idx: int,
    window_radius: int,
    gen_length: int,
    gen_scores: torch.Tensor,
    tracking: Dict[str, Any],
    tokenizer,
) -> List[Dict[str, Any]]:
    """high-mass center 주변 ±w window의 토큰별 정보를 추출."""
    ws = max(0, center_idx - window_radius)
    we = min(gen_length, center_idx + window_radius + 1)

    final_x = tracking["final_x"]
    prompt_len = final_x.shape[1] - gen_length

    tokens = []
    for rel_i in range(ws, we):
        abs_i = prompt_len + rel_i
        token_id = int(final_x[0, abs_i].item())
        token_str = tokenizer.decode([token_id], skip_special_tokens=False).strip()
        raw_str = tokenizer.convert_ids_to_tokens([token_id])[0]

        tokens.append({
            "rel_index": rel_i,
            "global_index": abs_i,
            "token_string": token_str,
            "token_string_raw": raw_str,
            "rollout_score": float(gen_scores[rel_i].item()),
            "unmask_step": tracking["unmask_step"][rel_i],
            "unmask_confidence": tracking["unmask_confidence"][rel_i],
            "block_id": tracking["unmask_block_id"][rel_i],
            "block_size": tracking["unmask_block_size"][rel_i],
            "category": classify_token_gsm8k(raw_str),
        })

    return tokens


def compute_window_unmask_metrics(
    tokens: List[Dict[str, Any]],
    late_ratio: float = 0.25,
    late_top_n: int = 3,
) -> Dict[str, Any]:
    """window 내 unmask ordering 지표와 late-unmask 분석."""
    valid = [t for t in tokens if t["unmask_step"] is not None]
    if not valid:
        return {"num_tokens": len(tokens), "num_unmasked": 0}

    steps = [t["unmask_step"] for t in valid]
    all_cats = [t["category"] for t in valid]

    # semantic-bearing tokens
    semantic = [t for t in valid if t["category"] in SEMANTIC_CATEGORIES]
    sem_steps = [t["unmask_step"] for t in semantic]

    # --- Late-unmask tokens ---
    sorted_by_step = sorted(valid, key=lambda t: t["unmask_step"], reverse=True)
    n_late_ratio = max(1, int(len(valid) * late_ratio))
    n_late = max(n_late_ratio, late_top_n)
    late_tokens = sorted_by_step[:n_late]

    late_cats = [t["category"] for t in late_tokens]
    late_cat_counts = dict(Counter(late_cats))
    late_semantic_count = sum(1 for c in late_cats if c in SEMANTIC_CATEGORIES)

    # --- Category distribution ---
    all_cat_counts = dict(Counter(all_cats))

    # --- Unmask step stats (all tokens) ---
    all_mean = sum(steps) / len(steps)
    all_std = (sum((s - all_mean) ** 2 for s in steps) / len(steps)) ** 0.5

    # --- Semantic token unmask stats ---
    sem_metrics = {}
    if sem_steps:
        sem_mean = sum(sem_steps) / len(sem_steps)
        sem_std = (sum((s - sem_mean) ** 2 for s in sem_steps) / len(sem_steps)) ** 0.5
        pairwise_gaps = [abs(a - b) for a, b in combinations(sem_steps, 2)]
        sem_metrics = {
            "semantic_count": len(sem_steps),
            "semantic_mean_step": sem_mean,
            "semantic_std_step": sem_std,
            "semantic_min_step": min(sem_steps),
            "semantic_max_step": max(sem_steps),
            "semantic_step_range": max(sem_steps) - min(sem_steps),
            "semantic_mean_pairwise_gap": (
                sum(pairwise_gaps) / len(pairwise_gaps) if pairwise_gaps else 0.0
            ),
        }
    else:
        sem_metrics = {
            "semantic_count": 0,
            "semantic_mean_step": None,
            "semantic_std_step": None,
            "semantic_min_step": None,
            "semantic_max_step": None,
            "semantic_step_range": None,
            "semantic_mean_pairwise_gap": None,
        }

    return {
        "num_tokens": len(tokens),
        "num_unmasked": len(valid),
        "all_mean_step": all_mean,
        "all_std_step": all_std,
        "all_min_step": min(steps),
        "all_max_step": max(steps),
        "category_counts": all_cat_counts,
        **sem_metrics,
        "late_tokens": [
            {
                "token_string": t["token_string"],
                "unmask_step": t["unmask_step"],
                "confidence": t["unmask_confidence"],
                "category": t["category"],
            }
            for t in late_tokens
        ],
        "late_category_counts": late_cat_counts,
        "late_semantic_ratio": (
            late_semantic_count / len(late_tokens) if late_tokens else 0.0
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Control Windows (random / low-mass)
# ═══════════════════════════════════════════════════════════════════════════

def get_control_centers(
    gen_scores: torch.Tensor,
    top_k: int,
    window_radius: int,
    gen_length: int,
    seed: int,
) -> Dict[str, List[int]]:
    """random / low-mass control window center를 선택."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    # low-mass: bottom-k by rollout score
    bottom_k = min(top_k, gen_length)
    low_indices = torch.argsort(gen_scores, descending=False)[:bottom_k].tolist()

    # random: uniform sampling avoiding edges
    safe_lo = window_radius
    safe_hi = gen_length - window_radius - 1
    if safe_hi <= safe_lo:
        random_indices = [gen_length // 2] * top_k
    else:
        random_indices = []
        for _ in range(top_k):
            idx = int(torch.randint(safe_lo, safe_hi + 1, (1,), generator=rng).item())
            random_indices.append(idx)

    return {
        "random": random_indices,
        "low_mass": low_indices,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation Helpers
# ═══════════════════════════════════════════════════════════════════════════

def safe_mean(vals):
    return sum(vals) / len(vals) if vals else None


def safe_std(vals):
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def aggregate_category_counts(
    list_of_counts: List[Dict[str, int]],
) -> Dict[str, float]:
    """여러 window의 category counts를 합산하여 비율 반환."""
    merged = Counter()
    for c in list_of_counts:
        merged.update(c)
    total = sum(merged.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in merged.most_common()}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="High-mass neighborhood unmask tracking: equal-mass vs inverse-CDF",
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--sample-ids", type=str, default="",
                   help="Comma-separated sample IDs (e.g. '0,1,5')")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--steps-per-block", type=int, default=32)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--rollout-mode", type=str, default="sigmoid",
                   choices=["sigmoid", "sigmoid_inverted", "baseline"])
    p.add_argument("--top-k-high-mass", type=int, default=3)
    p.add_argument("--window-radius", type=int, default=8)
    p.add_argument("--late-ratio", type=float, default=0.25,
                   help="Late-unmask threshold: top X%% by step")
    p.add_argument("--late-top-n", type=int, default=3)
    p.add_argument("--out-dir", type=str,
                   default="results_high_mass_unmask_tracking")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}")
    model = (
        LLaDAModelLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch_dtype,
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=args.seed)

    # Sample IDs
    if args.sample_ids:
        sample_ids = sorted(set(int(x.strip()) for x in args.sample_ids.split(",") if x.strip()))
    else:
        sample_ids = list(range(args.start_id, args.start_id + args.num_samples))
    valid_ids = [i for i in sample_ids if 0 <= i < len(ds)]
    if not valid_ids:
        raise ValueError("유효한 sample id가 없습니다.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Output file tag
    tag = (
        f"batch{len(valid_ids)}_start{min(valid_ids)}"
        f"_N{args.num_blocks}_lam{args.lam}_{args.rollout_mode}"
        f"_topk{args.top_k_high_mass}_w{args.window_radius}"
    )

    # ── Buffers ──
    all_token_rows: List[Dict[str, Any]] = []
    all_window_rows: List[Dict[str, Any]] = []
    all_sample_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    # Aggregate buffers for enrichment analysis
    agg_high_mass_cats = {"equal_mass": [], "inverse_cdf": []}
    agg_late_cats = {"equal_mass": [], "inverse_cdf": []}
    agg_semantic_metrics = {"equal_mass": [], "inverse_cdf": []}

    agg_control_cats = {
        "random_equal": [], "random_inverse": [],
        "low_mass_equal": [], "low_mass_inverse": [],
    }
    agg_control_late_cats = {
        "random_equal": [], "random_inverse": [],
        "low_mass_equal": [], "low_mass_inverse": [],
    }

    for sid in tqdm(valid_ids, desc="Unmask tracking"):
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
            prompt_len = input_ids.shape[1]

            # ── Step 0: Rollout Score (shared) ──
            gen_scores = compute_step0_rollout(
                model, input_ids, args.gen_length, args.mask_id, args.rollout_mode,
            )

            # ── Block partitions ──
            blocks_equal = hybrid_cdf_chunking(gen_scores, args.num_blocks, args.lam, inverse=False)
            blocks_inverse = hybrid_cdf_chunking(gen_scores, args.num_blocks, args.lam, inverse=True)

            # ── Generation with tracking (equal-mass) ──
            tracking_eq = generate_with_unmask_tracking(
                model, input_ids, args.gen_length, args.mask_id,
                blocks_equal, args.steps_per_block,
                args.temperature, args.threshold,
            )

            # ── Generation with tracking (inverse-CDF) ──
            tracking_inv = generate_with_unmask_tracking(
                model, input_ids, args.gen_length, args.mask_id,
                blocks_inverse, args.steps_per_block,
                args.temperature, args.threshold,
            )

            # ── Top-k high-mass centers ──
            k = min(max(args.top_k_high_mass, 1), int(gen_scores.numel()))
            top_indices = torch.argsort(gen_scores, descending=True)[:k].tolist()
            top_indices = list(dict.fromkeys(int(i) for i in top_indices))

            # ── Control centers ──
            controls = get_control_centers(
                gen_scores, args.top_k_high_mass, args.window_radius,
                args.gen_length, args.seed + sid,
            )

            sample_record = {
                "sample_id": int(sid),
                "question": question[:300],
                "prompt_len": int(prompt_len),
                "blocks_equal": [(s, e) for s, e in blocks_equal],
                "blocks_inverse": [(s, e) for s, e in blocks_inverse],
                "nfe_equal": tracking_eq["nfe"],
                "nfe_inverse": tracking_inv["nfe"],
                "high_mass_centers": [],
            }

            for rank, center_idx in enumerate(top_indices, start=1):
                center_record = {
                    "rank": rank,
                    "center_rel_index": center_idx,
                    "center_rollout_score": float(gen_scores[center_idx].item()),
                    "window_start": max(0, center_idx - args.window_radius),
                    "window_end": min(args.gen_length, center_idx + args.window_radius + 1),
                }

                for method_name, tracking in [("equal_mass", tracking_eq), ("inverse_cdf", tracking_inv)]:
                    tokens = extract_window_tokens(
                        center_idx, args.window_radius, args.gen_length,
                        gen_scores, tracking, tokenizer,
                    )
                    metrics = compute_window_unmask_metrics(
                        tokens, args.late_ratio, args.late_top_n,
                    )

                    center_record[method_name] = {
                        "metrics": {k: v for k, v in metrics.items()
                                    if k not in ("late_tokens", "category_counts", "late_category_counts")},
                        "late_tokens": metrics.get("late_tokens", []),
                    }

                    # token-level CSV rows
                    for t in tokens:
                        all_token_rows.append({
                            "sample_id": sid,
                            "window_type": "high_mass",
                            "rank": rank,
                            "center_idx": center_idx,
                            "method": method_name,
                            **t,
                        })

                    # window-level CSV row
                    row = {
                        "sample_id": sid,
                        "window_type": "high_mass",
                        "rank": rank,
                        "center_idx": center_idx,
                        "center_rollout_score": float(gen_scores[center_idx].item()),
                        "method": method_name,
                    }
                    for mk, mv in metrics.items():
                        if mk in ("late_tokens", "category_counts", "late_category_counts"):
                            row[mk] = json.dumps(mv) if isinstance(mv, (dict, list)) else mv
                        else:
                            row[mk] = mv
                    all_window_rows.append(row)

                    # aggregate buffers
                    agg_high_mass_cats[method_name].append(metrics.get("category_counts", {}))
                    agg_late_cats[method_name].append(metrics.get("late_category_counts", {}))
                    if metrics.get("semantic_mean_step") is not None:
                        agg_semantic_metrics[method_name].append({
                            "mean_step": metrics["semantic_mean_step"],
                            "std_step": metrics["semantic_std_step"],
                            "range": metrics["semantic_step_range"],
                            "pairwise_gap": metrics["semantic_mean_pairwise_gap"],
                        })

                sample_record["high_mass_centers"].append(center_record)

            # ── Control windows ──
            for ctrl_type, ctrl_centers in controls.items():
                for ci, center_idx in enumerate(ctrl_centers):
                    for method_name, tracking in [("equal_mass", tracking_eq), ("inverse_cdf", tracking_inv)]:
                        tokens = extract_window_tokens(
                            center_idx, args.window_radius, args.gen_length,
                            gen_scores, tracking, tokenizer,
                        )
                        metrics = compute_window_unmask_metrics(
                            tokens, args.late_ratio, args.late_top_n,
                        )

                        for t in tokens:
                            all_token_rows.append({
                                "sample_id": sid,
                                "window_type": ctrl_type,
                                "rank": ci + 1,
                                "center_idx": center_idx,
                                "method": method_name,
                                **t,
                            })

                        row = {
                            "sample_id": sid,
                            "window_type": ctrl_type,
                            "rank": ci + 1,
                            "center_idx": center_idx,
                            "center_rollout_score": float(gen_scores[center_idx].item()),
                            "method": method_name,
                        }
                        for mk, mv in metrics.items():
                            if mk in ("late_tokens", "category_counts", "late_category_counts"):
                                row[mk] = json.dumps(mv) if isinstance(mv, (dict, list)) else mv
                            else:
                                row[mk] = mv
                        all_window_rows.append(row)

                        ctrl_key = f"{ctrl_type}_{method_name.split('_')[0]}"
                        if ctrl_key in agg_control_cats:
                            agg_control_cats[ctrl_key].append(metrics.get("category_counts", {}))
                            agg_control_late_cats[ctrl_key].append(metrics.get("late_category_counts", {}))

            all_sample_records.append(sample_record)

        except Exception as e:
            import traceback
            failures.append({"sample_id": int(sid), "error": str(e), "traceback": traceback.format_exc()})
            print(f"  [ERROR] Sample {sid}: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # Aggregate Analysis
    # ═══════════════════════════════════════════════════════════════════════

    aggregate = {}

    # --- Analysis 1: High-mass region category composition ---
    aggregate["category_composition"] = {
        "high_mass_equal": aggregate_category_counts(agg_high_mass_cats["equal_mass"]),
        "high_mass_inverse": aggregate_category_counts(agg_high_mass_cats["inverse_cdf"]),
    }
    for ctrl_key in agg_control_cats:
        if agg_control_cats[ctrl_key]:
            aggregate["category_composition"][ctrl_key] = aggregate_category_counts(
                agg_control_cats[ctrl_key]
            )

    # --- Analysis 2: Unmask ordering comparison (semantic tokens) ---
    for method_name in ["equal_mass", "inverse_cdf"]:
        ms = agg_semantic_metrics[method_name]
        if ms:
            aggregate[f"semantic_unmask_{method_name}"] = {
                "mean_step_avg": safe_mean([m["mean_step"] for m in ms]),
                "std_step_avg": safe_mean([m["std_step"] for m in ms]),
                "range_avg": safe_mean([m["range"] for m in ms]),
                "pairwise_gap_avg": safe_mean([m["pairwise_gap"] for m in ms]),
                "n_windows": len(ms),
            }

    # Paired delta
    eq_ms = agg_semantic_metrics["equal_mass"]
    inv_ms = agg_semantic_metrics["inverse_cdf"]
    n_paired = min(len(eq_ms), len(inv_ms))
    if n_paired > 0:
        deltas_std = [inv_ms[i]["std_step"] - eq_ms[i]["std_step"] for i in range(n_paired)]
        deltas_range = [inv_ms[i]["range"] - eq_ms[i]["range"] for i in range(n_paired)]
        deltas_gap = [inv_ms[i]["pairwise_gap"] - eq_ms[i]["pairwise_gap"] for i in range(n_paired)]
        aggregate["semantic_unmask_delta_inv_minus_eq"] = {
            "delta_std_step_mean": safe_mean(deltas_std),
            "delta_range_mean": safe_mean(deltas_range),
            "delta_pairwise_gap_mean": safe_mean(deltas_gap),
            "n_paired": n_paired,
            "interpretation": (
                "음수 = inverse가 semantic 토큰을 더 coherent하게 unmask"
            ),
        }

    # --- Analysis 3: Late-unmask enrichment ---
    aggregate["late_unmask_enrichment"] = {}
    for method_name in ["equal_mass", "inverse_cdf"]:
        late_cs = agg_late_cats[method_name]
        if late_cs:
            merged = aggregate_category_counts(late_cs)
            semantic_ratio = sum(merged.get(c, 0) for c in SEMANTIC_CATEGORIES)
            aggregate["late_unmask_enrichment"][f"high_mass_{method_name}"] = {
                "category_ratios": merged,
                "semantic_total_ratio": semantic_ratio,
            }

    for ctrl_key in agg_control_late_cats:
        late_cs = agg_control_late_cats[ctrl_key]
        if late_cs:
            merged = aggregate_category_counts(late_cs)
            semantic_ratio = sum(merged.get(c, 0) for c in SEMANTIC_CATEGORIES)
            aggregate["late_unmask_enrichment"][ctrl_key] = {
                "category_ratios": merged,
                "semantic_total_ratio": semantic_ratio,
            }

    # ═══════════════════════════════════════════════════════════════════════
    # Save Results
    # ═══════════════════════════════════════════════════════════════════════

    # JSON: aggregate + per-sample summary
    out_json = os.path.join(args.out_dir, f"unmask_tracking_{tag}.json")
    payload = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "sample_ids": valid_ids,
            "gen_length": args.gen_length,
            "num_blocks": args.num_blocks,
            "steps_per_block": args.steps_per_block,
            "lam": args.lam,
            "temperature": args.temperature,
            "threshold": args.threshold,
            "rollout_mode": args.rollout_mode,
            "top_k_high_mass": args.top_k_high_mass,
            "window_radius": args.window_radius,
            "late_ratio": args.late_ratio,
            "late_top_n": args.late_top_n,
        },
        "counts": {
            "samples_processed": len(all_sample_records),
            "failures": len(failures),
            "total_token_records": len(all_token_rows),
            "total_window_records": len(all_window_rows),
        },
        "aggregate": aggregate,
        "samples": all_sample_records,
        "failures": failures,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Saved] {out_json}")

    # CSV: token-level
    if all_token_rows:
        out_token_csv = os.path.join(args.out_dir, f"token_level_{tag}.csv")
        fieldnames = list(all_token_rows[0].keys())
        with open(out_token_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_token_rows)
        print(f"[Saved] {out_token_csv}")

    # CSV: window-level
    if all_window_rows:
        out_window_csv = os.path.join(args.out_dir, f"window_summary_{tag}.csv")
        fieldnames = list(all_window_rows[0].keys())
        with open(out_window_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_window_rows)
        print(f"[Saved] {out_window_csv}")

    # ═══════════════════════════════════════════════════════════════════════
    # Print Summary
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 72)
    print("  HIGH-MASS NEIGHBORHOOD UNMASK TRACKING RESULTS")
    print("=" * 72)

    # Analysis 1
    print("\n[Analysis 1] High-mass window category composition")
    print("-" * 60)
    for window_type in ["high_mass_equal", "high_mass_inverse"]:
        cats = aggregate.get("category_composition", {}).get(window_type, {})
        if cats:
            print(f"  {window_type}:")
            for c, r in sorted(cats.items(), key=lambda x: -x[1]):
                bar = "█" * int(r * 40)
                print(f"    {c:20s} {r*100:5.1f}% {bar}")

    # Analysis 2
    print("\n[Analysis 2] Semantic token unmask ordering")
    print("-" * 60)
    for method in ["equal_mass", "inverse_cdf"]:
        key = f"semantic_unmask_{method}"
        if key in aggregate:
            m = aggregate[key]
            print(f"  {method}:")
            print(f"    mean step      = {m['mean_step_avg']:.2f}" if m['mean_step_avg'] else "    mean step      = N/A")
            print(f"    std step       = {m['std_step_avg']:.2f}" if m['std_step_avg'] else "    std step       = N/A")
            print(f"    step range     = {m['range_avg']:.2f}" if m['range_avg'] else "    step range     = N/A")
            print(f"    pairwise gap   = {m['pairwise_gap_avg']:.2f}" if m['pairwise_gap_avg'] else "    pairwise gap   = N/A")

    delta_key = "semantic_unmask_delta_inv_minus_eq"
    if delta_key in aggregate:
        d = aggregate[delta_key]
        print(f"\n  Delta (inverse - equal):  [음수 = inverse가 더 coherent]")
        for k, v in d.items():
            if k not in ("n_paired", "interpretation"):
                sign = "+" if v and v > 0 else ""
                val_str = f"{sign}{v:.3f}" if v is not None else "N/A"
                print(f"    {k:30s} = {val_str}")

    # Analysis 3
    print("\n[Analysis 3] Late-unmask enrichment (semantic ratio)")
    print("-" * 60)
    enrich = aggregate.get("late_unmask_enrichment", {})
    for key in sorted(enrich.keys()):
        info = enrich[key]
        sem_r = info.get("semantic_total_ratio", 0)
        print(f"  {key:30s}  semantic={sem_r*100:5.1f}%")
        cats = info.get("category_ratios", {})
        for c, r in sorted(cats.items(), key=lambda x: -x[1])[:5]:
            print(f"    {c:20s} {r*100:5.1f}%")

    print("\n" + "=" * 72)
    print(f"  Processed {len(all_sample_records)} samples, {len(failures)} failures")
    print(f"  Results: {args.out_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
