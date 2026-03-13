"""
Block-Free Confidence Maturation Probe
=======================================

목적:
  Block partition 영향을 완전히 제거하고, diffusion 모델 자체가
  Step-0 high-mass region을 어떻게 다루는지 분석.

세팅:
  Block-free: 매 step에서 전체 masked 위치 중
  confidence 최고 토큰을 글로벌하게 1개씩 unmask.
  각 token 위치의 confidence trajectory를 step마다 기록.

실험 1 — Confidence Maturation:
  high-mass window가 random/low-mass window보다
  confidence 0.5/0.8/0.9 도달 step, stabilization step이 더 늦은지.

실험 2 — Token Category Probe:
  high-mass window 내 최종 token의 category enrichment.
  number, operator, relation_word 등 structure-bearing token 비율 비교.

실험 3 — Coupling Probe:
  window 내 token 간 confidence trajectory correlation,
  convergence step synchronization, pairwise stabilization gap.
  high-mass window가 더 strongly coupled dynamics를 보이는지.

핵심 메시지:
  high-mass region은 단순히 중요한 토큰이 많은 곳이 아니라,
  여러 semantic-bearing token이 공동으로 늦게 수렴해야 하는 coupled region.
  → scheduler에서 해당 구간에 더 큰 block (더 많은 joint refinement budget)이 필요한 이유.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

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
)
from gsm8k_high_mass_unmask_tracking_eval import (
    classify_token_gsm8k,
    SEMANTIC_CATEGORIES,
    compute_step0_rollout,
    get_control_centers,
)


# ═══════════════════════════════════════════════════════════════════════════
# Block-Free Generation with Trajectory Tracking
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_block_free_with_trajectory(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    tokens_per_step: int = 1,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Block-free global confidence-based unmasking + per-position trajectory tracking.

    매 step마다:
      1) 전체 masked 위치에 대해 (confidence, predicted_token) 기록
      2) confidence top-k masked 위치를 unmask

    Returns:
      final_x: 최종 생성 결과 (1, prompt_len+gen_length)
      trajectories: gen_length 크기 리스트. 각 원소 = [(step, conf, pred_id), ...]
      unmask_step: 각 위치의 unmask된 global step
      unmask_confidence: unmask 시점의 confidence
      total_steps: 총 forward pass 수
    """
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    trajectories: List[List[Tuple[int, float, int]]] = [[] for _ in range(gen_length)]
    unmask_step: List[Optional[int]] = [None] * gen_length
    unmask_confidence: List[Optional[float]] = [None] * gen_length

    total_steps = 0

    for step in range(gen_length):
        gen_mask = (x[0, prompt_len:prompt_len + gen_length] == mask_id)
        remaining = int(gen_mask.sum().item())
        if remaining == 0:
            break

        total_steps += 1

        outputs = model(x)
        logits = outputs.logits

        if temperature > 0:
            logits_noisy = add_gumbel_noise(logits, temperature)
        else:
            logits_noisy = logits

        x0 = torch.argmax(logits_noisy, dim=-1)
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        conf = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

        gen_conf = conf[0, prompt_len:prompt_len + gen_length]
        gen_x0 = x0[0, prompt_len:prompt_len + gen_length]

        gen_conf_cpu = gen_conf.cpu().float().numpy()
        gen_x0_cpu = gen_x0.cpu().numpy()

        for rp in gen_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            trajectories[rp].append(
                (step, float(gen_conf_cpu[rp]), int(gen_x0_cpu[rp]))
            )

        neg_inf = torch.finfo(gen_conf.dtype).min
        conf_masked = torch.where(
            gen_mask, gen_conf,
            torch.tensor(neg_inf, device=device, dtype=gen_conf.dtype),
        )

        k = min(tokens_per_step, remaining)
        _, top_rel = torch.topk(conf_masked, k=int(k))

        for rp in top_rel.tolist():
            if unmask_step[rp] is None:
                unmask_step[rp] = step
                unmask_confidence[rp] = float(gen_conf_cpu[rp])
                x[0, prompt_len + rp] = gen_x0[rp]

    return {
        "final_x": x,
        "trajectories": trajectories,
        "unmask_step": unmask_step,
        "unmask_confidence": unmask_confidence,
        "total_steps": total_steps,
        "prompt_len": prompt_len,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Position-Level Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_position_metrics(
    traj: List[Tuple[int, float, int]],
    unmask_step_val: Optional[int],
    unmask_conf_val: Optional[float],
) -> Dict[str, Any]:
    """단일 position의 trajectory에서 maturation metrics 계산."""
    null_result = {
        "traj_length": 0,
        "step_first_05": None, "step_first_08": None, "step_first_09": None,
        "convergence_step": None, "last_change_step": None,
        "unmask_step": unmask_step_val, "unmask_confidence": unmask_conf_val,
        "conf_mean": None, "conf_max": None, "conf_min": None,
        "conf_std": None, "conf_final": None,
        "num_pred_changes": None,
    }
    if not traj:
        return null_result

    confs = [c for _, c, _ in traj]
    preds = [p for _, _, p in traj]

    step_05 = next((s for s, c, _ in traj if c >= 0.5), None)
    step_08 = next((s for s, c, _ in traj if c >= 0.8), None)
    step_09 = next((s for s, c, _ in traj if c >= 0.9), None)

    # Convergence step: first step from which prediction equals final and never changes
    final_pred = preds[-1]
    conv_idx = len(preds) - 1
    for t in range(len(preds) - 2, -1, -1):
        if preds[t] == final_pred:
            conv_idx = t
        else:
            break
    convergence_step = traj[conv_idx][0]

    # Last change step: last step where predicted token differs from previous
    lc_idx = 0
    num_changes = 0
    for t in range(1, len(preds)):
        if preds[t] != preds[t - 1]:
            lc_idx = t
            num_changes += 1
    last_change_step = traj[lc_idx][0]

    mean_c = sum(confs) / len(confs)

    return {
        "traj_length": len(traj),
        "step_first_05": step_05,
        "step_first_08": step_08,
        "step_first_09": step_09,
        "convergence_step": convergence_step,
        "last_change_step": last_change_step,
        "unmask_step": unmask_step_val,
        "unmask_confidence": unmask_conf_val,
        "conf_mean": mean_c,
        "conf_max": max(confs),
        "conf_min": min(confs),
        "conf_std": (sum((c - mean_c) ** 2 for c in confs) / len(confs)) ** 0.5,
        "conf_final": confs[-1],
        "num_pred_changes": num_changes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Window-Level Metrics
# ═══════════════════════════════════════════════════════════════════════════

MATURATION_KEYS = [
    "step_first_05", "step_first_08", "step_first_09",
    "convergence_step", "last_change_step", "unmask_step",
    "conf_mean", "conf_max", "conf_std", "conf_final",
    "traj_length", "num_pred_changes",
]


def compute_window_maturation(
    positions: List[int],
    pos_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Exp 1: window 내 position maturation metrics의 평균/std."""
    result: Dict[str, Any] = {}
    for key in MATURATION_KEYS:
        vals = [pos_metrics[p][key] for p in positions if pos_metrics[p][key] is not None]
        if vals:
            m = sum(vals) / len(vals)
            result[f"w_{key}_mean"] = m
            result[f"w_{key}_std"] = (
                (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
                if len(vals) > 1 else 0.0
            )
        else:
            result[f"w_{key}_mean"] = None
            result[f"w_{key}_std"] = None
    return result


def compute_window_categories(
    positions: List[int],
    final_x: torch.Tensor,
    prompt_len: int,
    gen_scores: torch.Tensor,
    tokenizer,
) -> Dict[str, Any]:
    """Exp 2: window 내 최종 token의 category 분포."""
    cats: List[str] = []
    for rp in positions:
        tid = int(final_x[0, prompt_len + rp].item())
        raw = tokenizer.convert_ids_to_tokens([tid])[0]
        cats.append(classify_token_gsm8k(raw))

    cat_counts = dict(Counter(cats))
    total = len(cats)
    cat_ratios = {k: v / total for k, v in cat_counts.items()} if total > 0 else {}
    semantic_count = sum(1 for c in cats if c in SEMANTIC_CATEGORIES)

    return {
        "category_counts": cat_counts,
        "category_ratios": cat_ratios,
        "semantic_ratio": semantic_count / total if total > 0 else 0.0,
        "total_tokens": total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Coupling Probe
# ═══════════════════════════════════════════════════════════════════════════

def _pearson_r(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 5:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    sx = (sum((a - mx) ** 2 for a in x) / n) ** 0.5
    sy = (sum((b - my) ** 2 for b in y) / n) ** 0.5
    if sx < 1e-10 or sy < 1e-10:
        return None
    return cov / (sx * sy)


def compute_window_coupling(
    positions: List[int],
    trajectories: List[List[Tuple[int, float, int]]],
    pos_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Exp 3: window 내 token 간 coupling 지표.

    - Pairwise confidence trajectory Pearson r
    - Pairwise Δconfidence correlation
    - Convergence step synchronization (std, range, pairwise gap)
    """
    valid_pos = [p for p in positions if trajectories[p]]
    if len(valid_pos) < 2:
        return {
            "coupling_n_tokens": len(valid_pos),
            "coupling_valid": False,
        }

    # Pairwise trajectory correlations
    conf_corrs: List[float] = []
    delta_corrs: List[float] = []

    for pi, pj in combinations(valid_pos, 2):
        steps_i = {s: c for s, c, _ in trajectories[pi]}
        steps_j = {s: c for s, c, _ in trajectories[pj]}
        shared = sorted(set(steps_i) & set(steps_j))

        if len(shared) >= 5:
            ci = [steps_i[s] for s in shared]
            cj = [steps_j[s] for s in shared]

            r = _pearson_r(ci, cj)
            if r is not None:
                conf_corrs.append(r)

            if len(shared) >= 6:
                di = [ci[t + 1] - ci[t] for t in range(len(ci) - 1)]
                dj = [cj[t + 1] - cj[t] for t in range(len(cj) - 1)]
                dr = _pearson_r(di, dj)
                if dr is not None:
                    delta_corrs.append(dr)

    # Convergence step synchronization
    conv_steps = [
        pos_metrics[p]["convergence_step"]
        for p in valid_pos
        if pos_metrics[p]["convergence_step"] is not None
    ]
    unmask_steps = [
        pos_metrics[p]["unmask_step"]
        for p in valid_pos
        if pos_metrics[p]["unmask_step"] is not None
    ]

    result: Dict[str, Any] = {
        "coupling_n_tokens": len(valid_pos),
        "coupling_valid": True,
        "n_conf_corr_pairs": len(conf_corrs),
        "n_delta_corr_pairs": len(delta_corrs),
    }

    if conf_corrs:
        result["mean_pairwise_conf_corr"] = sum(conf_corrs) / len(conf_corrs)
        result["median_pairwise_conf_corr"] = float(np.median(conf_corrs))
        result["std_pairwise_conf_corr"] = float(np.std(conf_corrs))
    else:
        result["mean_pairwise_conf_corr"] = None
        result["median_pairwise_conf_corr"] = None
        result["std_pairwise_conf_corr"] = None

    if delta_corrs:
        result["mean_pairwise_delta_corr"] = sum(delta_corrs) / len(delta_corrs)
        result["median_pairwise_delta_corr"] = float(np.median(delta_corrs))
    else:
        result["mean_pairwise_delta_corr"] = None
        result["median_pairwise_delta_corr"] = None

    def _sync_metrics(steps: List[int], prefix: str):
        if len(steps) >= 2:
            m = sum(steps) / len(steps)
            result[f"{prefix}_mean"] = m
            result[f"{prefix}_std"] = (
                (sum((s - m) ** 2 for s in steps) / len(steps)) ** 0.5
            )
            result[f"{prefix}_range"] = max(steps) - min(steps)
            gaps = [abs(a - b) for a, b in combinations(steps, 2)]
            result[f"{prefix}_pairwise_gap"] = sum(gaps) / len(gaps)
        else:
            for sfx in ("_mean", "_std", "_range", "_pairwise_gap"):
                result[f"{prefix}{sfx}"] = None

    _sync_metrics(conv_steps, "convergence")
    _sync_metrics(unmask_steps, "unmask")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_mean(vals):
    return sum(vals) / len(vals) if vals else None


def _safe_std(vals):
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else None
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def _aggregate_cat_counts(list_of_counts: List[Dict[str, int]]) -> Dict[str, float]:
    merged = Counter()
    for c in list_of_counts:
        merged.update(c)
    total = sum(merged.values())
    return {k: v / total for k, v in merged.most_common()} if total else {}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Block-free confidence maturation probe: high-mass vs control windows",
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--sample-ids", type=str, default="")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--tokens-per-step", type=int, default=1,
                   help="1 = 가장 순수한 신호 (느림). 4~8 = 빠르지만 coarser.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--rollout-mode", type=str, default="sigmoid",
                   choices=["sigmoid", "sigmoid_inverted", "baseline"])
    p.add_argument("--top-k-high-mass", type=int, default=3)
    p.add_argument("--window-radius", type=int, default=8)
    p.add_argument("--out-dir", type=str, default="results_block_free_probe")
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

    if args.sample_ids:
        sample_ids = sorted(set(int(x.strip()) for x in args.sample_ids.split(",") if x.strip()))
    else:
        sample_ids = list(range(args.start_id, args.start_id + args.num_samples))
    valid_ids = [i for i in sample_ids if 0 <= i < len(ds)]
    if not valid_ids:
        raise ValueError("유효한 sample id가 없습니다.")

    os.makedirs(args.out_dir, exist_ok=True)

    tag = (
        f"batch{len(valid_ids)}_start{min(valid_ids)}"
        f"_tps{args.tokens_per_step}_{args.rollout_mode}"
        f"_topk{args.top_k_high_mass}_w{args.window_radius}"
    )

    # ── Aggregate buffers ──
    # Exp 1: maturation
    agg_maturation: Dict[str, List[Dict[str, Any]]] = {
        "high_mass": [], "random": [], "low_mass": [],
    }
    # Exp 2: category
    agg_cat_counts: Dict[str, List[Dict[str, int]]] = {
        "high_mass": [], "random": [], "low_mass": [],
    }
    agg_semantic_ratios: Dict[str, List[float]] = {
        "high_mass": [], "random": [], "low_mass": [],
    }
    # Exp 3: coupling
    agg_coupling: Dict[str, List[Dict[str, Any]]] = {
        "high_mass": [], "random": [], "low_mass": [],
    }

    all_position_rows: List[Dict[str, Any]] = []
    all_window_rows: List[Dict[str, Any]] = []
    sample_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    est_fwd_per_sample = args.gen_length // args.tokens_per_step
    print(f"\nBlock-free probe: ~{est_fwd_per_sample} forward passes/sample × {len(valid_ids)} samples")

    for sid in tqdm(valid_ids, desc="Block-free probe"):
        try:
            sample = ds[int(sid)]
            question = sample["question"]

            if args.no_chat_template:
                prompt_str = question
            else:
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    add_generation_prompt=True, tokenize=False,
                )

            input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)
            prompt_len = input_ids.shape[1]

            # ── Step 0: Rollout score ──
            gen_scores = compute_step0_rollout(
                model, input_ids, args.gen_length, args.mask_id, args.rollout_mode,
            )

            # ── Block-free generation with trajectory ──
            gen_result = generate_block_free_with_trajectory(
                model, input_ids, args.gen_length, args.mask_id,
                args.tokens_per_step, args.temperature,
            )

            trajectories = gen_result["trajectories"]
            final_x = gen_result["final_x"]
            total_steps = gen_result["total_steps"]

            # ── Per-position metrics ──
            pos_metrics: List[Dict[str, Any]] = []
            for rp in range(args.gen_length):
                pm = compute_position_metrics(
                    trajectories[rp],
                    gen_result["unmask_step"][rp],
                    gen_result["unmask_confidence"][rp],
                )
                pos_metrics.append(pm)

            # ── Windows ──
            k = min(args.top_k_high_mass, args.gen_length)
            top_indices = torch.argsort(gen_scores, descending=True)[:k].tolist()
            top_indices = list(dict.fromkeys(int(i) for i in top_indices))

            controls = get_control_centers(
                gen_scores, args.top_k_high_mass, args.window_radius,
                args.gen_length, args.seed + sid,
            )

            windows_spec: List[Tuple[str, int, int]] = []
            for rank, ci in enumerate(top_indices, 1):
                windows_spec.append(("high_mass", rank, ci))
            for rank, ci in enumerate(controls["random"], 1):
                windows_spec.append(("random", rank, ci))
            for rank, ci in enumerate(controls["low_mass"], 1):
                windows_spec.append(("low_mass", rank, ci))

            sample_summary = {
                "sample_id": int(sid),
                "question": question[:300],
                "total_steps": total_steps,
                "nfe": total_steps + 1,
            }

            for wtype, rank, center_idx in windows_spec:
                ws = max(0, center_idx - args.window_radius)
                we = min(args.gen_length, center_idx + args.window_radius + 1)
                positions = list(range(ws, we))

                # Exp 1: Maturation
                mat = compute_window_maturation(positions, pos_metrics)
                agg_maturation[wtype].append(mat)

                # Exp 2: Category
                cat = compute_window_categories(
                    positions, final_x, prompt_len, gen_scores, tokenizer,
                )
                agg_cat_counts[wtype].append(cat["category_counts"])
                agg_semantic_ratios[wtype].append(cat["semantic_ratio"])

                # Exp 3: Coupling
                coup = compute_window_coupling(positions, trajectories, pos_metrics)
                agg_coupling[wtype].append(coup)

                # Position-level CSV rows
                for rp in positions:
                    tid = int(final_x[0, prompt_len + rp].item())
                    raw_tok = tokenizer.convert_ids_to_tokens([tid])[0]
                    dec_tok = tokenizer.decode([tid], skip_special_tokens=False).strip() or raw_tok
                    cat_label = classify_token_gsm8k(raw_tok)

                    all_position_rows.append({
                        "sample_id": sid,
                        "window_type": wtype,
                        "rank": rank,
                        "center_idx": center_idx,
                        "rel_pos": rp,
                        "token_string": dec_tok,
                        "category": cat_label,
                        "rollout_score": float(gen_scores[rp].item()),
                        **{mk: pos_metrics[rp][mk] for mk in [
                            "unmask_step", "unmask_confidence",
                            "step_first_05", "step_first_08", "step_first_09",
                            "convergence_step", "last_change_step",
                            "conf_mean", "conf_max", "conf_std", "conf_final",
                            "traj_length", "num_pred_changes",
                        ]},
                    })

                # Window-level CSV row
                window_row: Dict[str, Any] = {
                    "sample_id": sid,
                    "window_type": wtype,
                    "rank": rank,
                    "center_idx": center_idx,
                    "center_rollout_score": float(gen_scores[center_idx].item()),
                    "window_size": len(positions),
                    "semantic_ratio": cat["semantic_ratio"],
                }
                window_row.update(mat)
                for ck in [
                    "mean_pairwise_conf_corr", "median_pairwise_conf_corr",
                    "std_pairwise_conf_corr",
                    "mean_pairwise_delta_corr", "median_pairwise_delta_corr",
                    "convergence_mean", "convergence_std",
                    "convergence_range", "convergence_pairwise_gap",
                    "unmask_mean", "unmask_std", "unmask_range", "unmask_pairwise_gap",
                ]:
                    window_row[ck] = coup.get(ck)
                window_row["category_counts"] = json.dumps(cat["category_counts"])
                all_window_rows.append(window_row)

            # Free large trajectory data
            del trajectories
            sample_summaries.append(sample_summary)

        except Exception as e:
            import traceback
            failures.append({"sample_id": int(sid), "error": str(e), "tb": traceback.format_exc()})
            print(f"  [ERROR] Sample {sid}: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # Aggregate Analysis
    # ═══════════════════════════════════════════════════════════════════════

    aggregate: Dict[str, Any] = {}

    # ── Exp 1: Maturation ──
    aggregate["exp1_maturation"] = {}
    for wtype in ["high_mass", "random", "low_mass"]:
        mats = agg_maturation[wtype]
        if not mats:
            continue
        summary = {}
        for key in [f"w_{k}_mean" for k in MATURATION_KEYS]:
            vals = [m[key] for m in mats if m.get(key) is not None]
            summary[key] = {"mean": _safe_mean(vals), "std": _safe_std(vals), "n": len(vals)}
        aggregate["exp1_maturation"][wtype] = summary

    # Paired delta: high_mass - random, high_mass - low_mass
    for ctrl in ["random", "low_mass"]:
        n_pairs = min(len(agg_maturation["high_mass"]), len(agg_maturation[ctrl]))
        if n_pairs == 0:
            continue
        delta_key = f"delta_high_mass_minus_{ctrl}"
        aggregate["exp1_maturation"][delta_key] = {}
        for mkey in ["w_step_first_08_mean", "w_step_first_09_mean",
                      "w_convergence_step_mean", "w_unmask_step_mean"]:
            hv = [m[mkey] for m in agg_maturation["high_mass"][:n_pairs] if m[mkey] is not None]
            cv = [m[mkey] for m in agg_maturation[ctrl][:n_pairs] if m[mkey] is not None]
            n_both = min(len(hv), len(cv))
            if n_both > 0:
                deltas = [hv[i] - cv[i] for i in range(n_both)]
                aggregate["exp1_maturation"][delta_key][mkey] = {
                    "delta_mean": _safe_mean(deltas),
                    "delta_std": _safe_std(deltas),
                    "n": n_both,
                    "interpretation": "양수 = high-mass가 더 늦게 도달/수렴",
                }

    # ── Exp 2: Category enrichment ──
    aggregate["exp2_category"] = {}
    for wtype in ["high_mass", "random", "low_mass"]:
        cats = agg_cat_counts[wtype]
        sems = agg_semantic_ratios[wtype]
        if not cats:
            continue
        aggregate["exp2_category"][wtype] = {
            "category_ratios": _aggregate_cat_counts(cats),
            "semantic_ratio_mean": _safe_mean(sems),
            "semantic_ratio_std": _safe_std(sems),
        }

    # ── Exp 3: Coupling ──
    aggregate["exp3_coupling"] = {}
    for wtype in ["high_mass", "random", "low_mass"]:
        coups = [c for c in agg_coupling[wtype] if c.get("coupling_valid")]
        if not coups:
            continue
        summary = {}
        for ckey in [
            "mean_pairwise_conf_corr", "mean_pairwise_delta_corr",
            "convergence_std", "convergence_range", "convergence_pairwise_gap",
            "unmask_std", "unmask_range", "unmask_pairwise_gap",
        ]:
            vals = [c[ckey] for c in coups if c.get(ckey) is not None]
            summary[ckey] = {"mean": _safe_mean(vals), "std": _safe_std(vals), "n": len(vals)}
        aggregate["exp3_coupling"][wtype] = summary

    # Coupling delta
    for ctrl in ["random", "low_mass"]:
        h_coups = [c for c in agg_coupling["high_mass"] if c.get("coupling_valid")]
        c_coups = [c for c in agg_coupling[ctrl] if c.get("coupling_valid")]
        n_pairs = min(len(h_coups), len(c_coups))
        if n_pairs == 0:
            continue
        delta_key = f"delta_high_mass_minus_{ctrl}"
        aggregate["exp3_coupling"][delta_key] = {}
        for ckey in ["mean_pairwise_conf_corr", "mean_pairwise_delta_corr",
                      "convergence_std", "convergence_pairwise_gap"]:
            hv = [c[ckey] for c in h_coups[:n_pairs] if c.get(ckey) is not None]
            cv = [c[ckey] for c in c_coups[:n_pairs] if c.get(ckey) is not None]
            n_both = min(len(hv), len(cv))
            if n_both > 0:
                deltas = [hv[i] - cv[i] for i in range(n_both)]
                interp = {
                    "mean_pairwise_conf_corr": "양수 = high-mass가 더 coupled",
                    "mean_pairwise_delta_corr": "양수 = high-mass가 더 coupled",
                    "convergence_std": "음수 = high-mass가 더 synchronized",
                    "convergence_pairwise_gap": "음수 = high-mass가 더 synchronized",
                }
                aggregate["exp3_coupling"][delta_key][ckey] = {
                    "delta_mean": _safe_mean(deltas),
                    "n": n_both,
                    "interpretation": interp.get(ckey, ""),
                }

    # ═══════════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════════

    out_json = os.path.join(args.out_dir, f"block_free_probe_{tag}.json")
    payload = {
        "config": {
            "model": args.model, "dtype": args.dtype, "device": args.device,
            "seed": args.seed, "sample_ids": valid_ids,
            "gen_length": args.gen_length, "tokens_per_step": args.tokens_per_step,
            "temperature": args.temperature, "rollout_mode": args.rollout_mode,
            "top_k_high_mass": args.top_k_high_mass, "window_radius": args.window_radius,
        },
        "counts": {
            "samples": len(sample_summaries), "failures": len(failures),
            "position_records": len(all_position_rows),
            "window_records": len(all_window_rows),
        },
        "aggregate": aggregate,
        "sample_summaries": sample_summaries,
        "failures": failures,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Saved] {out_json}")

    if all_position_rows:
        out_pos_csv = os.path.join(args.out_dir, f"position_level_{tag}.csv")
        with open(out_pos_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_position_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_position_rows)
        print(f"[Saved] {out_pos_csv}")

    if all_window_rows:
        out_win_csv = os.path.join(args.out_dir, f"window_summary_{tag}.csv")
        with open(out_win_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_window_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_window_rows)
        print(f"[Saved] {out_win_csv}")

    # ═══════════════════════════════════════════════════════════════════════
    # Print Summary
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 76)
    print("  BLOCK-FREE CONFIDENCE MATURATION PROBE — RESULTS")
    print("=" * 76)

    # Exp 1
    print("\n[Exp 1] Confidence Maturation (window-level means)")
    print("-" * 72)
    header = f"  {'metric':32s} {'high_mass':>12s} {'random':>12s} {'low_mass':>12s}"
    print(header)
    print("  " + "-" * 68)
    for mkey in ["w_step_first_08_mean", "w_step_first_09_mean",
                  "w_convergence_step_mean", "w_unmask_step_mean",
                  "w_conf_mean_mean", "w_num_pred_changes_mean"]:
        vals = []
        for wt in ["high_mass", "random", "low_mass"]:
            info = aggregate.get("exp1_maturation", {}).get(wt, {}).get(mkey, {})
            v = info.get("mean") if isinstance(info, dict) else None
            vals.append(f"{v:.2f}" if v is not None else "N/A")
        print(f"  {mkey:32s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s}")

    for ctrl in ["random", "low_mass"]:
        dk = f"delta_high_mass_minus_{ctrl}"
        deltas = aggregate.get("exp1_maturation", {}).get(dk, {})
        if deltas:
            print(f"\n  Delta (high_mass - {ctrl}):")
            for mkey, info in deltas.items():
                dm = info.get("delta_mean")
                sign = "+" if dm and dm > 0 else ""
                val = f"{sign}{dm:.2f}" if dm is not None else "N/A"
                print(f"    {mkey:30s} = {val:>8s}  ({info.get('interpretation', '')})")

    # Exp 2
    print(f"\n[Exp 2] Token Category Enrichment")
    print("-" * 72)
    for wt in ["high_mass", "random", "low_mass"]:
        info = aggregate.get("exp2_category", {}).get(wt, {})
        sem = info.get("semantic_ratio_mean")
        sem_str = f"{sem*100:.1f}%" if sem is not None else "N/A"
        print(f"  {wt:15s}  semantic_ratio = {sem_str}")
        cats = info.get("category_ratios", {})
        for c, r in sorted(cats.items(), key=lambda x: -x[1])[:5]:
            bar = "█" * int(r * 40)
            print(f"    {c:20s} {r*100:5.1f}% {bar}")

    # Exp 3
    print(f"\n[Exp 3] Coupling Probe")
    print("-" * 72)
    header = f"  {'metric':32s} {'high_mass':>12s} {'random':>12s} {'low_mass':>12s}"
    print(header)
    print("  " + "-" * 68)
    for ckey in ["mean_pairwise_conf_corr", "mean_pairwise_delta_corr",
                  "convergence_std", "convergence_pairwise_gap"]:
        vals = []
        for wt in ["high_mass", "random", "low_mass"]:
            info = aggregate.get("exp3_coupling", {}).get(wt, {}).get(ckey, {})
            v = info.get("mean") if isinstance(info, dict) else None
            vals.append(f"{v:.4f}" if v is not None else "N/A")
        print(f"  {ckey:32s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s}")

    for ctrl in ["random", "low_mass"]:
        dk = f"delta_high_mass_minus_{ctrl}"
        deltas = aggregate.get("exp3_coupling", {}).get(dk, {})
        if deltas:
            print(f"\n  Delta (high_mass - {ctrl}):")
            for ckey, info in deltas.items():
                dm = info.get("delta_mean")
                sign = "+" if dm and dm > 0 else ""
                val = f"{sign}{dm:.4f}" if dm is not None else "N/A"
                print(f"    {ckey:30s} = {val:>10s}  ({info.get('interpretation', '')})")

    print("\n" + "=" * 76)
    print(f"  Processed {len(sample_summaries)} samples, {len(failures)} failures")
    print(f"  tokens_per_step={args.tokens_per_step}")
    print(f"  Results: {args.out_dir}/")
    print("=" * 76)


if __name__ == "__main__":
    main()
