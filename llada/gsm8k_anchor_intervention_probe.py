"""
Anchor Intervention Probe: Oracle Reveal & Delay Unmask
=======================================================

Step-0 high-mass window가 주변 토큰의 anchor 역할을 하는지 인과적으로 검증.

실험 A — Oracle Reveal:
  Window 안 토큰을 baseline final output(pseudo-target)으로 즉시 공개.
  주변(neighbor) 토큰의 confidence lift / entropy drop / stabilization 단축 측정.
  → "anchor 정보가 있으면 나머지가 얼마나 빨리 좋아지는가"

실험 B — Delay Unmask:
  Window 안 토큰의 unmask를 처음 d step 동안 강제 지연.
  주변 토큰의 confidence 손실 / entropy 증가 / stabilization 지연 측정.
  → "anchor를 늦췄을 때 주변이 얼마나 손해보는가"

핵심 지표:
  ΔC(k, R): 주변 confidence lift (intervention − baseline) at step k
  ΔH(k, R): 주변 entropy drop (intervention − baseline) at step k
  Δstep:    stabilization step reduction for neighbors

비교:
  high-mass oracle vs low-mass oracle → high-mass가 더 큰 lift이면 anchor 역할 확인
  high-mass delay vs baseline         → delay로 인한 neighbor 손해가 크면 anchor 의존성 확인
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_hybrid_cdf_eval import add_gumbel_noise
from gsm8k_high_mass_unmask_tracking_eval import (
    classify_token_gsm8k,
    SEMANTIC_CATEGORIES,
    compute_step0_rollout,
    get_control_centers,
)


# ═══════════════════════════════════════════════════════════════════════════
# Generation with Intervention Support
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_block_free_intervention(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    tokens_per_step: int = 1,
    temperature: float = 0.0,
    oracle_reveal: Optional[Dict[int, int]] = None,
    delay_positions: Optional[Set[int]] = None,
    delay_steps: int = 0,
) -> Dict[str, Any]:
    """Block-free generation with optional oracle reveal / delay intervention.

    Modes:
      baseline:  oracle_reveal=None, delay_positions=None
      oracle:    oracle_reveal={rel_pos: target_token_id, ...}
      delay:     delay_positions={rel_pos, ...}, delay_steps=d

    Trajectory format: (step, confidence, entropy, predicted_token_id)
    """
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    if oracle_reveal:
        for rp, tid in oracle_reveal.items():
            x[0, prompt_len + rp] = tid

    trajectories: List[List[Tuple[int, float, float, int]]] = [
        [] for _ in range(gen_length)
    ]
    unmask_step: List[Optional[int]] = [None] * gen_length
    unmask_confidence: List[Optional[float]] = [None] * gen_length

    if oracle_reveal:
        for rp in oracle_reveal:
            unmask_step[rp] = -1
            unmask_confidence[rp] = 1.0

    if delay_positions is None:
        delay_positions = set()

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

        probs_f64 = F.softmax(logits.to(torch.float64), dim=-1)
        conf = torch.gather(probs_f64, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

        gen_logits_f32 = logits[0, prompt_len:prompt_len + gen_length].float()
        gen_probs_f32 = F.softmax(gen_logits_f32, dim=-1)
        gen_entropy = -(gen_probs_f32 * gen_probs_f32.clamp(min=1e-10).log()).sum(dim=-1)

        gen_conf = conf[0, prompt_len:prompt_len + gen_length]
        gen_x0 = x0[0, prompt_len:prompt_len + gen_length]

        gen_conf_np = gen_conf.cpu().float().numpy()
        gen_entropy_np = gen_entropy.cpu().numpy()
        gen_x0_np = gen_x0.cpu().numpy()

        for rp in gen_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            trajectories[rp].append(
                (step, float(gen_conf_np[rp]), float(gen_entropy_np[rp]), int(gen_x0_np[rp]))
            )

        available_mask = gen_mask.clone()
        if delay_positions and step < delay_steps:
            for rp in delay_positions:
                if 0 <= rp < gen_length:
                    available_mask[rp] = False

        n_avail = int(available_mask.sum().item())
        if n_avail == 0:
            continue

        neg_inf = torch.finfo(gen_conf.dtype).min
        conf_avail = torch.where(
            available_mask, gen_conf,
            torch.tensor(neg_inf, device=device, dtype=gen_conf.dtype),
        )

        k = min(tokens_per_step, n_avail)
        _, top_rel = torch.topk(conf_avail, k=int(k))

        for rp in top_rel.tolist():
            if unmask_step[rp] is None:
                unmask_step[rp] = step
                unmask_confidence[rp] = float(gen_conf_np[rp])
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
# Neighbor Definition & Trajectory Lookup
# ═══════════════════════════════════════════════════════════════════════════

def get_neighbor_positions(
    window_positions: List[int],
    gen_length: int,
    neighbor_radius: int,
) -> List[int]:
    """Window 바깥, 경계에서 ±neighbor_radius 이내의 위치."""
    if not window_positions:
        return []
    wset = set(window_positions)
    ws = min(window_positions)
    we = max(window_positions) + 1
    neighbors = []
    for pos in range(max(0, ws - neighbor_radius), ws):
        if pos not in wset:
            neighbors.append(pos)
    for pos in range(we, min(gen_length, we + neighbor_radius)):
        if pos not in wset:
            neighbors.append(pos)
    return neighbors


def _traj_at_step(
    traj: List[Tuple[int, float, float, int]], step: int,
) -> Optional[Tuple[float, float]]:
    """Return (confidence, entropy) at the given step, or None if position was already unmasked."""
    for s, c, e, _ in traj:
        if s == step:
            return (c, e)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Intervention Delta Metrics
# ═══════════════════════════════════════════════════════════════════════════

K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def compute_delta_curves(
    baseline_trajs: List[List],
    interv_trajs: List[List],
    neighbor_positions: List[int],
    k_values: List[int] = K_VALUES,
) -> Dict[str, Any]:
    """ΔC(k) and ΔH(k) curves for neighbor positions."""
    curve: Dict[str, Any] = {}
    for k in k_values:
        dc, dh = [], []
        for pos in neighbor_positions:
            b = _traj_at_step(baseline_trajs[pos], k)
            iv = _traj_at_step(interv_trajs[pos], k)
            if b is not None and iv is not None:
                dc.append(iv[0] - b[0])
                dh.append(iv[1] - b[1])
        curve[k] = {
            "delta_conf_mean": _sm(dc), "delta_conf_std": _ss(dc),
            "delta_entropy_mean": _sm(dh), "delta_entropy_std": _ss(dh),
            "n_valid": len(dc),
        }
    return curve


def compute_position_metrics_4t(
    traj: List[Tuple[int, float, float, int]],
    unmask_step_val: Optional[int],
) -> Dict[str, Any]:
    """Position-level maturation from 4-tuple trajectory."""
    if not traj:
        return {
            "step_first_05": None, "step_first_08": None, "step_first_09": None,
            "convergence_step": None, "unmask_step": unmask_step_val,
        }
    confs = [c for _, c, _, _ in traj]
    preds = [p for _, _, _, p in traj]

    step_05 = next((s for s, c, _, _ in traj if c >= 0.5), None)
    step_08 = next((s for s, c, _, _ in traj if c >= 0.8), None)
    step_09 = next((s for s, c, _, _ in traj if c >= 0.9), None)

    final_pred = preds[-1]
    conv_idx = len(preds) - 1
    for t in range(len(preds) - 2, -1, -1):
        if preds[t] == final_pred:
            conv_idx = t
        else:
            break

    return {
        "step_first_05": step_05,
        "step_first_08": step_08,
        "step_first_09": step_09,
        "convergence_step": traj[conv_idx][0],
        "unmask_step": unmask_step_val,
    }


def compute_stabilization_deltas(
    base_metrics: List[Dict],
    interv_metrics: List[Dict],
    neighbor_positions: List[int],
) -> Dict[str, Any]:
    """Δstep_first_0.8, Δstep_first_0.9, Δconvergence for neighbors."""
    d08, d09, dconv = [], [], []
    for p in neighbor_positions:
        bm, im = base_metrics[p], interv_metrics[p]
        if bm["step_first_08"] is not None and im["step_first_08"] is not None:
            d08.append(im["step_first_08"] - bm["step_first_08"])
        if bm["step_first_09"] is not None and im["step_first_09"] is not None:
            d09.append(im["step_first_09"] - bm["step_first_09"])
        if bm["convergence_step"] is not None and im["convergence_step"] is not None:
            dconv.append(im["convergence_step"] - bm["convergence_step"])
    return {
        "delta_step_08_mean": _sm(d08), "delta_step_08_std": _ss(d08), "n_08": len(d08),
        "delta_step_09_mean": _sm(d09), "delta_step_09_std": _ss(d09), "n_09": len(d09),
        "delta_conv_mean": _sm(dconv), "delta_conv_std": _ss(dconv), "n_conv": len(dconv),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sm(v):
    return sum(v) / len(v) if v else None

def _ss(v):
    if len(v) < 2:
        return 0.0 if len(v) == 1 else None
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Anchor intervention probe: oracle reveal & delay unmask",
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
    p.add_argument("--tokens-per-step", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--rollout-mode", type=str, default="sigmoid",
                   choices=["sigmoid", "sigmoid_inverted", "baseline"])
    p.add_argument("--top-k-high-mass", type=int, default=3)
    p.add_argument("--window-radius", type=int, default=4)
    p.add_argument("--neighbor-radius", type=int, default=8)
    p.add_argument("--delay-steps", type=int, default=20,
                   help="Delay unmask: window tokens blocked for first d steps")
    p.add_argument("--k-values", type=str, default="1,2,4,8,16,32,64",
                   help="Steps after intervention to measure ΔC/ΔH")
    p.add_argument("--skip-low-mass-delay", action="store_true",
                   help="Low-mass delay control을 건너뛰어 실행 시간 단축")
    p.add_argument("--out-dir", type=str, default="results_anchor_intervention")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    k_values = [int(x.strip()) for x in args.k_values.split(",")]

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
        f"_nr{args.neighbor_radius}_d{args.delay_steps}"
    )

    # Intervention configurations:
    #   oracle_high, oracle_low : oracle reveal for high-mass / low-mass windows
    #   delay_high, [delay_low] : delay unmask for high-mass / [low-mass] windows
    intervention_types = ["oracle_high", "oracle_low", "delay_high"]
    if not args.skip_low_mass_delay:
        intervention_types.append("delay_low")

    n_runs_per_sample = 1 + args.top_k_high_mass * len(intervention_types)
    fwd_per_run = args.gen_length // args.tokens_per_step
    print(f"\n~{n_runs_per_sample} runs/sample × {fwd_per_run} fwd/run × {len(valid_ids)} samples"
          f" ≈ {n_runs_per_sample * fwd_per_run * len(valid_ids)} total forward passes")

    # ── Aggregate buffers ──
    # key → list of per-window dicts
    agg_delta_curves: Dict[str, List[Dict]] = defaultdict(list)
    agg_stab_deltas: Dict[str, List[Dict]] = defaultdict(list)

    all_window_rows: List[Dict[str, Any]] = []
    sample_summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sid in tqdm(valid_ids, desc="Anchor intervention"):
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

            # ── Baseline generation ──
            base_result = generate_block_free_intervention(
                model, input_ids, args.gen_length, args.mask_id,
                args.tokens_per_step, args.temperature,
            )
            base_trajs = base_result["trajectories"]
            base_final = base_result["final_x"]

            base_pos_metrics = [
                compute_position_metrics_4t(base_trajs[rp], base_result["unmask_step"][rp])
                for rp in range(args.gen_length)
            ]

            # ── Windows ──
            k_top = min(args.top_k_high_mass, args.gen_length)
            high_centers = torch.argsort(gen_scores, descending=True)[:k_top].tolist()
            high_centers = list(dict.fromkeys(int(i) for i in high_centers))

            controls = get_control_centers(
                gen_scores, args.top_k_high_mass, args.window_radius,
                args.gen_length, args.seed + sid,
            )
            low_centers = controls["low_mass"]

            sample_summary = {
                "sample_id": int(sid),
                "question": question[:300],
                "base_total_steps": base_result["total_steps"],
            }

            def _run_interventions(centers, mass_tag):
                """Run oracle + delay for a list of window centers."""
                for rank, center in enumerate(centers, 1):
                    ws = max(0, center - args.window_radius)
                    we = min(args.gen_length, center + args.window_radius + 1)
                    w_positions = list(range(ws, we))
                    neighbors = get_neighbor_positions(
                        w_positions, args.gen_length, args.neighbor_radius,
                    )
                    if not neighbors:
                        continue

                    # Pseudo-targets from baseline final output
                    oracle_map = {
                        rp: int(base_final[0, prompt_len + rp].item())
                        for rp in w_positions
                    }

                    # ── Oracle reveal ──
                    oracle_tag = f"oracle_{mass_tag}"
                    oracle_result = generate_block_free_intervention(
                        model, input_ids, args.gen_length, args.mask_id,
                        args.tokens_per_step, args.temperature,
                        oracle_reveal=oracle_map,
                    )
                    oracle_pos_metrics = [
                        compute_position_metrics_4t(
                            oracle_result["trajectories"][rp],
                            oracle_result["unmask_step"][rp],
                        )
                        for rp in range(args.gen_length)
                    ]

                    dc_oracle = compute_delta_curves(
                        base_trajs, oracle_result["trajectories"], neighbors, k_values,
                    )
                    ds_oracle = compute_stabilization_deltas(
                        base_pos_metrics, oracle_pos_metrics, neighbors,
                    )

                    agg_delta_curves[oracle_tag].append(dc_oracle)
                    agg_stab_deltas[oracle_tag].append(ds_oracle)

                    _save_window_row(
                        sid, mass_tag, "oracle", rank, center,
                        dc_oracle, ds_oracle, neighbors, k_values,
                    )
                    del oracle_result

                    # ── Delay unmask ──
                    delay_tag = f"delay_{mass_tag}"
                    if delay_tag not in intervention_types:
                        continue

                    delay_result = generate_block_free_intervention(
                        model, input_ids, args.gen_length, args.mask_id,
                        args.tokens_per_step, args.temperature,
                        delay_positions=set(w_positions),
                        delay_steps=args.delay_steps,
                    )
                    delay_pos_metrics = [
                        compute_position_metrics_4t(
                            delay_result["trajectories"][rp],
                            delay_result["unmask_step"][rp],
                        )
                        for rp in range(args.gen_length)
                    ]

                    dc_delay = compute_delta_curves(
                        base_trajs, delay_result["trajectories"], neighbors, k_values,
                    )
                    ds_delay = compute_stabilization_deltas(
                        base_pos_metrics, delay_pos_metrics, neighbors,
                    )

                    agg_delta_curves[delay_tag].append(dc_delay)
                    agg_stab_deltas[delay_tag].append(ds_delay)

                    _save_window_row(
                        sid, mass_tag, "delay", rank, center,
                        dc_delay, ds_delay, neighbors, k_values,
                    )
                    del delay_result

            def _save_window_row(sid, mass_tag, interv_type, rank, center, dc, ds, neighbors, kv):
                row: Dict[str, Any] = {
                    "sample_id": sid,
                    "mass_type": mass_tag,
                    "intervention": interv_type,
                    "rank": rank,
                    "center_idx": center,
                    "center_rollout": float(gen_scores[center].item()),
                    "n_neighbors": len(neighbors),
                }
                for k in kv:
                    entry = dc.get(k, {})
                    row[f"dC_k{k}"] = entry.get("delta_conf_mean")
                    row[f"dH_k{k}"] = entry.get("delta_entropy_mean")
                    row[f"n_k{k}"] = entry.get("n_valid")
                row.update(ds)
                all_window_rows.append(row)

            _run_interventions(high_centers, "high")
            _run_interventions(low_centers, "low")

            del base_trajs, base_result
            sample_summaries.append(sample_summary)

        except Exception as e:
            import traceback
            failures.append({"sample_id": int(sid), "error": str(e), "tb": traceback.format_exc()})
            print(f"  [ERROR] Sample {sid}: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # Aggregate
    # ═══════════════════════════════════════════════════════════════════════

    aggregate: Dict[str, Any] = {}

    # ΔC / ΔH curves aggregated across windows
    for tag_key, curves_list in agg_delta_curves.items():
        if not curves_list:
            continue
        agg_curve: Dict[str, Any] = {}
        for k in k_values:
            dcs = [c[k]["delta_conf_mean"] for c in curves_list if c.get(k, {}).get("delta_conf_mean") is not None]
            dhs = [c[k]["delta_entropy_mean"] for c in curves_list if c.get(k, {}).get("delta_entropy_mean") is not None]
            agg_curve[k] = {
                "delta_conf": {"mean": _sm(dcs), "std": _ss(dcs), "n": len(dcs)},
                "delta_entropy": {"mean": _sm(dhs), "std": _ss(dhs), "n": len(dhs)},
            }
        aggregate[f"delta_curves_{tag_key}"] = agg_curve

    # Stabilization deltas aggregated
    for tag_key, stabs_list in agg_stab_deltas.items():
        if not stabs_list:
            continue
        for metric_key in ["delta_step_08_mean", "delta_step_09_mean", "delta_conv_mean"]:
            vals = [s[metric_key] for s in stabs_list if s.get(metric_key) is not None]
            aggregate.setdefault(f"stab_{tag_key}", {})[metric_key] = {
                "mean": _sm(vals), "std": _ss(vals), "n": len(vals),
            }

    # Oracle effect comparison: high vs low
    for k in k_values:
        h_vals = [c[k]["delta_conf_mean"] for c in agg_delta_curves.get("oracle_high", [])
                  if c.get(k, {}).get("delta_conf_mean") is not None]
        l_vals = [c[k]["delta_conf_mean"] for c in agg_delta_curves.get("oracle_low", [])
                  if c.get(k, {}).get("delta_conf_mean") is not None]
        n_p = min(len(h_vals), len(l_vals))
        if n_p > 0:
            deltas = [h_vals[i] - l_vals[i] for i in range(n_p)]
            aggregate.setdefault("oracle_high_minus_low", {})[k] = {
                "delta_conf_diff_mean": _sm(deltas), "n": n_p,
                "interpretation": "양수 = high-mass oracle이 더 큰 anchor 효과",
            }

    # ═══════════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════════

    out_json = os.path.join(args.out_dir, f"anchor_intervention_{tag}.json")
    payload = {
        "config": {
            "model": args.model, "dtype": args.dtype, "device": args.device,
            "seed": args.seed, "sample_ids": valid_ids,
            "gen_length": args.gen_length, "tokens_per_step": args.tokens_per_step,
            "temperature": args.temperature, "rollout_mode": args.rollout_mode,
            "top_k_high_mass": args.top_k_high_mass,
            "window_radius": args.window_radius,
            "neighbor_radius": args.neighbor_radius,
            "delay_steps": args.delay_steps,
            "k_values": k_values,
        },
        "counts": {
            "samples": len(sample_summaries), "failures": len(failures),
            "window_records": len(all_window_rows),
        },
        "aggregate": aggregate,
        "sample_summaries": sample_summaries,
        "failures": failures,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Saved] {out_json}")

    if all_window_rows:
        out_csv = os.path.join(args.out_dir, f"window_deltas_{tag}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_window_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_window_rows)
        print(f"[Saved] {out_csv}")

    # ═══════════════════════════════════════════════════════════════════════
    # Print Summary
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  ANCHOR INTERVENTION PROBE — RESULTS")
    print("=" * 80)

    # ΔC curves
    for tag_key in sorted(agg_delta_curves.keys()):
        agg_c = aggregate.get(f"delta_curves_{tag_key}", {})
        if not agg_c:
            continue
        print(f"\n[{tag_key}] ΔC(k) curve  (neighbor confidence lift)")
        print(f"  {'k':>5s}  {'ΔC mean':>10s}  {'ΔC std':>10s}  {'ΔH mean':>10s}  {'n':>5s}")
        print("  " + "-" * 50)
        for k in k_values:
            entry = agg_c.get(k, {})
            dc = entry.get("delta_conf", {})
            dh = entry.get("delta_entropy", {})
            dc_m = f"{dc['mean']:+.4f}" if dc.get("mean") is not None else "N/A"
            dc_s = f"{dc['std']:.4f}" if dc.get("std") is not None else "N/A"
            dh_m = f"{dh['mean']:+.4f}" if dh.get("mean") is not None else "N/A"
            n = dc.get("n", 0)
            print(f"  {k:>5d}  {dc_m:>10s}  {dc_s:>10s}  {dh_m:>10s}  {n:>5d}")

    # Stabilization deltas
    print(f"\n[Stabilization Step Deltas]")
    print(f"  {'intervention':20s}  {'Δstep_0.8':>12s}  {'Δstep_0.9':>12s}  {'Δconv':>12s}")
    print("  " + "-" * 60)
    for tag_key in sorted(agg_stab_deltas.keys()):
        stab = aggregate.get(f"stab_{tag_key}", {})
        d08 = stab.get("delta_step_08_mean", {}).get("mean")
        d09 = stab.get("delta_step_09_mean", {}).get("mean")
        dcv = stab.get("delta_conv_mean", {}).get("mean")
        d08_s = f"{d08:+.1f}" if d08 is not None else "N/A"
        d09_s = f"{d09:+.1f}" if d09 is not None else "N/A"
        dcv_s = f"{dcv:+.1f}" if dcv is not None else "N/A"
        print(f"  {tag_key:20s}  {d08_s:>12s}  {d09_s:>12s}  {dcv_s:>12s}")

    # Oracle high vs low comparison
    ohl = aggregate.get("oracle_high_minus_low", {})
    if ohl:
        print(f"\n[Oracle Effect: High-mass − Low-mass]")
        print(f"  {'k':>5s}  {'ΔΔC mean':>12s}  {'n':>5s}")
        print("  " + "-" * 30)
        for k in k_values:
            entry = ohl.get(k, {})
            v = entry.get("delta_conf_diff_mean")
            vs = f"{v:+.4f}" if v is not None else "N/A"
            print(f"  {k:>5d}  {vs:>12s}  {entry.get('n', 0):>5d}")
        print("  (양수 = high-mass window가 더 큰 anchor 효과)")

    # Interpretation guide
    print(f"\n[Interpretation Guide]")
    print(f"  Oracle ΔC > 0:  anchor 공개 → 주변 confidence 상승 ✓")
    print(f"  Oracle ΔH < 0:  anchor 공개 → 주변 entropy 감소 ✓")
    print(f"  Oracle Δstep < 0: anchor 공개 → 주변 0.8/0.9 도달 단축 ✓")
    print(f"  Delay  ΔC < 0:  anchor 지연 → 주변 confidence 하락 ✓")
    print(f"  Delay  ΔH > 0:  anchor 지연 → 주변 entropy 증가 ✓")
    print(f"  Delay  Δstep > 0: anchor 지연 → 주변 0.8/0.9 도달 지연 ✓")
    print(f"  High − Low > 0: high-mass가 low-mass보다 더 강한 anchor ✓")

    print("\n" + "=" * 80)
    print(f"  Processed {len(sample_summaries)} samples, {len(failures)} failures")
    print(f"  Results: {args.out_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
