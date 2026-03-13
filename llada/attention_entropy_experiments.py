"""
Attention entropy experiments for hub-token hypothesis.

Experiment 1: Attention Fan-in Entropy (Step 0)
  - Compute A(i, j) from mean attention over layers/heads at Step 0.
  - Compute per-position entropy H_i = -sum_j A(i,j) log A(i,j).
  - Compare H_i between high-mass and low-mass groups, where groups are
    defined by Step 0 deep-layer rollout scores.

Experiment 2: Entropy Collapse in Pure Global Diffusion
  - Decode with global (non-block) diffusion: all masked positions compete
    globally by confidence each step.
  - Track per-position predictive entropy at every step.
  - Compare average entropy trajectories of high-mass vs low-mass groups.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM


def get_depth_adaptive_rollout_matrix(attentions: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    with torch.no_grad():
        if attentions[0].dim() == 4:
            avg_attn = [a.mean(dim=1) for a in attentions]
        else:
            avg_attn = list(attentions)

        num_layers = len(avg_attn)
        res_attn: List[torch.Tensor] = []
        for i, a in enumerate(avg_attn):
            eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)
            slope = 0.5
            mid = num_layers / 2
            depth_arg = i - mid
            alpha = 0.5 * torch.sigmoid(
                torch.tensor(slope * depth_arg, device=a.device, dtype=a.dtype)
            )
            res_attn.append((1.0 - alpha) * eye + alpha * a)

        rollout = res_attn[0]
        for i in range(1, len(res_attn)):
            rollout = torch.matmul(res_attn[i], rollout)
        return rollout[0]  # (T,)


def per_position_entropy(dist: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = dist.clamp(min=eps)
    return -(p * torch.log(p)).sum(dim=-1)


def build_global_step_schedule(total_tokens: int, steps: int) -> List[int]:
    if steps <= 0:
        return [total_tokens]
    base = total_tokens // steps
    rem = total_tokens % steps
    schedule = [base + (1 if i < rem else 0) for i in range(steps)]
    schedule = [x for x in schedule if x > 0]
    if sum(schedule) < total_tokens:
        schedule.append(total_tokens - sum(schedule))
    return schedule


@torch.no_grad()
def step0_attention_entropy_and_mass(
    model,
    input_ids: torch.Tensor,
    gen_length: int,
    mask_id: int,
) -> Dict[str, torch.Tensor]:
    device = model.device
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length

    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    out = model(x, output_attentions=True)
    attentions = out.attentions  # tuple[L], each (B,H,T,T)

   # 수정된 부분: Deep-layer rollout 매트릭스 위에서 모두 계산!
    rollout_matrix = get_depth_adaptive_rollout_matrix(attentions).to(torch.float64)  # (T, T)
    
    # 1. Row 기준 엔트로피 (i번째 토큰이 얼마나 넓게 문맥을 바라보는가: Fan-in)
    entropy_all = per_position_entropy(rollout_matrix).to(torch.float64)  # (T,)
    
    # 2. Column 기준 합계 (다른 토큰들이 얼마나 i번째 토큰을 주목하는가: Semantic Mass)
    rollout_scores = rollout_matrix.sum(dim=0)  # (T,)

    return {
        "prompt_len": torch.tensor(prompt_len, device=device),
        "entropy_gen": entropy_all[prompt_len:prompt_len + gen_length].detach(),
        "rollout_gen": rollout_scores[prompt_len:prompt_len + gen_length].detach(),
    }


def select_high_low_groups(
    scores: torch.Tensor,
    high_pct: float,
    low_pct: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = scores.numel()
    k_high = max(1, int(n * high_pct))
    k_low = max(1, int(n * low_pct))

    sorted_idx = torch.argsort(scores, descending=True)
    high_idx = sorted_idx[:k_high]
    low_idx = torch.argsort(scores, descending=False)[:k_low]
    return high_idx, low_idx


@torch.no_grad()
def run_global_diffusion_tracking(
    model,
    input_ids: torch.Tensor,
    gen_length: int,
    mask_id: int,
    steps: int,
) -> Dict[str, Any]:
    device = model.device
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length

    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    schedule = build_global_step_schedule(gen_length, steps)
    entropy_traj = []
    confidence_traj = []
    unmasked_count = []

    for k in schedule:
        out = model(x)
        logits = out.logits  # (1,T,V)
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        ent = per_position_entropy(probs)[0, prompt_len:prompt_len + gen_length]  # (G,)
        conf = probs[0, prompt_len:prompt_len + gen_length].max(dim=-1).values  # (G,)

        entropy_traj.append(ent.detach().to(torch.float64).cpu().numpy())
        confidence_traj.append(conf.detach().to(torch.float64).cpu().numpy())

        gen_mask = (x[0, prompt_len:prompt_len + gen_length] == mask_id)  # (G,)
        masked_idx = torch.nonzero(gen_mask, as_tuple=False).squeeze(-1)
        if masked_idx.numel() == 0:
            break

        k_eff = min(k, masked_idx.numel())
        masked_conf = conf[masked_idx]
        top_local = torch.topk(masked_conf, k=k_eff, largest=True).indices
        chosen_gen_idx = masked_idx[top_local]

        preds = logits.argmax(dim=-1)[0, prompt_len:prompt_len + gen_length]  # (G,)
        x[0, prompt_len + chosen_gen_idx] = preds[chosen_gen_idx]

        unmasked_count.append(int((x[0, prompt_len:prompt_len + gen_length] != mask_id).sum().item()))

    # final snapshot (optional final step after all/last update)
    out = model(x)
    logits = out.logits
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    ent = per_position_entropy(probs)[0, prompt_len:prompt_len + gen_length]
    conf = probs[0, prompt_len:prompt_len + gen_length].max(dim=-1).values
    entropy_traj.append(ent.detach().to(torch.float64).cpu().numpy())
    confidence_traj.append(conf.detach().to(torch.float64).cpu().numpy())
    unmasked_count.append(int((x[0, prompt_len:prompt_len + gen_length] != mask_id).sum().item()))

    return {
        "entropy_traj": np.stack(entropy_traj, axis=0),       # (S, G)
        "confidence_traj": np.stack(confidence_traj, axis=0), # (S, G)
        "steps_effective": int(len(entropy_traj)),
        "unmasked_count": unmasked_count,
    }


def get_prompt(task: str, sample: Dict[str, Any]) -> str:
    if task == "gsm8k":
        return sample["question"]
    if task == "humaneval":
        return sample["prompt"]
    raise ValueError(f"Unsupported task: {task}")


def load_task(task: str, seed: int, num_samples: int):
    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    elif task == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
    else:
        raise ValueError(f"Unsupported task: {task}")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    return ds


def plot_exp1_entropy_box(
    high_vals: List[float],
    low_vals: List[float],
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.boxplot([low_vals, high_vals], labels=["Low-mass", "High-mass"], patch_artist=True)
    ax.set_ylabel("Attention Fan-in Entropy")
    ax.set_title("Exp1: Step0 Fan-in Entropy by Mass Group")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_exp2_entropy_curves(
    mean_high: np.ndarray,
    mean_low: np.ndarray,
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(mean_high))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, mean_low, label="Low-mass", linewidth=2)
    ax.plot(x, mean_high, label="High-mass", linewidth=2)
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Average Predictive Entropy")
    ax.set_title("Exp2: Entropy Collapse (Pure Global Diffusion)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Attention entropy experiments (hub-token validation)")
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--task", type=str, default="humaneval", choices=["gsm8k", "humaneval"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--sample-ids", type=str, default=None, help="Comma-separated IDs in shuffled split")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--global-steps", type=int, default=256)
    p.add_argument("--high-pct", type=float, default=0.2)
    p.add_argument("--low-pct", type=float, default=0.2)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--out-dir", type=str, default="results_entropy_hub")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Info] Loading model: {args.model} on {args.device} ({args.dtype})")
    model = LLaDAModelLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ds = load_task(args.task, args.seed, args.num_samples)
    if args.sample_ids:
        sample_ids = [int(x.strip()) for x in args.sample_ids.split(",") if x.strip()]
    else:
        sample_ids = list(range(len(ds)))

    exp1_high_entropy: List[float] = []
    exp1_low_entropy: List[float] = []
    exp2_high_curves: List[np.ndarray] = []
    exp2_low_curves: List[np.ndarray] = []
    per_sample = []

    for sid in sample_ids:
        sample = ds[sid]
        prompt_text = get_prompt(args.task, sample)
        if not args.no_chat_template:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                tokenize=False,
            )

        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(args.device)

        # Step 0: entropy + rollout
        step0 = step0_attention_entropy_and_mass(
            model=model,
            input_ids=input_ids,
            gen_length=args.gen_length,
            mask_id=args.mask_id,
        )
        entropy_gen = step0["entropy_gen"]   # (G,)
        rollout_gen = step0["rollout_gen"]   # (G,)
        high_idx, low_idx = select_high_low_groups(
            scores=rollout_gen, high_pct=args.high_pct, low_pct=args.low_pct
        )

        exp1_high_entropy.extend(entropy_gen[high_idx].detach().cpu().tolist())
        exp1_low_entropy.extend(entropy_gen[low_idx].detach().cpu().tolist())

        # Experiment 2: global diffusion tracking
        tracked = run_global_diffusion_tracking(
            model=model,
            input_ids=input_ids,
            gen_length=args.gen_length,
            mask_id=args.mask_id,
            steps=args.global_steps,
        )
        ent_traj = tracked["entropy_traj"]  # (S, G)
        high_curve = ent_traj[:, high_idx.detach().cpu().numpy()].mean(axis=1)
        low_curve = ent_traj[:, low_idx.detach().cpu().numpy()].mean(axis=1)
        exp2_high_curves.append(high_curve)
        exp2_low_curves.append(low_curve)

        per_sample.append(
            {
                "sample_id": int(sid),
                "exp1_high_mean_entropy": float(
                    np.mean(entropy_gen[high_idx].detach().to(torch.float64).cpu().numpy())
                ),
                "exp1_low_mean_entropy": float(
                    np.mean(entropy_gen[low_idx].detach().to(torch.float64).cpu().numpy())
                ),
                "exp2_steps_effective": int(tracked["steps_effective"]),
                "exp2_high_curve": high_curve.tolist(),
                "exp2_low_curve": low_curve.tolist(),
            }
        )
        print(
            f"[Sample {sid}] Exp1 high={per_sample[-1]['exp1_high_mean_entropy']:.4f}, "
            f"low={per_sample[-1]['exp1_low_mean_entropy']:.4f}, "
            f"steps={tracked['steps_effective']}"
        )

    # Exp1 plots/stats
    exp1_plot = os.path.join(args.out_dir, "exp1_fan_in_entropy_boxplot.png")
    plot_exp1_entropy_box(exp1_high_entropy, exp1_low_entropy, exp1_plot)

    exp1_summary = {
        "high_mean": float(np.mean(exp1_high_entropy)) if exp1_high_entropy else None,
        "low_mean": float(np.mean(exp1_low_entropy)) if exp1_low_entropy else None,
        "high_std": float(np.std(exp1_high_entropy)) if exp1_high_entropy else None,
        "low_std": float(np.std(exp1_low_entropy)) if exp1_low_entropy else None,
        "high_count": len(exp1_high_entropy),
        "low_count": len(exp1_low_entropy),
    }

    # Exp2 aggregate curve (align by min length)
    min_len = min(min(len(c) for c in exp2_high_curves), min(len(c) for c in exp2_low_curves))
    high_arr = np.stack([c[:min_len] for c in exp2_high_curves], axis=0)
    low_arr = np.stack([c[:min_len] for c in exp2_low_curves], axis=0)
    mean_high = high_arr.mean(axis=0)
    mean_low = low_arr.mean(axis=0)

    exp2_plot = os.path.join(args.out_dir, "exp2_entropy_collapse_curves.png")
    plot_exp2_entropy_curves(mean_high, mean_low, exp2_plot)

    exp2_summary = {
        "steps_aligned": int(min_len),
        "high_curve_mean": mean_high.tolist(),
        "low_curve_mean": mean_low.tolist(),
        "high_start_end": [float(mean_high[0]), float(mean_high[-1])],
        "low_start_end": [float(mean_low[0]), float(mean_low[-1])],
    }

    out_json = os.path.join(args.out_dir, "attention_entropy_experiments.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "exp1_summary": exp1_summary,
                "exp2_summary": exp2_summary,
                "per_sample": per_sample,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[Done]")
    print(f"[Saved] {exp1_plot}")
    print(f"[Saved] {exp2_plot}")
    print(f"[Saved] {out_json}")


if __name__ == "__main__":
    main()
