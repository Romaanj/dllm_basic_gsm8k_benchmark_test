"""
GSM8K 샘플에 대해 Original vs Inverse Hybrid CDF를 비교 시각화하는 스크립트.

Original: attention 집중 → 작은 블록 (equal-mass)
Inverse:  attention 집중 → 큰 블록 (contextual thinking budget 가설)

Usage:
  python gsm8k_hybrid_inverse_cdf_plot.py --sample-id 0 --lam 0.7 --num-blocks 8
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_hybrid_cdf_eval import (
    get_depth_adaptive_rollout,
    get_baseline_rollout,
    hybrid_cdf_chunking,
)


@torch.no_grad()
def get_step0_attentions(model, input_ids, gen_length, mask_id):
    device = model.device
    prompt_len = input_ids.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = input_ids.clone()
    outputs = model(x, output_attentions=True)
    return outputs.attentions, prompt_len, gen_length


def plot_original_vs_inverse(
    gen_scores: torch.Tensor,
    gen_length: int,
    num_blocks: int,
    lam: float,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """
    Original hybrid CDF vs Inverse hybrid CDF 비교 시각화 (5-panel).

    Panel 0: 5개 CDF 커브 (attn, inverse_attn, uniform, hybrid, inverse_hybrid)
    Panel 1: Raw attention rollout score + inverse score
    Panel 2: Original hybrid CDF 블록 분할
    Panel 3: Inverse hybrid CDF 블록 분할
    Panel 4: 블록 크기 직접 비교 bar chart
    """
    scores = gen_scores.detach().cpu().to(torch.float64).clamp(min=0)
    total_mass = scores.sum().item()
    x = np.arange(1, gen_length + 1)

    if total_mass < 1e-12:
        attn_cdf_np = np.linspace(1 / gen_length, 1.0, gen_length)
        inv_attn_cdf_np = attn_cdf_np.copy()
    else:
        attn_cdf_np = (torch.cumsum(scores, dim=0) / total_mass).numpy()
        inv_scores = 1.0 / (scores + 1e-10)
        inv_attn_cdf_np = (torch.cumsum(inv_scores, dim=0) / inv_scores.sum()).numpy()

    uniform_cdf = x / gen_length
    hybrid_cdf = lam * attn_cdf_np + (1.0 - lam) * uniform_cdf
    inverse_hybrid_cdf = lam * inv_attn_cdf_np + (1.0 - lam) * uniform_cdf

    blocks_orig = hybrid_cdf_chunking(gen_scores, num_blocks=num_blocks, lam=lam, inverse=False)
    blocks_inv = hybrid_cdf_chunking(gen_scores, num_blocks=num_blocks, lam=lam, inverse=True)
    sizes_orig = [e - s for s, e in blocks_orig]
    sizes_inv = [e - s for s, e in blocks_inv]

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=False)

    # ── Panel 0: CDF curves ──
    ax = axes[0]
    ax.plot(x, attn_cdf_np, label="Attention CDF", color="C1", linewidth=1.2, alpha=0.7)
    ax.plot(x, inv_attn_cdf_np, label="Inverse Attention CDF", color="C2", linewidth=1.2, alpha=0.7)
    ax.plot(x, uniform_cdf, label="Uniform CDF", color="C0", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.plot(x, hybrid_cdf, label=f"Hybrid CDF (λ={lam})", color="C3", linewidth=2.0)
    ax.plot(x, inverse_hybrid_cdf, label=f"Inverse Hybrid CDF (λ={lam})", color="C4", linewidth=2.0, linestyle="-.")

    for k in range(1, num_blocks):
        ax.axhline(y=k / num_blocks, color="gray", linewidth=0.4, alpha=0.3)

    ax.set_ylabel("CDF value")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Original vs Inverse Hybrid CDF {title_suffix}\n"
        f"λ={lam}, N={num_blocks}"
    )

    # ── Panel 1: Attention scores (original + inverse) ──
    ax = axes[1]
    scores_np = scores.numpy()
    inv_scores_np = (1.0 / (scores + 1e-10)).numpy()
    inv_scores_normalized = inv_scores_np / inv_scores_np.sum() * total_mass

    ax.fill_between(x, 0, scores_np, alpha=0.3, color="C1", label="Attention score")
    ax.plot(x, scores_np, color="C1", linewidth=0.8)

    ax2 = ax.twinx()
    ax2.fill_between(x, 0, inv_scores_normalized, alpha=0.2, color="C2", label="Inverse score (scaled)")
    ax2.plot(x, inv_scores_normalized, color="C2", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Inverse score (scaled)", color="C2", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="C2")

    for start, end in blocks_orig[:-1]:
        ax.axvline(x=end, color="C3", linewidth=0.8, linestyle=":", alpha=0.6)
    for start, end in blocks_inv[:-1]:
        ax.axvline(x=end, color="C4", linewidth=0.8, linestyle="-.", alpha=0.6)

    ax.set_ylabel("Attention score", color="C1", fontsize=9)
    ax.tick_params(axis="y", labelcolor="C1")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Attention rollout score (orange) vs Inverse score (green)")

    # ── Panel 2: Original hybrid CDF blocks ──
    ax = axes[2]
    centers = [(s + e) / 2 for s, e in blocks_orig]
    widths = sizes_orig
    colors = plt.cm.RdPu(np.linspace(0.3, 0.8, len(blocks_orig)))
    ax.bar(centers, widths, width=widths, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    for i, (s, e) in enumerate(blocks_orig):
        ax.text((s + e) / 2, e - s + 0.5, f"{e - s}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Block size")
    ax.set_title(f"Original Hybrid CDF (λ={lam}) → sizes={sizes_orig}")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Inverse hybrid CDF blocks ──
    ax = axes[3]
    centers = [(s + e) / 2 for s, e in blocks_inv]
    widths = sizes_inv
    colors = plt.cm.YlGn(np.linspace(0.3, 0.8, len(blocks_inv)))
    ax.bar(centers, widths, width=widths, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    for i, (s, e) in enumerate(blocks_inv):
        ax.text((s + e) / 2, e - s + 0.5, f"{e - s}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Block size")
    ax.set_title(f"Inverse Hybrid CDF (λ={lam}) → sizes={sizes_inv}")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Side-by-side block size comparison ──
    ax = axes[4]
    n_blocks_orig = len(blocks_orig)
    n_blocks_inv = len(blocks_inv)
    n_max = max(n_blocks_orig, n_blocks_inv)
    bar_x = np.arange(n_max)
    bar_w = 0.35

    padded_orig = sizes_orig + [0] * (n_max - n_blocks_orig)
    padded_inv = sizes_inv + [0] * (n_max - n_blocks_inv)

    ax.bar(bar_x - bar_w / 2, padded_orig, bar_w, label="Original", color="C3", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(bar_x + bar_w / 2, padded_inv, bar_w, label="Inverse", color="C4", alpha=0.8, edgecolor="black", linewidth=0.5)

    for i, (vo, vi) in enumerate(zip(padded_orig, padded_inv)):
        if vo > 0:
            ax.text(i - bar_w / 2, vo + 0.3, str(vo), ha="center", va="bottom", fontsize=7, color="C3")
        if vi > 0:
            ax.text(i + bar_w / 2, vi + 0.3, str(vi), ha="center", va="bottom", fontsize=7, color="C4")

    ax.set_xlabel("Block index")
    ax.set_ylabel("Block size")
    ax.set_title("Block size comparison: Original vs Inverse")
    ax.set_xticks(bar_x)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] original vs inverse CDF plot → {out_path}")


def plot_inverse_lambda_comparison(
    gen_scores: torch.Tensor,
    gen_length: int,
    num_blocks: int,
    lam_list: list,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """
    여러 λ에 대해 inverse hybrid CDF를 겹쳐 그려 비교한다.
    """
    scores = gen_scores.detach().cpu().to(torch.float64).clamp(min=0)
    total_mass = scores.sum().item()
    x = np.arange(1, gen_length + 1)

    if total_mass < 1e-12:
        inv_attn_cdf = np.linspace(1 / gen_length, 1.0, gen_length)
    else:
        inv_scores = 1.0 / (scores + 1e-10)
        inv_attn_cdf = (torch.cumsum(inv_scores, dim=0) / inv_scores.sum()).numpy()

    uniform_cdf = x / gen_length

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: CDF curves
    ax = axes[0]
    ax.plot(x, inv_attn_cdf, label="Inverse Attention CDF (λ=1)", color="C2", linewidth=1.2, alpha=0.6)
    ax.plot(x, uniform_cdf, label="Uniform CDF (λ=0)", color="C0", linewidth=1.2, linestyle="--", alpha=0.6)
    for lam in lam_list:
        hybrid = lam * inv_attn_cdf + (1.0 - lam) * uniform_cdf
        ax.plot(x, hybrid, label=f"λ={lam}", linewidth=1.5)

    for k in range(1, num_blocks):
        ax.axhline(y=k / num_blocks, color="gray", linewidth=0.4, alpha=0.4)

    ax.set_ylabel("CDF value")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Inverse Hybrid CDF — λ comparison (N={num_blocks}) {title_suffix}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: block sizes for each λ
    ax = axes[1]
    bar_x = None
    bar_w = 0.8 / len(lam_list)
    for j, lam in enumerate(lam_list):
        blocks = hybrid_cdf_chunking(gen_scores, num_blocks=num_blocks, lam=lam, inverse=True)
        sizes = [e - s for s, e in blocks]
        n = len(blocks)
        if bar_x is None:
            bar_x = np.arange(n)
        offset = (j - len(lam_list) / 2 + 0.5) * bar_w
        ax.bar(bar_x[:n] + offset, sizes, bar_w, label=f"λ={lam}", alpha=0.8, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Block index")
    ax.set_ylabel("Block size")
    ax.set_title("Inverse Hybrid CDF — block sizes by λ")
    ax.set_xticks(bar_x if bar_x is not None else [])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] inverse lambda comparison plot → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="GSM8K: Original vs Inverse Hybrid CDF 비교 시각화",
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--sample-id", type=int, default=2)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--lam", type=float, default=0.7, help="hybrid CDF의 λ 값")
    p.add_argument(
        "--rollout-mode", type=str, default="sigmoid",
        choices=["sigmoid", "sigmoid_inverted", "baseline"],
    )
    p.add_argument("--out-dir", type=str, default="results_hybrid_inverse_cdf/cdf_plots")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    if args.sample_id < 0 or args.sample_id >= len(ds):
        raise ValueError(f"sample-id {args.sample_id} out of range (0..{len(ds)-1})")

    sample = ds[int(args.sample_id)]
    question = sample["question"]
    print(f"[Sample {args.sample_id}] Question:\n{question}\n")

    if args.no_chat_template:
        prompt_str = question
    else:
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True, tokenize=False,
        )

    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)

    attentions, prompt_len, gen_length = get_step0_attentions(
        model=model, input_ids=input_ids,
        gen_length=args.gen_length, mask_id=args.mask_id,
    )

    if args.rollout_mode == "sigmoid":
        rollout_scores = get_depth_adaptive_rollout(attentions).to(torch.float64)
    elif args.rollout_mode == "sigmoid_inverted":
        rollout_scores = get_depth_adaptive_rollout(attentions, invert_depth=True).to(torch.float64)
    else:
        rollout_scores = get_baseline_rollout(attentions).to(torch.float64)

    gen_scores = rollout_scores[prompt_len: prompt_len + gen_length]

    os.makedirs(args.out_dir, exist_ok=True)
    title_suffix = f"(id={args.sample_id}, prompt_len={prompt_len}, gen_len={gen_length})"

    # 1) 메인: Original vs Inverse 비교 (5-panel)
    out_path = os.path.join(
        args.out_dir,
        f"orig_vs_inverse_id{args.sample_id}_lam{args.lam}_N{args.num_blocks}.png",
    )
    plot_original_vs_inverse(
        gen_scores=gen_scores,
        gen_length=gen_length,
        num_blocks=args.num_blocks,
        lam=args.lam,
        out_path=out_path,
        title_suffix=title_suffix,
    )

    # 2) λ 비교: inverse CDF에서 여러 λ 비교
    out_path_comp = os.path.join(
        args.out_dir,
        f"inverse_lambda_comparison_id{args.sample_id}_N{args.num_blocks}.png",
    )
    plot_inverse_lambda_comparison(
        gen_scores=gen_scores,
        gen_length=gen_length,
        num_blocks=args.num_blocks,
        lam_list=[0.3, 0.5, 0.7, 0.9],
        out_path=out_path_comp,
        title_suffix=title_suffix,
    )


if __name__ == "__main__":
    main()
