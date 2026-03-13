"""
GSM8K 샘플에 대해 attention_cdf, uniform_cdf, hybrid_cdf를 시각화하는 스크립트.

Usage:
  python gsm8k_hybrid_cdf_plot.py --sample-id 0 --lam 0.5 --num-blocks 8
"""

import argparse
import json
import os
from typing import Tuple

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
def get_step0_attentions(
    model,
    input_ids: torch.Tensor,
    gen_length: int,
    mask_id: int,
) -> Tuple[Tuple[torch.Tensor, ...], int, int]:
    device = model.device
    prompt_len = input_ids.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = input_ids.clone()

    outputs = model(x, output_attentions=True)
    return outputs.attentions, prompt_len, gen_length


def plot_three_cdfs(
    gen_scores: torch.Tensor,
    gen_length: int,
    num_blocks: int,
    lam: float,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """
    attention_cdf, uniform_cdf, hybrid_cdf 세 개를 한 그래프에 그린다.
    블록 경계도 세로선으로 표시.
    """
    scores = gen_scores.detach().cpu().to(torch.float64).clamp(min=0)
    total_mass = scores.sum().item()

    x = np.arange(1, gen_length + 1)

    if total_mass < 1e-12:
        attn_cdf = np.linspace(1 / gen_length, 1.0, gen_length)
    else:
        attn_cdf = (torch.cumsum(scores, dim=0) / total_mass).numpy()

    uniform_cdf = x / gen_length
    hybrid_cdf = lam * attn_cdf + (1.0 - lam) * uniform_cdf

    # 블록 경계 계산: hybrid CDF 기반
    blocks_hybrid = hybrid_cdf_chunking(gen_scores, num_blocks=num_blocks, lam=lam)
    block_sizes_hybrid = [e - s for s, e in blocks_hybrid]

    # 블록 경계 계산: pure attention CDF 기반 (λ=1.0)
    blocks_attn = hybrid_cdf_chunking(gen_scores, num_blocks=num_blocks, lam=1.0)
    block_sizes_attn = [e - s for s, e in blocks_attn]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    # --- subplot 0: 세 CDF 겹쳐 그리기 ---
    axes[0].plot(x, attn_cdf, label="Attention CDF", color="C1", linewidth=1.5)
    axes[0].plot(x, uniform_cdf, label="Uniform CDF", color="C0", linewidth=1.5, linestyle="--")
    axes[0].plot(x, hybrid_cdf, label=f"Hybrid CDF (λ={lam})", color="C3", linewidth=2)

    # N등분 수평선
    for k in range(1, num_blocks):
        axes[0].axhline(y=k / num_blocks, color="gray", linewidth=0.5, alpha=0.5)

    # 블록 경계 세로선 (hybrid=빨간, attn=주황 점선)
    for start, end in blocks_hybrid[:-1]:
        axes[0].axvline(x=end, color="C3", linewidth=0.8, linestyle=":", alpha=0.7)
    for start, end in blocks_attn[:-1]:
        axes[0].axvline(x=end, color="C1", linewidth=0.6, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("CDF value")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"Hybrid CDF visualization {title_suffix}\n"
        f"hybrid blocks={len(blocks_hybrid)}, sizes={block_sizes_hybrid}"
    )

    # --- subplot 1: attention score (raw) ---
    axes[1].fill_between(
        x, 0, scores.numpy(), alpha=0.4, color="C1", label="Attention rollout score"
    )
    axes[1].plot(x, scores.numpy(), color="C1", linewidth=0.8)
    for start, end in blocks_hybrid[:-1]:
        axes[1].axvline(x=end, color="C3", linewidth=0.8, linestyle=":", alpha=0.7)
    for start, end in blocks_attn[:-1]:
        axes[1].axvline(x=end, color="C1", linewidth=0.6, linestyle="--", alpha=0.4)
    axes[1].set_ylabel("Rollout score")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # --- subplot 2: attention CDF 기반 블록 분할 bar chart ---
    block_centers_attn = [(s + e) / 2 for s, e in blocks_attn]
    block_widths_attn = [e - s for s, e in blocks_attn]
    colors_attn = plt.cm.Oranges(np.linspace(0.3, 0.8, len(blocks_attn)))
    axes[2].bar(
        block_centers_attn, block_widths_attn, width=block_widths_attn,
        color=colors_attn, edgecolor="black", linewidth=0.5, alpha=0.8,
    )
    for i, (s, e) in enumerate(blocks_attn):
        axes[2].text(
            (s + e) / 2, e - s + 0.5,
            f"{e - s}",
            ha="center", va="bottom", fontsize=8,
        )
    axes[2].set_ylabel("Block size")
    axes[2].set_title(f"Attention CDF only (λ=1.0) → sizes={block_sizes_attn}")
    axes[2].grid(True, alpha=0.3, axis="y")

    # --- subplot 3: hybrid CDF 기반 블록 분할 bar chart ---
    block_centers_hybrid = [(s + e) / 2 for s, e in blocks_hybrid]
    block_widths_hybrid = [e - s for s, e in blocks_hybrid]
    colors_hybrid = plt.cm.RdPu(np.linspace(0.3, 0.8, len(blocks_hybrid)))
    axes[3].bar(
        block_centers_hybrid, block_widths_hybrid, width=block_widths_hybrid,
        color=colors_hybrid, edgecolor="black", linewidth=0.5, alpha=0.8,
    )
    for i, (s, e) in enumerate(blocks_hybrid):
        axes[3].text(
            (s + e) / 2, e - s + 0.5,
            f"{e - s}",
            ha="center", va="bottom", fontsize=8,
        )
    axes[3].set_ylabel("Block size")
    axes[3].set_xlabel("Generation token index")
    axes[3].set_title(f"Hybrid CDF (λ={lam}) → sizes={block_sizes_hybrid}")
    axes[3].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] hybrid CDF plot → {out_path}")


def plot_lambda_comparison(
    gen_scores: torch.Tensor,
    gen_length: int,
    num_blocks: int,
    lam_list: list,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """
    여러 λ에 대해 hybrid_cdf를 겹쳐 그려서 비교한다.
    """
    scores = gen_scores.detach().cpu().to(torch.float64).clamp(min=0)
    total_mass = scores.sum().item()

    x = np.arange(1, gen_length + 1)

    if total_mass < 1e-12:
        attn_cdf = np.linspace(1 / gen_length, 1.0, gen_length)
    else:
        attn_cdf = (torch.cumsum(scores, dim=0) / total_mass).numpy()

    uniform_cdf = x / gen_length

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, attn_cdf, label="Attention CDF (λ=1)", color="C1", linewidth=1.2, alpha=0.6)
    ax.plot(x, uniform_cdf, label="Uniform CDF (λ=0)", color="C0", linewidth=1.2, linestyle="--", alpha=0.6)

    for lam in lam_list:
        hybrid = lam * attn_cdf + (1.0 - lam) * uniform_cdf
        ax.plot(x, hybrid, label=f"λ={lam}", linewidth=1.5)

    for k in range(1, num_blocks):
        ax.axhline(y=k / num_blocks, color="gray", linewidth=0.4, alpha=0.4)

    ax.set_xlabel("Generation token index")
    ax.set_ylabel("CDF value")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Hybrid CDF comparison (N={num_blocks}) {title_suffix}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] lambda comparison plot → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GSM8K: visualize attention_cdf, uniform_cdf, hybrid_cdf",
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--sample-id", type=int, default=2)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-blocks", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.5, help="hybrid CDF의 λ 값")
    p.add_argument(
        "--rollout-mode", type=str, default="sigmoid",
        choices=["sigmoid", "sigmoid_inverted", "baseline"],
    )
    p.add_argument(
        "--out-dir", type=str, default="results_hybrid_cdf/cdf_plots",
    )
    return p.parse_args()


def main() -> None:
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

    # 1) 메인 시각화: attention_cdf + uniform_cdf + hybrid_cdf + 블록 경계
    out_path = os.path.join(
        args.out_dir,
        f"hybrid_cdf_gsm8k_id{args.sample_id}_lam{args.lam}_N{args.num_blocks}.png",
    )
    plot_three_cdfs(
        gen_scores=gen_scores,
        gen_length=gen_length,
        num_blocks=args.num_blocks,
        lam=args.lam,
        out_path=out_path,
        title_suffix=title_suffix,
    )

    # 2) λ 비교: 여러 λ를 한 그래프에
    out_path_comp = os.path.join(
        args.out_dir,
        f"hybrid_cdf_lambda_comparison_id{args.sample_id}_N{args.num_blocks}.png",
    )
    plot_lambda_comparison(
        gen_scores=gen_scores,
        gen_length=gen_length,
        num_blocks=args.num_blocks,
        lam_list=[0.3, 0.5, 0.7, 0.9],
        out_path=out_path_comp,
        title_suffix=title_suffix,
    )


if __name__ == "__main__":
    main()
