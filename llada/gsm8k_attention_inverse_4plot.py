"""
GSM8K 샘플에서 attention score / CDF와 inverse score / inverse CDF만 시각화하는 스크립트.

Usage:
  python gsm8k_attention_inverse_4plot.py --sample-id 0 --gen-length 256
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


def plot_attention_inverse_4panels(
    gen_scores: torch.Tensor,
    gen_length: int,
    out_path: str,
    title_suffix: str = "",
) -> None:
    scores = gen_scores.detach().cpu().to(torch.float64).clamp(min=0)
    x = np.arange(1, gen_length + 1)

    total_mass = scores.sum().item()
    if total_mass < 1e-12:
        attn_cdf = np.linspace(1 / gen_length, 1.0, gen_length)
        attn_scores_np = np.zeros(gen_length, dtype=np.float64)
    else:
        attn_scores_np = scores.numpy()
        attn_cdf = (torch.cumsum(scores, dim=0) / total_mass).numpy()

    inv_scores = 1.0 / (scores + 1e-10)
    inv_sum = inv_scores.sum().item()
    if inv_sum < 1e-12:
        inv_scores_np = np.zeros(gen_length, dtype=np.float64)
        inv_cdf = np.linspace(1 / gen_length, 1.0, gen_length)
    else:
        inv_scores_np = inv_scores.numpy()
        inv_cdf = (torch.cumsum(inv_scores, dim=0) / inv_scores.sum()).numpy()

    uniform_cdf = x / gen_length

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)

    # Panel 1: Attention score
    ax = axes[0, 0]
    ax.fill_between(x, 0, attn_scores_np, alpha=0.30, color="C1", label="Attention score")
    ax.plot(x, attn_scores_np, color="C1", linewidth=0.9)
    ax.set_title("Attention score")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Panel 2: Attention CDF
    ax = axes[0, 1]
    ax.plot(x, attn_cdf, color="C1", linewidth=1.8, label="Attention CDF")
    ax.plot(x, uniform_cdf, color="C0", linewidth=1.1, linestyle="--", alpha=0.6, label="Uniform CDF")
    ax.set_title("Attention CDF")
    ax.set_ylabel("CDF value")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # Panel 3: Inverse attention score
    ax = axes[1, 0]
    ax.fill_between(x, 0, inv_scores_np, alpha=0.22, color="C2", label="Inverse attention score")
    ax.plot(x, inv_scores_np, color="C2", linewidth=0.9)
    ax.set_title("Inverse attention score: 1 / (score + eps)")
    ax.set_xlabel("Generation token index")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Panel 4: Inverse attention CDF
    ax = axes[1, 1]
    ax.plot(x, inv_cdf, color="C2", linewidth=1.8, label="Inverse attention CDF")
    ax.plot(x, uniform_cdf, color="C0", linewidth=1.1, linestyle="--", alpha=0.6, label="Uniform CDF")
    ax.set_title("Inverse attention CDF")
    ax.set_xlabel("Generation token index")
    ax.set_ylabel("CDF value")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"Attention / Inverse Attention 4-Panel Plot {title_suffix}", y=1.02, fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] 4-panel attention/inverse plot -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="GSM8K: attention score/CDF + inverse score/CDF 4-panel plot",
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--sample-id", type=int, default=100)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--seed", type=int, default=42)
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
    out_path = os.path.join(
        args.out_dir,
        f"attn_inverse_4panel_id{args.sample_id}_rollout-{args.rollout_mode}.png",
    )
    title_suffix = f"(id={args.sample_id}, prompt_len={prompt_len}, gen_len={gen_length})"
    plot_attention_inverse_4panels(
        gen_scores=gen_scores,
        gen_length=gen_length,
        out_path=out_path,
        title_suffix=title_suffix,
    )


if __name__ == "__main__":
    main()
