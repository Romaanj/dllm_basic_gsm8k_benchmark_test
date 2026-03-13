"""
원본 rollout score vs inverse score 기반 CDF/블록 분할 비교 시각화.

레이아웃(2열 x 3행):
- 왼쪽: rollout score -> original CDF -> 블록 사이즈 히스토그램
- 오른쪽: inverse score -> inverse CDF -> 블록 사이즈 히스토그램
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_hybrid_cdf_eval import StreamingRollout, hybrid_cdf_chunking


def get_prompt_text(
    tokenizer: AutoTokenizer,
    sample: Dict[str, Any],
    task: str,
    no_chat_template: bool,
) -> str:
    if task == "humaneval":
        text = str(sample["prompt"])
    else:
        text = str(sample["question"])

    if no_chat_template:
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )


@torch.no_grad()
def extract_step0_gen_scores(
    model,
    input_ids: torch.Tensor,
    gen_length: int,
    mask_id: int,
    rollout_mode: str,
) -> torch.Tensor:
    prompt_len = input_ids.shape[1]
    device = model.device

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = input_ids.clone()

    invert_depth = rollout_mode == "sigmoid_inverted"
    hook_mode = "baseline" if rollout_mode == "baseline" else "sigmoid"

    core_model = model.model if hasattr(model, "model") else model
    blocks_list = core_model.transformer.blocks
    streaming = StreamingRollout(
        num_layers=len(blocks_list), mode=hook_mode, invert_depth=invert_depth
    )
    streaming.register(blocks_list)
    try:
        _ = model(x, output_attentions=True)
    finally:
        streaming.remove()

    scores = streaming.get_scores()
    if scores is None:
        raise RuntimeError("rollout score extraction failed (scores is None)")
    return scores.to(torch.float64)[prompt_len : prompt_len + gen_length].clone()


def make_cdf(scores: torch.Tensor) -> np.ndarray:
    s = scores.detach().cpu().to(torch.float64).clamp(min=0)
    total = float(s.sum().item())
    if total < 1e-12:
        n = int(s.numel())
        return (np.arange(1, n + 1) / n).astype(np.float64)
    return (torch.cumsum(s, dim=0) / total).cpu().numpy()


def block_sizes(blocks: List[Tuple[int, int]]) -> List[int]:
    return [int(e - s) for s, e in blocks]


def draw_score_panel(ax, values: np.ndarray, blocks: List[Tuple[int, int]], title: str, color: str) -> None:
    x = np.arange(values.shape[0])
    ax.fill_between(x, 0, values, color=color, alpha=0.25)
    ax.plot(x, values, color=color, linewidth=1.0)
    for _, end in blocks[:-1]:
        ax.axvline(end - 0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25, axis="y")


def draw_cdf_panel(ax, cdf: np.ndarray, blocks: List[Tuple[int, int]], num_blocks: int, title: str, color: str) -> None:
    x = np.arange(cdf.shape[0])
    ax.plot(x, cdf, color=color, linewidth=1.8)
    for k in range(1, num_blocks):
        ax.axhline(k / num_blocks, color="gray", linewidth=0.5, alpha=0.35)
    for _, end in blocks[:-1]:
        ax.axvline(end - 0.5, color="black", linestyle=":", linewidth=0.7, alpha=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("CDF")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)


def draw_hist_panel(ax, blocks: List[Tuple[int, int]], title: str, cmap_name: str) -> None:
    centers = [(s + e - 1) / 2 for s, e in blocks]
    widths = [e - s for s, e in blocks]
    heights = widths
    colors = plt.get_cmap(cmap_name)(np.linspace(0.35, 0.85, len(blocks)))

    ax.bar(
        centers,
        heights,
        width=widths,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    for c, h in zip(centers, heights):
        ax.text(c, h + 0.6, str(int(h)), ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Block size")
    ax.set_xlabel("Generation token index")
    ax.grid(True, alpha=0.25, axis="y")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize rollout vs inverse score CDF-based block partition"
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "humaneval"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-id", type=int, default=0)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument(
        "--rollout-mode", type=str, default="sigmoid",
        choices=["sigmoid", "sigmoid_inverted", "baseline"],
    )
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--eps", type=float, default=1e-6, help="inverse score 안정화용 epsilon")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--out-dir", type=str, default="results_rollout_inverse_compare")
    p.add_argument("--out-name", type=str, default="")
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
            args.model, trust_remote_code=True, torch_dtype=torch_dtype
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.task == "humaneval":
        print("Loading HumanEval test split...")
        ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=args.seed)
    else:
        print("Loading GSM8K test split...")
        ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=args.seed)

    if args.sample_id < 0 or args.sample_id >= len(ds):
        raise ValueError(f"sample_id={args.sample_id} out of range [0, {len(ds)-1}]")

    sample = ds[int(args.sample_id)]
    prompt_str = get_prompt_text(
        tokenizer=tokenizer,
        sample=sample,
        task=args.task,
        no_chat_template=args.no_chat_template,
    )
    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)

    gen_scores = extract_step0_gen_scores(
        model=model,
        input_ids=input_ids,
        gen_length=args.gen_length,
        mask_id=args.mask_id,
        rollout_mode=args.rollout_mode,
    )
    gen_scores_np = gen_scores.detach().cpu().numpy()

    inv_scores = 1.0 / (gen_scores.clamp(min=0).to(torch.float64) + float(args.eps))
    inv_scores_np = inv_scores.detach().cpu().numpy()
    inv_scores_plot = inv_scores_np / max(float(inv_scores_np.max()), 1e-12)

    cdf_org = make_cdf(gen_scores)
    cdf_inv = make_cdf(inv_scores)

    blocks_org = hybrid_cdf_chunking(
        gen_scores=gen_scores, num_blocks=args.num_blocks, lam=1.0, inverse=False
    )
    blocks_inv = hybrid_cdf_chunking(
        gen_scores=gen_scores, num_blocks=args.num_blocks, lam=1.0, inverse=True
    )
    sizes_org = block_sizes(blocks_org)
    sizes_inv = block_sizes(blocks_inv)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex="col")

    draw_score_panel(
        axes[0, 0],
        gen_scores_np,
        blocks_org,
        title="Original deep-layer rollout score",
        color="#d95f02",
    )
    draw_cdf_panel(
        axes[1, 0],
        cdf_org,
        blocks_org,
        num_blocks=args.num_blocks,
        title=f"Original CDF (N={args.num_blocks})",
        color="#e7298a",
    )
    draw_hist_panel(
        axes[2, 0],
        blocks_org,
        title=f"Original CDF block-size histogram\nsizes={sizes_org}",
        cmap_name="RdPu",
    )

    draw_score_panel(
        axes[0, 1],
        inv_scores_plot,
        blocks_inv,
        title=f"Inverse score (1/(score+{args.eps:g})) [normalized]",
        color="#1b9e77",
    )
    draw_cdf_panel(
        axes[1, 1],
        cdf_inv,
        blocks_inv,
        num_blocks=args.num_blocks,
        title=f"Inverse CDF (N={args.num_blocks})",
        color="#66a61e",
    )
    draw_hist_panel(
        axes[2, 1],
        blocks_inv,
        title=f"Inverse CDF block-size histogram\nsizes={sizes_inv}",
        cmap_name="YlGn",
    )

    for col in range(2):
        axes[2, col].set_xlim(-0.5, args.gen_length - 0.5)

    task_id = str(sample.get("task_id", ""))
    fig.suptitle(
        f"Rollout vs Inverse CDF block partition (task={args.task}, sample={args.sample_id}, gen_len={args.gen_length})\n"
        f"prompt_len={int(input_ids.shape[1])}, rollout_mode={args.rollout_mode}"
        + (f", task_id={task_id}" if task_id else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = args.out_name or f"rollout_vs_inverse_task-{args.task}_sid-{args.sample_id}_L{args.gen_length}_N{args.num_blocks}.png"
    out_path = os.path.join(args.out_dir, out_name)
    plt.savefig(out_path, dpi=160)
    plt.close()

    meta = {
        "task": args.task,
        "sample_id": int(args.sample_id),
        "gen_length": int(args.gen_length),
        "num_blocks": int(args.num_blocks),
        "rollout_mode": args.rollout_mode,
        "blocks_original": blocks_org,
        "blocks_inverse": blocks_inv,
        "sizes_original": sizes_org,
        "sizes_inverse": sizes_inv,
    }
    meta_path = out_path.replace(".png", ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print(f"[Saved] figure -> {out_path}")
    print(f"[Saved] meta   -> {meta_path}")


if __name__ == "__main__":
    main()

