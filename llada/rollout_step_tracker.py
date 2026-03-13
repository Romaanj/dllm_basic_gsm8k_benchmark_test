"""
Rollout Score Step Tracker
==========================
블록 기반 디코딩 (앞→뒤, 블록당 1-token-per-step) 중 매 스텝마다
sigmoid depth-adaptive attention rollout score를 추출하여,
스텝 진행에 따른 score 변화를 heatmap으로 시각화합니다.

x축: 토큰 인덱스 (생성 구간)
y축: 디코딩 스텝
색상: rollout score
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM


# ═══════════════════════════════════════════════════════════════════════════
# Sigmoid Depth-Adaptive Rollout (gsm8k_equal_mass_eval.py에서 가져옴)
# ═══════════════════════════════════════════════════════════════════════════

def get_depth_adaptive_rollout(
    attentions: Tuple[torch.Tensor, ...],
    invert_depth: bool = False,
) -> torch.Tensor:
    """Sigmoid Depth-Adaptive Rollout → (T,) score vector."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Block-based 1-token-per-step 디코딩 + 매 스텝 rollout 추적
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_with_rollout_tracking(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_length: int = 32,
    mask_id: int = 126336,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    블록 기반 디코딩 (앞→뒤, 블록당 스텝당 1토큰 unmask) 하면서
    매 스텝 sigmoid depth-adaptive rollout score를 추적.

    Returns:
        dict with:
          - x: 최종 시퀀스
          - rollout_matrix: (num_steps, gen_length) numpy array
          - unmask_order: [(step, token_pos)] 리스트
          - nfe: 총 forward pass 횟수
    """
    device = model.device
    prompt_len = prompt.shape[1]

    assert gen_length % block_length == 0, (
        f"gen_length({gen_length})는 block_length({block_length})의 배수여야 합니다."
    )
    num_blocks = gen_length // block_length

    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    all_rollouts = []
    unmask_order = []
    nfe = 0
    step = 0

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            mask_idx = (x == mask_id).clone()
            mask_idx[:, block_end:] = False

            outputs = model(x, output_attentions=True)
            nfe += 1
            logits = outputs.logits
            attentions = outputs.attentions

            rollout = get_depth_adaptive_rollout(attentions)
            gen_rollout = rollout[prompt_len: prompt_len + gen_length]
            all_rollouts.append(gen_rollout.cpu().float().numpy())

            del attentions
            if device.type == "cuda":
                torch.cuda.empty_cache()

            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0 = torch.argmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_idx, x0, x)
            neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=device, dtype=x0_p.dtype)
            confidence = torch.where(mask_idx, x0_p, neg_inf)

            best_pos = torch.argmax(confidence, dim=1).item()
            x[0, best_pos] = x0[0, best_pos]

            gen_pos = best_pos - prompt_len
            unmask_order.append((step, gen_pos))
            step += 1

            if step >= gen_length:
                break

        if step >= gen_length:
            break

    rollout_matrix = np.stack(all_rollouts, axis=0) if all_rollouts else np.zeros((0, gen_length))

    return {
        "x": x,
        "rollout_matrix": rollout_matrix,
        "unmask_order": unmask_order,
        "nfe": nfe,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Heatmap 시각화
# ═══════════════════════════════════════════════════════════════════════════

def plot_rollout_heatmap(
    rollout_matrix: np.ndarray,
    unmask_order: List[Tuple[int, int]],
    sample_id: int,
    out_path: str,
    gen_length: int = 256,
    block_length: int = 32,
    question: str = "",
    gen_text_preview: str = "",
) -> None:
    num_steps = rollout_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(
        rollout_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        origin="upper",
    )
    plt.colorbar(im, ax=ax, label="Rollout Score")

    if unmask_order:
        steps_arr = [s for s, p in unmask_order]
        pos_arr = [p for s, p in unmask_order]
        ax.scatter(pos_arr, steps_arr, c="red", s=4, alpha=0.6, zorder=5, label="unmasked token")
        ax.legend(loc="upper right", fontsize=8)

    for b in range(1, gen_length // block_length):
        ax.axvline(x=b * block_length - 0.5, color="white", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_xlabel("Token Index (generation region)", fontsize=11)
    ax.set_ylabel("Decoding Step", fontsize=11)
    title = f"Sample {sample_id}: Rollout Score Evolution (block-based, {num_steps} steps)"
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Rollout Score Step Tracker")
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32, help="블록당 토큰 수 (fixed block)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--no-chat-template", action="store_true")

    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument(
        "--sample-ids", type=str, default=None,
        help="Comma-separated GSM8K sample indices (0-based, before shuffle)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="results_equal_mass/rollout_tracking")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        args.device = "cuda:3" if torch.cuda.is_available() else "cpu"

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  Rollout Score Step Tracker")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Device:      {args.device} ({args.dtype})")
    print(f"  Gen length:  {args.gen_length}")
    print(f"  Block length: {args.block_length}")
    print(f"  Temperature: {args.temperature}")
    print("=" * 60)

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
    ds = load_dataset("openai/gsm8k", "main", split="test")

    if args.sample_ids is not None:
        sample_indices = [int(x.strip()) for x in args.sample_ids.split(",")]
    else:
        ds_shuffled = ds.shuffle(seed=args.seed)
        sample_indices = list(range(min(args.num_samples, len(ds))))
        ds = ds_shuffled

    print(f"  Tracking {len(sample_indices)} samples\n")

    all_results = []

    for i, idx in enumerate(sample_indices):
        sample = ds[idx]
        question = sample["question"]
        gold_answer = sample["answer"].split("####")[-1].strip() if "####" in sample["answer"] else sample["answer"]

        print(f"[{i+1}/{len(sample_indices)}] Sample {idx}: {question[:80]}...")

        if not args.no_chat_template:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True, tokenize=False,
            )
        else:
            prompt_str = question

        input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(args.device)
        prompt_len = input_ids.shape[1]

        t0 = time.perf_counter()
        result = generate_with_rollout_tracking(
            model=model,
            prompt=input_ids,
            gen_length=args.gen_length,
            block_length=args.block_length,
            mask_id=args.mask_id,
            temperature=args.temperature,
        )
        elapsed = time.perf_counter() - t0

        gen_ids = result["x"][0, prompt_len:].cpu().tolist()
        gen_ids_clean = [t for t in gen_ids if t != args.mask_id]
        gen_text = tokenizer.decode(gen_ids_clean, skip_special_tokens=True)

        print(f"  NFE: {result['nfe']} | Steps: {result['rollout_matrix'].shape[0]} | Time: {elapsed:.1f}s")
        print(f"  Gold: {gold_answer}")
        print(f"  Gen:  {gen_text[:120]}...")

        heatmap_path = os.path.join(args.out_dir, f"rollout_heatmap_sample_{idx}.png")
        plot_rollout_heatmap(
            rollout_matrix=result["rollout_matrix"],
            unmask_order=result["unmask_order"],
            sample_id=idx,
            out_path=heatmap_path,
            gen_length=args.gen_length,
            block_length=args.block_length,
            question=question,
            gen_text_preview=gen_text[:200],
        )

        all_results.append({
            "sample_id": idx,
            "question": question,
            "gold": gold_answer,
            "gen_text_preview": gen_text[:300],
            "nfe": result["nfe"],
            "num_steps": result["rollout_matrix"].shape[0],
            "elapsed_sec": round(elapsed, 2),
            "unmask_order": result["unmask_order"],
        })

    json_path = os.path.join(args.out_dir, "rollout_step_tracker_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved: {json_path}")
    print("Done!")


if __name__ == "__main__":
    main()
