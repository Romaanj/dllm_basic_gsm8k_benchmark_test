import argparse
import json
import os
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_equal_mass_eval import (
    get_baseline_rollout,
    get_depth_adaptive_rollout,
    generate_equal_mass,
)


@torch.no_grad()
def get_step0_attentions(
    model,
    input_ids: torch.Tensor,
    gen_length: int,
    mask_id: int,
) -> Tuple[Tuple[torch.Tensor, ...], int, int]:
    """
    Equal-mass generation의 Step 0와 동일한 형태로 입력을 만들어
    attention tuple을 추출한다.

    Returns:
        attentions: Tuple[Tensor, ...]
        prompt_len: int
        gen_length: int
    """
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


def compute_gen_rollout_scores(
    attentions: Tuple[torch.Tensor, ...],
    prompt_len: int,
    gen_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    attention으로부터 rollout score를 계산하고 generation 구간만 반환.
    Returns:
        x: (T,) generation 토큰 상대 인덱스
        gen_scores_baseline, gen_scores_sigmoid, gen_scores_sigmoid_inv: (T,)
    """
    with torch.no_grad():
        rollout_baseline = get_baseline_rollout(attentions).to(torch.float64)
        rollout_sigmoid = get_depth_adaptive_rollout(
            attentions, invert_depth=False
        ).to(torch.float64)
        rollout_sigmoid_inv = get_depth_adaptive_rollout(
            attentions, invert_depth=True
        ).to(torch.float64)

        gen_slice = slice(prompt_len, prompt_len + gen_length)
        gen_scores_baseline = rollout_baseline[gen_slice].clone()
        gen_scores_sigmoid = rollout_sigmoid[gen_slice].clone()
        gen_scores_sigmoid_inv = rollout_sigmoid_inv[gen_slice].clone()

        x = torch.arange(gen_length, device=gen_scores_baseline.device)

    return x, gen_scores_baseline, gen_scores_sigmoid, gen_scores_sigmoid_inv


def compute_rollout_comparison_stats(
    gen_scores_baseline: torch.Tensor,
    gen_scores_sigmoid: torch.Tensor,
    gen_scores_sigmoid_inv: torch.Tensor,
) -> dict:
    """세 rollout 스코어 벡터 간 상관계수·차이 통계."""
    b = gen_scores_baseline.cpu().numpy()
    s = gen_scores_sigmoid.cpu().numpy()
    si = gen_scores_sigmoid_inv.cpu().numpy()

    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0

    def spearman(x, y):
        return float(
            np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
        )

    return {
        "baseline_vs_sigmoid": {
            "pearson": pearson(b, s),
            "spearman": spearman(b, s),
            "mae": float(np.abs(b - s).mean()),
            "rel_mae": float(np.abs(b - s).mean() / (np.abs(b).mean() + 1e-12)),
        },
        "baseline_vs_inv_sigmoid": {
            "pearson": pearson(b, si),
            "spearman": spearman(b, si),
            "mae": float(np.abs(b - si).mean()),
            "rel_mae": float(np.abs(b - si).mean() / (np.abs(b).mean() + 1e-12)),
        },
        "sigmoid_vs_inv_sigmoid": {
            "pearson": pearson(s, si),
            "spearman": spearman(s, si),
            "mae": float(np.abs(s - si).mean()),
            "rel_mae": float(np.abs(s - si).mean() / (np.abs(s).mean() + 1e-12)),
        },
    }


def plot_rollout_scores(
    x: torch.Tensor,
    gen_scores_baseline: torch.Tensor,
    gen_scores_sigmoid: torch.Tensor,
    gen_scores_sigmoid_inv: torch.Tensor,
    out_path: str,
    title_suffix: str = "",
    peak_indices: Optional[Sequence[int]] = None,
    peak_tokens: Optional[Sequence[str]] = None,
) -> None:
    """
    generation 구간에 대해 토큰 인덱스 vs score 그래프를 저장한다.
    필요하면 peak 위치에 토큰 문자열을 같이 annotate 한다.
    """
    x_np = x.cpu().numpy()

    # 세 그래프를 분리해서 그리되, x축은 공유
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(
        x_np,
        gen_scores_baseline.cpu().numpy(),
        color="C0",
    )
    axes[0].set_ylabel("Baseline")
    axes[0].grid(True)
    axes[0].set_title(f"GSM8K rollout scores on generation region {title_suffix}")

    axes[1].plot(
        x_np,
        gen_scores_sigmoid.cpu().numpy(),
        color="C1",
    )
    axes[1].set_ylabel("Sigmoid\n(deep)")
    axes[1].grid(True)

    axes[2].plot(
        x_np,
        gen_scores_sigmoid_inv.cpu().numpy(),
        color="C2",
    )
    axes[2].set_ylabel("Inv-sigmoid\n(shallow)")
    axes[2].set_xlabel("Generation token index")
    axes[2].grid(True)

    # 피크 위치에 토큰 annotate (주로 sigmoid subplot에 표시)
    if peak_indices is not None and peak_tokens is not None:
        for idx, tok in zip(peak_indices, peak_tokens):
            if idx < 0 or idx >= len(x_np):
                continue
            y_val = gen_scores_sigmoid[idx].item()
            axes[1].scatter(
                [x_np[idx]],
                [y_val],
                color="red",
                s=20,
                zorder=3,
            )
            axes[1].annotate(
                tok,
                (x_np[idx], y_val),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
                rotation=45,
            )

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] rollout plot → {out_path}")


def plot_sigmoid_and_cdf(
    x: torch.Tensor,
    gen_scores_sigmoid: torch.Tensor,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """
    Sigmoid rollout score와, 해당 score 기반 CDF를 두 개의 서브플롯으로 시각화.
    """
    x_np = x.cpu().numpy()
    y_sig = gen_scores_sigmoid.cpu().numpy()

    # Equal-mass chunking에서 사용하는 것과 비슷하게, 음수는 잘라내고 CDF 계산
    y_clamped = np.clip(y_sig, a_min=0.0, a_max=None)
    cdf = np.cumsum(y_clamped)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(x_np, y_sig, color="C1")
    axes[0].set_ylabel("Sigmoid score")
    axes[0].grid(True)
    axes[0].set_title(
        f"Sigmoid rollout score and CDF on generation region {title_suffix}"
    )

    axes[1].plot(x_np, cdf, color="C3")
    axes[1].set_ylabel("CDF")
    axes[1].set_xlabel("Generation token index")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True)

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] sigmoid score + CDF plot → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GSM8K: plot step-0 attention rollout scores on generation region for a given sample id.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
    )
    p.add_argument(
        "--sample-id",
        type=int,
        default=4,
        help="GSM8K test split에서 사용할 샘플 인덱스 (0-based).",
    )
    p.add_argument(
        "--gen-length",
        type=int,
        default=256,
        help="Equal-mass eval에서 사용하는 generation 길이와 맞추면 비교가 쉽다.",
    )
    p.add_argument(
        "--mask-id",
        type=int,
        default=126336,
        help="gsm8k_equal_mass_eval.py에서 사용하는 mask_id와 동일하게 둘 것.",
    )
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="지정 시 chat template을 사용하지 않고 question 문자열만 사용.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="GSM8K 셔플 seed (eval 스크립트와 동일하게 맞추고 싶을 때 사용).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results_equal_mass/rollout_tracking",
        help="그래프를 저장할 디렉토리.",
    )
    p.add_argument(
        "--num-blocks",
        type=int,
        default=8,
        help="equal_mass generation에서 사용할 블록 개수.",
    )
    p.add_argument(
        "--steps-per-block",
        type=int,
        default=32,
        help="equal_mass generation에서 블록당 스텝 수.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="confidence threshold (equal_mass_eval과 동일).",
    )
    p.add_argument(
        "--min-block-size",
        type=int,
        default=4,
        help="equal_mass chunking 최소 블록 크기.",
    )
    p.add_argument(
        "--max-block-size",
        type=int,
        default=48,
        help="equal_mass chunking 최대 블록 크기.",
    )
    p.add_argument(
        "--rollout-mode",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "sigmoid_inverted", "baseline"],
        help="generation에서 사용할 rollout 모드.",
    )
    p.add_argument(
        "--top-k-peaks",
        type=int,
        default=5,
        help="어떤 토큰이 왔는지 보고 싶은 상위 peak 개수 (sigmoid 기준).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        args.device = "cuda:3" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}")
    model = (
        LLaDAModelLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print("Loading GSM8K test split...")
    ds = (
        load_dataset("openai/gsm8k", "main", split="test")
        .shuffle(seed=args.seed)
    )
    if args.sample_id < 0 or args.sample_id >= len(ds):
        raise ValueError(
            f"sample-id {args.sample_id} is out of range (0 <= id < {len(ds)})."
        )
    sample = ds[int(args.sample_id)]
    question = sample["question"]
    print(f"[Sample {args.sample_id}] Question:\n{question}\n")

    if args.no_chat_template:
        prompt_str = question
    else:
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=False,
        )

    input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(
        model.device
    )

    # Step 0 attentions (rollout score 시각화용)
    attentions, prompt_len, gen_length = get_step0_attentions(
        model=model,
        input_ids=input_ids,
        gen_length=args.gen_length,
        mask_id=args.mask_id,
    )

    # Step 0 rollout score (generation 구간)
    (
        x,
        gen_scores_baseline,
        gen_scores_sigmoid,
        gen_scores_sigmoid_inv,
    ) = compute_gen_rollout_scores(
        attentions=attentions,
        prompt_len=prompt_len,
        gen_length=gen_length,
    )

    # rollout 비교 통계 계산 및 저장·출력
    stats = compute_rollout_comparison_stats(
        gen_scores_baseline, gen_scores_sigmoid, gen_scores_sigmoid_inv
    )
    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = os.path.join(
        args.out_dir, f"rollout_comparison_stats_id{args.sample_id}.json"
    )
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[Saved] rollout comparison stats → {stats_path}")

    print("\n--- Rollout Comparison ---")
    for pair, v in stats.items():
        print(
            f"  {pair}: Pearson={v['pearson']:.4f}, Spearman={v['spearman']:.4f}, MAE={v['mae']:.6f}"
        )
    print()

    # Block diffusion equal_mass 방식으로 실제 generation 수행
    print("Running equal-mass generation (block diffusion)...")
    out_ids, nfe, info = generate_equal_mass(
        model=model,
        tokenizer=tokenizer,
        prompt=input_ids,
        gen_length=args.gen_length,
        mask_id=args.mask_id,
        num_blocks=args.num_blocks,
        steps_per_block=args.steps_per_block,
        temperature=args.temperature,
        threshold=args.threshold,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        rollout_mode=args.rollout_mode,
        verbose=False,
    )

    gen_token_ids = out_ids[0, prompt_len : prompt_len + gen_length]
    gen_tokens = tokenizer.convert_ids_to_tokens(gen_token_ids.tolist())

    # "유독 강하게 피크"인 위치를 sigmoid rollout 기준 top-k로 선택
    k = min(args.top_k_peaks, gen_length)
    topk_vals, topk_idx = torch.topk(gen_scores_sigmoid, k=k)
    peak_indices = sorted(int(i) for i in topk_idx.cpu().tolist())

    # 사람이 보기 좋은 토큰 표시용 문자열 (decode 기반)
    peak_tokens = []

    # 피크 위치 & 토큰을 별도 JSON으로 저장 (수치 분석용)
    peak_entries = []
    for i in peak_indices:
        tok_id = int(gen_token_ids[i].item())
        raw_tok = gen_tokens[i]
        # 단일 토큰을 decode해서 사람이 읽기 쉬운 문자열로 변환
        decoded = tokenizer.decode([tok_id], skip_special_tokens=True)
        decoded_clean = decoded.replace("\n", " ").replace("\t", " ").strip()
        label = decoded_clean if decoded_clean else raw_tok
        peak_tokens.append(label)

        entry = {
            "rel_index": int(i),
            "abs_index": int(prompt_len + i),
            "baseline_score": float(gen_scores_baseline[i].item()),
            "sigmoid_score": float(gen_scores_sigmoid[i].item()),
            "inverted_sigmoid_score": float(gen_scores_sigmoid_inv[i].item()),
            "token_id": tok_id,
            "token_str": raw_tok,
            "token_text_decoded": decoded,
            "token_label_for_plot": label,
        }
        peak_entries.append(entry)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(
        args.out_dir, f"rollout_peaks_gsm8k_id{args.sample_id}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(peak_entries, f, ensure_ascii=False, indent=2)
    print(f"[Saved] peak-token mapping → {json_path}")

    # 그래프 저장 (sigmoid subplot에 피크 + 토큰 annotate)
    fname = f"rollout_scores_gsm8k_id{args.sample_id}.png"
    out_path = os.path.join(args.out_dir, fname)
    title_suffix = (
        f"(id={args.sample_id}, prompt_len={prompt_len}, gen_len={gen_length})"
    )

    plot_rollout_scores(
        x=x,
        gen_scores_baseline=gen_scores_baseline,
        gen_scores_sigmoid=gen_scores_sigmoid,
        gen_scores_sigmoid_inv=gen_scores_sigmoid_inv,
        out_path=out_path,
        title_suffix=title_suffix,
        peak_indices=peak_indices,
        peak_tokens=peak_tokens,
    )

    # Sigmoid score와 그 CDF만 따로 보는 플롯도 추가로 저장
    fname_sig_cdf = f"rollout_sigmoid_and_cdf_gsm8k_id{args.sample_id}.png"
    out_path_sig_cdf = os.path.join(args.out_dir, fname_sig_cdf)
    plot_sigmoid_and_cdf(
        x=x,
        gen_scores_sigmoid=gen_scores_sigmoid,
        out_path=out_path_sig_cdf,
        title_suffix=title_suffix,
    )


if __name__ == "__main__":
    main()

