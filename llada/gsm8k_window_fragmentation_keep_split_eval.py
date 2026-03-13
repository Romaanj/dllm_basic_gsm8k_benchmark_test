import argparse
import csv
import json
import os
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
    select_transfer_index_threshold,
    select_transfer_index_topk,
)


def fixed_blocks(gen_length: int, num_blocks: int) -> List[Tuple[int, int]]:
    if num_blocks <= 0:
        return [(0, gen_length)]
    num_blocks = min(num_blocks, gen_length)
    base = gen_length // num_blocks
    rem = gen_length % num_blocks
    sizes = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    blocks: List[Tuple[int, int]] = []
    cur = 0
    for s in sizes:
        nxt = min(gen_length, cur + s)
        if nxt > cur:
            blocks.append((cur, nxt))
        cur = nxt
    if blocks and blocks[-1][1] < gen_length:
        blocks[-1] = (blocks[-1][0], gen_length)
    return blocks if blocks else [(0, gen_length)]


def make_keep_whole_blocks(
    base_blocks: List[Tuple[int, int]],
    window_start: int,
    window_end: int,
) -> List[Tuple[int, int]]:
    if not base_blocks:
        return []

    # Window fully inside a single existing block:
    # keep-whole should preserve baseline partition exactly.
    block_idx_start = None
    block_idx_end = None
    end_pos = max(window_start, window_end - 1)
    for i, (s, e) in enumerate(base_blocks):
        if s <= window_start < e:
            block_idx_start = i
        if s <= end_pos < e:
            block_idx_end = i
    if block_idx_start is not None and block_idx_start == block_idx_end:
        return base_blocks[:]

    # Local surgery for cross-block windows:
    # 1) add window boundaries (ws, we), 2) remove boundaries inside (ws, we)
    # so the window becomes one single block while outside structure stays intact.
    boundaries = {0}
    for _, e in base_blocks:
        boundaries.add(int(e))
    boundaries.add(int(window_start))
    boundaries.add(int(window_end))

    kept = sorted(
        b for b in boundaries if not (window_start < b < window_end)
    )
    out: List[Tuple[int, int]] = []
    for i in range(len(kept) - 1):
        s = kept[i]
        e = kept[i + 1]
        if e > s:
            out.append((s, e))
    return out


def make_force_split_blocks(
    keep_blocks: List[Tuple[int, int]],
    window_start: int,
    window_end: int,
) -> List[Tuple[int, int]]:
    if not keep_blocks:
        return []

    split_at = (window_start + window_end) // 2
    if split_at <= window_start:
        split_at = window_start + 1
    if split_at >= window_end:
        split_at = window_end - 1

    out: List[Tuple[int, int]] = []
    split_done = False
    for s, e in keep_blocks:
        if s <= window_start and window_end <= e:
            local_split = split_at
            if local_split <= s:
                local_split = s + 1
            if local_split >= e:
                local_split = e - 1
            if s < local_split < e:
                out.append((s, local_split))
                out.append((local_split, e))
                split_done = True
            else:
                out.append((s, e))
        else:
            out.append((s, e))

    return out if split_done else keep_blocks[:]


@torch.no_grad()
def compute_step0_rollout(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    rollout_mode: str = "sigmoid",
) -> torch.Tensor:
    prompt_len = prompt.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_len] = prompt.clone()

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
        return torch.ones(gen_length, dtype=torch.float64) / gen_length
    return scores.to(torch.float64)[prompt_len : prompt_len + gen_length]


@torch.no_grad()
def generate_with_window_tracking(
    model,
    tokenizer,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    blocks: List[Tuple[int, int]],
    steps_per_block: int,
    temperature: float,
    threshold: Optional[float],
) -> Dict[str, Any]:
    device = model.device
    prompt_len = prompt.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = prompt.clone()

    unmask_step: List[Optional[int]] = [None] * gen_length
    unmask_confidence: List[Optional[float]] = [None] * gen_length
    proposal_history: List[List[int]] = [[] for _ in range(gen_length)]

    decode_iter = 0
    nfe = 0
    for block_idx, (bs_rel, be_rel) in enumerate(blocks):
        block_start = prompt_len + bs_rel
        block_end = prompt_len + be_rel
        block_mask = (x[:, block_start:block_end] == mask_id)
        if block_mask.sum() == 0:
            continue

        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        step_i = 0
        while True:
            remaining = (x[:, block_start:block_end] == mask_id).sum().item()
            if remaining == 0:
                break

            decode_iter += 1
            nfe += 1

            mask_idx = (x == mask_id)
            mask_idx[:, block_end:] = False

            logits = model(x).logits
            logits_noisy = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)
            probs = F.softmax(logits.to(torch.float64), dim=-1)
            score = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            x0 = torch.where(mask_idx, x0, x)
            neg_inf = torch.tensor(
                torch.finfo(score.dtype).min, device=device, dtype=score.dtype
            )
            confidence = torch.where(mask_idx, score, neg_inf)

            # Token proposal trajectory for still-masked generation tokens
            masked_abs = torch.where(mask_idx[0])[0]
            for abs_pos in masked_abs.tolist():
                rel = abs_pos - prompt_len
                if 0 <= rel < gen_length:
                    proposal_history[rel].append(int(x0[0, abs_pos].item()))

            if threshold is not None:
                transfer_index = select_transfer_index_threshold(
                    confidence, mask_idx, threshold
                )
            else:
                max_i = num_transfer.size(1) - 1
                si = min(step_i, max_i)
                transfer_index = select_transfer_index_topk(
                    confidence, mask_idx, num_transfer[:, si]
                )

            newly = transfer_index & (x == mask_id)
            newly_abs = torch.where(newly[0])[0]
            for abs_pos in newly_abs.tolist():
                rel = abs_pos - prompt_len
                if 0 <= rel < gen_length and unmask_step[rel] is None:
                    unmask_step[rel] = decode_iter
                    unmask_confidence[rel] = float(confidence[0, abs_pos].item())

            x[transfer_index] = x0[transfer_index]
            step_i += 1

    gen_ids = x[0, prompt_len : prompt_len + gen_length]
    final_text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
    return {
        "final_x": x,
        "final_text": final_text,
        "nfe": nfe,
        "unmask_step": unmask_step,
        "unmask_confidence": unmask_confidence,
        "proposal_history": proposal_history,
        "blocks": blocks,
    }


def get_window_centers(
    gen_scores: torch.Tensor,
    top_k: int,
    window_radius: int,
    seed: int,
) -> Dict[str, List[int]]:
    gen_length = int(gen_scores.numel())
    k = min(max(top_k, 1), gen_length)

    high = torch.argsort(gen_scores, descending=True)[:k].tolist()
    low = torch.argsort(gen_scores, descending=False)[:k].tolist()

    rng = torch.Generator()
    rng.manual_seed(seed)
    safe_lo = window_radius
    safe_hi = gen_length - window_radius - 1
    if safe_hi <= safe_lo:
        rnd = [gen_length // 2 for _ in range(k)]
    else:
        rnd = [
            int(torch.randint(safe_lo, safe_hi + 1, (1,), generator=rng).item())
            for _ in range(k)
        ]

    return {
        "high_mass": [int(i) for i in high],
        "low_mass": [int(i) for i in low],
        "random": [int(i) for i in rnd],
    }


def variance(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    m = sum(vals) / len(vals)
    return float(sum((v - m) ** 2 for v in vals) / len(vals))


def normalize_windows(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not windows:
        return []
    valid = [(int(s), int(e)) for s, e in windows if e > s]
    if not valid:
        return []
    valid.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = [valid[0]]
    for s, e in valid[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def make_keep_whole_blocks_multi(
    base_blocks: List[Tuple[int, int]],
    windows: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    out = base_blocks[:]
    for ws, we in normalize_windows(windows):
        out = make_keep_whole_blocks(out, ws, we)
    return out


def make_force_split_blocks_multi(
    keep_blocks: List[Tuple[int, int]],
    windows: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    out = keep_blocks[:]
    for ws, we in normalize_windows(windows):
        out = make_force_split_blocks(out, ws, we)
    return out


def window_union_indices(windows: List[Tuple[int, int]]) -> List[int]:
    idx = set()
    for s, e in normalize_windows(windows):
        for i in range(s, e):
            idx.add(i)
    return sorted(idx)


def compute_window_metrics(
    tracking: Dict[str, Any],
    window_start: int,
    window_end: int,
    tau: float,
) -> Dict[str, Any]:
    idxs = list(range(window_start, window_end))
    confs = [
        tracking["unmask_confidence"][i]
        for i in idxs
        if tracking["unmask_confidence"][i] is not None
    ]

    local_pmr = None
    if confs:
        local_pmr = float(sum(1 for c in confs if c < tau) / len(confs))

    final_local_conf = float(sum(confs) / len(confs)) if confs else None

    # Flip count: proposal token changed while token was still masked
    flips_per_token: List[int] = []
    for i in idxs:
        hist = tracking["proposal_history"][i]
        flips = 0
        for j in range(1, len(hist)):
            if hist[j] != hist[j - 1]:
                flips += 1
        flips_per_token.append(flips)

    total_flips = int(sum(flips_per_token))
    mean_flips = float(total_flips / len(flips_per_token)) if flips_per_token else 0.0

    # Neighbor mismatch: adjacent tokens with different PMR labels
    pmr_label = []
    for i in idxs:
        c = tracking["unmask_confidence"][i]
        pmr_label.append(None if c is None else (c < tau))
    mismatch_pairs = 0
    valid_pairs = 0
    for a, b in zip(pmr_label[:-1], pmr_label[1:]):
        if a is None or b is None:
            continue
        valid_pairs += 1
        if a != b:
            mismatch_pairs += 1
    mismatch_ratio = (
        float(mismatch_pairs / valid_pairs) if valid_pairs > 0 else None
    )

    return {
        "local_pmr": local_pmr,
        "final_local_confidence": final_local_conf,
        "neighbor_total_flip_count": total_flips,
        "neighbor_mean_flip_count": mean_flips,
        "neighbor_confidence_variance": variance([float(c) for c in confs]) if confs else None,
        "neighbor_mismatch_ratio": mismatch_ratio,
        "window_len": len(idxs),
    }


def compute_multi_window_metrics(
    tracking: Dict[str, Any],
    windows: List[Tuple[int, int]],
    tau: float,
) -> Dict[str, Any]:
    idxs = window_union_indices(windows)
    confs = [
        tracking["unmask_confidence"][i]
        for i in idxs
        if tracking["unmask_confidence"][i] is not None
    ]

    local_pmr = None
    if confs:
        local_pmr = float(sum(1 for c in confs if c < tau) / len(confs))

    final_local_conf = float(sum(confs) / len(confs)) if confs else None

    flips_per_token: List[int] = []
    for i in idxs:
        hist = tracking["proposal_history"][i]
        flips = 0
        for j in range(1, len(hist)):
            if hist[j] != hist[j - 1]:
                flips += 1
        flips_per_token.append(flips)

    total_flips = int(sum(flips_per_token))
    mean_flips = float(total_flips / len(flips_per_token)) if flips_per_token else 0.0

    pmr_label = []
    for i in idxs:
        c = tracking["unmask_confidence"][i]
        pmr_label.append(None if c is None else (c < tau))
    mismatch_pairs = 0
    valid_pairs = 0
    for a, b in zip(pmr_label[:-1], pmr_label[1:]):
        if a is None or b is None:
            continue
        valid_pairs += 1
        if a != b:
            mismatch_pairs += 1
    mismatch_ratio = float(mismatch_pairs / valid_pairs) if valid_pairs > 0 else None

    return {
        "local_pmr": local_pmr,
        "final_local_confidence": final_local_conf,
        "neighbor_total_flip_count": total_flips,
        "neighbor_mean_flip_count": mean_flips,
        "neighbor_confidence_variance": variance([float(c) for c in confs]) if confs else None,
        "neighbor_mismatch_ratio": mismatch_ratio,
        "window_len": len(idxs),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-block window fragmentation: keep-whole vs force-split"
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-ids", type=str, default="")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--steps-per-block", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--rollout-mode", type=str, default="sigmoid", choices=["sigmoid", "sigmoid_inverted", "baseline"])
    p.add_argument("--window-radius", type=int, default=4)
    p.add_argument("--top-k-per-type", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.8, help="PMR threshold")
    p.add_argument("--out-dir", type=str, default="results_window_fragmentation_keep_split")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

    print(f"Loading model: {args.model}")
    model = (
        LLaDAModelLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=dtype_map[args.dtype]
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
    sample_ids = [i for i in sample_ids if 0 <= i < len(ds)]
    if not sample_ids:
        raise ValueError("유효한 sample id가 없습니다.")

    os.makedirs(args.out_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sid in tqdm(sample_ids, desc="keep-whole vs force-split"):
        try:
            q = ds[int(sid)]["question"]
            if args.no_chat_template:
                prompt_str = q
            else:
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)

            gen_scores = compute_step0_rollout(
                model=model,
                prompt=input_ids,
                gen_length=args.gen_length,
                mask_id=args.mask_id,
                rollout_mode=args.rollout_mode,
            )
            centers = get_window_centers(
                gen_scores=gen_scores,
                top_k=args.top_k_per_type,
                window_radius=args.window_radius,
                seed=args.seed + int(sid),
            )
            base = fixed_blocks(args.gen_length, args.num_blocks)

            for wtype, wcenters in centers.items():
                windows = [
                    (
                        max(0, int(c) - args.window_radius),
                        min(args.gen_length, int(c) + args.window_radius + 1),
                    )
                    for c in wcenters
                ]
                windows = normalize_windows(windows)

                keep_blocks = make_keep_whole_blocks_multi(base, windows)
                split_blocks = make_force_split_blocks_multi(keep_blocks, windows)

                track_keep = generate_with_window_tracking(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=input_ids,
                    gen_length=args.gen_length,
                    mask_id=args.mask_id,
                    blocks=keep_blocks,
                    steps_per_block=args.steps_per_block,
                    temperature=args.temperature,
                    threshold=args.threshold,
                )
                track_split = generate_with_window_tracking(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=input_ids,
                    gen_length=args.gen_length,
                    mask_id=args.mask_id,
                    blocks=split_blocks,
                    steps_per_block=args.steps_per_block,
                    temperature=args.temperature,
                    threshold=args.threshold,
                )

                m_keep = compute_multi_window_metrics(track_keep, windows, args.tau)
                m_split = compute_multi_window_metrics(track_split, windows, args.tau)

                row = {
                    "sample_id": int(sid),
                    "window_type": wtype,
                    "top_k_requested": int(args.top_k_per_type),
                    "num_windows_effective": int(len(windows)),
                    "centers": json.dumps([int(c) for c in wcenters]),
                    "windows": json.dumps([[int(s), int(e)] for s, e in windows]),
                    "keep_local_pmr": m_keep["local_pmr"],
                    "split_local_pmr": m_split["local_pmr"],
                    "delta_local_pmr_split_minus_keep": (
                        None
                        if m_keep["local_pmr"] is None or m_split["local_pmr"] is None
                        else float(m_split["local_pmr"] - m_keep["local_pmr"])
                    ),
                    "keep_final_local_confidence": m_keep["final_local_confidence"],
                    "split_final_local_confidence": m_split["final_local_confidence"],
                    "delta_final_local_confidence_split_minus_keep": (
                        None
                        if m_keep["final_local_confidence"] is None
                        or m_split["final_local_confidence"] is None
                        else float(
                            m_split["final_local_confidence"] - m_keep["final_local_confidence"]
                        )
                    ),
                    "keep_neighbor_total_flip_count": m_keep["neighbor_total_flip_count"],
                    "split_neighbor_total_flip_count": m_split["neighbor_total_flip_count"],
                    "keep_neighbor_confidence_variance": m_keep["neighbor_confidence_variance"],
                    "split_neighbor_confidence_variance": m_split["neighbor_confidence_variance"],
                    "keep_neighbor_mismatch_ratio": m_keep["neighbor_mismatch_ratio"],
                    "split_neighbor_mismatch_ratio": m_split["neighbor_mismatch_ratio"],
                    "keep_total_sentence": track_keep["final_text"],
                    "split_total_sentence": track_split["final_text"],
                }
                rows.append(row)

                detailed.append(
                    {
                        "sample_id": int(sid),
                        "window_type": wtype,
                        "centers": [int(c) for c in wcenters],
                        "windows": [[int(s), int(e)] for s, e in windows],
                        "base_blocks": base,
                        "keep_blocks": keep_blocks,
                        "split_blocks": split_blocks,
                        "keep_metrics": m_keep,
                        "split_metrics": m_split,
                        "keep_total_sentence": track_keep["final_text"],
                        "split_total_sentence": track_split["final_text"],
                    }
                )
        except Exception as e:
            failures.append({"sample_id": int(sid), "error": str(e)})

    # aggregate by window type
    def mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
        vs = [float(v) for v in vals if v is not None]
        return float(sum(vs) / len(vs)) if vs else None

    aggregate: Dict[str, Any] = {}
    for wtype in ["high_mass", "low_mass", "random"]:
        subset = [r for r in rows if r["window_type"] == wtype]
        aggregate[wtype] = {
            "count": len(subset),
            "keep_local_pmr_mean": mean_or_none([r["keep_local_pmr"] for r in subset]),
            "split_local_pmr_mean": mean_or_none([r["split_local_pmr"] for r in subset]),
            "delta_local_pmr_mean": mean_or_none(
                [r["delta_local_pmr_split_minus_keep"] for r in subset]
            ),
            "keep_final_local_conf_mean": mean_or_none(
                [r["keep_final_local_confidence"] for r in subset]
            ),
            "split_final_local_conf_mean": mean_or_none(
                [r["split_final_local_confidence"] for r in subset]
            ),
            "delta_final_local_conf_mean": mean_or_none(
                [r["delta_final_local_confidence_split_minus_keep"] for r in subset]
            ),
            "keep_neighbor_flip_mean": mean_or_none(
                [float(r["keep_neighbor_total_flip_count"]) for r in subset]
            ),
            "split_neighbor_flip_mean": mean_or_none(
                [float(r["split_neighbor_total_flip_count"]) for r in subset]
            ),
            "keep_neighbor_conf_var_mean": mean_or_none(
                [r["keep_neighbor_confidence_variance"] for r in subset]
            ),
            "split_neighbor_conf_var_mean": mean_or_none(
                [r["split_neighbor_confidence_variance"] for r in subset]
            ),
            "keep_neighbor_mismatch_mean": mean_or_none(
                [r["keep_neighbor_mismatch_ratio"] for r in subset]
            ),
            "split_neighbor_mismatch_mean": mean_or_none(
                [r["split_neighbor_mismatch_ratio"] for r in subset]
            ),
        }

    tag = (
        f"batch{len(sample_ids)}_start{min(sample_ids)}_N{args.num_blocks}"
        f"_w{args.window_radius}_k{args.top_k_per_type}_{args.rollout_mode}"
    )
    out_json = os.path.join(args.out_dir, f"keep_split_window_fragmentation_{tag}.json")
    out_csv = os.path.join(args.out_dir, f"keep_split_window_fragmentation_rows_{tag}.csv")

    payload = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "sample_ids": sample_ids,
            "gen_length": args.gen_length,
            "num_blocks": args.num_blocks,
            "steps_per_block": args.steps_per_block,
            "threshold": args.threshold,
            "tau": args.tau,
            "window_radius": args.window_radius,
            "top_k_per_type": args.top_k_per_type,
            "rollout_mode": args.rollout_mode,
            "comparison": {
                "keep_whole": "For K windows, remove boundaries inside each window",
                "force_split": "After keep-whole, insert one boundary per window (midpoint)",
            },
        },
        "counts": {
            "rows": len(rows),
            "failures": len(failures),
        },
        "aggregate_by_window_type": aggregate,
        "rows": rows,
        "details": detailed,
        "failures": failures,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_json}")

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Saved] {out_csv}")

    print("\n=== Aggregate (split - keep) by window type ===")
    for wtype, m in aggregate.items():
        print(
            f"{wtype:10s} | "
            f"dPMR={m['delta_local_pmr_mean']} | "
            f"dConf={m['delta_final_local_conf_mean']}"
        )


if __name__ == "__main__":
    main()
