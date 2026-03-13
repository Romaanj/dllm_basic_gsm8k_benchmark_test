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
from gsm8k_hybrid_cdf_eval import StreamingRollout, add_gumbel_noise


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
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=model.device
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


def local_entropy_stats(
    entropy_gen: torch.Tensor, mask_gen: torch.Tensor, center: int, window_radius: int
) -> Tuple[float, Optional[float], int]:
    g = int(entropy_gen.numel())
    s = max(0, int(center) - int(window_radius))
    e = min(g, int(center) + int(window_radius) + 1)
    if s >= e:
        return 0.0, None, 0

    local_mask = mask_gen[s:e]
    count = int(local_mask.sum().item())
    if count == 0:
        return 0.0, None, 0

    local_vals = entropy_gen[s:e][local_mask]
    summ = float(local_vals.sum().item())
    mean = float(summ / count)
    return summ, mean, count


@torch.no_grad()
def run_baseline_entropy_tracking(
    model,
    tokenizer,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    window_radius: int,
    rollout_scores: torch.Tensor,
    high_indices: List[int],
    low_indices: List[int],
    temperature: float,
) -> Dict[str, Any]:
    device = model.device
    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length

    x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.clone()

    high_set = set(int(i) for i in high_indices)
    low_set = set(int(i) for i in low_indices)

    step_rows: List[Dict[str, Any]] = []
    entropy_history: List[torch.Tensor] = []
    mask_history: List[torch.Tensor] = []

    for step in range(1, gen_length + 1):
        mask_idx = (x == mask_id)
        mask_idx[:, :prompt_len] = False
        masked_abs = torch.where(mask_idx[0])[0]
        if masked_abs.numel() == 0:
            break

        logits = model(x).logits
        logits_fp32 = logits.to(torch.float32)
        log_probs = F.log_softmax(logits_fp32, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)

        entropy_gen = entropy[0, prompt_len : prompt_len + gen_length].detach().cpu()
        mask_gen = mask_idx[0, prompt_len : prompt_len + gen_length].detach().cpu()
        entropy_history.append(entropy_gen)
        mask_history.append(mask_gen)

        masked_entropy_vals = entropy[0, masked_abs]
        global_entropy_sum_pre = float(masked_entropy_vals.sum().item())
        masked_count_pre = int(masked_abs.numel())
        global_entropy_mean_pre = float(global_entropy_sum_pre / masked_count_pre)

        logits_noisy = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_noisy, dim=-1)
        score = torch.gather(
            probs.to(torch.float64), dim=-1, index=x0.unsqueeze(-1)
        ).squeeze(-1)
        x0 = torch.where(mask_idx, x0, x)

        neg_inf = torch.tensor(
            torch.finfo(score.dtype).min, device=device, dtype=score.dtype
        )
        confidence = torch.where(mask_idx, score, neg_inf)
        unmask_abs = int(torch.argmax(confidence, dim=1)[0].item())
        unmask_rel = int(unmask_abs - prompt_len)
        confidence_val = float(confidence[0, unmask_abs].item())
        token_id = int(x0[0, unmask_abs].item())
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)

        row = {
            "step": int(step),
            "unmasked_rel_pos": int(unmask_rel),
            "unmasked_abs_pos": int(unmask_abs),
            "unmasked_token_id": int(token_id),
            "unmasked_token_text": token_text,
            "unmask_confidence": confidence_val,
            "rollout_score": float(rollout_scores[unmask_rel].item()),
            "is_high_rollout": int(unmask_rel in high_set),
            "is_low_rollout": int(unmask_rel in low_set),
            "global_entropy_sum_pre": global_entropy_sum_pre,
            "global_entropy_mean_pre": global_entropy_mean_pre,
            "masked_count_pre": masked_count_pre,
        }
        step_rows.append(row)
        x[0, unmask_abs] = x0[0, unmask_abs]

    # final empty-mask state as the "post" state for last step
    entropy_history.append(torch.zeros(gen_length, dtype=torch.float32))
    mask_history.append(torch.zeros(gen_length, dtype=torch.bool))

    for i, row in enumerate(step_rows):
        pre_sum = float(row["global_entropy_sum_pre"])
        pre_mean = float(row["global_entropy_mean_pre"])
        post_sum = float(step_rows[i + 1]["global_entropy_sum_pre"]) if i + 1 < len(step_rows) else 0.0
        post_mean = float(step_rows[i + 1]["global_entropy_mean_pre"]) if i + 1 < len(step_rows) else None
        post_count = int(step_rows[i + 1]["masked_count_pre"]) if i + 1 < len(step_rows) else 0

        center = int(row["unmasked_rel_pos"])
        local_sum_pre, local_mean_pre, local_count_pre = local_entropy_stats(
            entropy_history[i], mask_history[i], center, window_radius
        )
        local_sum_post, local_mean_post, local_count_post = local_entropy_stats(
            entropy_history[i + 1], mask_history[i + 1], center, window_radius
        )

        row["global_entropy_sum_post"] = post_sum
        row["global_entropy_sum_delta"] = float(post_sum - pre_sum)
        row["global_entropy_mean_post"] = post_mean
        row["global_entropy_mean_delta"] = (
            None if post_mean is None else float(post_mean - pre_mean)
        )
        row["masked_count_post"] = post_count

        row["local_entropy_sum_pre"] = local_sum_pre
        row["local_entropy_sum_post"] = local_sum_post
        row["local_entropy_sum_delta"] = float(local_sum_post - local_sum_pre)
        row["local_entropy_mean_pre"] = local_mean_pre
        row["local_entropy_mean_post"] = local_mean_post
        row["local_entropy_mean_delta"] = (
            None if (local_mean_pre is None or local_mean_post is None)
            else float(local_mean_post - local_mean_pre)
        )
        row["local_masked_count_pre"] = local_count_pre
        row["local_masked_count_post"] = local_count_post

    final_ids = x[0, prompt_len : prompt_len + gen_length]
    final_text = tokenizer.decode(final_ids.tolist(), skip_special_tokens=True)
    return {"step_rows": step_rows, "final_text": final_text}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline one-token unmasking entropy event analysis"
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "humaneval"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-ids", type=str, default="")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--rollout-mode", type=str, default="sigmoid", choices=["sigmoid", "sigmoid_inverted", "baseline"])
    p.add_argument("--window-radius", type=int, default=4)
    p.add_argument("--top-k-per-type", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="results_baseline_entropy_events")
    return p.parse_args()


def mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in vals if v is not None]
    return float(sum(filtered) / len(filtered)) if filtered else None


def get_prompt(task: str, sample: Dict[str, Any]) -> str:
    if task == "gsm8k":
        return str(sample["question"])
    if task == "humaneval":
        return str(sample["prompt"])
    raise ValueError(f"Unsupported task: {task}")


def load_task_dataset(task: str, seed: int):
    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    elif task == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
    else:
        raise ValueError(f"Unsupported task: {task}")
    return ds.shuffle(seed=seed)


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

    print(f"Loading dataset: {args.task}")
    ds = load_task_dataset(args.task, args.seed)
    if args.sample_ids:
        sample_ids = sorted(set(int(x.strip()) for x in args.sample_ids.split(",") if x.strip()))
    else:
        sample_ids = list(range(args.start_id, args.start_id + args.num_samples))
    sample_ids = [i for i in sample_ids if 0 <= i < len(ds)]
    if not sample_ids:
        raise ValueError("유효한 sample id가 없습니다.")

    os.makedirs(args.out_dir, exist_ok=True)

    all_step_rows: List[Dict[str, Any]] = []
    sample_details: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sid in tqdm(sample_ids, desc="baseline entropy events"):
        try:
            sample = ds[int(sid)]
            q = get_prompt(args.task, sample)
            if args.no_chat_template:
                prompt_str = q
            else:
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)

            rollout_scores = compute_step0_rollout(
                model=model,
                prompt=input_ids,
                gen_length=args.gen_length,
                mask_id=args.mask_id,
                rollout_mode=args.rollout_mode,
            )
            k = min(max(1, int(args.top_k_per_type)), int(args.gen_length))
            high_idx = torch.argsort(rollout_scores, descending=True)[:k].tolist()
            low_idx = torch.argsort(rollout_scores, descending=False)[:k].tolist()

            tracked = run_baseline_entropy_tracking(
                model=model,
                tokenizer=tokenizer,
                prompt=input_ids,
                gen_length=args.gen_length,
                mask_id=args.mask_id,
                window_radius=args.window_radius,
                rollout_scores=rollout_scores,
                high_indices=high_idx,
                low_indices=low_idx,
                temperature=args.temperature,
            )

            sample_rows = tracked["step_rows"]
            for row in sample_rows:
                row["sample_id"] = int(sid)
            all_step_rows.extend(sample_rows)

            sample_details.append(
                {
                    "sample_id": int(sid),
                    "task": args.task,
                    "question": q,
                    "high_indices": [int(i) for i in high_idx],
                    "low_indices": [int(i) for i in low_idx],
                    "rollout_scores": [float(v) for v in rollout_scores.tolist()],
                    "final_text": tracked["final_text"],
                    "step_rows": sample_rows,
                }
            )
        except Exception as e:
            failures.append({"sample_id": int(sid), "error": str(e)})

    high_rows = [r for r in all_step_rows if int(r["is_high_rollout"]) == 1]
    low_rows = [r for r in all_step_rows if int(r["is_low_rollout"]) == 1]

    aggregate = {
        "all_steps_count": int(len(all_step_rows)),
        "high_event_count": int(len(high_rows)),
        "low_event_count": int(len(low_rows)),
        "high_global_delta_mean": mean_or_none([r["global_entropy_sum_delta"] for r in high_rows]),
        "low_global_delta_mean": mean_or_none([r["global_entropy_sum_delta"] for r in low_rows]),
        "high_local_delta_mean": mean_or_none([r["local_entropy_sum_delta"] for r in high_rows]),
        "low_local_delta_mean": mean_or_none([r["local_entropy_sum_delta"] for r in low_rows]),
        "high_global_mean_delta_mean": mean_or_none([r["global_entropy_mean_delta"] for r in high_rows]),
        "low_global_mean_delta_mean": mean_or_none([r["global_entropy_mean_delta"] for r in low_rows]),
        "high_local_mean_delta_mean": mean_or_none([r["local_entropy_mean_delta"] for r in high_rows]),
        "low_local_mean_delta_mean": mean_or_none([r["local_entropy_mean_delta"] for r in low_rows]),
    }

    tag = (
        f"batch{len(sample_ids)}_start{min(sample_ids)}_L{args.gen_length}"
        f"_w{args.window_radius}_k{args.top_k_per_type}_{args.rollout_mode}"
        f"_task-{args.task}"
    )
    out_json = os.path.join(args.out_dir, f"baseline_entropy_events_{tag}.json")
    out_steps_csv = os.path.join(args.out_dir, f"baseline_entropy_events_steps_{tag}.csv")
    out_events_csv = os.path.join(args.out_dir, f"baseline_entropy_events_highlow_{tag}.csv")

    payload = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "task": args.task,
            "seed": args.seed,
            "sample_ids": sample_ids,
            "gen_length": args.gen_length,
            "mask_id": args.mask_id,
            "temperature": args.temperature,
            "rollout_mode": args.rollout_mode,
            "window_radius": args.window_radius,
            "top_k_per_type": args.top_k_per_type,
            "decode_policy": "global confidence argmax, one token per step",
        },
        "aggregate": aggregate,
        "counts": {
            "samples": len(sample_ids),
            "step_rows": len(all_step_rows),
            "failures": len(failures),
        },
        "sample_details": sample_details,
        "failures": failures,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_json}")

    if all_step_rows:
        with open(out_steps_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_step_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_step_rows)
        print(f"[Saved] {out_steps_csv}")

        highlow_rows = [r for r in all_step_rows if int(r["is_high_rollout"]) == 1 or int(r["is_low_rollout"]) == 1]
        with open(out_events_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(highlow_rows[0].keys()))
            writer.writeheader()
            writer.writerows(highlow_rows)
        print(f"[Saved] {out_events_csv}")

    print("\n=== High/Low entropy delta summary ===")
    print(
        f"high_global_delta_mean={aggregate['high_global_delta_mean']} | "
        f"low_global_delta_mean={aggregate['low_global_delta_mean']}"
    )
    print(
        f"high_local_delta_mean={aggregate['high_local_delta_mean']} | "
        f"low_local_delta_mean={aggregate['low_local_delta_mean']}"
    )


if __name__ == "__main__":
    main()
