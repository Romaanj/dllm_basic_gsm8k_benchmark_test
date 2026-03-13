"""
Step0 high-rollout 토큰 추적 (global confidence baseline)
=========================================================

목표:
1) 디코딩은 baseline: 매 step global confidence argmax 1토큰 unmask
2) Step0 rollout에서 high-rollout 위치(top-k)의 최종 디코딩 토큰 확인
3) 각 high-rollout 위치 주변 ±W 토큰 컨텍스트 확인
4) high-rollout gen 토큰이 step0 rollout matrix 기준으로 어떤 prompt 토큰에
   높은 attention을 두는지 추적
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_hybrid_cdf_eval import StreamingRollout, add_gumbel_noise


def clean_token_text(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n").strip()


@torch.no_grad()
def get_step0_rollout(
    model,
    input_ids: torch.Tensor,
    gen_length: int,
    mask_id: int,
    rollout_mode: str = "sigmoid",
) -> Dict[str, Any]:
    device = model.device
    prompt_len = input_ids.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = input_ids.clone()

    invert_depth = rollout_mode == "sigmoid_inverted"
    hook_mode = "baseline" if rollout_mode == "baseline" else "sigmoid"

    core_model = model.model if hasattr(model, "model") else model
    blocks_list = core_model.transformer.blocks
    num_layers = len(blocks_list)

    streaming = StreamingRollout(
        num_layers=num_layers, mode=hook_mode, invert_depth=invert_depth
    )
    streaming.register(blocks_list)

    try:
        _ = model(x, output_attentions=True)
    finally:
        streaming.remove()

    scores = streaming.get_scores()
    result: Dict[str, Any] = {"prompt_len": prompt_len}

    if scores is None:
        result["scores"] = None
        result["gen_scores"] = None
        return result

    scores_f64 = scores.to(torch.float64)
    result["scores"] = scores_f64
    result["gen_scores"] = scores_f64[prompt_len: prompt_len + gen_length].clone()
    if streaming.rollout is not None:
        result["matrix"] = streaming.rollout[0].to(torch.float64).clone()
    return result


@torch.no_grad()
def generate_global_confidence_baseline(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = prompt.clone()

    unmask_step: List[int] = [-1] * gen_length
    unmask_confidence: List[float] = [float("nan")] * gen_length
    nfe = 0

    for step in range(1, gen_length + 1):
        mask_idx = x == mask_id
        mask_idx[:, :prompt_len] = False
        if mask_idx.sum().item() == 0:
            break

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
        unmask_abs = int(torch.argmax(confidence, dim=1)[0].item())
        unmask_rel = unmask_abs - prompt_len

        x[0, unmask_abs] = x0[0, unmask_abs]
        unmask_step[unmask_rel] = step
        unmask_confidence[unmask_rel] = float(confidence[0, unmask_abs].item())
        nfe += 1

    info: Dict[str, Any] = {
        "decode_policy": "global confidence argmax, one token per step",
        "unmask_step": unmask_step,
        "unmask_confidence": unmask_confidence,
    }
    return x, nfe, info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step0 high-rollout token trace with global confidence baseline"
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:3")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "humaneval"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--rollout-mode", type=str, default="sigmoid",
                   choices=["sigmoid", "sigmoid_inverted", "baseline"])
    p.add_argument("--top-k-high", type=int, default=5,
                   help="high-rollout로 볼 생성 위치 개수")
    p.add_argument("--window-size", type=int, default=4,
                   help="high 위치 주변 +-W 토큰 컨텍스트")
    p.add_argument("--top-k-prompt", type=int, default=10,
                   help="각 high gen 위치에서 추적할 prompt top-k")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--out-dir", type=str, default="results_step0_high_rollout_trace")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    os.makedirs(args.out_dir, exist_ok=True)

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
    num_samples = min(args.num_samples, len(ds))

    records: List[Dict[str, Any]] = []

    for sid in tqdm(range(num_samples), desc="Tracing samples"):
        sample = ds[int(sid)]
        if args.task == "humaneval":
            question = sample["prompt"]
            if args.no_chat_template:
                prompt_str = question
            else:
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        else:
            question = sample["question"]
            if args.no_chat_template:
                prompt_str = question
            else:
                prompt_str = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    add_generation_prompt=True,
                    tokenize=False,
                )

        input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)
        prompt_len = int(input_ids.shape[1])
        prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        rollout_data = get_step0_rollout(
            model=model,
            input_ids=input_ids,
            gen_length=args.gen_length,
            mask_id=args.mask_id,
            rollout_mode=args.rollout_mode,
        )
        gen_scores_t = rollout_data.get("gen_scores", None)
        matrix = rollout_data.get("matrix", None)
        if gen_scores_t is None or matrix is None:
            print(f"  [WARNING] sample {sid}: rollout/matrix unavailable, skipped")
            continue

        gen_output, nfe, gen_info = generate_global_confidence_baseline(
            model=model,
            prompt=input_ids,
            gen_length=args.gen_length,
            mask_id=args.mask_id,
            temperature=args.temperature,
        )

        gen_token_ids = gen_output[0, prompt_len: prompt_len + args.gen_length].tolist()
        gen_token_strs = tokenizer.convert_ids_to_tokens(gen_token_ids)
        gen_decoded = [
            tokenizer.decode([int(tid)], skip_special_tokens=False).strip() or raw
            for raw, tid in zip(gen_token_strs, gen_token_ids)
        ]
        final_generation_text_raw = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
        final_generation_text_clean = tokenizer.decode(gen_token_ids, skip_special_tokens=True)
        if final_generation_text_clean.strip():
            final_generation_text = final_generation_text_clean
        else:
            # clean decode가 빈 문자열이면 non-special 토큰만 이어붙인 fallback 사용
            fallback_tokens = [
                clean_token_text(tok)
                for tok in gen_token_strs
                if not str(tok).startswith("<|")
            ]
            final_generation_text = "".join(fallback_tokens).strip()

        gen_scores = gen_scores_t.detach().cpu().numpy()
        top_k = min(args.top_k_high, len(gen_scores))
        high_idx = np.argsort(gen_scores)[-top_k:][::-1]

        cross_attn = matrix[prompt_len: prompt_len + args.gen_length, :prompt_len]
        cross_np = cross_attn.detach().cpu().numpy()

        high_records: List[Dict[str, Any]] = []
        for rank, i in enumerate(high_idx, start=1):
            i = int(i)
            left = max(0, i - args.window_size)
            right = min(args.gen_length - 1, i + args.window_size)

            local_tokens = []
            for j in range(left, right + 1):
                local_tokens.append({
                    "rel_pos": int(j),
                    "token_raw": gen_token_strs[j],
                    "token_decoded": gen_decoded[j],
                    "token_clean": clean_token_text(gen_token_strs[j]),
                    "rollout_score": float(gen_scores[j]),
                    "is_target_high": bool(j == i),
                    "unmask_step": int(gen_info["unmask_step"][j]),
                    "unmask_confidence": float(gen_info["unmask_confidence"][j]),
                })

            row = cross_np[i]
            p_top = min(args.top_k_prompt, len(row))
            prompt_top_idx = np.argsort(row)[-p_top:][::-1]
            prompt_attn_top = []
            for pidx in prompt_top_idx:
                pidx = int(pidx)
                tok = prompt_tokens[pidx]
                prompt_attn_top.append({
                    "prompt_index": pidx,
                    "prompt_token_raw": tok,
                    "prompt_token_clean": clean_token_text(tok),
                    "attention": float(row[pidx]),
                })

            high_records.append({
                "rank": int(rank),
                "gen_index": i,
                "rollout_score": float(gen_scores[i]),
                "decoded_token": gen_decoded[i],
                "token_raw": gen_token_strs[i],
                "token_clean": clean_token_text(gen_token_strs[i]),
                "unmask_step": int(gen_info["unmask_step"][i]),
                "unmask_confidence": float(gen_info["unmask_confidence"][i]),
                "window_tokens": local_tokens,
                "top_prompt_tokens_for_this_gen": prompt_attn_top,
            })

        avg_prompt_attn = cross_np[high_idx, :].mean(axis=0)
        global_prompt_top_idx = np.argsort(avg_prompt_attn)[-args.top_k_prompt:][::-1]
        global_prompt_top = []
        for pidx in global_prompt_top_idx:
            pidx = int(pidx)
            tok = prompt_tokens[pidx]
            global_prompt_top.append({
                "prompt_index": pidx,
                "prompt_token_raw": tok,
                "prompt_token_clean": clean_token_text(tok),
                "avg_attention_from_high_gen": float(avg_prompt_attn[pidx]),
            })

        records.append({
            "sample_id": int(sid),
            "task": args.task,
            "task_id": str(sample.get("task_id", "")),
            "question": question,
            "final_generation_text": final_generation_text,
            "final_generation_text_clean": final_generation_text_clean,
            "final_generation_text_raw": final_generation_text_raw,
            "prompt_len": prompt_len,
            "gen_length": int(args.gen_length),
            "decode_policy": gen_info["decode_policy"],
            "nfe": int(nfe),
            "high_rollout_topk": int(top_k),
            "window_size": int(args.window_size),
            "high_rollout_gen_tokens": high_records,
            "global_top_prompt_tokens_from_high_gen": global_prompt_top,
        })

    out = {
        "args": vars(args),
        "num_records": len(records),
        "records": records,
    }
    out_path = os.path.join(args.out_dir, "step0_high_rollout_trace.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[Saved] {out_path}")
    print(f"[Done] traced {len(records)} samples.")


if __name__ == "__main__":
    main()
