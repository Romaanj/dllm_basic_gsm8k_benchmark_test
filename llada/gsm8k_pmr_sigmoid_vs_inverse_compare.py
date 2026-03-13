import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_equal_mass_high_mass_probe import (
    build_topk_high_mass_report,
    trace_equal_mass_generation,
)


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "min": None,
            "max": None,
        }
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(t.numel()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "p50": float(torch.quantile(t, 0.5).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def compute_sample_metrics(
    top_report: List[Dict[str, Any]],
    tau: float,
) -> Dict[str, Any]:
    top_conf: List[float] = []
    top_block_sizes: List[float] = []
    block_low_conf_ratios: List[float] = []
    next_block_sizes: List[float] = []
    next_block_low_conf_ratios: List[float] = []

    for item in top_report:
        top_tok = item["top_token"]
        conf = top_tok["unmask_confidence"]
        if conf is not None:
            top_conf.append(float(conf))

        blk = item["block_info"]
        top_block_sizes.append(float(blk["block_size"]))

        block_tokens = item["block_tokens_unmask_confidence"]
        block_conf = [
            float(r["unmask_confidence"])
            for r in block_tokens
            if r["unmask_confidence"] is not None
        ]
        if block_conf:
            low_ratio = sum(1 for c in block_conf if c < tau) / len(block_conf)
            block_low_conf_ratios.append(float(low_ratio))

        next_info = item.get("next_block_info")
        if next_info is not None:
            next_block_sizes.append(float(next_info["block_size"]))
            next_tokens = item.get("next_block_tokens_unmask_confidence") or []
            next_conf = [
                float(r["unmask_confidence"])
                for r in next_tokens
                if r.get("unmask_confidence") is not None
            ]
            if next_conf:
                next_low_ratio = sum(1 for c in next_conf if c < tau) / len(next_conf)
                next_block_low_conf_ratios.append(float(next_low_ratio))

    pmr = None
    if top_conf:
        pmr = float(sum(1 for c in top_conf if c < tau) / len(top_conf))

    return {
        "topk_high_mass_conf_mean": safe_mean(top_conf),
        "pmr_topk": pmr,
        "topk_token_block_size_mean": safe_mean(top_block_sizes),
        "block_internal_low_conf_ratio_mean": safe_mean(block_low_conf_ratios),
        "next_block_size_mean": safe_mean(next_block_sizes),
        "next_block_internal_low_conf_ratio_mean": safe_mean(next_block_low_conf_ratios),
        "topk_count_effective": len(top_conf),
    }


def run_trace(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    chunking_mode: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    trace = trace_equal_mass_generation(
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
        chunking_mode=chunking_mode,
        lam=args.lam,
    )
    top_report = build_topk_high_mass_report(trace, top_k=args.top_k_high_mass)
    return trace, top_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare PMR-style metrics between sigmoid_cdf(equal_mass) and inverse_cdf."
        )
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:3")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--no-chat-template", action="store_true")

    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--steps-per-block", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--min-block-size", type=int, default=1)
    p.add_argument("--max-block-size", type=int, default=80)
    p.add_argument(
        "--rollout-mode",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "sigmoid_inverted", "baseline"],
    )
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--top-k-high-mass", type=int, default=10)
    p.add_argument(
        "--tau",
        type=float,
        default=0.8,
        help="PMR threshold: unmask_confidence < tau 를 premature로 간주.",
    )

    p.add_argument("--out-dir", type=str, default="results_equal_mass/high_mass_probe")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading HumanEval test split...")
    ds = load_dataset("openai/humaneval", split="test").shuffle(seed=args.seed)
    end_id = min(args.start_id + args.num_samples, len(ds))
    sample_ids = list(range(args.start_id, end_id))
    if not sample_ids:
        raise ValueError("선택된 샘플이 없습니다. start-id / num-samples를 확인하세요.")

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sid in tqdm(sample_ids, desc="PMR compare (sigmoid_cdf vs inverse_cdf)"):
        try:
            sample = ds[int(sid)]
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

            trace_sig, top_sig = run_trace(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                args=args,
                chunking_mode="equal_mass",
            )
            trace_inv, top_inv = run_trace(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                args=args,
                chunking_mode="inverse_cdf",
            )

            m_sig = compute_sample_metrics(top_sig, tau=args.tau)
            m_inv = compute_sample_metrics(top_inv, tau=args.tau)

            row = {
                "sample_id": int(sid),
                "sigmoid_topk_conf_mean": m_sig["topk_high_mass_conf_mean"],
                "inverse_topk_conf_mean": m_inv["topk_high_mass_conf_mean"],
                "delta_topk_conf_mean_inv_minus_sig": (
                    None
                    if m_sig["topk_high_mass_conf_mean"] is None
                    or m_inv["topk_high_mass_conf_mean"] is None
                    else float(
                        m_inv["topk_high_mass_conf_mean"] - m_sig["topk_high_mass_conf_mean"]
                    )
                ),
                "sigmoid_pmr_topk": m_sig["pmr_topk"],
                "inverse_pmr_topk": m_inv["pmr_topk"],
                "delta_pmr_topk_inv_minus_sig": (
                    None
                    if m_sig["pmr_topk"] is None or m_inv["pmr_topk"] is None
                    else float(m_inv["pmr_topk"] - m_sig["pmr_topk"])
                ),
                "sigmoid_topk_block_size_mean": m_sig["topk_token_block_size_mean"],
                "inverse_topk_block_size_mean": m_inv["topk_token_block_size_mean"],
                "delta_topk_block_size_mean_inv_minus_sig": (
                    None
                    if m_sig["topk_token_block_size_mean"] is None
                    or m_inv["topk_token_block_size_mean"] is None
                    else float(
                        m_inv["topk_token_block_size_mean"]
                        - m_sig["topk_token_block_size_mean"]
                    )
                ),
                "sigmoid_block_low_conf_ratio_mean": m_sig["block_internal_low_conf_ratio_mean"],
                "inverse_block_low_conf_ratio_mean": m_inv["block_internal_low_conf_ratio_mean"],
                "delta_block_low_conf_ratio_mean_inv_minus_sig": (
                    None
                    if m_sig["block_internal_low_conf_ratio_mean"] is None
                    or m_inv["block_internal_low_conf_ratio_mean"] is None
                    else float(
                        m_inv["block_internal_low_conf_ratio_mean"]
                        - m_sig["block_internal_low_conf_ratio_mean"]
                    )
                ),
                "sigmoid_next_block_size_mean": m_sig["next_block_size_mean"],
                "inverse_next_block_size_mean": m_inv["next_block_size_mean"],
                "delta_next_block_size_mean_inv_minus_sig": (
                    None
                    if m_sig["next_block_size_mean"] is None
                    or m_inv["next_block_size_mean"] is None
                    else float(
                        m_inv["next_block_size_mean"] - m_sig["next_block_size_mean"]
                    )
                ),
                "sigmoid_next_block_low_conf_ratio_mean": m_sig[
                    "next_block_internal_low_conf_ratio_mean"
                ],
                "inverse_next_block_low_conf_ratio_mean": m_inv[
                    "next_block_internal_low_conf_ratio_mean"
                ],
                "delta_next_block_low_conf_ratio_mean_inv_minus_sig": (
                    None
                    if m_sig["next_block_internal_low_conf_ratio_mean"] is None
                    or m_inv["next_block_internal_low_conf_ratio_mean"] is None
                    else float(
                        m_inv["next_block_internal_low_conf_ratio_mean"]
                        - m_sig["next_block_internal_low_conf_ratio_mean"]
                    )
                ),
                "sigmoid_nfe": trace_sig["nfe"],
                "inverse_nfe": trace_inv["nfe"],
                "sigmoid_topk_effective": m_sig["topk_count_effective"],
                "inverse_topk_effective": m_inv["topk_count_effective"],
            }
            rows.append(row)
        except Exception as e:
            failures.append({"sample_id": int(sid), "error": str(e)})

    # 디버깅: 전부 실패하면 summary mean이 전부 None으로 보임
    print(f"\n[PMR compare] rows={len(rows)}, failures={len(failures)}")
    if failures and not rows:
        print("[PMR compare] 모든 샘플 실패. 첫 오류 예시:")
        for f in failures[:3]:
            print(f"  sample_id={f['sample_id']}: {f['error']}")
    elif failures:
        print(f"[PMR compare] 일부 실패 {len(failures)}건 (첫 오류): {failures[0]}")

    def collect(col: str) -> List[float]:
        out = []
        for r in rows:
            v = r[col]
            if v is not None:
                out.append(float(v))
        return out

    summary = {
        "sigmoid_cdf": {
            "topk_conf_mean": summarize(collect("sigmoid_topk_conf_mean")),
            "pmr_topk": summarize(collect("sigmoid_pmr_topk")),
            "topk_block_size_mean": summarize(collect("sigmoid_topk_block_size_mean")),
            "block_low_conf_ratio_mean": summarize(collect("sigmoid_block_low_conf_ratio_mean")),
            "next_block_size_mean": summarize(collect("sigmoid_next_block_size_mean")),
            "next_block_low_conf_ratio_mean": summarize(
                collect("sigmoid_next_block_low_conf_ratio_mean")
            ),
        },
        "inverse_cdf": {
            "topk_conf_mean": summarize(collect("inverse_topk_conf_mean")),
            "pmr_topk": summarize(collect("inverse_pmr_topk")),
            "topk_block_size_mean": summarize(collect("inverse_topk_block_size_mean")),
            "block_low_conf_ratio_mean": summarize(collect("inverse_block_low_conf_ratio_mean")),
            "next_block_size_mean": summarize(collect("inverse_next_block_size_mean")),
            "next_block_low_conf_ratio_mean": summarize(
                collect("inverse_next_block_low_conf_ratio_mean")
            ),
        },
        "delta_inv_minus_sigmoid": {
            "topk_conf_mean": summarize(collect("delta_topk_conf_mean_inv_minus_sig")),
            "pmr_topk": summarize(collect("delta_pmr_topk_inv_minus_sig")),
            "topk_block_size_mean": summarize(
                collect("delta_topk_block_size_mean_inv_minus_sig")
            ),
            "block_low_conf_ratio_mean": summarize(
                collect("delta_block_low_conf_ratio_mean_inv_minus_sig")
            ),
            "next_block_size_mean": summarize(
                collect("delta_next_block_size_mean_inv_minus_sig")
            ),
            "next_block_low_conf_ratio_mean": summarize(
                collect("delta_next_block_low_conf_ratio_mean_inv_minus_sig")
            ),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    tag = (
        f"pmr_compare_{len(sample_ids)}samples_start{args.start_id}"
        f"_N{args.num_blocks}_spb{args.steps_per_block}_th{args.threshold}"
        f"_tau{args.tau}_{args.rollout_mode}_lam{args.lam}"
    )
    out_csv = os.path.join(args.out_dir, f"{tag}.csv")
    out_json = os.path.join(args.out_dir, f"{tag}.json")

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "start_id": args.start_id,
            "num_samples_requested": args.num_samples,
            "num_samples_used": len(sample_ids),
            "gen_length": args.gen_length,
            "num_blocks": args.num_blocks,
            "steps_per_block": args.steps_per_block,
            "temperature": args.temperature,
            "threshold": args.threshold,
            "tau": args.tau,
            "rollout_mode": args.rollout_mode,
            "lam": args.lam,
            "top_k_high_mass": args.top_k_high_mass,
            "use_chat_template": not args.no_chat_template,
            "modes_compared": ["sigmoid_cdf(equal_mass)", "inverse_cdf"],
        },
        "summary": summary,
        "counts": {
            "rows": len(rows),
            "failures": len(failures),
        },
        "failures": failures,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if rows:
        print(f"[Saved] {out_csv}")
    else:
        print("[WARNING] CSV 미저장: 성공한 row가 0개입니다. failures를 JSON에서 확인하세요.")
    print(f"[Saved] {out_json}")
    if not rows:
        print("\n=== PMR Compare Summary ===")
        print("(성공 샘플이 없어 요약 mean을 계산할 수 없습니다.)")
        return
    print("\n=== PMR Compare Summary (inverse - sigmoid) ===")
    print(
        "topk_conf_mean:",
        summary["delta_inv_minus_sigmoid"]["topk_conf_mean"]["mean"],
    )
    print(
        "pmr_topk:",
        summary["delta_inv_minus_sigmoid"]["pmr_topk"]["mean"],
    )
    print(
        "topk_block_size_mean:",
        summary["delta_inv_minus_sigmoid"]["topk_block_size_mean"]["mean"],
    )
    print(
        "block_low_conf_ratio_mean:",
        summary["delta_inv_minus_sigmoid"]["block_low_conf_ratio_mean"]["mean"],
    )
    print(
        "next_block_size_mean:",
        summary["delta_inv_minus_sigmoid"]["next_block_size_mean"]["mean"],
    )
    print(
        "next_block_low_conf_ratio_mean:",
        summary["delta_inv_minus_sigmoid"]["next_block_low_conf_ratio_mean"]["mean"],
    )


if __name__ == "__main__":
    main()
