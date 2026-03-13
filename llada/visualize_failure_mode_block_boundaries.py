"""
Failure mode visualization for block boundary strategies.

Compares block boundaries from:
  1) Fixed block
  2) AdaBlock delimiter heuristic
  3) Hybrid inverse-CDF

The script runs the model on selected prompts, extracts three boundary sets on the
same generated-token axis, and saves overlay figures.
"""

import argparse
import html
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_depth_adaptive_rollout(
    attentions: Tuple[torch.Tensor, ...],
    invert_depth: bool = False,
) -> torch.Tensor:
    with torch.no_grad():
        if attentions[0].dim() == 4:
            avg_attn = [a.mean(dim=1) for a in attentions]
        else:
            avg_attn = list(attentions)

        num_layers = len(avg_attn)
        res_attn: List[torch.Tensor] = []
        for i, a in enumerate(avg_attn):
            eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype)
            slope = 0.5
            mid = num_layers / 2
            depth_arg = (mid - i) if invert_depth else (i - mid)
            alpha = 0.5 * torch.sigmoid(
                torch.tensor(slope * depth_arg, device=a.device, dtype=a.dtype)
            )
            res_attn.append((1.0 - alpha) * eye + alpha * a)

        rollout = res_attn[0]
        for i in range(1, len(res_attn)):
            rollout = torch.matmul(res_attn[i], rollout)
        return rollout[0].sum(dim=0)


def hybrid_cdf_chunking(
    gen_scores: torch.Tensor,
    num_blocks: int,
    lam: float = 1.0,
    inverse: bool = True,
) -> List[Tuple[int, int]]:
    gen_length = gen_scores.numel()
    if num_blocks <= 0 or num_blocks > gen_length:
        return [(0, gen_length)]

    scores = gen_scores.detach().cpu().to(torch.float64).clamp(min=0)
    total_mass = scores.sum().item()

    if total_mass < 1e-12:
        block_size = gen_length // num_blocks
        boundaries = [i * block_size for i in range(num_blocks)]
        boundaries.append(gen_length)
        boundaries = sorted(set(boundaries))
        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    if inverse:
        inv_scores = 1.0 / (scores + 1e-10)
        attn_cdf = torch.cumsum(inv_scores, dim=0) / inv_scores.sum()
    else:
        attn_cdf = torch.cumsum(scores, dim=0) / total_mass

    uniform_cdf = torch.arange(1, gen_length + 1, dtype=torch.float64) / gen_length
    hybrid_cdf = lam * attn_cdf + (1.0 - lam) * uniform_cdf

    boundaries = [0]
    for k in range(1, num_blocks):
        threshold = k / num_blocks
        candidates = torch.where(hybrid_cdf >= threshold)[0]
        boundary = candidates[0].item() + 1 if candidates.numel() > 0 else gen_length
        if boundary >= gen_length:
            break
        boundaries.append(boundary)

    if boundaries[-1] != gen_length:
        boundaries.append(gen_length)
    boundaries = sorted(set(boundaries))
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def compute_adablock_length(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    prompt_len: int,
    gen_length: int,
    generated_length: int,
    init_block_length: int,
    delimiter_ids: Sequence[int],
    delimiter_threshold: float,
) -> int:
    block_start = prompt_len + generated_length
    remaining_length = gen_length - generated_length
    window_size = min(int(0.25 * gen_length), remaining_length)

    window_tokens = predicted_tokens[0, block_start:block_start + window_size]
    delimiter_mask = torch.zeros_like(window_tokens, dtype=torch.bool)
    for token_id in delimiter_ids:
        delimiter_mask |= (window_tokens == token_id)

    if not torch.any(delimiter_mask):
        return min(init_block_length, remaining_length)

    delimiter_pos = block_start + torch.nonzero(delimiter_mask).squeeze(-1)
    delimiter_logits = logits[0, delimiter_pos, predicted_tokens[0, delimiter_pos]]
    log_sum_exp = torch.logsumexp(logits[0, delimiter_pos, :], dim=-1)
    delimiter_conf = torch.exp(delimiter_logits - log_sum_exp)

    max_conf, best_idx = torch.max(delimiter_conf, dim=0)
    best_delim_pos = delimiter_pos[best_idx].item()

    if max_conf.item() >= delimiter_threshold:
        block_len = best_delim_pos - block_start + 1
    else:
        block_len = min(init_block_length, remaining_length)
    return block_len


def get_transfer_index_threshold(
    logits: torch.Tensor,
    predicted_tokens: torch.Tensor,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x0 = predicted_tokens
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)
    transfer_index = mask_index & (confidence >= threshold)

    max_conf_idx = torch.argmax(confidence, dim=1, keepdim=True)
    force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_idx, True)
    transfer_index = (transfer_index | force_mask) & mask_index
    return x0, transfer_index


@dataclass
class SampleResult:
    sample_id: int
    prompt_text: str
    fixed_blocks: List[Tuple[int, int]]
    adablock_blocks: List[Tuple[int, int]]
    inverse_blocks: List[Tuple[int, int]]
    high_coupling_spans: List[Tuple[int, int]]
    fixed_token_ids: List[int]
    adablock_token_ids: List[int]
    inverse_token_ids: List[int]


def contiguous_spans(mask: torch.Tensor) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, mask.numel()))
    return spans


def draw_block_row(
    ax,
    blocks: List[Tuple[int, int]],
    y: float,
    label: str,
    colors: Sequence[str],
) -> None:
    import matplotlib.patches as mpatches

    for i, (s, e) in enumerate(blocks):
        rect = mpatches.Rectangle(
            (s, y - 0.28),
            e - s,
            0.56,
            facecolor=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        ax.add_patch(rect)
    ax.text(-8, y, label, ha="right", va="center", fontsize=10, fontweight="bold")


def render_sample_plot(
    out_path: str,
    sample: SampleResult,
    gen_length: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.8))
    colors = plt.cm.Set3.colors

    # High-coupling region shading
    for s, e in sample.high_coupling_spans:
        ax.axvspan(s, e, color="#ffe082", alpha=0.35, lw=0)

    draw_block_row(ax, sample.fixed_blocks, 2.0, "Fixed", colors)
    draw_block_row(ax, sample.adablock_blocks, 1.0, "AdaBlock", colors)
    draw_block_row(ax, sample.inverse_blocks, 0.0, "Inverse-CDF", colors)

    ax.set_xlim(-12, gen_length + 1)
    ax.set_ylim(-0.7, 2.7)
    ax.set_xlabel("Generated token index")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2)
    ax.set_title(f"Sample {sample.sample_id} | boundary comparison")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_blocks_html(
    out_path: str,
    sample: SampleResult,
    tokenizer,
) -> None:
    palette = [
        "#e3f2fd", "#fff3e0", "#e8f5e9", "#fce4ec",
        "#ede7f6", "#e0f2f1", "#fff8e1", "#efebe9",
        "#f1f8e9", "#f3e5f5",
    ]

    def blocks_to_html(token_ids: List[int], blocks: List[Tuple[int, int]]) -> str:
        spans = []
        for i, (s, e) in enumerate(blocks):
            ids = token_ids[s:e]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            safe = html.escape(text).replace("\n", "<br>\n")
            color = palette[i % len(palette)]
            tok_count = max(0, e - s)
            spans.append(
                f'<span class="blk" style="background:{color}" '
                f'title="Block {i}: [{s}, {e}) / {tok_count} tok">{safe}</span>'
            )
        return "".join(spans)

    fixed_html = blocks_to_html(sample.fixed_token_ids, sample.fixed_blocks)
    adablock_html = blocks_to_html(sample.adablock_token_ids, sample.adablock_blocks)
    inverse_html = blocks_to_html(sample.inverse_token_ids, sample.inverse_blocks)

    hcr_text = ", ".join([f"[{s},{e})" for s, e in sample.high_coupling_spans]) or "(none)"
    prompt_preview = html.escape(sample.prompt_text[:1200])

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Failure Mode Text Blocks - Sample {sample.sample_id}</title>
  <style>
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      max-width: 1100px; margin: 20px auto; padding: 0 16px;
      line-height: 1.45;
    }}
    .card {{
      border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; margin-bottom: 16px;
      background: #fff;
    }}
    .name {{ font-weight: 700; margin-bottom: 8px; }}
    .blk {{
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 4px;
      padding: 0 1px;
      white-space: pre-wrap;
    }}
    .meta {{ color: #444; font-size: 0.95em; }}
    code {{ background: #f6f8fa; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h2>Sample {sample.sample_id} - Text Block Coloring</h2>
  <p class="meta">
    High-coupling spans: <code>{hcr_text}</code>
  </p>
  <p class="meta">
    Hover each colored span to inspect block index and token range.
  </p>
  <div class="card">
    <div class="name">Prompt preview</div>
    <div>{prompt_preview}</div>
  </div>
  <div class="card">
    <div class="name">Fixed</div>
    <div>{fixed_html}</div>
  </div>
  <div class="card">
    <div class="name">AdaBlock</div>
    <div>{adablock_html}</div>
  </div>
  <div class="card">
    <div class="name">Inverse-CDF</div>
    <div>{inverse_html}</div>
  </div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)


def get_prompt_from_doc(task: str, doc: Dict) -> str:
    # Keep prompt format simple and explicit for boundary analysis.
    if task == "gsm8k":
        return doc["question"]
    if task == "humaneval":
        return doc["prompt"]
    if task == "minerva_math":
        return "Problem:\n" + doc["problem"] + "\n\nSolution:"
    raise ValueError(f"Unsupported task: {task}")


def load_task_split(task: str):
    if task == "gsm8k":
        return load_dataset("openai/gsm8k", "main", split="test")
    if task == "humaneval":
        return load_dataset("openai/openai_humaneval", split="test")
    if task == "minerva_math":
        return load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    raise ValueError(f"Unsupported task: {task}")


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@torch.no_grad()
def decode_with_blocks(
    model,
    x_init: torch.Tensor,
    prompt_len: int,
    blocks: List[Tuple[int, int]],
    temperature: float,
    threshold: float,
    mask_id: int,
) -> torch.Tensor:
    x = x_init.clone()
    for block_start_rel, block_end_rel in blocks:
        block_start = prompt_len + block_start_rel
        block_end = prompt_len + block_end_rel

        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

            out_step = model(x)
            logits_step = out_step.logits
            preds_step = torch.argmax(
                add_gumbel_noise(logits_step, temperature=temperature), dim=-1
            )
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = False
            x0, transfer_index = get_transfer_index_threshold(
                logits=logits_step,
                predicted_tokens=preds_step,
                mask_index=mask_index,
                x=x,
                threshold=threshold,
            )
            x[transfer_index] = x0[transfer_index]
    return x


@torch.no_grad()
def run_single_sample(
    model,
    tokenizer,
    prompt_text: str,
    sample_id: int,
    gen_length: int,
    fixed_block_length: int,
    num_blocks: int,
    lam: float,
    temperature: float,
    threshold: float,
    init_block_length: int,
    delimiter_ids: Sequence[int],
    delimiter_threshold: float,
    use_chat_template: bool,
) -> SampleResult:
    if use_chat_template:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )

    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]
    mask_id = 126336

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=model.device
    )
    x[:, :prompt_len] = input_ids

    # Step 0 forward for rollout and inverse-cdf
    outputs = model(x, output_attentions=True)
    rollout_scores = get_depth_adaptive_rollout(outputs.attentions).to(torch.float64)
    gen_scores = rollout_scores[prompt_len:prompt_len + gen_length]

    inverse_blocks = hybrid_cdf_chunking(
        gen_scores=gen_scores, num_blocks=num_blocks, lam=lam, inverse=True
    )

    # High-coupling region from top quantile scores
    score_threshold = torch.quantile(gen_scores, 0.8)
    high_mask = gen_scores >= score_threshold
    high_spans = contiguous_spans(high_mask)

    # Fixed boundaries
    assert gen_length % fixed_block_length == 0, "gen_length must divide fixed_block_length"
    fixed_blocks = [
        (i, i + fixed_block_length) for i in range(0, gen_length, fixed_block_length)
    ]

    # Decode fixed and inverse strategies with their own boundaries.
    x_fixed = decode_with_blocks(
        model=model,
        x_init=x,
        prompt_len=prompt_len,
        blocks=fixed_blocks,
        temperature=temperature,
        threshold=threshold,
        mask_id=mask_id,
    )
    x_inverse = decode_with_blocks(
        model=model,
        x_init=x,
        prompt_len=prompt_len,
        blocks=inverse_blocks,
        temperature=temperature,
        threshold=threshold,
        mask_id=mask_id,
    )

    # AdaBlock boundaries (boundary extraction by running its block loop)
    x_adablock = x.clone()
    generated_length = 0
    adablock_blocks: List[Tuple[int, int]] = []

    while generated_length < gen_length:
        out = model(x_adablock)
        logits = out.logits
        logits_noisy = add_gumbel_noise(logits, temperature=temperature)
        predicted_tokens = torch.argmax(logits_noisy, dim=-1)

        block_len = compute_adablock_length(
            logits=logits,
            predicted_tokens=predicted_tokens,
            prompt_len=prompt_len,
            gen_length=gen_length,
            generated_length=generated_length,
            init_block_length=init_block_length,
            delimiter_ids=delimiter_ids,
            delimiter_threshold=delimiter_threshold,
        )

        block_start_rel = generated_length
        block_end_rel = min(generated_length + block_len, gen_length)
        adablock_blocks.append((block_start_rel, block_end_rel))

        block_start = prompt_len + block_start_rel
        block_end = prompt_len + block_end_rel

        # Fill current block to progress to the next block (threshold-based)
        while True:
            if (x_adablock[:, block_start:block_end] == mask_id).sum() == 0:
                break
            out_step = model(x_adablock)
            logits_step = out_step.logits
            preds_step = torch.argmax(add_gumbel_noise(logits_step, temperature=temperature), dim=-1)

            mask_index = (x_adablock == mask_id)
            mask_index[:, block_end:] = False
            x0, transfer_index = get_transfer_index_threshold(
                logits=logits_step,
                predicted_tokens=preds_step,
                mask_index=mask_index,
                x=x_adablock,
                threshold=threshold,
            )
            x_adablock[transfer_index] = x0[transfer_index]

        generated_length = block_end_rel

    return SampleResult(
        sample_id=sample_id,
        prompt_text=prompt_text,
        fixed_blocks=fixed_blocks,
        adablock_blocks=adablock_blocks,
        inverse_blocks=inverse_blocks,
        high_coupling_spans=high_spans,
        fixed_token_ids=x_fixed[0, prompt_len:prompt_len + gen_length].detach().cpu().tolist(),
        adablock_token_ids=x_adablock[0, prompt_len:prompt_len + gen_length].detach().cpu().tolist(),
        inverse_token_ids=x_inverse[0, prompt_len:prompt_len + gen_length].detach().cpu().tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize failure modes: fixed vs AdaBlock vs inverse-CDF boundaries"
    )
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--task", type=str, default="humaneval", choices=["gsm8k", "humaneval", "minerva_math"])
    parser.add_argument("--sample-ids", type=str, default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--fixed-block-length", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--init-block-length", type=int, default=32)
    parser.add_argument("--delimiter-ids", type=str, default="198")
    parser.add_argument("--delimiter-threshold", type=float, default=0.3)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--format", type=str, default="both", choices=["plot", "text", "both"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results_failure_mode")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:3" if torch.cuda.is_available() else "cpu"
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    os.makedirs(args.out_dir, exist_ok=True)
    sample_ids = parse_int_list(args.sample_ids)
    delimiter_ids = parse_int_list(args.delimiter_ids)

    print(f"[Info] Loading model: {args.model} ({args.dtype}) on {args.device}")
    model = LLaDAModelLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[Info] Loading dataset: {args.task}")
    ds = load_task_split(args.task)

    all_results = []
    for sid in sample_ids:
        doc = ds[sid]
        prompt = get_prompt_from_doc(args.task, doc)
        print(f"[Run] sample_id={sid}")
        result = run_single_sample(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            sample_id=sid,
            gen_length=args.gen_length,
            fixed_block_length=args.fixed_block_length,
            num_blocks=args.num_blocks,
            lam=args.lam,
            temperature=args.temperature,
            threshold=args.threshold,
            init_block_length=args.init_block_length,
            delimiter_ids=delimiter_ids,
            delimiter_threshold=args.delimiter_threshold,
            use_chat_template=not args.no_chat_template,
        )
        all_results.append(result)

        do_plot = (args.format in ("plot", "both")) and (not args.no_plot)
        if do_plot:
            fig_path = os.path.join(args.out_dir, f"boundary_overlay_sample{sid}.png")
            try:
                render_sample_plot(fig_path, result, args.gen_length)
                print(f"[Saved] {fig_path}")
            except ModuleNotFoundError as e:
                print(f"[Warn] Plot skipped (missing dependency): {e}")

        if args.format in ("text", "both"):
            html_path = os.path.join(args.out_dir, f"boundary_overlay_sample{sid}.html")
            render_blocks_html(html_path, result, tokenizer)
            print(f"[Saved] {html_path}")

    json_path = os.path.join(args.out_dir, "boundary_overlay_metadata.json")
    serializable = [
        {
            "sample_id": r.sample_id,
            "fixed_blocks": r.fixed_blocks,
            "adablock_blocks": r.adablock_blocks,
            "inverse_blocks": r.inverse_blocks,
            "high_coupling_spans": r.high_coupling_spans,
            "fixed_token_ids": r.fixed_token_ids,
            "adablock_token_ids": r.adablock_token_ids,
            "inverse_token_ids": r.inverse_token_ids,
        }
        for r in all_results
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
