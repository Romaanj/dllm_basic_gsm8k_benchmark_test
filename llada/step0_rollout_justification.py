"""
Step 0 Deep Layer Attention Rollout Score 정당성 검증 실험
==========================================================

목적:
  Step 0에서 계산한 deep layer attention rollout score가
  생성 영역의 구조적 중요도를 사전에(pre-planning) 포착한다는 것을 입증.

실험 1 — Gini Coefficient:
  어텐션 질량 집중도를 수학적으로 정량화.
  Gini ≈ 0 → 균등 분포(모든 토큰이 1/L 어텐션),
  Gini ≈ 1 → 극단적 집중.
  0.6–0.8 범위이면 "질량 집중 현상"이 통계적으로 유의미.

실험 2 — Syntax-Aware Heatmap:
  어텐션 피크가 발생하는 토큰의 구문 속성을 분류.
  수학 도메인: 연산자(=, +), 숫자, 단위($, km)
  코드 도메인: 제어 키워드(def, if, return), 특수문자(:, (, ))
  → 구조적으로 중요한 토큰에 어텐션 질량이 편중되는지 검증.

실험 3 — Prompt-Target Cross Correlation:
  Rollout matrix에서 generation → prompt 어텐션을 추출하여
  Step 0 어텐션 피크가 프롬프트의 어느 부분과 강하게 연결되는지 분석.
  예: "How many" → 생성 영역의 숫자 위치에 질량 집중
  → 모델이 이미 '답안의 구조'를 설계하고 있다는 증거.

핵심 설계:
  토큰 생성은 global confidence argmax 1토큰 unmasking (attention 무관 베이스라인)으로 수행.
  → "step 0 attention과 독립적으로 생성된 토큰"에서 구조적 대응을 보이므로
    순환 논증(circular reasoning)을 원천 차단.

Usage:
  # 50샘플 Gini + 3샘플 상세분석 (기본)
  python step0_rollout_justification.py --num-samples 50 --detail-samples 0 1 2

  # 빠른 테스트 (5샘플)
  python step0_rollout_justification.py --num-samples 5 --detail-samples 0 1
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling_llada import LLaDAModelLM
from gsm8k_hybrid_cdf_eval import (
    StreamingRollout,
    add_gumbel_noise,
)


# ═══════════════════════════════════════════════════════════════════════════
# Step 0 Attention Extraction
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_step0_rollout(
    model,
    input_ids: torch.Tensor,
    gen_length: int,
    mask_id: int,
    rollout_mode: str = "sigmoid",
    return_matrix: bool = False,
) -> Dict[str, Any]:
    """Step 0 forward pass → StreamingRollout으로 rollout score 및 matrix 추출.

    return_matrix=True이면 (seq_len, seq_len) 행렬도 반환하여
    prompt-target cross correlation 분석에 사용.
    """
    device = model.device
    prompt_len = input_ids.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = input_ids.clone()

    torch.cuda.empty_cache()

    invert_depth = rollout_mode == "sigmoid_inverted"
    hook_mode = "baseline" if rollout_mode == "baseline" else "sigmoid"

    core_model = model.model if hasattr(model, "model") else model
    blocks_list = core_model.transformer.blocks
    num_layers = len(blocks_list)

    streaming = StreamingRollout(
        num_layers=num_layers, mode=hook_mode, invert_depth=invert_depth,
    )
    streaming.register(blocks_list)

    try:
        outputs = model(x, output_attentions=True)
    finally:
        streaming.remove()

    scores = streaming.get_scores()

    result: Dict[str, Any] = {
        "prompt_len": prompt_len,
    }

    if scores is not None:
        scores_f64 = scores.to(torch.float64)
        result["scores"] = scores_f64
        result["gen_scores"] = scores_f64[prompt_len: prompt_len + gen_length].clone()
    else:
        result["scores"] = None
        result["gen_scores"] = None

    if return_matrix and streaming.rollout is not None:
        result["matrix"] = streaming.rollout[0].to(torch.float64).clone()

    del outputs, streaming
    torch.cuda.empty_cache()

    return result


@torch.no_grad()
def generate_global_confidence_baseline(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    mask_id: int,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """Global confidence argmax로 매 step 1개 토큰만 unmask."""
    device = model.device
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length), mask_id,
        dtype=torch.long, device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    nfe = 0
    for _ in range(gen_length):
        mask_idx = (x == mask_id)
        mask_idx[:, :prompt_len] = False
        if mask_idx.sum().item() == 0:
            break

        outputs = model(x)
        logits = outputs.logits

        logits_noisy = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_noisy, dim=-1)

        probs = torch.softmax(logits.to(torch.float64), dim=-1)
        score = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        x0 = torch.where(mask_idx, x0, x)

        neg_inf = torch.tensor(
            torch.finfo(score.dtype).min, device=device, dtype=score.dtype
        )
        confidence = torch.where(mask_idx, score, neg_inf)
        unmask_abs = int(torch.argmax(confidence, dim=1)[0].item())
        x[0, unmask_abs] = x0[0, unmask_abs]
        nfe += 1

    info = {
        "decode_policy": "global confidence argmax, one token per step",
    }
    return x, nfe, info


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Gini Coefficient
# ═══════════════════════════════════════════════════════════════════════════

def compute_gini(scores: torch.Tensor) -> float:
    """1D scores 벡터의 Gini Coefficient.

    G = (2 Σ_i i·y_sorted[i]) / (n·Σy) − (n+1)/n
    """
    y = scores.detach().cpu().to(torch.float64).clamp(min=0)
    n = y.numel()
    if n == 0 or y.sum() < 1e-12:
        return 0.0

    sorted_y, _ = torch.sort(y)
    index = torch.arange(1, n + 1, dtype=torch.float64)
    gini = (2.0 * (index * sorted_y).sum() / (n * sorted_y.sum())) - (n + 1.0) / n
    return float(gini)


def compute_topk_concentration(scores: torch.Tensor, k_ratios: List[float] = None) -> Dict[str, float]:
    """Top-k% 위치가 전체 질량의 몇 %를 점유하는지 계산.

    k_ratios: e.g. [0.05, 0.10, 0.20] → top-5%, top-10%, top-20%
    """
    if k_ratios is None:
        k_ratios = [0.05, 0.10, 0.20]

    s = scores.detach().cpu().to(torch.float64).clamp(min=0)
    n = s.numel()
    total = s.sum().item()
    if total < 1e-12:
        return {f"top_{int(r*100)}pct": 0.0 for r in k_ratios}

    sorted_s, _ = torch.sort(s, descending=True)
    result = {}
    for r in k_ratios:
        k = max(1, int(n * r))
        mass = sorted_s[:k].sum().item() / total
        result[f"top_{int(r*100)}pct"] = mass
    return result


def plot_gini_distribution(
    gini_values: List[float],
    concentration: Dict[str, List[float]],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mean_g = np.mean(gini_values)
    std_g = np.std(gini_values)

    # Panel 0: Histogram
    ax = axes[0]
    ax.hist(gini_values, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(mean_g, color="red", linewidth=2, linestyle="--",
               label=f"Mean = {mean_g:.4f}")
    ax.axvspan(0.6, 0.8, alpha=0.15, color="green", label="Target [0.6, 0.8]")
    ax.set_xlabel("Gini Coefficient")
    ax.set_ylabel("Count")
    ax.set_title("Step 0 Attention Rollout — Gini Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 1: Box plot
    ax = axes[1]
    bp = ax.boxplot(gini_values, vert=True, widths=0.5, patch_artist=True)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.6)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title(
        f"mean={mean_g:.4f} ± {std_g:.4f}\n"
        f"min={min(gini_values):.4f}, max={max(gini_values):.4f}"
    )
    ax.grid(True, alpha=0.3)

    # Panel 2: Concentration ratios
    ax = axes[2]
    for key, vals in concentration.items():
        label = key.replace("_", " ").replace("pct", "%")
        ax.hist(vals, bins=25, alpha=0.5, label=f"{label}: mean={np.mean(vals):.3f}")
    ax.set_xlabel("Mass Fraction")
    ax.set_ylabel("Count")
    ax.set_title("Top-k% Attention Mass Concentration")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] Gini distribution → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Syntax-Aware Token Classification & Heatmap
# ═══════════════════════════════════════════════════════════════════════════

MATH_OPERATORS = {"=", "+", "-", "*", "/", "×", "÷", "<", ">", "≤", "≥", "≠", "==", "!=", "<=", ">="}
MATH_UNITS = {
    "$", "km", "miles", "mile", "hours", "hour", "hr", "min", "minutes", "minute",
    "days", "day", "weeks", "week", "months", "month", "years", "year",
    "kg", "lb", "lbs", "oz", "cm", "m", "ft", "inch", "inches",
    "%", "percent", "dollars", "dollar", "cents", "cent", "each", "per", "total",
}
CODE_KEYWORDS = {
    "def", "if", "else", "elif", "return", "for", "while", "class",
    "import", "from", "with", "as", "try", "except", "finally",
    "yield", "lambda", "pass", "break", "continue", "print",
    "and", "or", "not", "in", "is", "True", "False", "None",
}


def classify_token(token_str: str) -> str:
    """토큰 문자열을 구문 카테고리로 분류.

    Returns one of:
      number, operator, unit, keyword, punctuation, delimiter, text, special
    """
    t = token_str.strip()
    t_clean = t.lstrip("ĠĊ▁")

    if not t_clean:
        return "special"

    
    if "####" in t_clean or t_clean in ("<<", ">>"):
        return "delimiter"
    if t_clean in ("\n", "\\n") or t_clean.startswith("<|") or t_clean.startswith("<|"):
        return "delimiter"

    if re.match(r"^-?\d[\d,]*\.?\d*$", t_clean):
        return "number"

    if t_clean in MATH_OPERATORS:
        return "operator"

    if t_clean.lower() in MATH_UNITS:
        return "unit"

    if t_clean.lower() in CODE_KEYWORDS:
        return "keyword"

    if all(c in "()[]{}:;,.<>!?@#$%^&*-+=/" for c in t_clean):
        return "punctuation"

    return "text"


CATEGORY_COLORS = {
    "number":      "#E74C3C",
    "operator":    "#E67E22",
    "unit":        "#9B59B6",
    "keyword":     "#2ECC71",
    "punctuation": "#3498DB",
    "delimiter":   "#1ABC9C",
    "text":        "#BDC3C7",
    "special":     "#7F8C8D",
}

STRUCTURAL_CATEGORIES = {"number", "operator", "unit", "keyword", "punctuation", "delimiter"}


def plot_syntax_heatmap(
    gen_scores: torch.Tensor,
    token_strings: List[str],
    categories: List[str],
    out_path: str,
    top_k: int = 20,
    title_suffix: str = "",
) -> None:
    """Attention score bar chart + 토큰 구문 카테고리 색상 코딩."""
    scores_np = gen_scores.detach().cpu().numpy()
    n = len(scores_np)
    topk_idx = np.argsort(scores_np)[-top_k:][::-1]
    topk_sorted = sorted(topk_idx)

    fig, axes = plt.subplots(3, 1, figsize=(16, 13))

    # Panel 0: Full attention score colored by category
    ax = axes[0]
    colors = [CATEGORY_COLORS.get(c, "#BDC3C7") for c in categories]
    ax.bar(range(n), scores_np, color=colors, width=1.0, edgecolor="none")
    for idx in topk_idx:
        ax.plot(idx, scores_np[idx], "v", color="black", markersize=5)
    handles = [mpatches.Patch(color=col, label=cat) for cat, col in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=4)
    ax.set_ylabel("Attention Rollout Score")
    ax.set_xlabel("Generation Token Index")
    ax.set_title(f"Syntax-Aware Attention Heatmap {title_suffix}")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 1: Top-k peak tokens detail (horizontal bar)
    ax = axes[1]
    labels = [f"[{i}] {token_strings[i]}" for i in topk_sorted]
    vals = [scores_np[i] for i in topk_sorted]
    bar_colors = [CATEGORY_COLORS.get(categories[i], "#BDC3C7") for i in topk_sorted]
    ax.barh(range(len(topk_sorted)), vals, color=bar_colors,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(topk_sorted)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Attention Rollout Score")
    ax.set_title(f"Top-{top_k} Peak Tokens by Syntax Category")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: Attention mass enrichment (mass%/count% per category)
    ax = axes[2]
    total_mass = scores_np.sum()
    cat_mass_frac = defaultdict(float)
    cat_count = Counter(categories)
    for i, c in enumerate(categories):
        cat_mass_frac[c] += scores_np[i] / total_mass if total_mass > 0 else 0

    all_cats = sorted(CATEGORY_COLORS.keys())
    mass_pcts = [cat_mass_frac.get(c, 0) * 100 for c in all_cats]
    count_pcts = [cat_count.get(c, 0) / n * 100 for c in all_cats]
    enrichment = [m / max(cp, 0.01) for m, cp in zip(mass_pcts, count_pcts)]

    x_bar = np.arange(len(all_cats))
    w = 0.28
    ax.bar(x_bar - w, mass_pcts, w, label="Attention Mass %",
           color=[CATEGORY_COLORS[c] for c in all_cats], edgecolor="black", linewidth=0.5)
    ax.bar(x_bar, count_pcts, w, label="Token Count %",
           color=[CATEGORY_COLORS[c] for c in all_cats], alpha=0.4,
           edgecolor="black", linewidth=0.5)

    ax2 = ax.twinx()
    ax2.plot(x_bar + w, enrichment, "D-", color="red", markersize=5, label="Enrichment (mass/count)")
    ax2.axhline(1.0, color="red", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Enrichment Ratio", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax.set_xticks(x_bar)
    ax.set_xticklabels(all_cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Attention Mass vs Token Count by Category (Enrichment)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] Syntax heatmap → {out_path}")


def plot_category_attention_summary(
    all_categories: List[List[str]],
    all_scores: List[torch.Tensor],
    out_path: str,
) -> None:
    """여러 샘플에 대해 카테고리별 어텐션 질량/카운트 비율 및 enrichment 집계."""
    n_samples = len(all_categories)

    per_sample_mass = defaultdict(list)
    per_sample_count = defaultdict(list)

    for cats, scores in zip(all_categories, all_scores):
        s = scores.detach().cpu().numpy()
        total = s.sum()
        n = len(cats)
        cat_m = defaultdict(float)
        cat_c = Counter(cats)
        for i, c in enumerate(cats):
            cat_m[c] += s[i] / total if total > 0 else 0
        for c in CATEGORY_COLORS:
            per_sample_mass[c].append(cat_m.get(c, 0.0))
            per_sample_count[c].append(cat_c.get(c, 0) / n)

    all_cats = sorted(CATEGORY_COLORS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 0: Average attention mass fraction
    ax = axes[0]
    means_mass = [np.mean(per_sample_mass[c]) * 100 for c in all_cats]
    stds_mass = [np.std(per_sample_mass[c]) * 100 for c in all_cats]
    bar_colors = [CATEGORY_COLORS[c] for c in all_cats]
    ax.bar(all_cats, means_mass, yerr=stds_mass, color=bar_colors,
           edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_ylabel("Attention Mass (%)")
    ax.set_title(f"Avg Attention Mass by Category (n={n_samples})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Panel 1: Average token count fraction
    ax = axes[1]
    means_count = [np.mean(per_sample_count[c]) * 100 for c in all_cats]
    stds_count = [np.std(per_sample_count[c]) * 100 for c in all_cats]
    ax.bar(all_cats, means_count, yerr=stds_count, color=bar_colors,
           edgecolor="black", linewidth=0.5, capsize=3, alpha=0.6)
    ax.set_ylabel("Token Count (%)")
    ax.set_title(f"Avg Token Count by Category (n={n_samples})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Panel 2: Enrichment ratio (mass%/count%) per category
    ax = axes[2]
    enrichments = {c: [] for c in all_cats}
    for i in range(n_samples):
        for c in all_cats:
            m = per_sample_mass[c][i]
            cnt = per_sample_count[c][i]
            enrichments[c].append(m / cnt if cnt > 0.001 else 0.0)
    means_enrich = [np.mean(enrichments[c]) for c in all_cats]
    stds_enrich = [np.std(enrichments[c]) for c in all_cats]
    ax.bar(all_cats, means_enrich, yerr=stds_enrich, color=bar_colors,
           edgecolor="black", linewidth=0.5, capsize=3)
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.5, label="Enrichment = 1.0")
    ax.set_ylabel("Enrichment (mass/count)")
    ax.set_title(f"Attention Enrichment by Category (n={n_samples})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] Category attention summary → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Prompt-Target Cross Correlation
# ═══════════════════════════════════════════════════════════════════════════

def compute_prompt_target_attention(
    rollout_matrix: torch.Tensor,
    prompt_len: int,
    gen_length: int,
) -> torch.Tensor:
    """Rollout matrix에서 generation → prompt 어텐션 부분행렬 추출.

    Returns: (gen_length, prompt_len)
    rollout_matrix[i, j] = position i가 position j에 갖는 누적 어텐션.
    """
    return rollout_matrix[prompt_len: prompt_len + gen_length, :prompt_len].clone()


QUESTION_WORDS = {
    "how", "what", "which", "who", "where", "when", "why",
    "many", "much", "total", "find", "calculate", "compute",
    "each", "every", "all", "if", "more", "less", "than",
    "cost", "price", "number", "amount", "sum", "difference",
}


def compute_prompt_keyword_attention(
    cross_attn: torch.Tensor,
    prompt_tokens: List[str],
    gen_scores: torch.Tensor,
    top_k_gen: int = 10,
) -> Dict[str, Any]:
    """Top-k generation 피크 위치에서 프롬프트 토큰 중 어느 토큰에 집중하는지 분석."""
    scores_np = gen_scores.detach().cpu().numpy()
    topk_gen_idx = np.argsort(scores_np)[-top_k_gen:][::-1]

    cross_np = cross_attn.detach().cpu().numpy()
    avg_prompt_attn = cross_np[topk_gen_idx, :].mean(axis=0)

    top_prompt = np.argsort(avg_prompt_attn)[::-1][:20]
    top_prompt_tokens = [
        {
            "index": int(i),
            "token": prompt_tokens[i] if i < len(prompt_tokens) else f"[{i}]",
            "attention": float(avg_prompt_attn[i]),
        }
        for i in top_prompt
    ]

    keyword_attn = defaultdict(float)
    for i, tok in enumerate(prompt_tokens):
        clean = tok.replace("Ġ", "").replace("▁", "").lower().strip()
        if clean in QUESTION_WORDS:
            keyword_attn[clean] += float(avg_prompt_attn[i])

    return {
        "top_prompt_tokens": top_prompt_tokens,
        "keyword_attention": dict(keyword_attn),
    }


def plot_prompt_target_heatmap(
    cross_attn: torch.Tensor,
    prompt_tokens: List[str],
    gen_scores: torch.Tensor,
    gen_token_strings: Optional[List[str]],
    out_path: str,
    top_k_gen: int = 10,
    title_suffix: str = "",
) -> None:
    """Generation top-k 피크 위치가 프롬프트의 어느 부분에 집중하는지 시각화."""
    scores_np = gen_scores.detach().cpu().numpy()
    topk_gen_idx = np.argsort(scores_np)[-top_k_gen:][::-1]
    topk_sorted = sorted(topk_gen_idx)

    cross_np = cross_attn.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Panel 0: Aggregated prompt importance
    ax = axes[0]
    avg_prompt_attn = cross_np.mean(axis=0)
    ax.bar(range(len(avg_prompt_attn)), avg_prompt_attn,
           color="steelblue", width=1.0)

    top5_prompt = np.argsort(avg_prompt_attn)[-7:][::-1]
    for idx in top5_prompt:
        if idx >= len(prompt_tokens):
            continue
        label = prompt_tokens[idx].replace("Ġ", " ").replace("▁", " ").strip()
        ax.annotate(
            label, (idx, avg_prompt_attn[idx]),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7, rotation=45, color="red",
            arrowprops=dict(arrowstyle="-", color="red", alpha=0.5),
        )

    ax.set_xlabel("Prompt Token Index")
    ax.set_ylabel("Average Attention from Gen")
    ax.set_title(f"Prompt Token Importance (avg over all gen positions) {title_suffix}")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 1: Heatmap — top-k gen peaks → prompt tokens
    ax = axes[1]
    sub_matrix = cross_np[topk_sorted, :]
    row_sums = sub_matrix.sum(axis=1, keepdims=True)
    sub_matrix_norm = sub_matrix / np.maximum(row_sums, 1e-12)

    im = ax.imshow(sub_matrix_norm, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    y_labels = []
    for i in topk_sorted:
        if gen_token_strings and i < len(gen_token_strings):
            tok = gen_token_strings[i][:10]
            y_labels.append(f"gen[{i}] '{tok}'")
        else:
            y_labels.append(f"gen[{i}]")
    ax.set_yticks(range(len(topk_sorted)))
    ax.set_yticklabels(y_labels, fontsize=8)

    step = max(1, len(prompt_tokens) // 40)
    xticks = list(range(0, len(prompt_tokens), step))
    xlabels = [prompt_tokens[i].replace("Ġ", " ").replace("▁", " ").strip()[:10]
               for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=6, rotation=45, ha="right")
    ax.set_title(f"Top-{top_k_gen} Gen Peaks → Prompt (row-normalized)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: Question keyword attention aggregation
    ax = axes[2]
    keyword_attn = defaultdict(float)
    avg_from_peaks = cross_np[list(topk_sorted), :].mean(axis=0)
    for i, tok in enumerate(prompt_tokens):
        clean = tok.replace("Ġ", "").replace("▁", "").lower().strip()
        if clean in QUESTION_WORDS:
            keyword_attn[clean] += float(avg_from_peaks[i])

    if keyword_attn:
        sorted_kw = sorted(keyword_attn.items(), key=lambda x: -x[1])
        kw_names = [k for k, _ in sorted_kw]
        kw_vals = [v for _, v in sorted_kw]
        ax.barh(range(len(kw_names)), kw_vals, color="coral", edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(kw_names)))
        ax.set_yticklabels(kw_names, fontsize=9)
        ax.set_xlabel("Attention from Top-k Gen Peaks")
        ax.set_title("Question Keyword Attention (from gen peaks → prompt)")
    else:
        ax.text(0.5, 0.5, "No question keywords found in prompt",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("Question Keyword Attention")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Saved] Prompt-Target cross correlation → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Step 0 Deep Layer Attention Rollout Score 정당성 검증 실험",
    )
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-samples", type=int, default=50,
                   help="Gini 분석에 사용할 GSM8K 샘플 수")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--rollout-mode", type=str, default="sigmoid",
                   choices=["sigmoid", "sigmoid_inverted", "baseline"])
    p.add_argument("--top-k-peaks", type=int, default=15)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--out-dir", type=str, default="results_step0_justification")
    p.add_argument("--detail-samples", type=int, nargs="+", default=[0, 1, 2],
                   help="상세 시각화(syntax heatmap + cross correlation)를 생성할 샘플 ID")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    os.makedirs(args.out_dir, exist_ok=True)

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

    num_samples = min(args.num_samples, len(ds))
    detail_set = set(args.detail_samples)

    # ── Aggregation buffers ──
    gini_values: List[float] = []
    concentration_buf: Dict[str, List[float]] = defaultdict(list)
    all_categories: List[List[str]] = []
    all_gen_scores: List[torch.Tensor] = []
    sample_records: List[Dict] = []

    for sid in tqdm(range(num_samples), desc="Analyzing samples"):
        sample = ds[int(sid)]
        question = sample["question"]

        if args.no_chat_template:
            prompt_str = question
        else:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True, tokenize=False,
            )

        input_ids = tokenizer(prompt_str, return_tensors="pt")["input_ids"].to(model.device)
        prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        need_matrix = sid in detail_set

        # ── Step 0 Rollout ──
        rollout_data = get_step0_rollout(
            model=model,
            input_ids=input_ids,
            gen_length=args.gen_length,
            mask_id=args.mask_id,
            rollout_mode=args.rollout_mode,
            return_matrix=need_matrix,
        )

        gen_scores = rollout_data["gen_scores"]
        prompt_len = rollout_data["prompt_len"]

        if gen_scores is None:
            print(f"  [WARNING] Sample {sid}: rollout returned None, skipping")
            continue

        # ── Experiment 1: Gini ──
        gini = compute_gini(gen_scores)
        gini_values.append(gini)

        conc = compute_topk_concentration(gen_scores)
        for k, v in conc.items():
            concentration_buf[k].append(v)

        # ── Generate actual tokens (attention-independent baseline) ──
        # global confidence argmax 1-token unmasking: step 0 attention과 완전 독립.
        # 이렇게 해야 "attention 피크 ↔ 구조적 토큰" 대응이 순환 논증이 아님을 보장.
        gen_output, nfe, gen_info = generate_global_confidence_baseline(
            model=model,
            prompt=input_ids,
            gen_length=args.gen_length,
            mask_id=args.mask_id,
            temperature=args.temperature,
        )

        gen_token_ids = gen_output[0, prompt_len: prompt_len + args.gen_length]
        gen_token_strs = tokenizer.convert_ids_to_tokens(gen_token_ids.tolist())
        gen_decoded = [
            tokenizer.decode([int(tid)], skip_special_tokens=False).strip() or raw
            for raw, tid in zip(gen_token_strs, gen_token_ids.tolist())
        ]

        # ── Experiment 2: Syntax Classification ──
        categories = [classify_token(t) for t in gen_token_strs]
        all_categories.append(categories)
        all_gen_scores.append(gen_scores.cpu().clone())

        # structural vs text attention mass ratio
        s_np = gen_scores.cpu().numpy()
        total_mass = s_np.sum()
        structural_mass = sum(
            s_np[i] for i, c in enumerate(categories) if c in STRUCTURAL_CATEGORIES
        )
        structural_ratio = structural_mass / total_mass if total_mass > 0 else 0

        record: Dict[str, Any] = {
            "sample_id": sid,
            "question": question[:200],
            "gini": gini,
            "concentration": conc,
            "prompt_len": prompt_len,
            "structural_attention_ratio": float(structural_ratio),
        }

        # ── Experiment 3: Cross Correlation (detail samples) ──
        if need_matrix and "matrix" in rollout_data:
            cross_attn = compute_prompt_target_attention(
                rollout_data["matrix"], prompt_len, args.gen_length,
            )

            cross_result = compute_prompt_keyword_attention(
                cross_attn, prompt_tokens, gen_scores,
                top_k_gen=args.top_k_peaks,
            )
            record["cross_correlation"] = cross_result

            plot_prompt_target_heatmap(
                cross_attn=cross_attn,
                prompt_tokens=prompt_tokens,
                gen_scores=gen_scores,
                gen_token_strings=gen_decoded,
                out_path=os.path.join(args.out_dir, f"cross_correlation_id{sid}.png"),
                top_k_gen=args.top_k_peaks,
                title_suffix=f"(sample={sid})",
            )

            del cross_attn

        # Detail syntax heatmap
        if sid in detail_set:
            plot_syntax_heatmap(
                gen_scores=gen_scores,
                token_strings=gen_decoded,
                categories=categories,
                out_path=os.path.join(args.out_dir, f"syntax_heatmap_id{sid}.png"),
                top_k=args.top_k_peaks,
                title_suffix=f"(sample={sid})",
            )

        sample_records.append(record)

        if (sid + 1) % 10 == 0:
            print(f"  [{sid+1}/{num_samples}] Gini mean={np.mean(gini_values):.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Aggregate Results
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 0 ATTENTION ROLLOUT JUSTIFICATION — RESULTS SUMMARY")
    print("=" * 70)

    # ── Exp 1: Gini ──
    if gini_values:
        mean_g = np.mean(gini_values)
        std_g = np.std(gini_values)
        in_range = sum(1 for g in gini_values if 0.6 <= g <= 0.8)
        print(f"\n[Exp 1] Gini Coefficient (n={len(gini_values)})")
        print(f"  Mean ± Std : {mean_g:.4f} ± {std_g:.4f}")
        print(f"  Median     : {np.median(gini_values):.4f}")
        print(f"  Range      : [{min(gini_values):.4f}, {max(gini_values):.4f}]")
        print(f"  In [0.6,0.8]: {in_range}/{len(gini_values)} ({100*in_range/len(gini_values):.1f}%)")
        for k, vals in concentration_buf.items():
            label = k.replace("_", " ").replace("pct", "%")
            print(f"  {label} mass : {np.mean(vals)*100:.1f}% ± {np.std(vals)*100:.1f}%")

        plot_gini_distribution(
            gini_values, dict(concentration_buf),
            out_path=os.path.join(args.out_dir, "gini_distribution.png"),
        )

    # ── Exp 2: Category ──
    if all_categories and all_gen_scores:
        print(f"\n[Exp 2] Syntax-Aware Analysis (n={len(all_categories)})")

        cat_mass_all = defaultdict(list)
        cat_count_all = defaultdict(list)
        structural_ratios = []

        for cats, scores in zip(all_categories, all_gen_scores):
            s = scores.numpy()
            total = s.sum()
            n = len(cats)
            cm = defaultdict(float)
            cc = Counter(cats)
            for i, c in enumerate(cats):
                cm[c] += s[i] / total if total > 0 else 0
            for c in CATEGORY_COLORS:
                cat_mass_all[c].append(cm.get(c, 0.0))
                cat_count_all[c].append(cc.get(c, 0) / n)

            struct_m = sum(cm.get(c, 0) for c in STRUCTURAL_CATEGORIES)
            structural_ratios.append(struct_m)

        print(f"\n  {'Category':12s} | {'Mass%':>8s} | {'Count%':>8s} | {'Enrichment':>10s}")
        print("  " + "-" * 48)
        for c in sorted(CATEGORY_COLORS.keys()):
            avg_m = np.mean(cat_mass_all[c]) * 100
            avg_c = np.mean(cat_count_all[c]) * 100
            enrich = avg_m / avg_c if avg_c > 0.01 else 0
            marker = " ★" if enrich > 1.5 else ""
            print(f"  {c:12s} | {avg_m:7.2f}% | {avg_c:7.2f}% | {enrich:9.2f}x{marker}")

        avg_struct = np.mean(structural_ratios) * 100
        print(f"\n  Structural tokens attention mass: {avg_struct:.1f}% (avg)")

        plot_category_attention_summary(
            all_categories, all_gen_scores,
            out_path=os.path.join(args.out_dir, "category_attention_summary.png"),
        )

    # ── Exp 3: Cross Correlation ──
    cross_records = [r for r in sample_records if "cross_correlation" in r]
    if cross_records:
        print(f"\n[Exp 3] Prompt-Target Cross Correlation (n={len(cross_records)})")
        for r in cross_records:
            print(f"\n  Sample {r['sample_id']}: \"{r['question'][:80]}...\"")
            kw_attn = r["cross_correlation"]["keyword_attention"]
            if kw_attn:
                sorted_kw = sorted(kw_attn.items(), key=lambda x: -x[1])[:5]
                print(f"    Keyword attention: {dict(sorted_kw)}")
            top5 = r["cross_correlation"]["top_prompt_tokens"][:5]
            for t in top5:
                tok_clean = t["token"].replace("Ġ", " ").replace("▁", " ").strip()
                print(f"    prompt[{t['index']:3d}] '{tok_clean}' → attn={t['attention']:.6f}")

    # ── Save JSON ──
    summary = {
        "args": vars(args),
        "gini": {
            "values": gini_values,
            "mean": float(np.mean(gini_values)) if gini_values else None,
            "std": float(np.std(gini_values)) if gini_values else None,
            "median": float(np.median(gini_values)) if gini_values else None,
            "min": float(min(gini_values)) if gini_values else None,
            "max": float(max(gini_values)) if gini_values else None,
        },
        "concentration": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                          for k, v in concentration_buf.items()},
        "samples": sample_records,
    }

    summary_path = os.path.join(args.out_dir, "step0_justification_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Saved] Full results → {summary_path}")
    print(f"[Saved] All plots  → {args.out_dir}/")


if __name__ == "__main__":
    main()
