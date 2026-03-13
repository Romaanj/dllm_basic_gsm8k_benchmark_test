"""
Equal-Mass Wins 블록 경계 시각화
===============================
equal_mass가 맞고 fixed_block이 틀린 케이스에서,
두 전략의 블록 분할 차이를 시각화합니다.

- fixed_block: [32]*8 고정 블록
- equal_mass: CSV의 block_sizes (CDF 기반 동적 블록)
- tokenizer: GSAI-ML/LLaDA-8B-Instruct
"""

import argparse
import ast
import csv
import html
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from transformers import AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_wins(wins_path: str) -> List[Dict]:
    with open(wins_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_block_sizes_by_id(csv_path: str) -> Dict[int, List[int]]:
    """CSV에서 equal_mass_sigmoid 행의 block_sizes를 id별로 로드"""
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["strategy"] != "equal_mass_sigmoid":
                continue
            sid = int(row["id"])
            bs = row.get("block_sizes", "").strip()
            if not bs:
                continue
            try:
                out[sid] = ast.literal_eval(bs)
            except (ValueError, SyntaxError):
                continue
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def _draw_block_row(
    ax,
    block_sizes: List[int],
    y_pos: float,
    label: str,
    colors: List[str],
    gen_length: int,
    show_labels: bool = True,
):
    """한 전략의 블록 경계를 수평 막대로 그림"""
    cum = 0
    for i, sz in enumerate(block_sizes):
        left = cum
        width = sz
        cum += sz
        color = colors[i % len(colors)]
        rect = mpatches.Rectangle(
            (left, y_pos - 0.35), width, 0.7, facecolor=color, edgecolor="black", linewidth=0.5
        )
        ax.add_patch(rect)
        if show_labels and sz >= 24:
            ax.text(left + width / 2, y_pos, str(sz), ha="center", va="center", fontsize=9)
    ax.text(-15, y_pos, label, ha="right", va="center", fontsize=11, fontweight="bold")


def get_text_blocks(
    tokenizer, gen_text: str, block_sizes: List[int]
) -> List[Tuple[str, int]]:
    """토큰화 후 block_sizes로 잘라 각 블록의 실제 텍스트 반환. (텍스트, 토큰수) 리스트."""
    enc = tokenizer(gen_text, return_tensors="pt", add_special_tokens=False)
    token_ids = enc["input_ids"][0].tolist()
    result = []
    cum = 0
    for sz in block_sizes:
        chunk_ids = token_ids[cum : cum + sz]
        cum += sz
        if not chunk_ids:
            result.append(("", 0))
            continue
        text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        result.append((text, len(chunk_ids)))
        if cum >= len(token_ids):
            break
    return result


def export_text_blocks_html(
    wins: List[Dict],
    block_sizes_map: Dict[int, List[int]],
    tokenizer,
    out_path: str,
    max_samples: int = 12,
    wins_type: str = "equal_mass",
    block_colors: Tuple[str, ...] = (
        "#e3f2fd", "#fff3e0", "#e8f5e9", "#fce4ec",
        "#f3e5f5", "#e0f2f1", "#fff8e1", "#efebe9",
    ),
) -> None:
    """각 샘플의 equal_mass_gen_text를 블록별로 색칠한 HTML 출력."""
    samples = []
    for w in wins:
        sid = w["sample_id"]
        if sid not in block_sizes_map:
            continue
        samples.append((w, block_sizes_map[sid]))
    if max_samples > 0:
        samples = samples[:max_samples]

    fixed_block_sizes = [32] * 8

    def _render_blocks(blocks: List[Tuple[str, int]], strategy_name: str) -> str:
        spans = []
        for i, (text, n_tok) in enumerate(blocks):
            color = block_colors[i % len(block_colors)]
            escaped = html.escape(text).replace("\n", "<br>\n")
            label = f"Block {i} ({n_tok} tok)"
            spans.append(
                f'<span class="block" style="background:{color}" title="{label}">{escaped}</span>'
            )
        return "".join(spans)

    rows = []
    for w, block_sizes in samples:
        question = w.get("question", "")
        gold = w.get("gold", "")

        em_text = w.get("equal_mass_gen_text", "")
        em_blocks = get_text_blocks(tokenizer, em_text, block_sizes)
        em_pred = w.get("equal_mass_pred", "")
        em_html = _render_blocks(em_blocks, "equal_mass")

        fx_text = w.get("fixed_gen_text", "")
        fx_blocks = get_text_blocks(tokenizer, fx_text, fixed_block_sizes)
        fx_pred = w.get("fixed_pred", "")
        fx_html = _render_blocks(fx_blocks, "fixed_block")

        rows.append(f"""
        <div class="sample">
            <h3>Sample {w["sample_id"]} | gold={gold}</h3>
            <p class="question"><b>Q:</b> {html.escape(question)}</p>
            <div class="strategy-section">
                <h4>equal_mass (pred={em_pred} {"✓" if str(em_pred) == str(gold) else ""})</h4>
                <p class="block-sizes">block_sizes: {block_sizes}</p>
                <div class="text-blocks">{em_html}</div>
            </div>
            <div class="strategy-section">
                <h4>fixed_block (pred={fx_pred} {"✓" if str(fx_pred) == str(gold) else ""})</h4>
                <p class="block-sizes">block_sizes: [32, 32, 32, 32, 32, 32, 32, 32]</p>
                <div class="text-blocks">{fx_html}</div>
            </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{"Fixed" if wins_type == "fixed" else "Equal-Mass"} Block Text Division</title>
    <style>
        body {{ font-family: sans-serif; max-width: 900px; margin: 20px auto; padding: 0 20px; }}
        .sample {{ margin: 30px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; }}
        .question {{ color: #333; margin-bottom: 12px; }}
        .strategy-section {{ margin: 20px 0; padding: 12px; background: #f9f9f9; border-radius: 6px; }}
        .strategy-section h4 {{ margin-top: 0; color: #333; }}
        .block-sizes {{ font-size: 0.9em; color: #666; margin-bottom: 8px; }}
        .text-blocks {{ line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; }}
        .text-blocks .block {{ padding: 0 2px; border-radius: 2px; border: 1px solid rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>{'Fixed Block Wins' if wins_type == 'fixed' else 'Equal-Mass Wins'}: 텍스트 블록 구간 비교</h1>
    <p>생성된 글자가 블록별로 어떻게 나뉘는지 표시 (색상=블록, 동일 색=동일 블록 인덱스)</p>
    {"".join(rows)}
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML 저장: {out_path}")


def plot_sample(
    ax,
    sample_id: int,
    equal_mass_blocks: List[int],
    gen_length: int = 256,
    n_blocks: int = 8,
):
    fixed_blocks = [32] * n_blocks
    colors = plt.cm.tab10.colors

    # equal_mass
    _draw_block_row(ax, equal_mass_blocks, 1.0, "equal_mass", colors, gen_length)
    # fixed
    _draw_block_row(ax, fixed_blocks, 0.0, "fixed_block", colors, gen_length)

    ax.set_xlim(-20, gen_length + 5)
    ax.set_ylim(-0.6, 1.6)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title(f"Sample {sample_id}", fontsize=10)


def main():
    parser = argparse.ArgumentParser(description="Equal-Mass Wins 블록 경계 시각화")
    parser.add_argument(
        "--wins",
        type=str,
        default=None,
        help="wins JSON 경로 (미지정시 --wins-type에 따라 자동 설정)",
    )
    parser.add_argument(
        "--wins-type",
        type=str,
        choices=["equal_mass", "fixed"],
        default="equal_mass",
        help="equal_mass: equal_mass 정답 케이스 / fixed: fixed_block 정답 케이스",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results_equal_mass", "gsm8k_equal_mass_N8_spb32_th0.9.csv"),
        help="gsm8k_equal_mass CSV 경로",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="출력 경로 (미지정시 --wins-type에 따라 자동 설정)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Tokenizer 모델 (블록당 토큰 미리보기용)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12,
        help="시각화할 최대 샘플 수 (0=전체)",
    )
    parser.add_argument("--gen-length", type=int, default=256, help="생성 길이 (토큰)")
    parser.add_argument(
        "--format",
        type=str,
        choices=["blocks", "text"],
        default="blocks",
        help="blocks: 막대 경계만 / text: 실제 글자 블록 구간 HTML",
    )
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results_equal_mass")
    if args.wins is None:
        wins_name = "equal_mass_wins_N8_spb32_th0.9.json" if args.wins_type == "equal_mass" else "fixed_wins_N8_spb32_th0.9.json"
        args.wins = os.path.join(results_dir, wins_name)
    if args.out is None:
        base = "equal_mass_block_boundaries" if args.wins_type == "equal_mass" else "fixed_block_boundaries"
        args.out = os.path.join(results_dir, base + ".png")

    print(f"Tokenizer 로드: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    wins = load_wins(args.wins)
    block_sizes_map = load_block_sizes_by_id(args.csv)

    if args.format == "text":
        out_html = args.out.replace(".png", ".html") if args.out.endswith(".png") else args.out + ".html"
        export_text_blocks_html(wins, block_sizes_map, tokenizer, out_html, args.max_samples, args.wins_type)
        return

    samples = []
    for w in wins:
        sid = w["sample_id"]
        if sid not in block_sizes_map:
            continue
        samples.append((sid, block_sizes_map[sid]))

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    n = len(samples)
    if n == 0:
        print("시각화할 샘플이 없습니다.")
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for idx, (sid, em_blocks) in enumerate(samples):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        plot_sample(ax, sid, em_blocks, gen_length=args.gen_length)

    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].axis("off")

    fig.suptitle("Equal-Mass vs Fixed Block boundaries (equal_mass: correct / fixed_block: wrong)", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"저장 완료: {args.out}")
    plt.close()


if __name__ == "__main__":
    main()
