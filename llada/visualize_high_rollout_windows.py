import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def normalize_token(tok: str) -> str:
    t = (tok or "").strip()
    if t == "":
        return "<sp>"
    if t == "\\n":
        return "<nl>"
    return t


def short_token(tok: str, max_len: int = 12) -> str:
    t = normalize_token(tok)
    return t if len(t) <= max_len else (t[: max_len - 1] + "…")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize high-rollout center/window tokens from trace JSON"
    )
    p.add_argument(
        "--trace-json",
        type=str,
        default="results_step0_high_rollout_trace/step0_high_rollout_trace.json",
    )
    p.add_argument("--out-dir", type=str, default="results_step0_high_rollout_trace")
    p.add_argument("--top-n-tokens", type=int, default=30)
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0이면 전체 샘플 시각화, 양수면 앞에서 N개 샘플만",
    )
    return p.parse_args()


def plot_token_frequency(center_counter: Counter, window_counter: Counter, out_path: str, top_n: int) -> None:
    vocab = [t for t, _ in (center_counter + window_counter).most_common(top_n)]
    if not vocab:
        return

    center_vals = [center_counter.get(t, 0) for t in vocab]
    window_vals = [window_counter.get(t, 0) for t in vocab]

    x = np.arange(len(vocab))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(12, len(vocab) * 0.4), 6))
    ax.bar(x - width / 2, center_vals, width=width, label="center(high)", color="#e74c3c")
    ax.bar(x + width / 2, window_vals, width=width, label="window(others)", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab, rotation=70, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Token Frequency: high center vs window neighbors")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_sample_window_grid(record: Dict[str, Any], out_path: str) -> None:
    highs: List[Dict[str, Any]] = record.get("high_rollout_gen_tokens", [])
    if not highs:
        return

    window_size = int(record.get("window_size", 0))
    cols = 2 * window_size + 1
    rows = len(highs)

    z = np.full((rows, cols), np.nan, dtype=np.float64)
    labels = [["" for _ in range(cols)] for _ in range(rows)]
    row_names = []

    for r, h in enumerate(highs):
        gen_idx = int(h["gen_index"])
        row_names.append(f"r{int(h['rank'])}@{gen_idx}")
        for wt in h.get("window_tokens", []):
            rel_pos = int(wt["rel_pos"])
            offset = rel_pos - gen_idx
            c = offset + window_size
            if 0 <= c < cols:
                z[r, c] = float(wt.get("rollout_score", 0.0))
                labels[r][c] = short_token(str(wt.get("token_clean", "")))

    finite_vals = z[np.isfinite(z)]
    vmax = float(np.percentile(finite_vals, 95)) if finite_vals.size > 0 else 1.0
    vmax = max(vmax, 1e-6)

    fig_h = max(4.5, rows * 0.65 + 2.5)
    fig_w = max(8.0, cols * 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(np.nan_to_num(z, nan=0.0), cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("rollout_score")

    x_labels = [str(i - window_size) for i in range(cols)]
    ax.set_xticks(range(cols))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Relative position from high token center")

    ax.set_yticks(range(rows))
    ax.set_yticklabels(row_names)
    ax.set_ylabel("High-rollout token rank@index")

    for r in range(rows):
        for c in range(cols):
            txt = labels[r][c]
            if txt:
                ax.text(c, r, txt, ha="center", va="center", fontsize=8, color="black")

    center_col = window_size
    ax.axvline(center_col - 0.5, color="black", linewidth=1.5, alpha=0.8)
    ax.axvline(center_col + 0.5, color="black", linewidth=1.5, alpha=0.8)

    sid = record.get("sample_id", "?")
    ax.set_title(f"Sample {sid}: high-rollout token windows")
    fig.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.trace_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict[str, Any]] = data.get("records", [])
    if args.max_samples > 0:
        records = records[: args.max_samples]

    center_counter: Counter = Counter()
    window_counter: Counter = Counter()

    for rec in records:
        sid = int(rec.get("sample_id", -1))
        highs = rec.get("high_rollout_gen_tokens", [])
        for h in highs:
            center_counter.update([normalize_token(str(h.get("token_clean", "")))])
            for wt in h.get("window_tokens", []):
                if bool(wt.get("is_target_high", False)):
                    continue
                window_counter.update([normalize_token(str(wt.get("token_clean", "")))])

        per_sample_path = os.path.join(args.out_dir, f"high_rollout_window_grid_id{sid}.png")
        plot_sample_window_grid(rec, per_sample_path)

    freq_path = os.path.join(args.out_dir, "high_rollout_window_token_frequency.png")
    plot_token_frequency(center_counter, window_counter, freq_path, top_n=args.top_n_tokens)

    print(f"[Saved] {freq_path}")
    print(f"[Done] visualized {len(records)} samples in {args.out_dir}")


if __name__ == "__main__":
    main()
