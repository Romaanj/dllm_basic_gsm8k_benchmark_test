import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_CONFIG = {
    "local_mean_delta": {
        "value_col": "local_entropy_mean_delta",
        "token_weight_col": "local_masked_count_pre",
        "label": "Local entropy mean delta (post - pre)",
        "title": "Local Entropy Mean Delta vs Window Radius",
    },
    "global_mean_delta": {
        "value_col": "global_entropy_mean_delta",
        "token_weight_col": "masked_count_pre",
        "label": "Global entropy mean delta (post - pre)",
        "title": "Global Entropy Mean Delta vs Window Radius",
    },
    "local_sum_delta": {
        "value_col": "local_entropy_sum_delta",
        "token_weight_col": None,
        "label": "Local entropy sum delta (post - pre)",
        "title": "Local Entropy Sum Delta vs Window Radius",
    },
    "global_sum_delta": {
        "value_col": "global_entropy_sum_delta",
        "token_weight_col": None,
        "label": "Global entropy sum delta (post - pre)",
        "title": "Global Entropy Sum Delta vs Window Radius",
    },
}


def parse_window_list(s: str) -> List[int]:
    out: List[int] = []
    for p in s.split(","):
        t = p.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("window list is empty")
    return out


def mean_std(vals: List[float], weights: Optional[List[float]] = None) -> Dict[str, Optional[float]]:
    if not vals:
        return {"mean": None, "std": None, "sem": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    n = int(arr.size)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.size != arr.size:
            raise ValueError("weights size must match vals size")
        w = np.clip(w, a_min=0.0, a_max=None)
        wsum = float(w.sum())
        if wsum <= 0.0:
            return {"mean": None, "std": None, "sem": None, "n": n}
        mean = float(np.sum(w * arr) / wsum)
        var = float(np.sum(w * (arr - mean) ** 2) / wsum)
        std = float(np.sqrt(max(0.0, var)))
        return {"mean": mean, "std": std, "sem": None, "n": n}
    std = float(arr.std(ddof=1)) if n >= 2 else 0.0
    sem = float(std / np.sqrt(n)) if n > 0 else None
    return {"mean": float(arr.mean()), "std": std, "sem": sem, "n": n}


def find_latest_highlow_csv(results_dir: str, window_radius: int, task: str) -> str:
    pattern_task = os.path.join(
        results_dir,
        f"baseline_entropy_events_highlow_*_w{window_radius}_*_task-{task}.csv",
    )
    matches = glob.glob(pattern_task)
    if not matches:
        # backward compatibility for older outputs without task suffix
        legacy_pattern = os.path.join(
            results_dir, f"baseline_entropy_events_highlow_*_w{window_radius}_*.csv"
        )
        matches = glob.glob(legacy_pattern)
    if not matches:
        raise FileNotFoundError(
            f"no highlow csv found for window={window_radius}, task={task}"
        )
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def load_value_by_group(
    csv_path: str,
    value_col: str,
    weight_col: Optional[str],
) -> Dict[str, Dict[str, List[float]]]:
    high_vals: List[float] = []
    low_vals: List[float] = []
    high_w: List[float] = []
    low_w: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get(value_col, "")
            if raw is None or raw == "":
                continue
            try:
                v = float(raw)
            except ValueError:
                continue

            w = 1.0
            if weight_col:
                wr = row.get(weight_col, "")
                if wr is None or wr == "":
                    continue
                try:
                    w = float(wr)
                except ValueError:
                    continue

            is_high = int(row.get("is_high_rollout", "0") or 0)
            is_low = int(row.get("is_low_rollout", "0") or 0)
            if is_high == 1:
                high_vals.append(v)
                high_w.append(w)
            if is_low == 1:
                low_vals.append(v)
                low_w.append(w)
    return {
        "high": {"vals": high_vals, "weights": high_w},
        "low": {"vals": low_vals, "weights": low_w},
    }


def plot_mean_with_sem(
    windows: List[int],
    high_stats: List[Dict[str, Optional[float]]],
    low_stats: List[Dict[str, Optional[float]]],
    out_path: str,
    y_label: str,
    title: str,
) -> None:
    x = np.asarray(windows, dtype=np.int64)
    high_mean = np.asarray([s["mean"] for s in high_stats], dtype=np.float64)
    low_mean = np.asarray([s["mean"] for s in low_stats], dtype=np.float64)
    high_sem = np.asarray([s["sem"] for s in high_stats], dtype=np.float64)
    low_sem = np.asarray([s["sem"] for s in low_stats], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if np.all(np.isfinite(low_sem)) and np.all(np.isfinite(high_sem)):
        ax.errorbar(
            x, low_mean, yerr=low_sem, marker="o", linewidth=2, capsize=4, label="Low rollout"
        )
        ax.errorbar(
            x, high_mean, yerr=high_sem, marker="o", linewidth=2, capsize=4, label="High rollout"
        )
    else:
        ax.plot(x, low_mean, marker="o", linewidth=2, label="Low rollout")
        ax.plot(x, high_mean, marker="o", linewidth=2, label="High rollout")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Window radius (W)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_grouped_box(
    windows: List[int],
    high_vals_by_w: List[List[float]],
    low_vals_by_w: List[List[float]],
    out_path: str,
    y_label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    positions: List[float] = []
    data: List[List[float]] = []
    labels: List[str] = []

    gap = 2.0
    for i, w in enumerate(windows):
        base = i * gap
        positions.extend([base - 0.3, base + 0.3])
        data.extend([low_vals_by_w[i], high_vals_by_w[i]])
        labels.extend([f"W={w}\nLow", f"W={w}\nHigh"])

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    for i, patch in enumerate(box["boxes"]):
        patch.set_alpha(0.6)
        if i % 2 == 0:
            patch.set_facecolor("C0")
        else:
            patch.set_facecolor("C1")

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot entropy delta for W sweep (high vs low rollout)"
    )
    p.add_argument("--results-dir", type=str, default="results_baseline_entropy_events")
    p.add_argument("--window-list", type=str, default="1,2,4,8,10")
    p.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "humaneval"])
    p.add_argument(
        "--metric",
        type=str,
        default="local_mean_delta",
        choices=list(METRIC_CONFIG.keys()),
    )
    p.add_argument("--value-col", type=str, default="local_entropy_mean_delta")
    p.add_argument(
        "--mean-mode",
        type=str,
        default="token_weighted_mean",
        choices=["event_mean", "token_weighted_mean"],
    )
    p.add_argument("--out-prefix", type=str, default="baseline_local_entropy_window_sweep")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    windows = parse_window_list(args.window_list)
    os.makedirs(args.results_dir, exist_ok=True)
    metric_cfg = METRIC_CONFIG[args.metric]
    value_col = args.value_col if args.value_col else metric_cfg["value_col"]

    high_vals_by_w: List[List[float]] = []
    low_vals_by_w: List[List[float]] = []
    high_stats: List[Dict[str, Optional[float]]] = []
    low_stats: List[Dict[str, Optional[float]]] = []
    file_map: Dict[int, str] = {}

    for w in windows:
        csv_path = find_latest_highlow_csv(args.results_dir, w, args.task)
        file_map[int(w)] = csv_path
        if args.mean_mode == "token_weighted_mean":
            weight_col = metric_cfg["token_weight_col"]
        else:
            weight_col = None
        grouped = load_value_by_group(
            csv_path=csv_path, value_col=value_col, weight_col=weight_col
        )

        high_vals = grouped["high"]["vals"]
        low_vals = grouped["low"]["vals"]
        high_weights = grouped["high"]["weights"]
        low_weights = grouped["low"]["weights"]
        high_vals_by_w.append(high_vals)
        low_vals_by_w.append(low_vals)
        high_stats.append(mean_std(high_vals, high_weights if weight_col else None))
        low_stats.append(mean_std(low_vals, low_weights if weight_col else None))

    mean_plot = os.path.join(args.results_dir, f"{args.out_prefix}_mean_sem.png")
    box_plot = os.path.join(args.results_dir, f"{args.out_prefix}_boxplot.png")
    summary_json = os.path.join(args.results_dir, f"{args.out_prefix}_summary.json")

    y_label = f"{metric_cfg['label']} ({args.mean_mode})"
    title = f"{metric_cfg['title']} ({args.task})"
    plot_mean_with_sem(
        windows, high_stats, low_stats, mean_plot, y_label=y_label, title=title
    )
    plot_grouped_box(
        windows,
        high_vals_by_w,
        low_vals_by_w,
        box_plot,
        y_label=metric_cfg["label"],
        title=f"Distribution by Window Radius (Low vs High, {args.metric})",
    )

    summary = {
        "windows": windows,
        "task": args.task,
        "metric": args.metric,
        "value_col": value_col,
        "mean_mode": args.mean_mode,
        "source_highlow_csv": file_map,
        "high_stats": high_stats,
        "low_stats": low_stats,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {mean_plot}")
    print(f"[Saved] {box_plot}")
    print(f"[Saved] {summary_json}")


if __name__ == "__main__":
    main()
