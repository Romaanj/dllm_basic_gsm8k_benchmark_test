"""
PMR 비교 결과 CSV를 읽어 original_cdf vs inverse_cdf 분포 히스토그램을 그린다.

요구 레이아웃:
- 1열(첫 패널): topk_conf_mean 비교
- 2열(두번째 패널): pmr_topk 비교
- 나머지 항목은 아래 행으로 계속 배치
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot original_cdf vs inverse_cdf histograms from pmr_compare csv"
    )
    p.add_argument(
        "--input-csv",
        type=str,
        default="results_equal_mass/high_mass_probe/pmr_compare_100samples_start0_N8_spb32_th0.9_tau0.8_sigmoid_lam1.0.csv",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results_equal_mass/high_mass_probe",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=20,
    )
    p.add_argument(
        "--out-name",
        type=str,
        default="pmr_compare_histograms_original_vs_inverse.png",
    )
    return p.parse_args()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float_array(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    vals = []
    for r in rows:
        v = r.get(key, "")
        if v is None or str(v).strip() == "":
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    if not vals:
        return np.array([], dtype=np.float64)
    return np.array(vals, dtype=np.float64)


def discover_metric_pairs(rows: List[Dict[str, str]]) -> List[Tuple[str, str, str]]:
    if not rows:
        return []
    cols = list(rows[0].keys())

    metric_suffixes = []
    for c in cols:
        if c.startswith("sigmoid_"):
            suffix = c[len("sigmoid_"):]
            inv_col = f"inverse_{suffix}"
            if inv_col in cols:
                metric_suffixes.append(suffix)

    # 요청 우선순위: topk_conf_mean, pmr_topk, 다음 블록 메트릭
    preferred = [
        "topk_conf_mean",
        "pmr_topk",
        "next_block_size_mean",
        "next_block_low_conf_ratio_mean",
    ]
    ordered = []
    seen = set()
    for s in preferred + metric_suffixes:
        if s in seen:
            continue
        sig = f"sigmoid_{s}"
        inv = f"inverse_{s}"
        if sig in cols and inv in cols:
            ordered.append((s, sig, inv))
            seen.add(s)
    return ordered


def nice_name(metric_suffix: str) -> str:
    return metric_suffix.replace("_", " ")


def main() -> None:
    args = parse_args()
    rows = read_csv_rows(args.input_csv)
    pairs = discover_metric_pairs(rows)
    if not pairs:
        raise RuntimeError("비교 가능한 sigmoid_/inverse_ 항목을 찾지 못했습니다.")

    n = len(pairs)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for idx, (suffix, sig_col, inv_col) in enumerate(pairs):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        sig_vals = to_float_array(rows, sig_col)
        inv_vals = to_float_array(rows, inv_col)
        if sig_vals.size == 0 and inv_vals.size == 0:
            ax.set_title(f"{nice_name(suffix)} (no data)")
            ax.axis("off")
            continue

        all_vals = np.concatenate([sig_vals, inv_vals]) if sig_vals.size and inv_vals.size else (
            sig_vals if sig_vals.size else inv_vals
        )
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        bins = np.linspace(vmin, vmax, args.bins + 1)

        ax.hist(
            sig_vals,
            bins=bins,
            alpha=0.55,
            color="#e7298a",
            edgecolor="black",
            linewidth=0.4,
            label=f"original_cdf ({sig_col})",
        )
        ax.hist(
            inv_vals,
            bins=bins,
            alpha=0.50,
            color="#66a61e",
            edgecolor="black",
            linewidth=0.4,
            label=f"inverse_cdf ({inv_col})",
        )

        sig_mean = float(np.mean(sig_vals)) if sig_vals.size else float("nan")
        inv_mean = float(np.mean(inv_vals)) if inv_vals.size else float("nan")
        ax.axvline(sig_mean, color="#c51b7d", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.axvline(inv_mean, color="#4d9221", linestyle="--", linewidth=1.2, alpha=0.9)

        ax.set_title(
            f"{nice_name(suffix)}\n"
            f"original_cdf mean={sig_mean:.4f}, inverse_cdf mean={inv_mean:.4f}",
            fontsize=10,
        )
        ax.set_xlabel(nice_name(suffix))
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(fontsize=8)

    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    fig.suptitle(
        "PMR Compare Histogram: original_cdf vs inverse_cdf",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()

