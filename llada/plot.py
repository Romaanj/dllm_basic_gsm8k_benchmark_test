import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams


def _configure_korean_font() -> bool:
    """Use a Korean-capable font if available; return whether enabled."""
    candidates = [
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "AppleGothic",
        "Malgun Gothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return True
    return False


HAS_KOREAN_FONT = _configure_korean_font()


def _plot_token_trajectory(ax, token_row, color, linewidth, alpha=1.0, label=None, zorder=2):
    traj = token_row.get("confidence_trajectory", [])
    if traj:
        xs = [p["iter"] for p in traj]
        ys = [p["conf"] for p in traj]
        ax.plot(
            xs, ys, color=color, linewidth=linewidth, alpha=alpha, label=label, zorder=zorder
        )
        return
    # trajectory가 없는 구버전 JSON 대응: 점 하나라도 표시
    it = token_row.get("unmask_decode_iter")
    cf = token_row.get("unmask_confidence")
    if it is not None and cf is not None:
        ax.plot(
            [it], [cf], marker="o", markersize=3.5, linestyle="None",
            color=color, alpha=alpha, label=label, zorder=zorder
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-1 high-mass token trajectory plotting")
    p.add_argument("--sample-id", type=int, default=0)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--steps-per-block", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--rollout-mode", type=str, default="sigmoid")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument(
        "--in-dir",
        type=str,
        default="results_equal_mass/high_mass_probe",
    )
    p.add_argument("--out", type=str, default="")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _build_default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    in_dir = Path(args.in_dir)
    base = (
        f"id{args.sample_id}_N{args.num_blocks}_spb{args.steps_per_block}"
        f"_th{args.threshold}_{args.rollout_mode}"
    )
    reg_path = in_dir / f"high_mass_probe_{base}.json"
    inv_path = in_dir / f"high_mass_probe_{base}_inverse_cdf_lam{args.lam}.json"
    return reg_path, inv_path


def _draw_panel(ax, top_report_row: dict, is_inverse: bool) -> None:
    tokens = top_report_row["block_tokens_unmask_confidence"]
    top = top_report_row["top_token"]

    for t in tokens:
        _plot_token_trajectory(ax, t, color="gray", linewidth=1.1, alpha=0.55, zorder=1)

    _plot_token_trajectory(
        ax,
        top,
        color="red",
        linewidth=3.2,
        alpha=1.0,
        label="Top-1 High-mass token",
        zorder=3,
    )

    top_iter = top.get("unmask_decode_iter")
    top_conf = top.get("unmask_confidence")
    if top_iter is not None and top_conf is not None:
        ax.axvline(top_iter, color="black", linestyle="--", alpha=0.8)
        ax.scatter([top_iter], [top_conf], color="black", s=28, zorder=4)
        ax.text(
            top_iter + 0.6,
            min(0.97, top_conf + 0.03),
            f"Unmasked @ {top_conf:.3f}",
            fontsize=10,
            color="black",
        )

    if is_inverse:
        if HAS_KOREAN_FONT:
            title = "Inverse CDF (우리 방식)\n큰 블록 -> 천천히 정확한 co-evolution"
        else:
            title = "Inverse CDF (Our method)\nLarge blocks -> slower and more accurate co-evolution"
    else:
        if HAS_KOREAN_FONT:
            title = "Regular (Equal-mass CDF)\n작은 블록 -> 조기 강제 언마스킹"
        else:
            title = "Regular (Equal-mass CDF)\nSmall blocks -> early forced unmasking"
    ax.set_title(title)
    ax.set_xlabel("Denoising Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    args = parse_args()
    reg_path, inv_path = _build_default_paths(args)

    if not reg_path.exists():
        raise FileNotFoundError(f"Regular JSON not found: {reg_path}")
    if not inv_path.exists():
        raise FileNotFoundError(f"Inverse JSON not found: {inv_path}")

    with reg_path.open(encoding="utf-8") as f:
        reg = json.load(f)
    with inv_path.open(encoding="utf-8") as f:
        inv = json.load(f)

    reg_top = reg["top_high_mass_report"][0]
    inv_top = inv["top_high_mass_report"][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    _draw_panel(ax1, reg_top, is_inverse=False)
    _draw_panel(ax2, inv_top, is_inverse=True)

    ax1.set_ylabel("Confidence")
    ax1.set_ylim(0, 1.02)
    sample_id = reg["config"]["sample_id"]
    plt.suptitle(
        f"Confidence Trajectory of High-Mass Token (Sample {sample_id})",
        fontsize=14,
    )
    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(f"trajectory_sample{sample_id}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()