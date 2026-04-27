"""Phase 6C Pareto-frontier plot for cost-vs-rescue tradeoff.

Reads results/enhanced_fusion/metrics/ablation_table.csv and produces a
publication-quality scatter of (FPR_on_benign_test, H2_strict_avg) for all
11 variants. Identifies Pareto-optimal variants — those for which no other
variant achieves both higher strict_avg AND lower FPR — and connects them
with a frontier line.

This is the methodological replacement for the OPERATIONAL_FPR_BUDGET=0.25
single-cutoff framing: the figure lets a reviewer pick any operating point
on the frontier and read off the (recall, FPR) tradeoff.

Run: python notebooks/pareto_frontier.py
Output: results/enhanced_fusion/figures/pareto_frontier.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABLATION_CSV = ROOT / "results" / "enhanced_fusion" / "metrics" / "ablation_table.csv"
FIG_DIR = ROOT / "results" / "enhanced_fusion" / "figures"
FIG_PATH = FIG_DIR / "pareto_frontier.png"

RECOMMENDED_VARIANT = "entropy_benign_p95"
FPR_GUIDES = (0.10, 0.15, 0.20, 0.25, 0.30)


def pareto_optimal_indices(fpr: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Return indices of points on the Pareto frontier (low FPR, high recall).

    A point is Pareto-optimal if no other point has both lower FPR AND higher
    recall. Ties on either dimension do not dominate.
    """
    n = len(fpr)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dominates = (fpr[j] <= fpr[i] and recall[j] >= recall[i]) and (
                fpr[j] < fpr[i] or recall[j] > recall[i]
            )
            if dominates:
                keep[i] = False
                break
    return np.where(keep)[0]


def main() -> None:
    df = pd.read_csv(ABLATION_CSV)
    fpr = df["avg_false_alert_rate"].to_numpy(dtype=float)
    recall = df["h2_strict_avg"].to_numpy(dtype=float)
    labels = df["variant"].tolist()

    front_idx = pareto_optimal_indices(fpr, recall)
    front_order = front_idx[np.argsort(fpr[front_idx])]
    front_fpr = fpr[front_order]
    front_recall = recall[front_order]
    dominated_mask = np.ones(len(df), dtype=bool)
    dominated_mask[front_order] = False

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(
        fpr[dominated_mask],
        recall[dominated_mask],
        s=110,
        c="#9aa0a6",
        edgecolors="#3c4043",
        linewidths=0.8,
        alpha=0.85,
        label="Dominated variants",
        zorder=2,
    )
    ax.plot(
        front_fpr,
        front_recall,
        color="#c0392b",
        linewidth=2.0,
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        front_fpr,
        front_recall,
        s=170,
        c="#c0392b",
        edgecolors="#5a1a12",
        linewidths=1.0,
        label="Pareto-optimal frontier",
        zorder=4,
    )

    rec_idx_arr = np.where(df["variant"] == RECOMMENDED_VARIANT)[0]
    if rec_idx_arr.size:
        rec_idx = int(rec_idx_arr[0])
        ax.scatter(
            [fpr[rec_idx]],
            [recall[rec_idx]],
            s=380,
            marker="*",
            c="#f1c40f",
            edgecolors="#7d6608",
            linewidths=1.4,
            label=f"Recommended: {RECOMMENDED_VARIANT}",
            zorder=5,
        )

    for i, lbl in enumerate(labels):
        dx, dy = 0.004, 0.012
        ax.annotate(
            lbl,
            xy=(fpr[i], recall[i]),
            xytext=(fpr[i] + dx, recall[i] + dy),
            fontsize=8,
            color="#202124",
            zorder=6,
        )

    for x in FPR_GUIDES:
        ax.axvline(x, color="#dadce0", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(
            x + 0.001,
            ax.get_ylim()[0] if False else 0.02,
            f"FPR={x:.2f}",
            fontsize=7,
            color="#5f6368",
            rotation=90,
            va="bottom",
        )

    ax.axhline(0.50, color="#1a73e8", linewidth=0.9, linestyle=":", alpha=0.7, zorder=1)
    ax.text(
        max(fpr) - 0.005,
        0.51,
        "H2-strict pass threshold (0.50)",
        fontsize=8,
        color="#1a73e8",
        ha="right",
        va="bottom",
    )

    ax.set_xlabel("Benign-test false-alert rate (FPR)", fontsize=11)
    ax.set_ylabel("H2-strict average rescue recall", fontsize=11)
    ax.set_title(
        "Phase 6C Pareto frontier: cost (FPR) vs zero-day rescue (H2-strict avg)\n"
        "across all 11 fusion variants on CICIoMT2024 LOO",
        fontsize=12,
    )
    ax.set_xlim(0.0, max(fpr) * 1.10)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)

    print(f"Wrote {FIG_PATH}")
    print()
    print("Pareto-optimal variants (sorted by FPR):")
    print(
        df.iloc[front_order][
            ["variant", "h2_strict_pass", "h2_strict_avg", "avg_false_alert_rate"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
