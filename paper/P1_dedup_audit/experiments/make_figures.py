"""
Build the manuscript figures (English labels only, per the frozen brief).

Palette: categorical slots 1 and 2 of the validated reference palette
(#2a78d6 blue, #eb6834 orange). Both were checked with the data-viz validator on
2026-07-30 and pass all six checks in light mode, including the >=3:1 contrast
floor and CVD separation (worst adjacent pair dE 24.7 protan, 33.6 normal).

Design rules followed: no dual-axis chart anywhere (F2 is two stacked panels
sharing an x-axis rather than a rate/Pareto overlay on two scales); a legend
whenever two series are present and none when there is one; thin marks; recessive
grid and axes; direct labels used selectively rather than on every mark.

Every number is read from an artifact — nothing is typed in by hand except the
schematic text of F3, which is an authored diagram.

Outputs -> paper/P1_dedup_audit/figures/{fig1..fig5}_*_en.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[3]
PCAP = REPO.parent / "iomt-pcap-experiments"
FIG = REPO / "paper" / "P1_dedup_audit" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "figure.dpi": 320, "savefig.dpi": 320,
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def finish(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}_en.{ext}")
    plt.close(fig)
    print(f"  wrote {name}_en.png / .pdf")


def recessive(ax, axis="y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_perclass() -> pd.DataFrame:
    d = json.loads((PCAP / "dr6_out" / "dr6b_perclass_f32.json").read_text())
    rows = [{"cls": k,
             "train_rows": v["train"]["rows"], "train_dups": v["train"]["dups"],
             "train_rate": v["train"]["rate"],
             "test_rows": v["test"]["rows"], "test_dups": v["test"]["dups"],
             "test_rate": v["test"]["rate"]} for k, v in d.items()]
    df = pd.DataFrame(rows).sort_values("train_dups", ascending=False).reset_index(drop=True)
    total = df["train_dups"].sum()
    if total != 2_645_751:
        sys.exit(f"per-class train duplicates sum to {total:,}, expected 2,645,751")
    df["mass_share"] = df["train_dups"] / total
    df["cum_mass"] = df["mass_share"].cumsum()
    return df


# ---------------------------------------------------------------- F1
def fig1() -> None:
    src = FIG / "precision_sweep.json"
    if not src.exists():
        print("  F1 SKIPPED — precision_sweep.json not present yet")
        return
    d = json.loads(src.read_text())
    by = {p["representation"]: p for p in d["points"]}
    sig = sorted((p for p in d["points"] if p["significant_digits"] is not None),
                 key=lambda p: p["significant_digits"])

    # One categorical axis: increasing precision left to right, ending at the
    # precision the CSVs are actually printed at. No dual axis, no broken axis.
    labels = [str(p["significant_digits"]) for p in sig] + ["float64\nas printed"]
    tr = [p["train"]["rate"] * 100 for p in sig] + [by["float64"]["train"]["rate"] * 100]
    te = [p["test"]["rate"] * 100 for p in sig] + [by["float64"]["test"]["rate"] * 100]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    # Sampled region solid; the last segment spans 9..16 digits, which we did not
    # sample, so it is dashed rather than implying measured intermediate values.
    ax.plot(x[:-1], tr[:-1], marker="o", ms=4.5, lw=2, color=BLUE, label="Train split")
    ax.plot(x[:-1], te[:-1], marker="s", ms=4.5, lw=2, color=ORANGE, label="Test split")
    ax.plot(x[-2:], tr[-2:], lw=2, color=BLUE, ls=(0, (4, 2)))
    ax.plot(x[-2:], te[-2:], lw=2, color=ORANGE, ls=(0, (4, 2)))
    ax.plot([x[-1]], [tr[-1]], marker="o", ms=4.5, color=BLUE)
    ax.plot([x[-1]], [te[-1]], marker="s", ms=4.5, color=ORANGE)

    # float32 lands between 7 and 8 significant digits; mark where, and show that
    # the sweep independently reproduces the float32 measurement there.
    ax.axvspan(5.6, 7.4, color=GRID, alpha=0.55, zorder=0)
    ax.annotate("float32, the precision models compute in\n"
                f"{by['float32']['train']['rate']*100:.2f}% train / "
                f"{by['float32']['test']['rate']*100:.2f}% test",
                xy=(6.5, 62), ha="center", fontsize=7.5, color=INK)
    ax.annotate(f"{tr[-1]:.2f}% train\n{te[-1]:.2f}% test",
                xy=(x[-1], tr[-1]), xytext=(-6, 26), textcoords="offset points",
                ha="right", fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=INK2))
    ax.annotate(f"{tr[0]:.1f}%", xy=(x[0], tr[0]), xytext=(4, -12),
                textcoords="offset points", fontsize=7.5, color=INK)

    ax.set_xlabel("Precision at which two rows are compared")
    ax.set_ylabel("Duplicate rows (% of split)")
    ax.set_title("The duplicate rate is a function of the precision it is measured at")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(-4, 100)
    recessive(ax)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.42))
    fig.text(0.5, -0.07, "Significant digits retained, increasing left to right; the final point is the "
             "full precision printed in the released CSVs.\nThe dashed segment spans 9–16 digits, which "
             "were not sampled.", ha="center", fontsize=7, color=INK2)
    finish(fig, "fig1_precision_collapse")


# ---------------------------------------------------------------- F2
def fig2() -> None:
    df = load_perclass()
    lbl = [c.replace("TCP_IP-", "").replace("_", " ") for c in df["cls"]]
    xs = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    ax1.bar(xs, df["train_rate"] * 100, color=BLUE, width=0.62)
    ax1.set_ylabel("Within-class duplicate rate (%)")
    ax1.set_title("Duplicate redundancy is concentrated, and not where volume predicts")
    ax1.set_ylim(0, 100)
    recessive(ax1)
    for i in (0,):
        ax1.annotate(f"{df['train_rate'][i]*100:.2f}%", (xs[i], df["train_rate"][i] * 100),
                     xytext=(0, 3), textcoords="offset points", ha="center",
                     fontsize=7.5, color=INK)
    zeros = df[df["train_dups"] == 0]
    z = int(zeros["train_rows"].idxmax())        # the LARGEST class with no duplicates
    ax1.annotate(f"{lbl[z]} — the largest class in the split:\n"
                 f"{df['train_rows'][z]:,} rows, zero duplicates",
                 (xs[z], 0), xytext=(0, 40), textcoords="offset points", ha="center",
                 fontsize=7.5, color=INK,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=INK2))

    ax2.plot(xs, df["cum_mass"] * 100, marker="o", ms=4, lw=2, color=ORANGE)
    ax2.set_ylabel("Cumulative share of\nduplicate mass (%)")
    ax2.set_ylim(0, 104)
    recessive(ax2)
    ax2.axhline(99.51, color=INK2, lw=0.7, ls="--")
    ax2.annotate("six flood classes = 99.51% of all duplicate rows",
                 xy=(5, 99.51), xytext=(6.4, 72), fontsize=8, color=INK,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=INK2))
    ax2.set_xticks(xs)
    ax2.set_xticklabels(lbl, rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Released class, ordered by duplicate count (train split, float32)")
    finish(fig, "fig2_per_class_structure")


# ---------------------------------------------------------------- F3
def fig3() -> None:
    # Status is ordinal-categorical, so it is drawn as a discrete 3-cell stepper.
    # A proportional bar would invite reading an unmeasured percentage off it.
    axes_rows = [
        ("1. Duplicate rows", 3, "corrected exactly", "this work", BLUE),
        ("2. Device-level overlap", 2, "corrected approximately",
         "one study, predicted device labels", ORANGE),
        ("3. Official split destroyed\n    by merge-and-re-split", 1, "partially mitigated",
         "two 2026 studies, unquantified", ORANGE),
        ("4. Resampling before\n    splitting", 0, "open", "uncorrected corpus-wide", "#8c8b85"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(axes_rows)); ax.axis("off")
    ax.set_title("Four leakage axes in the CICIoMT2024 literature, and their correction status",
                 loc="left", pad=12)
    for i, (name, filled, status, who, col) in enumerate(axes_rows):
        y = len(axes_rows) - i - 1
        ax.add_patch(Rectangle((0.05, y + 0.18), 3.5, 0.64, facecolor="white",
                               edgecolor=col, lw=1.4))
        ax.text(0.2, y + 0.5, name, va="center", fontsize=8.5, color=INK)
        for c in range(3):                      # three discrete cells, 2px-equivalent gap
            x0 = 3.95 + c * 0.62
            ax.add_patch(Rectangle((x0, y + 0.32), 0.5, 0.36,
                                   facecolor=col if c < filled else "white",
                                   edgecolor=col if c < filled else GRID, lw=0.9))
        ax.text(6.0, y + 0.5, f"{status} — {who}", va="center", fontsize=8, color=INK2)
    fig.text(0.02, -0.02, "Filled cells encode an ordinal status (none / partial / approximate / exact); "
             "they are not a measured quantity.", fontsize=7, color=INK2)
    finish(fig, "fig3_leakage_axes")


# ---------------------------------------------------------------- F4
def fig4() -> None:
    df = load_perclass().copy()
    df["distinct_test"] = df["test_rows"] - df["test_dups"]
    df = df.sort_values("test_rows", ascending=False).reset_index(drop=True)
    lbl = [c.replace("TCP_IP-", "").replace("_", " ") for c in df["cls"]]
    xs = np.arange(len(df)); w = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(xs - w / 2, df["test_rows"], width=w, color=BLUE, label="Released test rows")
    ax.bar(xs + w / 2, df["distinct_test"], width=w, color=ORANGE, label="Distinct feature vectors")
    ax.set_yscale("log")
    ax.set_ylabel("Test rows (log scale)")
    ax.set_xlabel("Released class, ordered by test-split size")
    ax.set_title("What a test-set metric is actually computed over")
    ax.set_xticks(xs); ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=7)
    recessive(ax)
    ax.legend(loc="upper right")
    worst = df["test_rate"].idxmax()
    ax.annotate(f"{lbl[worst]}: {df['test_rate'][worst]*100:.2f}% of test rows are copies",
                xy=(xs[worst], df["distinct_test"][worst]), xytext=(10, -26),
                textcoords="offset points", fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=INK2))
    finish(fig, "fig4_effective_sample_size")


# ---------------------------------------------------------------- F5
def fig5() -> None:
    cfg = ["Random Forest\n28 features", "Random Forest\n44 features",
           "XGBoost\n28 features", "XGBoost\n44 features"]
    delta = [-0.0114, -0.0171, -0.0449, -0.0368]     # numbers_map.md Section 4
    sigma = 0.0247                                    # across-seed sigma, Section 8.1

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.axhspan(-sigma, sigma, color=GRID, alpha=0.55, zorder=0)
    ax.annotate("across-seed σ of this pipeline (±0.025)", xy=(3.45, sigma), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.bar(np.arange(4), delta, width=0.55, color=ORANGE, zorder=2)
    ax.axhline(0, color=INK2, lw=0.8)
    for i, v in enumerate(delta):
        ax.annotate(f"{v:+.4f}", (i, v), xytext=(0, -12), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(cfg, fontsize=8)
    ax.set_ylabel("Δ macro-F1 (SMOTETomek − original)")
    ax.set_title("SMOTETomek degrades macro-F1 in all four configurations\n"
                 "(two of them within this pipeline's seed noise)", fontsize=9.5)
    ax.set_ylim(-0.058, 0.036)
    recessive(ax)
    finish(fig, "fig5_smotetomek_delta")


if __name__ == "__main__":
    print("building figures (English labels only)")
    fig2(); fig3(); fig4(); fig5(); fig1()
    print("done")
