#!/usr/bin/env python3
"""
Path B Week 1 — Multi-Seed Aggregation & Figures
=================================================

Aggregates the 5 per-seed ablation tables into a single summary CSV with
mean / std / min / max / p05 / p95 across seeds for every (variant) metric,
plus a long-form per-target summary, plus 3 figures.

N=5 seeds is small — we do NOT bootstrap; we report the empirical (min, max)
and 5%–95% percentile as a transparent range. README §15B will frame this
explicitly.

Inputs:
    results/enhanced_fusion/multi_seed/seed_<S>/metrics/
        ablation_table.csv       (11 rows × seed)
        per_target_results.csv   (55 rows × seed)

Outputs:
    results/enhanced_fusion/multi_seed_summary.csv          (variant-level)
    results/enhanced_fusion/multi_seed_per_target_summary.csv (variant×target)
    results/enhanced_fusion/multi_seed/figures/
        seed_stability_per_variant.png
        seed_stability_per_target.png
        multi_seed_pareto.png

Usage:
    cd ~/IoMT-Project && source venv/bin/activate
    python -u notebooks/multi_seed_aggregate.py
"""

import sys
import time
from pathlib import Path

# Headless matplotlib BEFORE pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("./results/enhanced_fusion/multi_seed")
SUMMARY_DIR = Path("./results/enhanced_fusion")
SEEDS = [1, 7, 42, 100, 1729]

H2_STRICT_ELIGIBLE = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "ARP_Spoofing",
]
H2_PASS_THRESHOLD = 0.70

VARIANT_ORDER = [
    "baseline_ae_p90", "baseline_ae_p95",
    "confidence_0.6", "confidence_0.7",
    "entropy_benign_p90", "entropy_benign_p95", "entropy_benign_p99",
    "ensemble_p90", "ensemble_p95",
    "conf07_ent_p95", "full_enhanced",
]

plt.rcParams["figure.dpi"]    = 110
plt.rcParams["savefig.dpi"]   = 220
plt.rcParams["savefig.bbox"]  = "tight"
plt.rcParams["font.size"]     = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_per_seed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all seeds' ablation_table.csv and per_target_results.csv into long-form DFs."""
    ablation_frames = []
    per_target_frames = []
    for seed in SEEDS:
        ab_path = OUTPUT_DIR / f"seed_{seed}" / "metrics" / "ablation_table.csv"
        pt_path = OUTPUT_DIR / f"seed_{seed}" / "metrics" / "per_target_results.csv"
        if not ab_path.exists() or not pt_path.exists():
            raise FileNotFoundError(
                f"[seed={seed}] missing per-seed metrics. Run multi_seed_fusion.py first."
            )
        ab = pd.read_csv(ab_path)
        pt = pd.read_csv(pt_path)
        # Each seed's ablation_table.csv already has a `seed` column from multi_seed_fusion.py
        ablation_frames.append(ab)
        per_target_frames.append(pt)

    ablation_long = pd.concat(ablation_frames, ignore_index=True)
    per_target_long = pd.concat(per_target_frames, ignore_index=True)

    # Sanity: 5 seeds × 11 variants = 55 ablation rows
    assert len(ablation_long) == 5 * 11, (
        f"expected 55 ablation rows (5 seeds × 11 variants), got {len(ablation_long)}"
    )
    # 5 seeds × 5 targets × 11 variants = 275 per-target rows
    assert len(per_target_long) == 5 * 5 * 11, (
        f"expected 275 per_target rows, got {len(per_target_long)}"
    )
    return ablation_long, per_target_long


def aggregate_variant_level(ablation_long: pd.DataFrame) -> pd.DataFrame:
    """Across-seed aggregation per variant (11 rows)."""
    # Numeric columns to summarize
    numeric_cols = [
        "h2_strict_pass_int", "h2_strict_avg",
        "h2_binary_pass_int", "h2_binary_avg",
        "avg_flag_rate", "avg_false_alert_rate",
    ]
    rows = []
    for variant in VARIANT_ORDER:
        sub = ablation_long[ablation_long["variant"] == variant]
        assert len(sub) == 5, f"variant {variant} has {len(sub)} seeds, expected 5"

        row = {"variant": variant, "variant_name": sub["variant_name"].iloc[0]}
        for col in numeric_cols:
            vals = sub[col].astype(float).values
            row[f"{col}_mean"]  = float(np.mean(vals))
            row[f"{col}_std"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{col}_min"]   = float(np.min(vals))
            row[f"{col}_max"]   = float(np.max(vals))
            row[f"{col}_p05"]   = float(np.percentile(vals, 5))
            row[f"{col}_p95"]   = float(np.percentile(vals, 95))
        # How many seeds achieved 4/4 H2-strict pass?
        row["strict_pass_count"] = int((sub["h2_strict_pass_int"] == 4).sum())
        # How many seeds achieved 5/5 H2-binary pass?
        row["binary_pass_count"] = int((sub["h2_binary_pass_int"] == 5).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_target(per_target_long: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, target) aggregation across seeds."""
    rows = []
    for variant in VARIANT_ORDER:
        for target in sorted(per_target_long["target"].unique()):
            sub = per_target_long[
                (per_target_long["variant"] == variant)
                & (per_target_long["target"] == target)
            ]
            assert len(sub) == 5
            row = {"variant": variant, "target": target}
            for col in ("h2_strict_rescue_recall", "h2_binary_recall",
                        "flag_rate_all", "false_alert_rate_benign"):
                vals = sub[col].astype(float).values
                # h2_strict_rescue_recall is NaN for ineligible/insufficient targets
                clean = vals[~np.isnan(vals)]
                if len(clean) == 0:
                    row[f"{col}_mean"] = float("nan")
                    row[f"{col}_std"]  = float("nan")
                    row[f"{col}_min"]  = float("nan")
                    row[f"{col}_max"]  = float("nan")
                    row[f"{col}_n_seeds"] = 0
                else:
                    row[f"{col}_mean"] = float(np.mean(clean))
                    row[f"{col}_std"]  = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
                    row[f"{col}_min"]  = float(np.min(clean))
                    row[f"{col}_max"]  = float(np.max(clean))
                    row[f"{col}_n_seeds"] = int(len(clean))
            rows.append(row)
    return pd.DataFrame(rows)


# ---- Figures ----------------------------------------------------------------
def fig_seed_stability_per_variant(ablation_long: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    """Bar chart with error bars showing strict_avg per variant across 5 seeds."""
    summary_indexed = summary.set_index("variant")
    means = [summary_indexed.loc[v, "h2_strict_avg_mean"] for v in VARIANT_ORDER]
    stds  = [summary_indexed.loc[v, "h2_strict_avg_std"]  for v in VARIANT_ORDER]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(VARIANT_ORDER))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#3a7bd5",
                  edgecolor="#1f4f8c", alpha=0.85, label="mean ± std")

    # Overlay individual seed dots
    for i, variant in enumerate(VARIANT_ORDER):
        seed_vals = ablation_long[ablation_long["variant"] == variant]["h2_strict_avg"].values
        ax.scatter([i] * len(seed_vals), seed_vals, color="#cf4a4a",
                   s=28, zorder=3, alpha=0.85,
                   label="individual seeds" if i == 0 else None)

    ax.axhline(H2_PASS_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"H2-strict threshold = {H2_PASS_THRESHOLD}")

    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in VARIANT_ORDER],
                       rotation=0, fontsize=8.5)
    ax.set_ylabel("H2-strict rescue recall (mean over 4 eligible targets)")
    ax.set_title("Seed stability per fusion variant (5 seeds)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log(f"  saved {out_path.name}")


def fig_seed_stability_per_target(per_target_long: pd.DataFrame, out_path: Path) -> None:
    """4-panel: per-eligible-target rescue recall across seeds, all 11 variants."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, target in zip(axes, H2_STRICT_ELIGIBLE):
        sub = per_target_long[per_target_long["target"] == target]
        x = np.arange(len(VARIANT_ORDER))

        # Each seed = one dot per variant
        for seed in SEEDS:
            sub_s = sub[sub["seed"] == seed].set_index("variant").reindex(VARIANT_ORDER)
            ax.scatter(x, sub_s["h2_strict_rescue_recall"].values,
                       label=f"seed {seed}", s=36, alpha=0.75)

        # Mean bar
        means = [
            sub[sub["variant"] == v]["h2_strict_rescue_recall"].mean()
            for v in VARIANT_ORDER
        ]
        ax.plot(x, means, color="black", linewidth=1.4, label="mean", alpha=0.85, zorder=4)

        ax.axhline(H2_PASS_THRESHOLD, color="red", linestyle="--", linewidth=1.0)
        ax.set_title(target, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", "\n") for v in VARIANT_ORDER],
                           rotation=0, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("rescue recall")

    axes[0].legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.95)
    fig.suptitle("Per-target H2-strict rescue recall stability (5 seeds × 11 variants × 4 eligible targets)",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log(f"  saved {out_path.name}")


def fig_multi_seed_pareto(ablation_long: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    """Scatter (FPR, strict_avg) with one point per (variant, seed) + per-variant mean overlay."""
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("tab20")

    summary_indexed = summary.set_index("variant")
    for i, variant in enumerate(VARIANT_ORDER):
        sub = ablation_long[ablation_long["variant"] == variant]
        color = cmap(i % 20)
        # Per-seed faint dots
        ax.scatter(sub["avg_false_alert_rate"], sub["h2_strict_avg"],
                   color=color, alpha=0.45, s=40, edgecolor="white", linewidth=0.5)
        # Mean as a larger marker
        ax.scatter(
            summary_indexed.loc[variant, "avg_false_alert_rate_mean"],
            summary_indexed.loc[variant, "h2_strict_avg_mean"],
            color=color, alpha=0.95, s=170, edgecolor="black", linewidth=1.0,
            label=variant,
        )
        # Annotate near the mean
        ax.annotate(
            variant.replace("_", " "),
            (summary_indexed.loc[variant, "avg_false_alert_rate_mean"],
             summary_indexed.loc[variant, "h2_strict_avg_mean"]),
            xytext=(5, 4), textcoords="offset points", fontsize=7.5, alpha=0.9,
        )

    ax.axhline(H2_PASS_THRESHOLD, color="black", linestyle="--", linewidth=1.0,
               label=f"H2-strict threshold ({H2_PASS_THRESHOLD})")
    ax.set_xlabel("Operational FPR on benign test rows (lower = better)")
    ax.set_ylabel("H2-strict rescue recall, mean over 4 targets (higher = better)")
    ax.set_title("Multi-seed Pareto frontier — large markers = mean across 5 seeds; small dots = individual seeds")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log(f"  saved {out_path.name}")


# ---- Main -------------------------------------------------------------------
def main() -> int:
    log("=" * 76)
    log("Path B Week 1 — Multi-seed aggregation & figures")
    log("=" * 76)

    ablation_long, per_target_long = load_per_seed()
    log(f"  loaded {len(ablation_long)} ablation rows, {len(per_target_long)} per-target rows")

    summary = aggregate_variant_level(ablation_long)
    summary.to_csv(SUMMARY_DIR / "multi_seed_summary.csv", index=False)
    log(f"  saved {SUMMARY_DIR}/multi_seed_summary.csv ({len(summary)} variants)")

    per_target_summary = aggregate_per_target(per_target_long)
    per_target_summary.to_csv(SUMMARY_DIR / "multi_seed_per_target_summary.csv", index=False)
    log(f"  saved {SUMMARY_DIR}/multi_seed_per_target_summary.csv ({len(per_target_summary)} rows)")

    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_seed_stability_per_variant(
        ablation_long, summary, figures_dir / "seed_stability_per_variant.png"
    )
    fig_seed_stability_per_target(
        per_target_long, figures_dir / "seed_stability_per_target.png"
    )
    fig_multi_seed_pareto(
        ablation_long, summary, figures_dir / "multi_seed_pareto.png"
    )

    # Headline summary line for run log
    ebp95 = summary[summary["variant"] == "entropy_benign_p95"].iloc[0]
    log("")
    log("=" * 76)
    log("HEADLINE — entropy_benign_p95 across 5 seeds:")
    log(
        f"  H2-strict avg: {ebp95['h2_strict_avg_mean']:.4f} ± "
        f"{ebp95['h2_strict_avg_std']:.4f} "
        f"(min={ebp95['h2_strict_avg_min']:.4f}, max={ebp95['h2_strict_avg_max']:.4f})"
    )
    log(
        f"  Strict pass count (seeds achieving 4/4): "
        f"{ebp95['strict_pass_count']}/5"
    )
    log(
        f"  Operational FPR: {ebp95['avg_false_alert_rate_mean']:.4f} ± "
        f"{ebp95['avg_false_alert_rate_std']:.4f}"
    )
    log("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
