#!/usr/bin/env python3
"""
Path B Week 2 — Task 2: Per-Fold KS Test for Benign Val→Test Entropy Shift
==========================================================================

Closes senior review item §1.5: §15C.10 reports a single aggregate KS = 0.0645
on E7 entropy over benign rows. This script breaks that aggregate down per
LOO fold to identify whether the shift is uniform across folds or driven by
specific ones.

For each of the 5 LOO targets, this script:
  1. Loads the LOO-XGBoost model (`loo_xgb_without_<target>.pkl`).
  2. Runs `predict_proba(X_val)` to obtain that fold's val probabilities.
     (NOTE: LOO val proba is NOT saved on disk — only test proba is — so we
      regenerate val proba inline. ~30s per fold × 5 folds ≈ 2.5 min.)
  3. Computes Shannon entropy on benign-val rows.
  4. Loads `loo_<target>_test_proba.npy`, computes entropy on benign-test rows.
  5. Runs scipy.stats.ks_2samp on the two benign entropy distributions.
  6. Frees the model (del + gc.collect()) before the next fold.

A 6th "AGGREGATE_E7" row uses E7's full-19-class val/test proba directly and
should reproduce KS ≈ 0.0645 from §15C.10.

Important interpretation note: at n_benign ≈ 38,546 (val) / 37,607 (test),
a KS statistic of 6%–7% will yield p < 1e-60 even though it represents a
small-to-moderate effect size. The KS *statistic* is the comparable signal,
NOT the p-value, at this sample size.

This task does NOT retrain anything. predict_proba is inference-only.

Inputs (read-only):
  preprocessed/full_features/X_val.npy
  preprocessed/full_features/y_val.csv, y_test.csv
  results/supervised/predictions/E7_val_proba.npy
  results/supervised/predictions/E7_test_proba.npy
  results/zero_day_loo/models/loo_xgb_without_<target>.pkl
  results/zero_day_loo/predictions/loo_<target>_test_proba.npy

Outputs:
  results/enhanced_fusion/ks_per_fold/
    ks_per_fold.csv     # 6 rows (5 LOO folds + 1 AGGREGATE_E7)
    ks_per_fold.png     # bar chart of KS per fold + horizontal aggregate line

Smoke gate:
    SMOKE=1 python notebooks/ks_per_fold.py
    (1 target = Recon_Ping_Sweep, no aggregate, no plot — should complete in <60s)

Full run:
    python notebooks/ks_per_fold.py
    (~3-4 min on M4, no GPU; ~150 MB peak per loaded model)
"""

# %% SECTION 0 — Imports
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9


# %% SECTION 1 — Configuration

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = ROOT / "preprocessed"
SUPERVISED_DIR = ROOT / "results" / "supervised"
LOO_DIR = ROOT / "results" / "zero_day_loo"
OUTPUT_DIR = ROOT / "results" / "enhanced_fusion" / "ks_per_fold"

ZERO_DAY_TARGETS = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "MQTT_DoS_Connect_Flood",
    "ARP_Spoofing",
]

SMOKE = bool(int(os.environ.get("SMOKE", "0")))


def log(msg: str, t0: float | None = None) -> None:
    elapsed = f" [+{time.time() - t0:6.1f}s]" if t0 is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)


# %% SECTION 2 — Pure helpers


def compute_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy of per-row probability vector. Higher = more uncertain.

    Copied verbatim from enhanced_fusion.py:338-341 to keep the script
    self-contained and avoid triggering enhanced_fusion's module-level pipeline
    on import.
    """
    p = np.clip(proba, 1e-10, 1.0)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)


def ks_metrics(target: str, val_ent_benign: np.ndarray, test_ent_benign: np.ndarray) -> dict:
    """Run two-sample KS on benign-val vs benign-test entropy and pack metrics."""
    ks_stat, ks_p = ks_2samp(val_ent_benign, test_ent_benign)
    return {
        "target": target,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "n_val_benign": int(len(val_ent_benign)),
        "n_test_benign": int(len(test_ent_benign)),
        "val_mean": float(val_ent_benign.mean()),
        "test_mean": float(test_ent_benign.mean()),
        "val_p95": float(np.percentile(val_ent_benign, 95.0)),
        "test_p95": float(np.percentile(test_ent_benign, 95.0)),
        "delta_mean": float(test_ent_benign.mean() - val_ent_benign.mean()),
    }


# %% SECTION 3 — Per-fold computation


def loo_fold_entropy(
    target: str, X_val: np.ndarray, benign_val_mask: np.ndarray, benign_test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (val_ent_benign, test_ent_benign) for one LOO fold.

    Loads the fold's XGBoost model, runs predict_proba on X_val, computes
    entropy on benign-val rows; loads the saved test proba, computes entropy
    on benign-test rows. Frees the model afterwards.
    """
    t0 = time.time()
    model_path = LOO_DIR / "models" / f"loo_xgb_without_{target}.pkl"
    test_proba_path = LOO_DIR / "predictions" / f"loo_{target}_test_proba.npy"
    log(f"  [{target}] loading model: {model_path.name}")
    model = joblib.load(model_path)

    log(f"  [{target}] predict_proba on X_val ({X_val.shape}) ...")
    val_proba = model.predict_proba(X_val).astype(np.float32, copy=False)
    val_entropy = compute_entropy(val_proba)
    val_ent_benign = val_entropy[benign_val_mask]
    del val_proba, val_entropy

    # Free the model BEFORE loading test proba — keeps peak memory predictable.
    del model
    gc.collect()

    log(f"  [{target}] loading test proba: {test_proba_path.name}")
    test_proba = np.load(test_proba_path)
    test_entropy = compute_entropy(test_proba)
    test_ent_benign = test_entropy[benign_test_mask]
    del test_proba, test_entropy
    gc.collect()

    log(
        f"  [{target}] benign val: n={len(val_ent_benign):,} mean={val_ent_benign.mean():.4f} "
        f"p95={np.percentile(val_ent_benign, 95):.4f}  |  "
        f"benign test: n={len(test_ent_benign):,} mean={test_ent_benign.mean():.4f} "
        f"p95={np.percentile(test_ent_benign, 95):.4f}",
        t0=t0,
    )
    return val_ent_benign, test_ent_benign


# %% SECTION 4 — Plot


def plot_ks_bar(df: pd.DataFrame, agg_ks: float, out_path: Path) -> None:
    """Per-fold KS statistic bar chart with the aggregate (E7) reference line."""
    fold_df = df[df["target"] != "AGGREGATE_E7"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(fold_df))
    bars = ax.bar(
        x, fold_df["ks_statistic"].values,
        color="#1f77b4", edgecolor="#0d3a66", linewidth=0.8, width=0.6, zorder=3,
    )
    ax.axhline(
        agg_ks, color="#c0392b", linestyle="--", linewidth=1.4, zorder=2,
        label=f"E7 aggregate (§15C.10)  KS = {agg_ks:.4f}",
    )

    # Annotate each bar with KS statistic + scientific-notation p-value.
    for i, (bar, row) in enumerate(zip(bars, fold_df.itertuples(index=False))):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.0015,
            f"KS={row.ks_statistic:.4f}\np={row.ks_pvalue:.1e}",
            ha="center", va="bottom", fontsize=8.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(fold_df["target"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("KS statistic (benign val vs benign test entropy)")
    ax.set_title(
        "Per-fold KS statistic for benign val→test entropy shift\n"
        "(LOO-XGBoost entropy on benign rows, seed=42 baseline)"
    )
    ax.set_ylim(0, max(fold_df["ks_statistic"].max(), agg_ks) * 1.35)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log(f"  saved → {out_path}")


# %% SECTION 5 — Main orchestration


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("=" * 76)
    log("Path B Week 2 — Task 2: Per-Fold KS Test (benign val→test entropy shift)")
    log(f"  SMOKE={int(SMOKE)}  output_dir={OUTPUT_DIR}")
    log("=" * 76)

    # Static loads.
    X_val = np.load(PREPROCESSED_DIR / "full_features" / "X_val.npy")
    y_val_df = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_val.csv")
    y_test_df = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_test.csv")
    label_col = "label" if "label" in y_val_df.columns else y_val_df.columns[0]
    y_val_labels = y_val_df[label_col].astype(str).values
    y_test_labels = y_test_df[label_col].astype(str).values
    benign_val_mask = (y_val_labels == "Benign")
    benign_test_mask = (y_test_labels == "Benign")
    log(
        f"  X_val={X_val.shape}  benign_val={int(benign_val_mask.sum()):,}  "
        f"benign_test={int(benign_test_mask.sum()):,}"
    )

    targets_to_run = [ZERO_DAY_TARGETS[0]] if SMOKE else ZERO_DAY_TARGETS
    if SMOKE:
        log(f"[SMOKE] limiting to 1 target: {targets_to_run[0]}")

    rows: List[dict] = []
    for tgt in targets_to_run:
        val_ent, test_ent = loo_fold_entropy(
            tgt, X_val, benign_val_mask, benign_test_mask,
        )
        rows.append(ks_metrics(tgt, val_ent, test_ent))
        log(
            f"  [{tgt}] KS={rows[-1]['ks_statistic']:.4f}  "
            f"p={rows[-1]['ks_pvalue']:.2e}  "
            f"Δmean={rows[-1]['delta_mean']:+.4f}",
            t0=t0,
        )

    # Aggregate row using E7 (full-19-class) entropy. Reproduces §15C.10 KS≈0.0645.
    if not SMOKE:
        log("Computing AGGREGATE_E7 row from E7 val/test proba ...")
        e7_val_proba = np.load(SUPERVISED_DIR / "predictions" / "E7_val_proba.npy")
        e7_test_proba = np.load(SUPERVISED_DIR / "predictions" / "E7_test_proba.npy")
        e7_val_ent_benign = compute_entropy(e7_val_proba)[benign_val_mask]
        e7_test_ent_benign = compute_entropy(e7_test_proba)[benign_test_mask]
        del e7_val_proba, e7_test_proba
        agg_row = ks_metrics("AGGREGATE_E7", e7_val_ent_benign, e7_test_ent_benign)
        rows.append(agg_row)
        log(
            f"  [AGGREGATE_E7] KS={agg_row['ks_statistic']:.4f}  "
            f"p={agg_row['ks_pvalue']:.2e}  Δmean={agg_row['delta_mean']:+.4f}  "
            f"(expected KS ≈ 0.0645 from §15C.10)",
            t0=t0,
        )

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "ks_per_fold.csv"
    df.to_csv(csv_path, index=False)
    log(f"  saved → {csv_path}  ({len(df)} rows)")

    log("")
    log("Per-fold KS table:")
    log("\n" + df[[
        "target", "ks_statistic", "ks_pvalue",
        "n_val_benign", "n_test_benign",
        "val_mean", "test_mean", "val_p95", "test_p95",
    ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log("")

    if not SMOKE and len(df) >= 2:
        agg_ks = float(df.loc[df["target"] == "AGGREGATE_E7", "ks_statistic"].iloc[0])
        plot_ks_bar(df, agg_ks, OUTPUT_DIR / "ks_per_fold.png")
    else:
        log("[SMOKE] skipping plot")

    log(f"DONE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
