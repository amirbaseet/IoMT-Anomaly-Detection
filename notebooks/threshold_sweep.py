#!/usr/bin/env python3
"""
Path B Week 2 — Task 1: Continuous Entropy Threshold Sweep
==========================================================

Closes senior review item §1.4: Phase 6C selected `entropy_benign_p95` from a
discrete grid {p90, p95, p97, p99}. This script sweeps continuously across 29
thresholds at 0.5-percentile resolution between p85.0 and p99.0 to identify
the empirical operating-point optimum.

For each threshold, applies the Phase 6C 5-case entropy fusion logic on the
seed=42 baseline LOO predictions and records:
  - h2_strict_avg     — mean rescue recall over 4 eligible targets
  - h2_strict_pass    — count of eligible targets at recall ≥ 0.70 (k/4)
  - h2_binary_avg     — mean any-alert recall over all 5 targets
  - avg_false_alert_rate — mean fusion-level FPR over 5 targets
  - per-target detail (rescue, binary, FPR)

Reproducibility guard: at percentile=95.0 the script asserts that
h2_strict_avg matches the canonical Phase 6C value (0.8035264623662012)
within 1e-9. If it doesn't, the copied fusion functions have drifted from
enhanced_fusion.py and the script aborts before sweeping. (Same guardrail
design as multi_seed_fusion.py:457-470.)

This task does NOT retrain anything. It re-applies fusion to saved arrays.

Inputs (read-only):
  preprocessed/full_features/y_test.csv, y_val.csv
  preprocessed/label_encoders.json
  results/supervised/predictions/E7_val_proba.npy
  results/zero_day_loo/predictions/loo_<target>_test_pred.npy
  results/zero_day_loo/predictions/loo_<target>_test_proba.npy
  results/zero_day_loo/models/loo_label_map_<target>.json
  results/unsupervised/scores/ae_test_mse.npy
  results/unsupervised/thresholds.json

Outputs:
  results/enhanced_fusion/threshold_sweep/
    sweep_table.csv             # 29 rows × ~10 cols (aggregate per threshold)
    sweep_per_target.csv        # 29×5 = 145 rows (per (threshold, target) detail)
    pareto_continuous.png       # scatter (FPR, strict_avg) + discrete-grid overlay
    strict_avg_vs_threshold.png # line plot of strict_avg vs percentile

Smoke gate:
    SMOKE=1 python notebooks/threshold_sweep.py
    (3 thresholds {p85, p92.5, p99}, no plots — should complete in <10s)

Full run:
    python notebooks/threshold_sweep.py
    (~30s on M4, no GPU)
"""

# %% SECTION 0 — Imports
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reproducibility / plot styling — match enhanced_fusion.py conventions.
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
UNSUPERVISED_DIR = ROOT / "results" / "unsupervised"
OUTPUT_DIR = ROOT / "results" / "enhanced_fusion" / "threshold_sweep"

ZERO_DAY_TARGETS = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "MQTT_DoS_Connect_Flood",
    "ARP_Spoofing",
]
# H2-strict eligibility: see Phase 6B for justification of MQTT_DoS_Connect_Flood
# exclusion (0% LOO→Benign rows ⇒ empty rescue denominator).
H2_STRICT_ELIGIBLE = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "ARP_Spoofing",
]
H2_STRICT_MIN_BENIGN_N = 30
H2_PASS_THRESHOLD = 0.70

# Reproducibility guardrail — entropy_benign_p95 strict avg under seed=42 baseline.
# Matches multi_seed_fusion.py:83-84.
SEED42_REFERENCE_STRICT_AVG_P95 = 0.8035264623662012
SEED42_REFERENCE_TOLERANCE = 1e-9

# Continuous percentile grid: 85.0, 85.5, …, 99.0  → 29 thresholds.
PERCENTILE_GRID = np.arange(85.0, 99.5, 0.5)
assert len(PERCENTILE_GRID) == 29, f"expected 29 percentiles, got {len(PERCENTILE_GRID)}"

# Discrete-grid overlay for the Pareto figure (matches Phase 6C ENTROPY_PERCENTILES).
DISCRETE_GRID_PERCENTILES = (90.0, 95.0, 97.0, 99.0)

DETECTED_CASES = (1, 2, 3, 5)

SMOKE = bool(int(os.environ.get("SMOKE", "0")))


def log(msg: str, t0: float | None = None) -> None:
    elapsed = f" [+{time.time() - t0:6.1f}s]" if t0 is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)


# %% SECTION 2 — Pure functions copied from existing scripts
#
# We deliberately copy rather than import because both source modules execute
# their full pipelines at import time. Re-running enhanced_fusion.py / pareto
# logic on import would defeat the "no retrain, no recompute" rule. The copies
# below are byte-identical to the source (by inspection); the reproducibility
# guard at p95 catches any silent drift.


# Copied verbatim from enhanced_fusion.py:338-341.
def compute_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy of per-row probability vector. Higher = more uncertain."""
    p = np.clip(proba, 1e-10, 1.0)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)


# Copied verbatim from enhanced_fusion.py:499-512.
def entropy_fusion(
    sup_pred: np.ndarray,
    ae_binary: np.ndarray,
    entropy: np.ndarray,
    ent_threshold: float,
    benign_id: int,
) -> np.ndarray:
    """High entropy ⇒ model confused ⇒ potential novel attack."""
    sup_attack = sup_pred != benign_id
    high_entropy = entropy > ent_threshold
    return np.where(sup_attack & ae_binary & ~high_entropy, 1,
           np.where(~sup_attack & ae_binary, 2,
           np.where(high_entropy & ae_binary, 2,
           np.where(high_entropy & ~ae_binary, 5,
           np.where(sup_attack & ~ae_binary & ~high_entropy, 3,
                                                              4)))))


# Copied verbatim from pareto_frontier.py:33-51.
def pareto_optimal_indices(fpr: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Indices of points on the Pareto frontier (low FPR, high recall)."""
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


# %% SECTION 3 — Load seed=42 baseline signals (precompute once)


def _extract_global_class_map(encoders: dict) -> Dict[str, int]:
    """Mirrors enhanced_fusion.py:_extract_global_class_map for schema robustness."""
    if all(isinstance(v, int) for v in encoders.values()):
        return {str(k): int(v) for k, v in encoders.items()}
    for key in ("label", "y", "class", "multiclass", "target"):
        if key in encoders and isinstance(encoders[key], dict):
            sub = encoders[key]
            if all(isinstance(v, int) for v in sub.values()):
                return {str(k): int(v) for k, v in sub.items()}
            if "classes_" in sub:
                return {str(c): i for i, c in enumerate(sub["classes_"])}
    if "classes_" in encoders:
        return {str(c): i for i, c in enumerate(encoders["classes_"])}
    raise ValueError(
        f"Cannot extract global class map from encoders.json — keys = "
        f"{list(encoders.keys())[:10]}"
    )


def load_baseline_signals() -> dict:
    """Load the seed=42 baseline arrays needed by the entropy-fusion sweep.

    Returns dict with:
      e7_val_entropy:        E7 softmax entropy on val (for threshold derivation)
      benign_val_mask:       boolean mask over y_val for "Benign" rows
      y_test_labels:         string labels for X_test rows (for target/benign masks)
      ae_test_binary:        boolean (ae_test_mse > AE_T_P90)
      loo_test_pred_global:  {target: int32 (N_test,) global-id-mapped predictions}
      loo_test_entropy:      {target: float32 (N_test,) Shannon entropy}
      global_benign_id:      int (Benign id in the full 19-class space)
    """
    t0 = time.time()
    log("Loading seed=42 baseline signals ...")

    # Label encoders → global class map.
    with open(PREPROCESSED_DIR / "label_encoders.json") as f:
        encoders = json.load(f)
    global_class_map = _extract_global_class_map(encoders)
    if "Benign" not in global_class_map:
        raise RuntimeError("Expected 'Benign' in global class map.")
    global_benign_id = global_class_map["Benign"]
    log(f"  global label space: {len(global_class_map)} classes; benign_id={global_benign_id}")

    # y_val / y_test as string labels.
    y_val_df = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_val.csv")
    y_test_df = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_test.csv")
    label_col = "label" if "label" in y_val_df.columns else y_val_df.columns[0]
    y_val_labels = y_val_df[label_col].astype(str).values
    y_test_labels = y_test_df[label_col].astype(str).values
    benign_val_mask = y_val_labels == "Benign"
    log(
        f"  y_val={len(y_val_labels):,}  benign_val={int(benign_val_mask.sum()):,}  "
        f"y_test={len(y_test_labels):,}"
    )

    # E7 val entropy → drives threshold percentile derivation.
    e7_val_proba = np.load(SUPERVISED_DIR / "predictions" / "E7_val_proba.npy")
    e7_val_entropy = compute_entropy(e7_val_proba)
    del e7_val_proba

    # AE test binary at p90.
    ae_test_mse = np.load(UNSUPERVISED_DIR / "scores" / "ae_test_mse.npy")
    with open(UNSUPERVISED_DIR / "thresholds.json") as f:
        ae_thresholds_doc = json.load(f)
    ae_t_p90 = float(ae_thresholds_doc["thresholds"]["p90"])
    ae_test_binary = (ae_test_mse > ae_t_p90)
    log(f"  AE p90 threshold = {ae_t_p90:.6f}  flag-rate(test)={float(ae_test_binary.mean()):.4f}")
    del ae_test_mse

    # Per-target LOO test predictions + entropy (all in global label space).
    loo_test_pred_global: Dict[str, np.ndarray] = {}
    loo_test_entropy: Dict[str, np.ndarray] = {}
    for tgt in ZERO_DAY_TARGETS:
        # Local class names → ordered by local index (the .npy proba columns).
        with open(LOO_DIR / "models" / f"loo_label_map_{tgt}.json") as f:
            label_map = json.load(f)
        # label_map is {class_name: local_idx}; invert to local→name list.
        local_to_name = sorted(label_map.items(), key=lambda kv: kv[1])
        local_to_global = np.array(
            [global_class_map[name] for name, _ in local_to_name], dtype=np.int32,
        )

        proba = np.load(LOO_DIR / "predictions" / f"loo_{tgt}_test_proba.npy")
        pred_local = np.load(LOO_DIR / "predictions" / f"loo_{tgt}_test_pred.npy")
        if proba.shape[1] != len(local_to_global):
            raise RuntimeError(
                f"{tgt}: proba has {proba.shape[1]} cols but label_map has "
                f"{len(local_to_global)} entries"
            )
        loo_test_entropy[tgt] = compute_entropy(proba)
        loo_test_pred_global[tgt] = local_to_global[pred_local]
        log(
            f"  {tgt:30s}  mean_ent={loo_test_entropy[tgt].mean():.4f}  "
            f"benign_pred_frac={float((loo_test_pred_global[tgt] == global_benign_id).mean()):.4f}"
        )
        del proba, pred_local

    log(f"signal loading: {time.time() - t0:.1f}s")
    return {
        "e7_val_entropy": e7_val_entropy,
        "benign_val_mask": benign_val_mask,
        "y_test_labels": y_test_labels,
        "ae_test_binary": ae_test_binary,
        "loo_test_pred_global": loo_test_pred_global,
        "loo_test_entropy": loo_test_entropy,
        "global_benign_id": int(global_benign_id),
    }


# %% SECTION 4 — Per-threshold evaluation


def evaluate_one_threshold(
    pct: float,
    ent_threshold: float,
    signals: dict,
) -> tuple[dict, List[dict]]:
    """Return (aggregate_row, per_target_rows) for one entropy threshold.

    Aggregate row matches the schema of enhanced_fusion.py's ablation_table
    for the entropy_benign_p* family: h2_strict_avg averaged over the
    H2_STRICT_ELIGIBLE subset with n_loo_benign ≥ 30; h2_binary_avg averaged
    over all 5 targets; avg_false_alert_rate averaged over all 5 targets.
    """
    benign_id = signals["global_benign_id"]
    ae_binary = signals["ae_test_binary"]
    y_test_labels = signals["y_test_labels"]
    benign_test_mask = (y_test_labels == "Benign")

    per_target_rows: List[dict] = []
    for tgt in ZERO_DAY_TARGETS:
        target_mask = (y_test_labels == tgt)
        sup_pred = signals["loo_test_pred_global"][tgt]
        entropy = signals["loo_test_entropy"][tgt]

        cases = entropy_fusion(
            sup_pred=sup_pred,
            ae_binary=ae_binary,
            entropy=entropy,
            ent_threshold=ent_threshold,
            benign_id=benign_id,
        )

        # H2-binary: any-alert recall over ALL target rows.
        target_cases = cases[target_mask]
        h2_binary = float(np.isin(target_cases, DETECTED_CASES).mean()) if target_mask.sum() else float("nan")

        # H2-strict: rescue recall on the LOO→Benign subset for this target.
        loo_benign_target_mask = target_mask & (sup_pred == benign_id)
        n_loo_benign = int(loo_benign_target_mask.sum())
        if n_loo_benign >= H2_STRICT_MIN_BENIGN_N:
            sub_cases = cases[loo_benign_target_mask]
            h2_strict = float(np.isin(sub_cases, DETECTED_CASES).mean())
        else:
            h2_strict = float("nan")

        # Operational FPR proxy: detected-case rate on benign-test (per target —
        # depends on per-target sup_pred and per-target entropy).
        false_alert = float(
            np.isin(cases[benign_test_mask], DETECTED_CASES).mean()
        ) if benign_test_mask.sum() else float("nan")

        per_target_rows.append({
            "percentile": pct,
            "ent_threshold": ent_threshold,
            "target": tgt,
            "n_target": int(target_mask.sum()),
            "n_loo_benign": n_loo_benign,
            "h2_strict_rescue_recall": h2_strict,
            "h2_binary_recall": h2_binary,
            "false_alert_rate_benign": false_alert,
        })

    pdf = pd.DataFrame(per_target_rows)

    # Aggregate H2-strict over eligible-and-evaluated subset.
    strict_sub = pdf[
        pdf["target"].isin(H2_STRICT_ELIGIBLE)
        & (pdf["n_loo_benign"] >= H2_STRICT_MIN_BENIGN_N)
        & pdf["h2_strict_rescue_recall"].notna()
    ]
    n_strict_evaluated = len(strict_sub)
    n_strict_pass = int((strict_sub["h2_strict_rescue_recall"] >= H2_PASS_THRESHOLD).sum())
    h2_strict_avg = float(strict_sub["h2_strict_rescue_recall"].mean()) if len(strict_sub) else float("nan")

    h2_binary_avg = float(pdf["h2_binary_recall"].mean())
    avg_false_alert_rate = float(pdf["false_alert_rate_benign"].mean())

    aggregate_row = {
        "percentile": pct,
        "ent_threshold": ent_threshold,
        "h2_strict_pass": f"{n_strict_pass}/4",
        "h2_strict_pass_int": n_strict_pass,
        "h2_strict_evaluated": n_strict_evaluated,
        "h2_strict_avg": h2_strict_avg,
        "h2_binary_avg": h2_binary_avg,
        "avg_false_alert_rate": avg_false_alert_rate,
    }
    return aggregate_row, per_target_rows


# %% SECTION 5 — Reproducibility guard


def assert_p95_reproduces(signals: dict) -> None:
    """Replay entropy_benign_p95 evaluation and assert strict_avg matches the
    canonical Phase 6C value within 1e-9. If this fails, the copied fusion
    functions have drifted from enhanced_fusion.py."""
    benign_val_entropy = signals["e7_val_entropy"][signals["benign_val_mask"]]
    p95_threshold = float(np.percentile(benign_val_entropy, 95.0))
    aggregate, _ = evaluate_one_threshold(95.0, p95_threshold, signals)
    actual = aggregate["h2_strict_avg"]
    diff = abs(actual - SEED42_REFERENCE_STRICT_AVG_P95)
    if diff > SEED42_REFERENCE_TOLERANCE:
        raise RuntimeError(
            f"[reproducibility guard] entropy_benign_p95 strict_avg drift!\n"
            f"  actual:    {actual!r}\n"
            f"  reference: {SEED42_REFERENCE_STRICT_AVG_P95!r}\n"
            f"  diff:      {diff:.3e}  (tolerance {SEED42_REFERENCE_TOLERANCE:.0e})\n"
            f"The copied fusion functions have drifted from enhanced_fusion.py.\n"
            f"Aborting before sweeping — the sweep results would be unreliable."
        )
    log(
        f"  ✓ p95 reproducibility guard PASSED  "
        f"(actual={actual!r}, diff={diff:.3e})"
    )


# %% SECTION 6 — Plots


def plot_pareto_continuous(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter (FPR, strict_avg) for all 29 thresholds with the discrete-grid
    overlay highlighted in distinct markers."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

    fpr = df["avg_false_alert_rate"].to_numpy(dtype=float)
    rec = df["h2_strict_avg"].to_numpy(dtype=float)
    pct = df["percentile"].to_numpy(dtype=float)

    # Pareto frontier among the 29 continuous points.
    front_idx = pareto_optimal_indices(fpr, rec)
    front_order = front_idx[np.argsort(fpr[front_idx])]
    dominated_mask = np.ones(len(df), dtype=bool)
    dominated_mask[front_order] = False

    ax.scatter(
        fpr[dominated_mask], rec[dominated_mask],
        s=70, c="#9aa0a6", edgecolors="#3c4043", linewidths=0.6, alpha=0.85,
        label="Continuous sweep (dominated)", zorder=2,
    )
    ax.plot(
        fpr[front_order], rec[front_order],
        color="#c0392b", linewidth=1.8, alpha=0.9, zorder=3,
    )
    ax.scatter(
        fpr[front_order], rec[front_order],
        s=110, c="#c0392b", edgecolors="white", linewidths=0.8,
        label="Continuous Pareto frontier", zorder=4,
    )

    # Discrete-grid overlay: {p90, p95, p97, p99}.
    grid_markers = {
        90.0: ("o", "#1f77b4", "p90 (discrete grid)"),
        95.0: ("s", "#2ca02c", "p95 (current operating point)"),
        97.0: ("D", "#9467bd", "p97 (discrete grid)"),
        99.0: ("^", "#ff7f0e", "p99 (discrete grid)"),
    }
    for grid_pct, (marker, color, label) in grid_markers.items():
        # Find the row in our continuous sweep whose percentile == grid_pct.
        match = df[df["percentile"] == grid_pct]
        if not len(match):
            continue
        row = match.iloc[0]
        ax.scatter(
            row["avg_false_alert_rate"], row["h2_strict_avg"],
            marker=marker, s=220, c=color, edgecolors="black", linewidths=1.4,
            zorder=5, label=label,
        )

    # Annotate the empirical optimum (max strict_avg).
    best_idx = int(np.nanargmax(rec))
    ax.annotate(
        f"empirical max\np{pct[best_idx]:.1f}\nstrict={rec[best_idx]:.4f}\nFPR={fpr[best_idx]:.4f}",
        xy=(fpr[best_idx], rec[best_idx]),
        xytext=(20, -50), textcoords="offset points",
        fontsize=9, ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#f57c00", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#f57c00", lw=0.8),
    )

    # 0.70 H2-strict pass-mark reference.
    ax.axhline(H2_PASS_THRESHOLD, color="#888", linestyle=":", linewidth=1.0,
               label=f"H2-strict pass mark ({H2_PASS_THRESHOLD:.2f})")

    ax.set_xlabel("Operational FPR on benign-test (fusion-level, mean over 5 targets)")
    ax.set_ylabel("H2-strict rescue avg (over 4 eligible targets)")
    ax.set_title(
        "Phase 6C addendum — continuous entropy threshold sweep\n"
        "29 thresholds at p85.0–p99.0 (Δ=0.5pp), seed=42 baseline"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log(f"  saved → {out_path}")


def plot_strict_avg_vs_threshold(df: pd.DataFrame, out_path: Path) -> None:
    """Strict avg as a function of percentile, with the 0.70 reference line and
    the current p95 vertical reference."""
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    pct = df["percentile"].to_numpy(dtype=float)
    rec = df["h2_strict_avg"].to_numpy(dtype=float)
    fpr = df["avg_false_alert_rate"].to_numpy(dtype=float)

    # Primary y-axis: strict avg.
    ax1.plot(pct, rec, color="#c0392b", linewidth=1.8, marker="o", markersize=4,
             label="H2-strict rescue avg")
    ax1.axhline(H2_PASS_THRESHOLD, color="#666", linestyle=":", linewidth=1.0,
                label=f"H2-strict pass mark ({H2_PASS_THRESHOLD:.2f})")
    ax1.axvline(95.0, color="#2ca02c", linestyle="--", linewidth=1.0,
                label="Current p95 operating point")
    ax1.set_xlabel("Benign-val entropy percentile")
    ax1.set_ylabel("H2-strict rescue avg", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")

    # Secondary y-axis: avg false alert rate.
    ax2 = ax1.twinx()
    ax2.plot(pct, fpr, color="#1f77b4", linewidth=1.4, alpha=0.7,
             marker="s", markersize=3, label="Avg FPR (benign-test)")
    ax2.set_ylabel("Mean false-alert rate on benign-test", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_title(
        "H2-strict rescue avg vs entropy threshold percentile\n"
        "Seed=42 baseline, 29 thresholds (p85.0..p99.0)"
    )
    ax1.grid(True, alpha=0.3)

    # Combine legends.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log(f"  saved → {out_path}")


# %% SECTION 7 — Main orchestration


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("=" * 76)
    log("Path B Week 2 — Task 1: Continuous Entropy Threshold Sweep")
    log(f"  SMOKE={int(SMOKE)}  output_dir={OUTPUT_DIR}")
    log("=" * 76)

    signals = load_baseline_signals()

    # Reproducibility guard FIRST — abort before sweeping if drift detected.
    log("Running p95 reproducibility guard ...")
    assert_p95_reproduces(signals)

    # Decide grid: full or smoke-shrunk.
    if SMOKE:
        grid = np.array([85.0, 92.5, 99.0])
        log(f"[SMOKE] sweeping {len(grid)} thresholds: {grid.tolist()}")
    else:
        grid = PERCENTILE_GRID
        log(f"Sweeping {len(grid)} thresholds at 0.5pp resolution from {grid[0]} to {grid[-1]}")

    # Derive thresholds from benign-val entropy.
    benign_val_entropy = signals["e7_val_entropy"][signals["benign_val_mask"]]
    log(
        f"  benign_val entropy: n={len(benign_val_entropy):,}  "
        f"mean={benign_val_entropy.mean():.4f}  median={np.median(benign_val_entropy):.4f}"
    )

    aggregate_rows: List[dict] = []
    per_target_all: List[dict] = []
    for pct in grid:
        ent_t = float(np.percentile(benign_val_entropy, pct))
        agg, ptr = evaluate_one_threshold(float(pct), ent_t, signals)
        aggregate_rows.append(agg)
        per_target_all.extend(ptr)
        log(
            f"  p={pct:5.1f}  ent_t={ent_t:.4f}  "
            f"strict_pass={agg['h2_strict_pass']}  strict_avg={agg['h2_strict_avg']:.4f}  "
            f"binary_avg={agg['h2_binary_avg']:.4f}  FPR={agg['avg_false_alert_rate']:.4f}",
            t0=t0,
        )

    sweep_df = pd.DataFrame(aggregate_rows)
    per_target_df = pd.DataFrame(per_target_all)

    # Save tables.
    sweep_csv = OUTPUT_DIR / "sweep_table.csv"
    per_target_csv = OUTPUT_DIR / "sweep_per_target.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    per_target_df.to_csv(per_target_csv, index=False)
    log(f"  saved → {sweep_csv}  ({len(sweep_df)} rows)")
    log(f"  saved → {per_target_csv}  ({len(per_target_df)} rows)")

    # Highlight: empirical max strict_avg + comparison to discrete grid.
    best_idx = int(sweep_df["h2_strict_avg"].idxmax())
    best_row = sweep_df.iloc[best_idx]
    log("")
    log(f"  empirical max strict_avg over the sweep:")
    log(
        f"    p={best_row['percentile']:.1f}  "
        f"strict_avg={best_row['h2_strict_avg']:.6f}  "
        f"strict_pass={best_row['h2_strict_pass']}  "
        f"FPR={best_row['avg_false_alert_rate']:.4f}"
    )
    p95_row = sweep_df[sweep_df["percentile"] == 95.0]
    if len(p95_row):
        p95 = p95_row.iloc[0]
        log("  reference p95 row:")
        log(
            f"    p=95.0  strict_avg={p95['h2_strict_avg']:.6f}  "
            f"strict_pass={p95['h2_strict_pass']}  FPR={p95['avg_false_alert_rate']:.4f}"
        )
    log("")

    # Plots — skipped in SMOKE mode.
    if not SMOKE:
        plot_pareto_continuous(sweep_df, OUTPUT_DIR / "pareto_continuous.png")
        plot_strict_avg_vs_threshold(sweep_df, OUTPUT_DIR / "strict_avg_vs_threshold.png")
    else:
        log("[SMOKE] skipping plots")

    log(f"DONE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
