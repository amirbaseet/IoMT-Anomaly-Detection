#!/usr/bin/env python3
"""
fusion_engine.py — Phase 6: Hybrid Fusion Engine & Zero-Day Simulation
========================================================================
Project : A Hybrid Supervised-Unsupervised Framework for Anomaly Detection
          and Zero-Day Attack Identification in IoMT Networks
Dataset : CICIoMT2024
Author  : Amro
Phase   : 6 (Fusion + Zero-Day Sim)
Version : 3 (H1 label-space bug fixed; verdict trichotomy applied throughout)

Combines:
  - Layer 1 (E7 / XGBoost) supervised predictions
  - Layer 2 (Autoencoder + Isolation Forest) unsupervised anomaly scores
into Layer 3 — a 4-case fusion decision engine — and evaluates
hypotheses H1 (fusion improves macro-F1; bootstrap-tested in 20-label
space) and H2 (AE catches what E7 misclassifies as benign on >= 50 % of
zero-day targets).

No retraining is performed. Pure post-hoc combination of saved arrays.

The `apply_fusion` and `fusion_classes` helpers are importable for
downstream phases (Phase 7 SHAP slicing).

Usage:
    cd ~/IoMT-Project
    source venv/bin/activate
    python notebooks/fusion_engine.py
"""

# %% [Section 1] Imports & Configuration ====================================
import warnings
warnings.filterwarnings("ignore")

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# ---- Paths ----------------------------------------------------------------
BASE_DIR         = Path(".").resolve()
SUPERVISED_DIR   = BASE_DIR / "results" / "supervised"
UNSUPERVISED_DIR = BASE_DIR / "results" / "unsupervised"
PREPROCESSED_DIR = BASE_DIR / "preprocessed"
OUTPUT_DIR       = BASE_DIR / "results" / "fusion"

CASE_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#bcbd22", 4: "#2ca02c"}
CASE_NAMES = {
    1: "Confirmed Alert",
    2: "Zero-Day Warning",
    3: "Low-Confidence Alert",
    4: "Clear",
}
CASE_DECISIONS  = {1: "confirmed_alert", 2: "zero_day_warning",
                   3: "low_confidence_alert", 4: "clear"}
CASE_CONFIDENCE = {1: "HIGH", 2: "MEDIUM_HIGH", 3: "MEDIUM_LOW", 4: "HIGH"}

# ---- Hypothesis test parameters ------------------------------------------
H1_BOOTSTRAP_ITERS = 200          # paired bootstrap iterations
H1_BOOTSTRAP_SEED  = 42
H2_RECALL_TARGET   = 0.70
H2_FRACTION_TARGET = 0.50
H2_MIN_SAMPLES     = 30           # min n_called_benign for reliable H2 metric

ZERO_DAY_TARGETS = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "MQTT_DoS_Connect_Flood",
    "ARP_Spoofing",
]

AE_THRESHOLD_PERCENTILES = [90, 95, 99]
PERCENTILES_SWEEP        = [50, 60, 70, 80, 85, 90, 92, 95, 97, 99]
RECOMMENDED_FPR_BUDGET   = 0.05   # selection criterion on val


# %% [Section 2] Helper functions (importable) ==============================
def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def md_table(df: pd.DataFrame, *, index: bool = False,
             floatfmt: str = ".4f") -> str:
    """Markdown table with graceful fallback if `tabulate` isn't installed."""
    try:
        return df.to_markdown(index=index, floatfmt=floatfmt)
    except Exception:
        return "```\n" + df.to_csv(index=index) + "```"


def npload(p: Path) -> np.ndarray:
    if not p.exists():
        sys.exit(f"[FATAL] Missing file: {p}")
    return np.load(p)


def normalise_if_binary(arr: np.ndarray) -> np.ndarray:
    """Coerce IF binary to {0=normal, 1=anomaly} regardless of source convention.

    sklearn IsolationForest.predict returns {-1, +1} (-1 = anomaly).
    Phase-5 README states the saved file is already converted, but we
    defend against either convention.
    """
    a = np.asarray(arr).astype(np.int8)
    uniq = set(np.unique(a).tolist())
    if uniq <= {-1, 1}:
        return (a == -1).astype(np.int8)
    if uniq <= {0, 1}:
        return a
    raise ValueError(f"Unexpected IF binary values: {uniq}")


def apply_fusion(sup_pred: np.ndarray, ae_binary: np.ndarray,
                 benign_id: int) -> np.ndarray:
    """Compute the 4-case fusion decision per sample.

    Cases:
      1 = Attack + Anomaly  -> Confirmed Alert     (HIGH)
      2 = Benign + Anomaly  -> Zero-Day Warning    (MEDIUM_HIGH)
      3 = Attack + Normal   -> Low-Confidence      (MEDIUM_LOW)
      4 = Benign + Normal   -> Clear               (HIGH)

    Args:
        sup_pred : (N,) supervised predicted class id
        ae_binary: (N,) 0 = normal, 1 = anomaly
        benign_id: class id of "Benign" in the multiclass encoding
    """
    sup_attack = sup_pred != benign_id
    ae_anom    = ae_binary == 1
    case = np.empty(len(sup_pred), dtype=np.int8)
    case[ sup_attack &  ae_anom] = 1
    case[~sup_attack &  ae_anom] = 2
    case[ sup_attack & ~ae_anom] = 3
    case[~sup_attack & ~ae_anom] = 4
    return case


def fusion_classes(sup_pred: np.ndarray, fusion_case: np.ndarray,
                   benign_id: int, zero_day_id: int) -> np.ndarray:
    """Final multiclass label per fusion case.

    Case 1, 3 -> use E7 predicted class
    Case 2    -> 'zero_day_unknown' (numeric id = `zero_day_id`)
    Case 4    -> Benign class id
    """
    out = sup_pred.copy().astype(np.int32)
    out[fusion_case == 2] = zero_day_id
    out[fusion_case == 4] = benign_id
    return out


def ae_binary_at(threshold: float, mse: np.ndarray) -> np.ndarray:
    return (mse > threshold).astype(np.int8)


def bootstrap_paired_f1(
    y_true: np.ndarray,
    preds_dict: dict,
    labels: list,
    n_boot: int = 200,
    seed: int = 42,
) -> dict:
    """Paired-resample bootstrap of macro-F1 for each prediction.

    Same resampled indices are reused across all predictions per iteration,
    enabling valid paired difference distributions and CIs.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    out = {name: np.empty(n_boot, dtype=np.float64) for name in preds_dict}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        for name, pred in preds_dict.items():
            out[name][b] = f1_score(
                y_b, pred[idx], labels=labels,
                average="macro", zero_division=0,
            )
        if (b + 1) % 50 == 0:
            print(f"    bootstrap iter {b + 1}/{n_boot}")
    return out


def ci_95(arr: np.ndarray) -> tuple:
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def h1_verdict_msg(passes: bool, delta_ci_hi: float) -> str:
    """Trichotomy verdict message for an H1 row."""
    if passes:
        return "PASS — Δ CI excludes 0 (positive)"
    if delta_ci_hi < 0:
        return "FAIL — Δ CI excludes 0 (negative; fusion hurts macro-F1)"
    return "FAIL — Δ CI spans 0"


# %% [Section 3] Main pipeline ==============================================
def main() -> None:
    for sub in ("fusion_results", "metrics", "figures"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
    })
    sns.set_style("whitegrid")

    banner("PHASE 6 — FUSION ENGINE & ZERO-DAY SIMULATION (v3)")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output  : {OUTPUT_DIR}")

    # ---- [3.1] Load inputs ------------------------------------------------
    banner("[3.1] Loading saved arrays from previous phases")

    # Supervised: only argmax preds are needed (proba arrays unused → skip)
    e7_val_pred  = npload(SUPERVISED_DIR / "predictions" / "E7_val_pred.npy")
    e7_test_pred = npload(SUPERVISED_DIR / "predictions" / "E7_test_pred.npy")

    # Unsupervised AE
    ae_val_mse     = npload(UNSUPERVISED_DIR / "scores" / "ae_val_mse.npy")
    ae_test_mse    = npload(UNSUPERVISED_DIR / "scores" / "ae_test_mse.npy")
    # Saved binary flags retained for compatibility but recomputed per threshold
    _ = npload(UNSUPERVISED_DIR / "scores" / "ae_val_binary.npy")
    _ = npload(UNSUPERVISED_DIR / "scores" / "ae_test_binary.npy")

    # Unsupervised IF
    if_val_binary  = normalise_if_binary(
        npload(UNSUPERVISED_DIR / "scores" / "if_val_binary.npy"))
    if_test_binary = normalise_if_binary(
        npload(UNSUPERVISED_DIR / "scores" / "if_test_binary.npy"))

    # Labels & encoders
    y_val  = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_val.csv")
    y_test = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_test.csv")
    with open(PREPROCESSED_DIR / "label_encoders.json") as f:
        label_encoders = json.load(f)
    with open(UNSUPERVISED_DIR / "thresholds.json") as f:
        ae_thresholds_raw = json.load(f)

    # Sanity
    n_val, n_test = len(y_val), len(y_test)
    for name, arr, expected_n in [
        ("e7_val_pred",   e7_val_pred,   n_val),
        ("e7_test_pred",  e7_test_pred,  n_test),
        ("ae_val_mse",    ae_val_mse,    n_val),
        ("ae_test_mse",   ae_test_mse,   n_test),
        ("if_val_binary", if_val_binary, n_val),
        ("if_test_binary",if_test_binary,n_test),
    ]:
        if arr.shape[0] != expected_n:
            sys.exit(f"[FATAL] {name} length {arr.shape[0]} != {expected_n}")

    print(f"  Val  samples : {n_val:>10,}")
    print(f"  Test samples : {n_test:>10,}")
    print(f"  Multiclass classes: {len(label_encoders['multiclass'])}")
    print(f"  AE threshold keys: {list(ae_thresholds_raw.keys())}")

    mc_inv     = {int(v): k for k, v in label_encoders["multiclass"].items()}
    n_classes  = len(label_encoders["multiclass"])
    BENIGN_ID  = label_encoders["multiclass"]["Benign"]
    ZERO_DAY_ID = n_classes  # one above range for fusion's pseudo-class
    print(f"  Benign class ID = {BENIGN_ID}, Zero-day pseudo-id = {ZERO_DAY_ID}")

    y_val_mc   = y_val["multiclass_label"].values.astype(int)
    y_test_mc  = y_test["multiclass_label"].values.astype(int)
    y_val_bin  = (y_val_mc  != BENIGN_ID).astype(int)
    y_test_bin = (y_test_mc != BENIGN_ID).astype(int)

    benign_val_mse = ae_val_mse[y_val_bin == 0]

    # ---- [3.2] Resolve thresholds ----------------------------------------
    def resolve_threshold(pct: int) -> float:
        for key in (f"p{pct}", str(pct), f"{pct}.0", str(float(pct))):
            if key in ae_thresholds_raw:
                v = ae_thresholds_raw[key]
                if isinstance(v, dict) and "value" in v:
                    return float(v["value"])
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return float(np.percentile(benign_val_mse, pct))

    THRESHOLDS = {pct: resolve_threshold(pct) for pct in AE_THRESHOLD_PERCENTILES}
    print("  AE thresholds in use:")
    for pct, val in THRESHOLDS.items():
        print(f"    p{pct}: {val:.4f}")

    # ---- [3.3] Apply fusion at each operating point ----------------------
    banner("[3.3] Applying fusion at p90, p95, p99 (and IF baseline)")
    fusion_variants_test, fusion_variants_val = {}, {}
    for pct, t in THRESHOLDS.items():
        name = f"AE_p{pct}"
        fusion_variants_test[name] = {
            "ae_binary": ae_binary_at(t, ae_test_mse), "threshold": t,
        }
        fusion_variants_val[name] = {
            "ae_binary": ae_binary_at(t, ae_val_mse), "threshold": t,
        }
    fusion_variants_test["IF"] = {"ae_binary": if_test_binary, "threshold": None}
    fusion_variants_val["IF"]  = {"ae_binary": if_val_binary,  "threshold": None}

    for v in fusion_variants_test.values():
        v["case"] = apply_fusion(e7_test_pred, v["ae_binary"], BENIGN_ID)
    for v in fusion_variants_val.values():
        v["case"] = apply_fusion(e7_val_pred, v["ae_binary"], BENIGN_ID)

    PRIMARY = f"AE_p{AE_THRESHOLD_PERCENTILES[0]}"
    print(f"  Primary fusion config: {PRIMARY}")

    # ---- [3.4] Case distribution -----------------------------------------
    banner("[3.4] Case distribution per variant (test set)")
    case_dist_rows = []
    for name, v in fusion_variants_test.items():
        counts = np.bincount(v["case"], minlength=5)[1:5]
        share  = counts / counts.sum() * 100
        row = {"variant": name, "threshold": v["threshold"]}
        for c in (1, 2, 3, 4):
            row[f"case{c}_n"]   = int(counts[c - 1])
            row[f"case{c}_pct"] = float(share[c - 1])
        case_dist_rows.append(row)
        print(f"  {name:<10}  C1={counts[0]:>8,} ({share[0]:5.2f}%)  "
              f"C2={counts[1]:>8,} ({share[1]:5.2f}%)  "
              f"C3={counts[2]:>8,} ({share[2]:5.2f}%)  "
              f"C4={counts[3]:>8,} ({share[3]:5.2f}%)")
    case_dist_df = pd.DataFrame(case_dist_rows)
    case_dist_df.to_csv(OUTPUT_DIR / "metrics" / "case_distribution.csv",
                        index=False)

    # ---- [3.5] Binary detection (Cases 1+2+3 vs Case 4) ------------------
    banner("[3.5] Binary anomaly detection — fusion vs E7-only")
    e7_test_binpred = (e7_test_pred != BENIGN_ID).astype(int)
    bin_rows = [{
        "variant"  : "E7_only",
        "accuracy" : accuracy_score(y_test_bin, e7_test_binpred),
        "precision": precision_score(y_test_bin, e7_test_binpred, zero_division=0),
        "recall"   : recall_score(y_test_bin, e7_test_binpred, zero_division=0),
        "f1"       : f1_score(y_test_bin, e7_test_binpred, zero_division=0),
        "mcc"      : matthews_corrcoef(y_test_bin, e7_test_binpred),
    }]
    for name, v in fusion_variants_test.items():
        flagged = (v["case"] != 4).astype(int)
        bin_rows.append({
            "variant"  : name,
            "accuracy" : accuracy_score(y_test_bin, flagged),
            "precision": precision_score(y_test_bin, flagged, zero_division=0),
            "recall"   : recall_score(y_test_bin, flagged, zero_division=0),
            "f1"       : f1_score(y_test_bin, flagged, zero_division=0),
            "mcc"      : matthews_corrcoef(y_test_bin, flagged),
        })
    bin_metric_df = pd.DataFrame(bin_rows)
    bin_metric_df.to_csv(
        OUTPUT_DIR / "metrics" / "fusion_vs_supervised_binary.csv", index=False)
    print(bin_metric_df.to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- [3.6] H1 — multiclass macro-F1 with bootstrap CI ----------------
    banner("[3.6] H1 — multiclass macro-F1 with paired bootstrap CI")

    # E7 metrics computed in the 20-label space for apples-to-apples
    # comparison with fusion variants (zero_day_unknown class included)
    labels_with_zd = list(range(n_classes)) + [ZERO_DAY_ID]

    e7_macro_f1 = f1_score(y_test_mc, e7_test_pred,
                           labels=labels_with_zd,
                           average="macro", zero_division=0)
    e7_mcc      = matthews_corrcoef(y_test_mc, e7_test_pred)
    e7_acc      = accuracy_score(y_test_mc, e7_test_pred)
    print(f"  E7 only        : macro-F1 = {e7_macro_f1:.4f}  "
          f"MCC = {e7_mcc:.4f}  acc = {e7_acc:.4f}")

    # Build dict of predictions for paired bootstrap (same y_true, same indices)
    preds_for_boot = {"E7_only": e7_test_pred.astype(np.int32)}
    fused_predictions = {}
    for name, v in fusion_variants_test.items():
        fp = fusion_classes(e7_test_pred, v["case"], BENIGN_ID, ZERO_DAY_ID)
        fused_predictions[name] = fp
        preds_for_boot[name] = fp

    print(f"  Running paired bootstrap "
          f"({H1_BOOTSTRAP_ITERS} iters, {len(preds_for_boot)} variants)…")
    boot = bootstrap_paired_f1(
        y_test_mc.astype(np.int32),
        preds_for_boot,
        labels=labels_with_zd,
        n_boot=H1_BOOTSTRAP_ITERS,
        seed=H1_BOOTSTRAP_SEED,
    )

    mc_macro_rows = []
    for name in fusion_variants_test:
        fused_pred = fused_predictions[name]
        f1m = f1_score(y_test_mc, fused_pred, labels=labels_with_zd,
                       average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_test_mc, fused_pred)
        acc = accuracy_score(y_test_mc, fused_pred)
        delta_dist = boot[name] - boot["E7_only"]
        delta_lo, delta_hi = ci_95(delta_dist)
        f1_lo, f1_hi       = ci_95(boot[name])
        mc_macro_rows.append({
            "variant"        : name,
            "macro_f1"       : f1m,
            "macro_f1_ci_lo" : f1_lo,
            "macro_f1_ci_hi" : f1_hi,
            "mcc"            : mcc,
            "accuracy"       : acc,
            "delta_f1_vs_E7" : f1m - e7_macro_f1,
            "delta_ci_lo"    : delta_lo,
            "delta_ci_hi"    : delta_hi,
            "h1_significant" : bool(delta_lo > 0),
        })
        flag = "✓ sig" if delta_lo > 0 else "✗ ns "
        print(f"  Fusion {name:<8}: F1 = {f1m:.4f} "
              f"[{f1_lo:.4f}, {f1_hi:.4f}]  Δ = {f1m - e7_macro_f1:+.4f} "
              f"[{delta_lo:+.4f}, {delta_hi:+.4f}]  {flag}")
    mc_macro_df = pd.DataFrame(mc_macro_rows)
    mc_macro_df.to_csv(
        OUTPUT_DIR / "metrics" / "fusion_vs_supervised.csv", index=False)

    # E7 baseline CI for the summary
    e7_f1_lo, e7_f1_hi = ci_95(boot["E7_only"])
    print(f"\n  E7 macro-F1 95% CI: [{e7_f1_lo:.4f}, {e7_f1_hi:.4f}]")

    # H1 verdict (primary + best across configs)
    primary_row = mc_macro_df[mc_macro_df["variant"] == PRIMARY].iloc[0]
    H1_PASS_PRIMARY = bool(primary_row["h1_significant"])
    best_row = mc_macro_df.iloc[mc_macro_df["macro_f1"].idxmax()]
    H1_PASS_BEST    = bool(best_row["h1_significant"])

    msg_primary = h1_verdict_msg(H1_PASS_PRIMARY,
                                 float(primary_row["delta_ci_hi"]))
    msg_best    = h1_verdict_msg(H1_PASS_BEST,
                                 float(best_row["delta_ci_hi"]))
    print(f"  H1 (primary={PRIMARY}): {msg_primary}")
    print(f"  H1 (best={best_row['variant']}): {msg_best}")

    # ---- [3.7] Per-class case distribution at all 3 thresholds -----------
    banner("[3.7] Per-class case distribution — three thresholds")
    class_names = [mc_inv[i] for i in range(n_classes)]
    per_class_heatmaps = {}  # variant -> (n_classes, 4) matrix
    for variant_name in [f"AE_p{p}" for p in AE_THRESHOLD_PERCENTILES]:
        case_arr = fusion_variants_test[variant_name]["case"]
        H = np.zeros((n_classes, 4))
        nonempty = []
        for ci in range(n_classes):
            mask = y_test_mc == ci
            if mask.sum() == 0:
                continue
            nonempty.append(ci)
            cic = case_arr[mask]
            for k in (1, 2, 3, 4):
                H[ci, k - 1] = (cic == k).mean() * 100
        # Sanity: nonempty rows must sum to 100
        sums = H[nonempty].sum(axis=1)
        np.testing.assert_allclose(
            sums, 100.0, atol=1e-4,
            err_msg=f"Per-class rows don't sum to 100% in {variant_name}")
        per_class_heatmaps[variant_name] = H

    # CSV: long format for primary heatmap (kept as the canonical reference)
    primary_H = per_class_heatmaps[PRIMARY]
    per_class_df = pd.DataFrame(
        primary_H, index=class_names,
        columns=[f"Case{i}_pct" for i in (1, 2, 3, 4)],
    )
    per_class_df["n_test"] = [int((y_test_mc == ci).sum())
                              for ci in range(n_classes)]
    per_class_df.to_csv(OUTPUT_DIR / "metrics" / "per_class_case_analysis.csv")
    print(per_class_df.to_string(float_format=lambda x: f"{x:6.2f}"))

    # ---- [3.8] Simulated zero-day under E7-blindness (H2) ----------------
    banner("[3.8] Simulated zero-day under E7-blindness (H2)")
    print("  NOTE: E7 is NOT retrained per target — true LOO would require "
          "5 separate E7 fits.\n  This measures: when E7 misclassifies a "
          "target attack as benign,\n  does the AE flag it as anomalous?")

    zero_day_rows = []
    for target in ZERO_DAY_TARGETS:
        if target not in label_encoders["multiclass"]:
            print(f"  [warn] {target} not in label encoder — skipping")
            continue
        tid  = label_encoders["multiclass"][target]
        mask = (y_test_mc == tid)
        n_t  = int(mask.sum())
        if n_t == 0:
            print(f"  [warn] {target} has no test samples")
            continue

        e7_correct       = (e7_test_pred[mask] == tid)
        e7_recall        = float(e7_correct.mean())
        e7_called_benign = (e7_test_pred[mask] == BENIGN_ID)
        n_called_benign  = int(e7_called_benign.sum())
        sufficient       = n_called_benign >= H2_MIN_SAMPLES

        row = {
            "target": target, "n_test": n_t,
            "e7_recall": e7_recall,
            "e7_called_benign_n": n_called_benign,
            "e7_called_benign_pct": float(n_called_benign / n_t * 100),
            "h2_sample_sufficient": sufficient,
        }

        for pct, t in THRESHOLDS.items():
            ae_bin_local       = ae_binary_at(t, ae_test_mse[mask])
            ae_recall_raw      = float(ae_bin_local.mean())  # auxiliary
            if n_called_benign > 0:
                missed_mse = ae_test_mse[mask][e7_called_benign]
                ae_recall_on_missed = float(
                    ae_binary_at(t, missed_mse).mean())
            else:
                ae_recall_on_missed = float("nan")

            case_local = apply_fusion(
                e7_test_pred[mask], ae_bin_local, BENIGN_ID)
            binary_detected = float((case_local != 4).mean())
            confirmed_or_zd = float(np.isin(case_local, [1, 2]).mean())

            row[f"ae_recall_p{pct}"]              = ae_recall_raw
            row[f"ae_recall_on_missed_p{pct}"]    = ae_recall_on_missed
            row[f"binary_detected_recall_p{pct}"] = binary_detected
            row[f"confirmed_or_zeroday_p{pct}"]   = confirmed_or_zd

        zero_day_rows.append(row)
        suff_tag = "" if sufficient else "  [INSUFFICIENT n_missed]"
        print(f"  {target:<28} n={n_t:>5,}  E7={e7_recall:.3f}  "
              f"E7→Benign={n_called_benign:>4d}  "
              f"AEonMissed: p90={row['ae_recall_on_missed_p90']:.3f} "
              f"p95={row['ae_recall_on_missed_p95']:.3f} "
              f"p99={row['ae_recall_on_missed_p99']:.3f}{suff_tag}")

    zd_df = pd.DataFrame(zero_day_rows)
    zd_df.to_csv(OUTPUT_DIR / "metrics" / "zero_day_results.csv", index=False)

    # H2 verdict on PRIMARY metric: ae_recall_on_missed across thresholds
    H2_per_target_primary = {}
    for r in zero_day_rows:
        if not r["h2_sample_sufficient"]:
            H2_per_target_primary[r["target"]] = {
                "value": float("nan"), "sufficient": False, "passes": False,
            }
            continue
        candidates = [r[f"ae_recall_on_missed_p{p}"]
                      for p in AE_THRESHOLD_PERCENTILES]
        candidates = [c for c in candidates if not np.isnan(c)]
        best = max(candidates) if candidates else float("nan")
        H2_per_target_primary[r["target"]] = {
            "value": float(best), "sufficient": True,
            "passes": bool(best >= H2_RECALL_TARGET),
        }

    n_pass_primary = sum(1 for d in H2_per_target_primary.values()
                         if d["passes"])
    n_total = len(H2_per_target_primary)
    H2_PASS = ((n_pass_primary / n_total) >= H2_FRACTION_TARGET
               if n_total else False)

    # Auxiliary: same verdict on raw ae_recall (closer to Phase 5's framing)
    H2_per_target_aux = {}
    for r in zero_day_rows:
        best = max(r[f"ae_recall_p{p}"] for p in AE_THRESHOLD_PERCENTILES)
        H2_per_target_aux[r["target"]] = {
            "value": float(best),
            "passes": bool(best >= H2_RECALL_TARGET),
        }
    n_pass_aux = sum(1 for d in H2_per_target_aux.values() if d["passes"])

    print(f"\n  H2 (PRIMARY: AE recall on E7-missed): "
          f"{n_pass_primary}/{n_total} → "
          f"{'PASS' if H2_PASS else 'FAIL'}")
    print(f"  H2 (auxiliary: raw AE per-class recall): "
          f"{n_pass_aux}/{n_total}")

    # ---- [3.9] Threshold sweep — selection on val, evaluation on test ----
    banner("[3.9] Threshold sweep — val for selection, test for reporting")
    val_rows, test_rows = [], []
    for pct in PERCENTILES_SWEEP:
        t = float(np.percentile(benign_val_mse, pct))
        # Validation
        ae_bin_v = ae_binary_at(t, ae_val_mse)
        case_v   = apply_fusion(e7_val_pred, ae_bin_v, BENIGN_ID)
        flag_v   = (case_v != 4).astype(int)
        val_rows.append({
            "percentile": pct, "threshold": t,
            "val_attack_recall": float(flag_v[y_val_bin == 1].mean()),
            "val_benign_fpr"   : float(flag_v[y_val_bin == 0].mean()),
            "val_binary_f1"    : float(f1_score(
                y_val_bin, flag_v, zero_division=0)),
        })
        # Test (reporting only — never used for selection)
        ae_bin_te = ae_binary_at(t, ae_test_mse)
        case_te   = apply_fusion(e7_test_pred, ae_bin_te, BENIGN_ID)
        flag_te   = (case_te != 4).astype(int)
        test_rows.append({
            "percentile": pct, "threshold": t,
            "test_attack_recall": float(flag_te[y_test_bin == 1].mean()),
            "test_benign_fpr"   : float(flag_te[y_test_bin == 0].mean()),
            "test_binary_f1"    : float(f1_score(
                y_test_bin, flag_te, zero_division=0)),
        })

    val_sweep_df  = pd.DataFrame(val_rows)
    test_sweep_df = pd.DataFrame(test_rows)
    sweep_df = val_sweep_df.merge(
        test_sweep_df.drop(columns=["threshold"]), on="percentile")
    sweep_df.to_csv(
        OUTPUT_DIR / "metrics" / "threshold_sensitivity.csv", index=False)
    print(sweep_df.to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    # SELECTION: max val attack recall with val FPR < budget (val-only)
    val_op_mask = val_sweep_df["val_benign_fpr"] < RECOMMENDED_FPR_BUDGET
    if val_op_mask.any():
        selected_val = val_sweep_df[val_op_mask].sort_values(
            "val_attack_recall", ascending=False).iloc[0]
        selected_pct = int(selected_val["percentile"])
        # Look up test performance at the val-selected percentile
        test_at_sel = test_sweep_df[
            test_sweep_df["percentile"] == selected_pct].iloc[0]
        print(f"\n  Recommended (selected on val, FPR<{RECOMMENDED_FPR_BUDGET}): "
              f"p{selected_pct}  "
              f"val_TPR={selected_val['val_attack_recall']:.4f} "
              f"val_FPR={selected_val['val_benign_fpr']:.4f}  →  "
              f"test_TPR={test_at_sel['test_attack_recall']:.4f} "
              f"test_FPR={test_at_sel['test_benign_fpr']:.4f}")
        recommended = {
            "percentile": selected_pct,
            "threshold" : float(selected_val["threshold"]),
            "val_TPR"   : float(selected_val["val_attack_recall"]),
            "val_FPR"   : float(selected_val["val_benign_fpr"]),
            "test_TPR"  : float(test_at_sel["test_attack_recall"]),
            "test_FPR"  : float(test_at_sel["test_benign_fpr"]),
            "test_F1"   : float(test_at_sel["test_binary_f1"]),
        }
    else:
        print("\n  No val percentile meets FPR budget; deployment threshold "
              "requires committee discussion.")
        recommended = None

    # ---- [3.10] Visualizations -------------------------------------------
    banner("[3.10] Generating figures")

    # Fig 1: case distribution per variant
    fig, ax = plt.subplots(figsize=(9, 5))
    variants = list(fusion_variants_test.keys())
    x = np.arange(len(variants))
    width = 0.2
    for i, c in enumerate((1, 2, 3, 4)):
        counts = [(fusion_variants_test[v]["case"] == c).sum()
                  for v in variants]
        ax.bar(x + (i - 1.5) * width, counts, width,
               label=f"Case {c}: {CASE_NAMES[c]}", color=CASE_COLORS[c])
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylabel("Number of test samples (log scale)")
    ax.set_title("Fusion case distribution by variant — test set")
    ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "case_distribution.png")
    plt.close(fig)

    # Fig 2: per-class heatmap × 3 thresholds
    fig, axes = plt.subplots(1, 3, figsize=(20, 9), sharey=True)
    for ax, pct in zip(axes, AE_THRESHOLD_PERCENTILES):
        H = per_class_heatmaps[f"AE_p{pct}"]
        sns.heatmap(
            H, annot=True, fmt=".1f", cmap="YlOrRd",
            xticklabels=[f"Case {i+1}" for i in range(4)],
            yticklabels=class_names if ax is axes[0] else False,
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "% of class samples"} if ax is axes[-1] else None,
            ax=ax, vmin=0, vmax=100,
        )
        ax.set_title(f"AE_p{pct}")
    fig.suptitle("Per-class fusion case distribution at three operating points",
                 y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "per_class_heatmap.png")
    plt.close(fig)

    # Fig 3: fusion macro-F1 with 95% CI error bars
    fig, ax = plt.subplots(figsize=(8, 5))
    labels_plot = ["E7 only"] + list(mc_macro_df["variant"])
    values      = [e7_macro_f1] + list(mc_macro_df["macro_f1"])
    ci_lo       = [e7_f1_lo]    + list(mc_macro_df["macro_f1_ci_lo"])
    ci_hi       = [e7_f1_hi]    + list(mc_macro_df["macro_f1_ci_hi"])
    err_lo = np.clip(np.array(values) - np.array(ci_lo), 0, None)
    err_hi = np.clip(np.array(ci_hi)  - np.array(values), 0, None)
    colors = ["#7f7f7f"] + ["#1f77b4"] * len(mc_macro_df)
    ax.bar(labels_plot, values, color=colors,
           yerr=[err_lo, err_hi], capsize=5,
           ecolor="#444", error_kw={"elinewidth": 1.2})
    ax.axhline(e7_macro_f1, ls="--", color="grey", alpha=0.7,
               label=f"E7 baseline = {e7_macro_f1:.4f}")
    for i, (v, lo, hi) in enumerate(zip(values, ci_lo, ci_hi)):
        ax.text(i, v + (hi - v) + 0.005,
                f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Macro-F1 (test, 20-class incl. zero_day_unknown)")
    ax.set_title("Fusion vs E7 macro-F1 with paired bootstrap 95% CI")
    y_min = min(ci_lo) - 0.03
    ax.set_ylim(max(0, y_min), 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "fusion_vs_supervised.png")
    plt.close(fig)

    # Fig 4: zero-day — primary metric (ae_recall_on_missed) per target
    fig, ax = plt.subplots(figsize=(11, 5))
    targets   = list(zd_df["target"])
    xs        = np.arange(len(targets))
    bar_w     = 0.18
    series = [
        ("E7 recall",             "e7_recall",                  "#7f7f7f"),
        ("AE on E7-missed p90",   "ae_recall_on_missed_p90",    "#1f77b4"),
        ("AE on E7-missed p95",   "ae_recall_on_missed_p95",    "#9467bd"),
        ("AE on E7-missed p99",   "ae_recall_on_missed_p99",    "#17becf"),
        ("Confirmed/zero-day p90","confirmed_or_zeroday_p90",   "#2ca02c"),
    ]
    for i, (label, col, colour) in enumerate(series):
        ax.bar(xs + (i - 2) * bar_w, zd_df[col].fillna(0), bar_w,
               label=label, color=colour)
    ax.axhline(H2_RECALL_TARGET, ls="--", color="red", alpha=0.7,
               label=f"H2 target = {H2_RECALL_TARGET:.2f}")
    # Mark insufficient-sample targets
    for i, r in zd_df.iterrows():
        if not r["h2_sample_sufficient"]:
            ax.text(i, 1.02, "n<30", ha="center", color="red", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(targets, rotation=20, ha="right")
    ax.set_ylim(0, 1.10); ax.set_ylabel("Recall")
    ax.set_title("Zero-day simulation — H2 primary metric "
                 "(AE recall conditional on E7 misclassification as benign)")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "zero_day_detection.png")
    plt.close(fig)

    # Fig 5: threshold sensitivity — val + test side-by-side
    fig, ax1 = plt.subplots(figsize=(9.5, 5))
    ax1.plot(sweep_df["percentile"], sweep_df["val_attack_recall"],
             "o-", color="#1f77b4", label="Val TPR")
    ax1.plot(sweep_df["percentile"], sweep_df["test_attack_recall"],
             "o--", color="#1f77b4", alpha=0.5, label="Test TPR")
    ax1.set_xlabel("AE threshold percentile (on benign-val MSE)")
    ax1.set_ylabel("Attack recall (TPR)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(sweep_df["percentile"], sweep_df["val_benign_fpr"],
             "s-", color="#d62728", label="Val FPR")
    ax2.plot(sweep_df["percentile"], sweep_df["test_benign_fpr"],
             "s--", color="#d62728", alpha=0.5, label="Test FPR")
    ax2.set_ylabel("Benign FPR", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.axhline(RECOMMENDED_FPR_BUDGET, ls="--",
                color="#d62728", alpha=0.4)
    if recommended is not None:
        ax1.axvline(recommended["percentile"], ls=":",
                    color="green", alpha=0.7,
                    label=f"Selected on val: p{recommended['percentile']}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", framealpha=0.9, fontsize=8)
    ax1.set_title("Threshold sensitivity — val for selection, test for reporting")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figures" / "threshold_sensitivity.png")
    plt.close(fig)

    print(f"  Saved 5 figures to {OUTPUT_DIR / 'figures'}")

    # ---- [3.11] Save fusion arrays + config + verdicts -------------------
    banner("[3.11] Persisting fusion arrays + verdicts")
    np.save(OUTPUT_DIR / "fusion_results" / "fusion_val_cases.npy",
            fusion_variants_val[PRIMARY]["case"])
    np.save(OUTPUT_DIR / "fusion_results" / "fusion_test_cases.npy",
            fusion_variants_test[PRIMARY]["case"])

    def to_decision_df(case_arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "case": case_arr,
            "decision":   [CASE_DECISIONS[int(c)]  for c in case_arr],
            "confidence": [CASE_CONFIDENCE[int(c)] for c in case_arr],
        })

    # Decoded CSVs are redundant w/ npy + dict — kept for inspection only
    to_decision_df(fusion_variants_val[PRIMARY]["case"]).to_csv(
        OUTPUT_DIR / "fusion_results" / "fusion_val_labels.csv", index=False)
    to_decision_df(fusion_variants_test[PRIMARY]["case"]).to_csv(
        OUTPUT_DIR / "fusion_results" / "fusion_test_labels.csv", index=False)

    config = {
        "phase": 6, "version": 3,
        "primary_variant": PRIMARY,
        "ae_thresholds": THRESHOLDS,
        "benign_id": int(BENIGN_ID),
        "zero_day_id": int(ZERO_DAY_ID),
        "n_val": int(n_val), "n_test": int(n_test),
        "n_classes": int(n_classes),
        "h1_bootstrap_iters": H1_BOOTSTRAP_ITERS,
        "h1_bootstrap_seed":  H1_BOOTSTRAP_SEED,
        "h2_recall_target":   H2_RECALL_TARGET,
        "h2_fraction_target": H2_FRACTION_TARGET,
        "h2_min_samples":     H2_MIN_SAMPLES,
        "zero_day_targets":   ZERO_DAY_TARGETS,
        "recommended_fpr_budget": RECOMMENDED_FPR_BUDGET,
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    verdicts = {
        "H1": {
            "description": ("Fusion improves multiclass macro-F1 over E7 with "
                            "paired-bootstrap 95% CI excluding zero "
                            "(20-class label space incl. zero_day_unknown; "
                            "E7 evaluated in same 20-label space)"),
            "e7_macro_f1_20class": float(e7_macro_f1),
            "e7_macro_f1_ci": [e7_f1_lo, e7_f1_hi],
            "fusion_macro_f1_primary": float(primary_row["macro_f1"]),
            "fusion_macro_f1_primary_ci": [
                float(primary_row["macro_f1_ci_lo"]),
                float(primary_row["macro_f1_ci_hi"]),
            ],
            "delta_primary": float(primary_row["delta_f1_vs_E7"]),
            "delta_primary_ci": [
                float(primary_row["delta_ci_lo"]),
                float(primary_row["delta_ci_hi"]),
            ],
            "best_variant": str(best_row["variant"]),
            "best_delta_ci": [
                float(best_row["delta_ci_lo"]),
                float(best_row["delta_ci_hi"]),
            ],
            "verdict_primary": "PASS" if H1_PASS_PRIMARY else "FAIL",
            "verdict_primary_msg": msg_primary,
            "verdict_best":    "PASS" if H1_PASS_BEST    else "FAIL",
            "verdict_best_msg": msg_best,
            "note": ("20-class macro-F1 penalises every false zero_day_unknown "
                     "alarm. See binary metrics for operational view."),
        },
        "H2": {
            "description": (
                f"AE recall on E7-misclassified samples >= {H2_RECALL_TARGET} "
                f"on at least {int(H2_FRACTION_TARGET * 100)}% of zero-day "
                f"targets (best threshold per target; targets with "
                f"n_called_benign < {H2_MIN_SAMPLES} flagged insufficient)"
            ),
            "primary_metric": "ae_recall_on_missed",
            "per_target_primary": H2_per_target_primary,
            "n_pass_primary": int(n_pass_primary),
            "n_total": int(n_total),
            "verdict": "PASS" if H2_PASS else "FAIL",
            "auxiliary_metric": "ae_recall (raw per-class)",
            "per_target_auxiliary": H2_per_target_aux,
            "n_pass_auxiliary": int(n_pass_aux),
        },
        "recommended_threshold": recommended,
    }
    with open(OUTPUT_DIR / "metrics" / "h1_h2_verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=2)

    # ---- [3.12] summary.md ------------------------------------------------
    banner("[3.12] Writing summary.md")
    with open(OUTPUT_DIR / "summary.md", "w") as f:
        f.write("# Phase 6 — Fusion Engine & Zero-Day Simulation Summary\n\n")
        f.write(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                "(v3 — H1 label-space bug fixed)_\n\n")

        f.write("## 1. Configuration\n\n")
        f.write(f"- Primary variant: **{PRIMARY}**\n")
        for pct, t in THRESHOLDS.items():
            f.write(f"- AE threshold p{pct}: `{t:.4f}`\n")
        f.write(f"- Benign class id: `{BENIGN_ID}` "
                f"| Zero-day pseudo-class id: `{ZERO_DAY_ID}`\n")
        f.write(f"- Test samples: `{n_test:,}` | Val samples: `{n_val:,}`\n")
        f.write(f"- Bootstrap iterations: {H1_BOOTSTRAP_ITERS} "
                f"(seed={H1_BOOTSTRAP_SEED})\n")
        f.write(f"- H1 evaluated in 20-label space "
                f"(includes `zero_day_unknown`; E7 scored in same space "
                f"for apples-to-apples comparison)\n")
        f.write(f"- H2 primary metric: AE recall on samples E7 misclassified "
                f"as benign\n")
        f.write(f"- H2 sample-size guard: n_called_benign >= "
                f"{H2_MIN_SAMPLES}\n\n")

        f.write("## 2. Case Distribution (test set)\n\n")
        f.write(md_table(case_dist_df, floatfmt=".2f"))
        f.write("\n\n")

        f.write("## 3. Fusion vs E7 — 20-class macro-F1 with bootstrap CI\n\n")
        f.write(f"E7 baseline macro-F1 (20-class): **{e7_macro_f1:.4f}** "
                f"[{e7_f1_lo:.4f}, {e7_f1_hi:.4f}] "
                f"| MCC: {e7_mcc:.4f} | acc: {e7_acc:.4f}\n\n")
        f.write(md_table(mc_macro_df, floatfmt=".4f"))
        f.write("\n\n")

        f.write("## 4. Binary Detection (Cases 1+2+3 vs Case 4)\n\n")
        f.write(md_table(bin_metric_df, floatfmt=".4f"))
        f.write("\n\n")

        f.write("## 5. Simulated Zero-Day under E7-Blindness\n\n")
        f.write("> **Methodological note.** This is *not* leave-one-attack-out "
                "in the strict sense — E7 is trained on all 19 classes, "
                "including the 5 targets. The simulation measures: when E7 "
                "misclassifies a target attack as benign, does the AE catch "
                "it? True LOO would require retraining E7 five times "
                "(deferred to future work).\n\n")
        cols = ["target", "n_test", "e7_recall",
                "e7_called_benign_n", "e7_called_benign_pct",
                "h2_sample_sufficient",
                "ae_recall_on_missed_p90", "ae_recall_on_missed_p95",
                "ae_recall_on_missed_p99",
                "ae_recall_p90", "ae_recall_p95", "ae_recall_p99",
                "binary_detected_recall_p90",
                "confirmed_or_zeroday_p90"]
        f.write(md_table(zd_df[cols], floatfmt=".3f"))
        f.write("\n\n")

        f.write("## 6. Hypothesis Verdicts\n\n")
        f.write("### H1 — Fusion improves macro-F1 (paired bootstrap)\n\n")
        f.write(f"- E7 baseline (20-class): {e7_macro_f1:.4f} "
                f"[{e7_f1_lo:.4f}, {e7_f1_hi:.4f}]\n")
        f.write(f"- Fusion ({PRIMARY}): {float(primary_row['macro_f1']):.4f} "
                f"[{float(primary_row['macro_f1_ci_lo']):.4f}, "
                f"{float(primary_row['macro_f1_ci_hi']):.4f}]\n")
        f.write(f"- Δ = {float(primary_row['delta_f1_vs_E7']):+.4f} "
                f"95% CI [{float(primary_row['delta_ci_lo']):+.4f}, "
                f"{float(primary_row['delta_ci_hi']):+.4f}]\n")
        f.write(f"- Best variant ({best_row['variant']}): "
                f"Δ CI [{float(best_row['delta_ci_lo']):+.4f}, "
                f"{float(best_row['delta_ci_hi']):+.4f}]\n")
        f.write(f"- **Verdict (primary): {msg_primary}**\n")
        f.write(f"- **Verdict (best variant): {msg_best}**\n\n")
        f.write("> 20-class macro-F1 penalises every false `zero_day_unknown` "
                "alarm equally. Binary detection (§4) is more representative "
                "of operational value.\n\n")

        f.write(f"### H2 — AE catches what E7 misses on "
                f"≥{int(H2_FRACTION_TARGET*100)}% of zero-day targets\n\n")
        f.write("**Primary metric: AE recall on samples E7 misclassified "
                "as benign.**\n\n")
        f.write(f"- Targets passing (best threshold, AE-on-missed ≥ "
                f"{H2_RECALL_TARGET}): **{n_pass_primary}/{n_total}**\n")
        for tgt, d in H2_per_target_primary.items():
            if not d["sufficient"]:
                f.write(f"  - ⚠ {tgt}: insufficient samples "
                        f"(n_called_benign < {H2_MIN_SAMPLES}); excluded\n")
            else:
                mark = "✓" if d["passes"] else "✗"
                f.write(f"  - {mark} {tgt}: best AE-on-missed = "
                        f"{d['value']:.3f}\n")
        f.write(f"- **Verdict: {'PASS ✓' if H2_PASS else 'FAIL ✗'}**\n\n")
        f.write("**Auxiliary (raw AE per-class recall, Phase-5 framing):** "
                f"{n_pass_aux}/{n_total} pass.\n\n")

        f.write("## 7. Threshold Sensitivity (val for selection, test for reporting)\n\n")
        f.write(md_table(sweep_df, floatfmt=".4f"))
        f.write("\n\n")

        f.write("## 8. Recommended Operating Threshold\n\n")
        if recommended is not None:
            f.write(f"Selected on val (FPR < {RECOMMENDED_FPR_BUDGET}): "
                    f"**p{recommended['percentile']}** "
                    f"(threshold = {recommended['threshold']:.4f})\n\n")
            f.write(f"- Val:  TPR = {recommended['val_TPR']:.4f}, "
                    f"FPR = {recommended['val_FPR']:.4f}\n")
            f.write(f"- Test: TPR = {recommended['test_TPR']:.4f}, "
                    f"FPR = {recommended['test_FPR']:.4f}, "
                    f"binary F1 = {recommended['test_F1']:.4f}\n\n")
        else:
            f.write(f"No val percentile achieves FPR < {RECOMMENDED_FPR_BUDGET}. "
                    "Trade-off discussion required in thesis Chapter 5.\n\n")

        f.write("## 9. Per-class case rates (primary variant)\n\n")
        f.write(md_table(per_class_df.reset_index().rename(
            columns={"index": "class"}), floatfmt=".2f"))
        f.write("\n\n")

        f.write("## 10. Files generated\n\n")
        f.write("- `fusion_results/fusion_{val,test}_cases.npy` — case arrays\n")
        f.write("- `fusion_results/fusion_{val,test}_labels.csv` — decoded "
                "(redundant w/ npy + dict, kept for inspection)\n")
        f.write("- `metrics/case_distribution.csv`\n")
        f.write("- `metrics/fusion_vs_supervised.csv` "
                "(macro-F1 + bootstrap CIs)\n")
        f.write("- `metrics/fusion_vs_supervised_binary.csv`\n")
        f.write("- `metrics/per_class_case_analysis.csv`\n")
        f.write("- `metrics/zero_day_results.csv`\n")
        f.write("- `metrics/threshold_sensitivity.csv` (val + test)\n")
        f.write("- `metrics/h1_h2_verdicts.json`\n")
        f.write("- `figures/*.png` (5 plots)\n")
        f.write("- `config.json`\n")

    print(f"\nDone. Output: {OUTPUT_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()