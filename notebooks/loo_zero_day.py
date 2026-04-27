#!/usr/bin/env python3
"""
Phase 6B — True Leave-One-Attack-Out Zero-Day Simulation
=========================================================
For each of 5 zero-day target classes, retrain XGBoost from scratch
WITHOUT that class, then evaluate fusion with the existing AE.

Corrects Phase 6's simulated LOO (which trained E7 on all 19 classes
and measured AE on samples E7 happened to misclassify).

Project: IoMT Hybrid Supervised-Unsupervised Anomaly Detection Framework
"""

# %% Imports & configuration
import os
import sys
import gc
import json
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
PREPROCESSED_DIR = Path("./preprocessed")
UNSUPERVISED_DIR = Path("./results/unsupervised")
FUSION_DIR       = Path("./results/fusion")
OUTPUT_DIR       = Path("./results/zero_day_loo")

MODELS_DIR      = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR     = OUTPUT_DIR / "metrics"
FIGURES_DIR     = OUTPUT_DIR / "figures"

for d in (MODELS_DIR, PREDICTIONS_DIR, METRICS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# XGBoost hyperparams (identical to E7 — only training data changes)
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
    objective="multi:softprob",
    eval_metric="mlogloss",
)

ZERO_DAY_TARGETS = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "MQTT_DoS_Connect_Flood",
    "ARP_Spoofing",
]

H2_THRESHOLD = 0.70
H2_MAJORITY  = 3  # ≥3 of 5 targets

# Plot styling
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt(v, p=3):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "—"
    return f"{v:.{p}f}"


# %% Section 1 — Load all inputs
log("=" * 70)
log("PHASE 6B — TRUE LEAVE-ONE-ATTACK-OUT ZERO-DAY SIMULATION")
log("=" * 70)

t_start = time.time()

log("Loading training data...")
X_train = np.load(PREPROCESSED_DIR / "full_features" / "X_train.npy")
y_train = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_train.csv")
log(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

log("Loading test data...")
X_test = np.load(PREPROCESSED_DIR / "full_features" / "X_test.npy")
y_test = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_test.csv")
log(f"  X_test:  {X_test.shape}  y_test:  {y_test.shape}")

log("Loading existing AE & IF scores (no retraining needed)...")
ae_test_mse    = np.load(UNSUPERVISED_DIR / "scores" / "ae_test_mse.npy")
if_test_scores = np.load(UNSUPERVISED_DIR / "scores" / "if_test_scores.npy")
log(f"  ae_test_mse: {ae_test_mse.shape}  if_test_scores: {if_test_scores.shape}")

assert ae_test_mse.shape[0] == X_test.shape[0], (
    f"AE scores ({ae_test_mse.shape[0]}) and X_test ({X_test.shape[0]}) misaligned"
)

log("Loading AE thresholds...")
with open(UNSUPERVISED_DIR / "thresholds.json") as f:
    thresholds_blob = json.load(f)
# thresholds.json may be:
#   - flat: {"p90":..., "p95":..., "p99":...}
#   - nested under "ae":         {"ae": {...}, "if": {...}}
#   - nested under "thresholds": {"thresholds": {...}, "evaluation_on_val": [...], ...}
if "thresholds" in thresholds_blob and isinstance(thresholds_blob["thresholds"], dict):
    ae_thresholds_raw = thresholds_blob["thresholds"]
elif "ae" in thresholds_blob and isinstance(thresholds_blob["ae"], dict):
    ae_thresholds_raw = thresholds_blob["ae"]
else:
    ae_thresholds_raw = thresholds_blob
# Keep only the p* numeric thresholds
ae_thresholds = {
    k: float(v) for k, v in ae_thresholds_raw.items()
    if isinstance(v, (int, float)) and k in ("p90", "p95", "p99")
}
if not ae_thresholds:
    raise RuntimeError(f"No AE p90/p95/p99 thresholds in {UNSUPERVISED_DIR / 'thresholds.json'}")
log(f"  AE thresholds: {ae_thresholds}")

log("Loading label encoders...")
with open(PREPROCESSED_DIR / "label_encoders.json") as f:
    label_encoders = json.load(f)
multiclass_map = label_encoders["multiclass"]      # name -> int
inv_multiclass_map = {int(v): k for k, v in multiclass_map.items()}
log(f"  Found {len(multiclass_map)} classes")

# Persist run config
config = {
    "phase": "6B",
    "description": "True Leave-One-Attack-Out Zero-Day Simulation",
    "random_state": RANDOM_STATE,
    "xgb_params": XGB_PARAMS,
    "zero_day_targets": ZERO_DAY_TARGETS,
    "ae_thresholds": ae_thresholds,
    "h2_threshold": H2_THRESHOLD,
    "h2_majority_required": H2_MAJORITY,
    "n_train": int(X_train.shape[0]),
    "n_test":  int(X_test.shape[0]),
    "n_features": int(X_train.shape[1]),
    "n_classes_full": len(multiclass_map),
    "started_at": datetime.now().isoformat(timespec="seconds"),
}
with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

log(f"Setup complete in {time.time()-t_start:.1f}s\n")


# %% Section 2 — LOO retraining loop
def run_loo_target(target: str):
    """Train XGBoost on all data EXCEPT `target`, predict on full test set.

    Returns (loo_pred_encoded, loo_proba, loo_label_map). Resumes if results exist.
    """
    log("\n" + "=" * 70)
    log(f"LOO TARGET: {target}")
    log("=" * 70)

    model_path = MODELS_DIR / f"loo_xgb_without_{target}.pkl"
    pred_path  = PREDICTIONS_DIR / f"loo_{target}_test_pred.npy"
    proba_path = PREDICTIONS_DIR / f"loo_{target}_test_proba.npy"
    map_path   = MODELS_DIR / f"loo_label_map_{target}.json"

    # Resume support
    if pred_path.exists() and proba_path.exists() and map_path.exists():
        log(f"  Resume: predictions exist for {target}, loading from disk...")
        loo_pred_encoded = np.load(pred_path)
        loo_proba = np.load(proba_path)
        with open(map_path) as f:
            raw = json.load(f)
        loo_label_map = {k: int(v) for k, v in raw.items()}
        log(f"  Loaded preds {loo_pred_encoded.shape}, proba {loo_proba.shape}, "
            f"{len(loo_label_map)} classes")
        return loo_pred_encoded, loo_proba, loo_label_map

    # 1. Build LOO training set
    target_mask = (y_train["label"] != target).values
    n_removed = int((~target_mask).sum())
    X_tr = X_train[target_mask]
    y_tr_labels = y_train.loc[target_mask, "label"].values

    remaining_classes = sorted(set(y_tr_labels))
    loo_label_map = {cls: i for i, cls in enumerate(remaining_classes)}
    y_tr_encoded = np.array([loo_label_map[c] for c in y_tr_labels], dtype=np.int32)
    n_classes_loo = len(remaining_classes)

    log(f"  Removed {n_removed:,} rows of {target}")
    log(f"  Training: {X_tr.shape[0]:,} rows × {X_tr.shape[1]} cols, "
        f"{n_classes_loo} classes (was {len(multiclass_map)})")

    # Sanity: target must really be gone
    assert target not in remaining_classes, f"{target} leaked into LOO training set"

    # 2. Train
    params = dict(XGB_PARAMS)
    params["num_class"] = n_classes_loo
    model = XGBClassifier(**params)

    t_train = time.time()
    model.fit(X_tr, y_tr_encoded)
    train_min = (time.time() - t_train) / 60
    log(f"  Training done in {train_min:.1f} min")

    # 3. Save model + label map immediately
    joblib.dump(model, model_path, compress=3)
    with open(map_path, "w") as f:
        json.dump(loo_label_map, f, indent=2)
    log(f"  Saved model -> {model_path.name}")

    # 4. Predict on full test
    t_pred = time.time()
    loo_pred_encoded = model.predict(X_test).astype(np.int32)
    loo_proba = model.predict_proba(X_test).astype(np.float32)
    log(f"  Prediction done in {time.time()-t_pred:.1f}s "
        f"(pred {loo_pred_encoded.shape}, proba {loo_proba.shape})")

    np.save(pred_path,  loo_pred_encoded)
    np.save(proba_path, loo_proba)
    log(f"  Saved predictions to {PREDICTIONS_DIR.name}/")

    # 5. Free memory before next fold
    del model, X_tr, y_tr_encoded, y_tr_labels
    gc.collect()

    return loo_pred_encoded, loo_proba, loo_label_map


# Run all 5 LOO folds
results = {}
for target in ZERO_DAY_TARGETS:
    pred_enc, proba, lmap = run_loo_target(target)
    results[target] = {"pred_encoded": pred_enc, "proba": proba, "label_map": lmap}


# %% Section 3 — Per-target zero-day evaluation
log("\n" + "=" * 70)
log("PER-TARGET EVALUATION")
log("=" * 70)

per_target_rows        = []  # rows for loo_results.csv
case_distribution_rows = []  # fusion case breakdowns
prediction_dist_rows   = []  # what LOO-E7 thinks held-out classes are
h2_records             = []  # for H2 verdict computation

for target in ZERO_DAY_TARGETS:
    log(f"\n--- {target} ---")
    pred_enc = results[target]["pred_encoded"]
    lmap     = results[target]["label_map"]
    inv_lmap = {v: k for k, v in lmap.items()}

    # Map LOO-encoded predictions back to original text labels
    loo_pred_labels = np.array([inv_lmap[int(p)] for p in pred_enc])

    target_test_mask = (y_test["label"] == target).values
    n_target_test = int(target_test_mask.sum())
    log(f"  Target test samples: {n_target_test:,}")

    if n_target_test == 0:
        log(f"  WARNING: no test samples for {target}, skipping")
        continue

    loo_preds_on_target = loo_pred_labels[target_test_mask]
    target_ae_mse = ae_test_mse[target_test_mask]

    # Sanity: LOO model must never predict the held-out class
    assert (loo_preds_on_target == target).sum() == 0, (
        f"LOO model predicted {target} — should be impossible (was removed from training)"
    )

    # 1. What does the blind E7 call the held-out attack?
    pred_counter = Counter(loo_preds_on_target.tolist())
    n_called_benign       = pred_counter.get("Benign", 0)
    n_called_other_attack = n_target_test - n_called_benign  # target itself can't appear
    log(f"  LOO-E7 → Benign:        {n_called_benign:,} "
        f"({100*n_called_benign/n_target_test:.1f}%)")
    log(f"  LOO-E7 → Other attack:  {n_called_other_attack:,} "
        f"({100*n_called_other_attack/n_target_test:.1f}%)")
    top5 = pred_counter.most_common(5)
    log(f"  Top-5 predictions: {top5}")

    for cls, cnt in pred_counter.items():
        prediction_dist_rows.append({
            "target": target,
            "predicted_as": cls,
            "count": int(cnt),
            "pct": float(100 * cnt / n_target_test),
        })

    # 2. AE recall — overall and on LOO-E7-missed (called Benign) samples
    benign_mask = (loo_preds_on_target == "Benign")
    n_benign = int(benign_mask.sum())

    ae_recall_per_thr        = {}
    ae_on_loo_missed_per_thr = {}

    for thr_name, thr_val in ae_thresholds.items():
        ae_anomaly_target = (target_ae_mse > thr_val)
        ae_recall = float(ae_anomaly_target.mean())
        ae_recall_per_thr[thr_name] = ae_recall

        if n_benign > 0:
            ae_on_missed = float((target_ae_mse[benign_mask] > thr_val).mean())
        else:
            ae_on_missed = float("nan")
        ae_on_loo_missed_per_thr[thr_name] = ae_on_missed

        log(f"  AE @ {thr_name}={thr_val:.4g}: recall={ae_recall:.3f} | "
            f"on LOO-E7-missed (n={n_benign}): "
            f"{'--' if np.isnan(ae_on_missed) else f'{ae_on_missed:.3f}'}")

    # 3. Fusion case distribution
    # Case 1: LOO-E7 says attack + AE anomaly → confirmed (wrong class but detected)
    # Case 2: LOO-E7 says benign + AE anomaly → ZERO-DAY warning
    # Case 3: LOO-E7 says attack + AE normal  → E7-only alert
    # Case 4: LOO-E7 says benign + AE normal  → missed entirely
    loo_is_attack = (loo_preds_on_target != "Benign")
    case_per_thr = {}
    for thr_name, thr_val in ae_thresholds.items():
        ae_anomaly = (target_ae_mse > thr_val)
        case1 = int(( loo_is_attack &  ae_anomaly).sum())
        case2 = int((~loo_is_attack &  ae_anomaly).sum())
        case3 = int(( loo_is_attack & ~ae_anomaly).sum())
        case4 = int((~loo_is_attack & ~ae_anomaly).sum())
        assert case1 + case2 + case3 + case4 == n_target_test, (
            f"Case sums mismatch at {thr_name}: "
            f"{case1+case2+case3+case4} != {n_target_test}"
        )
        binary_recall = (case1 + case2 + case3) / n_target_test  # any-alert recall

        case_per_thr[thr_name] = {
            "case1": case1, "case2": case2, "case3": case3, "case4": case4,
            "binary_recall": float(binary_recall),
        }
        case_distribution_rows.append({
            "target": target,
            "threshold": thr_name,
            "thr_value": float(thr_val),
            "case1_attack_and_anomaly": case1,
            "case2_benign_and_anomaly_zeroday": case2,
            "case3_attack_only": case3,
            "case4_missed_entirely": case4,
            "binary_recall": float(binary_recall),
            "n_target_test": n_target_test,
        })
        log(f"  Cases @ {thr_name}: C1={case1}, C2={case2} (zero-day), "
            f"C3={case3}, C4={case4} | binary recall={binary_recall:.3f}")

    # 4. Per-target row
    per_target_rows.append({
        "target": target,
        "n_test_samples": n_target_test,
        "loo_e7_recall": 0.0,  # cannot predict the held-out class — 0 by definition
        "pct_called_benign":       100 * n_called_benign       / n_target_test,
        "pct_called_other_attack": 100 * n_called_other_attack / n_target_test,
        "ae_recall_p90":   ae_recall_per_thr.get("p90", np.nan),
        "ae_recall_p95":   ae_recall_per_thr.get("p95", np.nan),
        "ae_recall_p99":   ae_recall_per_thr.get("p99", np.nan),
        "ae_on_missed_p90": ae_on_loo_missed_per_thr.get("p90", np.nan),
        "ae_on_missed_p95": ae_on_loo_missed_per_thr.get("p95", np.nan),
        "ae_on_missed_p99": ae_on_loo_missed_per_thr.get("p99", np.nan),
        "binary_recall_p90": case_per_thr.get("p90", {}).get("binary_recall", np.nan),
        "binary_recall_p95": case_per_thr.get("p95", {}).get("binary_recall", np.nan),
        "binary_recall_p99": case_per_thr.get("p99", {}).get("binary_recall", np.nan),
        "n_loo_e7_called_benign": n_benign,
    })

    h2_records.append({
        "target": target,
        "ae_recall_on_loo_missed_p90": ae_on_loo_missed_per_thr.get("p90", np.nan),
        "ae_recall_on_loo_missed_p95": ae_on_loo_missed_per_thr.get("p95", np.nan),
        "ae_recall_on_loo_missed_p99": ae_on_loo_missed_per_thr.get("p99", np.nan),
        "ae_recall_all_p95":           ae_recall_per_thr.get("p95", np.nan),
        "ae_recall_all_p90":           ae_recall_per_thr.get("p90", np.nan),
        "binary_recall_p95":           case_per_thr.get("p95", {}).get("binary_recall", np.nan),
    })

    # Free per-target proba (large)
    results[target]["proba"] = None
    gc.collect()


# %% Section 4 — Save core metrics & comparison vs Phase 6
results_df = pd.DataFrame(per_target_rows)
results_df.to_csv(METRICS_DIR / "loo_results.csv", index=False)
log(f"\nSaved {METRICS_DIR / 'loo_results.csv'}")

pred_dist_df = pd.DataFrame(prediction_dist_rows).sort_values(
    ["target", "count"], ascending=[True, False]
)
pred_dist_df.to_csv(METRICS_DIR / "loo_prediction_distribution.csv", index=False)
log(f"Saved {METRICS_DIR / 'loo_prediction_distribution.csv'}")

case_df = pd.DataFrame(case_distribution_rows)
case_df.to_csv(METRICS_DIR / "loo_case_distribution.csv", index=False)
log(f"Saved {METRICS_DIR / 'loo_case_distribution.csv'}")

# Comparison vs Phase 6 (best-effort — schema may differ)
phase6_csv = FUSION_DIR / "metrics" / "zero_day_results.csv"
cmp_df = None
if phase6_csv.exists():
    try:
        phase6_df = pd.read_csv(phase6_csv)
        # Try to find the columns we need; fall back gracefully
        col_e7      = next((c for c in ("e7_recall", "e7_recall_target",
                                        "phase6_e7_recall") if c in phase6_df.columns), None)
        col_ae_miss = next((c for c in ("ae_recall_on_missed_p95", "ae_on_missed_p95",
                                        "ae_recall_on_e7_missed_p95") if c in phase6_df.columns), None)
        col_binary  = next((c for c in ("binary_recall_p95", "fusion_binary_recall_p95",
                                        "binary_recall") if c in phase6_df.columns), None)

        rows = []
        for _, r in results_df.iterrows():
            target = r["target"]
            sub = phase6_df[phase6_df.get("target", pd.Series()) == target] \
                if "target" in phase6_df.columns else pd.DataFrame()
            rows.append({
                "target": target,
                "phase6_e7_recall":         float(sub[col_e7].iloc[0])      if (col_e7      and not sub.empty) else np.nan,
                "loo_e7_recall":            float(r["loo_e7_recall"]),
                "phase6_ae_on_missed_p95":  float(sub[col_ae_miss].iloc[0]) if (col_ae_miss and not sub.empty) else np.nan,
                "loo_ae_on_missed_p95":     float(r["ae_on_missed_p95"]),
                "phase6_binary_recall_p95": float(sub[col_binary].iloc[0])  if (col_binary  and not sub.empty) else np.nan,
                "loo_binary_recall_p95":    float(r["binary_recall_p95"]),
            })
        cmp_df = pd.DataFrame(rows)
        cmp_df.to_csv(METRICS_DIR / "loo_vs_phase6_comparison.csv", index=False)
        log(f"Saved comparison vs Phase 6 -> {METRICS_DIR / 'loo_vs_phase6_comparison.csv'}")
    except Exception as e:
        log(f"  Phase 6 comparison failed ({e}); skipping")
        cmp_df = None
else:
    log(f"Phase 6 results not found at {phase6_csv} — skipping comparison")


# %% Section 5 — H2 re-evaluation
log("\n" + "=" * 70)
log("H2 RE-EVALUATION (TRUE LOO)")
log("=" * 70)

def evaluate_h2(records, key, label):
    n_pass = 0
    detail = []
    for r in records:
        v = r.get(key, np.nan)
        passed = (not pd.isna(v)) and (float(v) >= H2_THRESHOLD)
        if passed:
            n_pass += 1
        detail.append({
            "target": r["target"],
            "value": (None if pd.isna(v) else float(v)),
            "passes": bool(passed),
        })
    verdict = "PASS" if n_pass >= H2_MAJORITY else "FAIL"
    log(f"  [{label}] {n_pass}/{len(records)} ≥ {H2_THRESHOLD} → H2 = {verdict}")
    for d in detail:
        marker = "✓" if d["passes"] else "✗"
        v_str = "n/a" if d["value"] is None else f"{d['value']:.3f}"
        log(f"      {marker} {d['target']:30s}  {v_str}")
    return {
        "criterion": label,
        "metric_key": key,
        "n_pass": n_pass,
        "n_total": len(records),
        "verdict": verdict,
        "details": detail,
    }


h2_eval = {
    "h2_strict_ae_on_loo_missed_p95": evaluate_h2(
        h2_records, "ae_recall_on_loo_missed_p95",
        "Strict: AE recall on LOO-missed @ p95"),
    "h2_strict_ae_on_loo_missed_p90": evaluate_h2(
        h2_records, "ae_recall_on_loo_missed_p90",
        "Strict: AE recall on LOO-missed @ p90"),
    "h2_relaxed_ae_all_p95": evaluate_h2(
        h2_records, "ae_recall_all_p95",
        "Relaxed: AE recall on all target samples @ p95"),
    "h2_relaxed_ae_all_p90": evaluate_h2(
        h2_records, "ae_recall_all_p90",
        "Relaxed: AE recall on all target samples @ p90"),
    "h2_binary_p95": evaluate_h2(
        h2_records, "binary_recall_p95",
        "Binary: any-alert recall (Cases 1+2+3) @ p95"),
}

with open(METRICS_DIR / "h2_loo_verdict.json", "w") as f:
    json.dump({
        "threshold": H2_THRESHOLD,
        "majority_rule": f"≥{H2_MAJORITY} of 5 targets",
        "evaluations": h2_eval,
        "h2_records": [
            {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
             for k, v in r.items()}
            for r in h2_records
        ],
    }, f, indent=2, default=float)
log(f"\nSaved {METRICS_DIR / 'h2_loo_verdict.json'}")


# %% Section 7 — Visualizations
log("\n" + "=" * 70)
log("BUILDING FIGURES")
log("=" * 70)

# 7.1 — LOO zero-day results bar chart
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(ZERO_DAY_TARGETS))
w = 0.27
loo_e7     = results_df.set_index("target").reindex(ZERO_DAY_TARGETS)["loo_e7_recall"].values
ae_p95     = results_df.set_index("target").reindex(ZERO_DAY_TARGETS)["ae_recall_p95"].values
binary_p95 = results_df.set_index("target").reindex(ZERO_DAY_TARGETS)["binary_recall_p95"].values

ax.bar(x - w, loo_e7,     w, label="LOO-E7 recall (≡ 0)", color="#cf4a4a")
ax.bar(x,     ae_p95,     w, label="AE recall (p95)",      color="#3a7bd5")
ax.bar(x + w, binary_p95, w, label="Fusion binary recall (p95)", color="#2aa876")
ax.axhline(H2_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
           label=f"H2 target = {H2_THRESHOLD}")

ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", "\n") for t in ZERO_DAY_TARGETS], fontsize=9)
ax.set_ylabel("Recall on held-out target class")
ax.set_title("Phase 6B — True Leave-One-Attack-Out Zero-Day Results")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "loo_zero_day_results.png", bbox_inches="tight")
plt.close()
log("  Saved loo_zero_day_results.png")

# 7.2 — Phase 6 vs Phase 6B comparison
if cmp_df is not None and not cmp_df.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cmp_df))
    w = 0.2
    ax.bar(x - 1.5*w, cmp_df["phase6_e7_recall"].fillna(0),         w,
           label="Phase 6 E7",                color="#9aa9bf")
    ax.bar(x - 0.5*w, cmp_df["loo_e7_recall"].fillna(0),            w,
           label="LOO E7",                    color="#cf4a4a")
    ax.bar(x + 0.5*w, cmp_df["phase6_ae_on_missed_p95"].fillna(0),  w,
           label="Phase 6 AE-on-missed (p95)",color="#7fbcef")
    ax.bar(x + 1.5*w, cmp_df["loo_ae_on_missed_p95"].fillna(0),     w,
           label="LOO AE-on-missed (p95)",    color="#3a7bd5")
    ax.axhline(H2_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"H2 target = {H2_THRESHOLD}")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in cmp_df["target"]], fontsize=9)
    ax.set_ylabel("Recall")
    ax.set_title("Phase 6 (Simulated) vs Phase 6B (True LOO)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=2, framealpha=0.95, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loo_vs_phase6_comparison.png", bbox_inches="tight")
    plt.close()
    log("  Saved loo_vs_phase6_comparison.png")
else:
    log("  Skipping loo_vs_phase6_comparison.png (no Phase 6 data)")

# 7.3 — LOO-E7 prediction distribution per target (stacked, top-N)
pivot = pred_dist_df.pivot_table(
    index="target", columns="predicted_as", values="pct", fill_value=0
).reindex(ZERO_DAY_TARGETS)

TOP_N = 6
class_totals = pivot.sum(axis=0).sort_values(ascending=False)
top_classes  = list(class_totals.head(TOP_N).index)
other_cols   = [c for c in pivot.columns if c not in top_classes]
if other_cols:
    pivot["Other"] = pivot[other_cols].sum(axis=1)
    pivot = pivot[top_classes + ["Other"]]
else:
    pivot = pivot[top_classes]

fig, ax = plt.subplots(figsize=(11, 6))
bottom = np.zeros(len(pivot))
cmap = plt.get_cmap("tab10")
for i, col in enumerate(pivot.columns):
    ax.bar(np.arange(len(pivot)), pivot[col].values, bottom=bottom,
           label=col, color=cmap(i % 10))
    bottom += pivot[col].values

ax.set_xticks(np.arange(len(pivot)))
ax.set_xticklabels([t.replace("_", "\n") for t in pivot.index], fontsize=9)
ax.set_ylabel("% of held-out test samples")
ax.set_title("What does the blind LOO-E7 think the held-out attack is?")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "loo_prediction_distribution.png", bbox_inches="tight")
plt.close()
log("  Saved loo_prediction_distribution.png")

# 7.4 — Fusion case distribution per target (p95)
case_p95 = (case_df[case_df["threshold"] == "p95"]
            .set_index("target").reindex(ZERO_DAY_TARGETS))
n_total = case_p95["n_target_test"].values

fig, ax = plt.subplots(figsize=(11, 6))
case_labels = [
    ("case1_attack_and_anomaly",         "Case 1: E7 attack + AE anomaly",       "#2aa876"),
    ("case2_benign_and_anomaly_zeroday", "Case 2: Zero-day warning",             "#3a7bd5"),
    ("case3_attack_only",                "Case 3: E7 only (AE normal)",          "#f4a259"),
    ("case4_missed_entirely",            "Case 4: Missed entirely",              "#cf4a4a"),
]
bottom = np.zeros(len(case_p95))
for col, lbl, color in case_labels:
    pct = 100 * case_p95[col].values / n_total
    ax.bar(np.arange(len(case_p95)), pct, bottom=bottom, label=lbl, color=color)
    bottom += pct

ax.set_xticks(np.arange(len(case_p95)))
ax.set_xticklabels([t.replace("_", "\n") for t in case_p95.index], fontsize=9)
ax.set_ylabel("% of held-out target samples")
ax.set_title("Fusion case distribution under true LOO (AE @ p95)")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "loo_case_distribution.png", bbox_inches="tight")
plt.close()
log("  Saved loo_case_distribution.png")


# %% Section 9 — summary.md
log("\nWriting summary.md...")

n_strict_pass_p95 = h2_eval["h2_strict_ae_on_loo_missed_p95"]["n_pass"]
n_strict_pass_p90 = h2_eval["h2_strict_ae_on_loo_missed_p90"]["n_pass"]
n_relaxed_p95     = h2_eval["h2_relaxed_ae_all_p95"]["n_pass"]
n_binary_p95      = h2_eval["h2_binary_p95"]["n_pass"]
overall_verdict   = h2_eval["h2_strict_ae_on_loo_missed_p95"]["verdict"]

lines = []
lines.append("# Phase 6B — True Leave-One-Attack-Out Zero-Day Results\n\n")
lines.append(f"_Run started:  {config['started_at']}_  \n")
lines.append(f"_Run finished: {datetime.now().isoformat(timespec='seconds')}_  \n")
lines.append(f"_Total runtime: {(time.time()-t_start)/60:.1f} min_\n\n")

lines.append("## 1. Setup\n\n")
lines.append("- Trained **5** XGBoost models, each excluding one target class entirely from training.\n")
lines.append("- Hyperparameters identical to E7; only the training data changes.\n")
lines.append("- The AE and IF were trained on benign-only data and remain **unchanged** — "
             "their scores carry over from Phase 6 unmodified.\n")
lines.append(f"- AE thresholds: `{ae_thresholds}`\n")
lines.append(f"- H2 criterion: AE recall ≥ {H2_THRESHOLD} on ≥ {H2_MAJORITY} of 5 held-out target classes.\n\n")

lines.append("## 2. Per-target results\n\n")
lines.append("| Target | n test | LOO-E7→Benign | LOO-E7→Other | AE recall (p95) | AE on LOO-missed (p95) | Binary recall (p95) |\n")
lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
for _, r in results_df.iterrows():
    lines.append(
        f"| {r['target']} | {int(r['n_test_samples']):,} | "
        f"{r['pct_called_benign']:.1f}% | {r['pct_called_other_attack']:.1f}% | "
        f"{fmt(r['ae_recall_p95'])} | {fmt(r['ae_on_missed_p95'])} | "
        f"{fmt(r['binary_recall_p95'])} |\n"
    )

lines.append("\n## 3. H2 re-evaluation under true LOO\n\n")
lines.append(f"**Primary verdict (strict, AE on LOO-missed @ p95): {overall_verdict}** "
             f"({n_strict_pass_p95}/5 targets ≥ {H2_THRESHOLD}).\n\n")
for k, e in h2_eval.items():
    lines.append(f"### {e['criterion']}\n\n")
    lines.append(f"**{e['verdict']}** — {e['n_pass']}/{e['n_total']} targets ≥ {H2_THRESHOLD}\n\n")
    for d in e["details"]:
        marker = "✓" if d["passes"] else "✗"
        v_str = "—" if d["value"] is None else f"{d['value']:.3f}"
        lines.append(f"- {marker} `{d['target']}` → {v_str}\n")
    lines.append("\n")

if cmp_df is not None and not cmp_df.empty:
    lines.append("## 4. Phase 6 (simulated) vs Phase 6B (true LOO)\n\n")
    lines.append("| Target | P6 E7 | LOO E7 | P6 AE-missed (p95) | LOO AE-missed (p95) | P6 Binary (p95) | LOO Binary (p95) |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for _, r in cmp_df.iterrows():
        lines.append(
            f"| {r['target']} | "
            f"{fmt(r['phase6_e7_recall'])} | {fmt(r['loo_e7_recall'])} | "
            f"{fmt(r['phase6_ae_on_missed_p95'])} | {fmt(r['loo_ae_on_missed_p95'])} | "
            f"{fmt(r['phase6_binary_recall_p95'])} | {fmt(r['loo_binary_recall_p95'])} |\n"
        )
    lines.append("\n")
else:
    lines.append("## 4. Phase 6 comparison\n\n")
    lines.append("_Phase 6 results CSV not found or had a different schema — comparison skipped._\n\n")

lines.append("## 5. What does the blind LOO-E7 think held-out attacks are?\n\n")
for target in ZERO_DAY_TARGETS:
    sub = (pred_dist_df[pred_dist_df["target"] == target]
           .sort_values("count", ascending=False).head(5))
    parts = [f"{r['predicted_as']} ({r['pct']:.1f}%)" for _, r in sub.iterrows()]
    lines.append(f"- **{target}** → {', '.join(parts)}\n")
lines.append("\n")

lines.append("## 6. Discussion\n\n")
lines.append(
    "Phase 6 reported H2 as FAIL (0/5) but was based on a simulated LOO — the supervised "
    "model E7 was trained on all 19 classes including each target, so the only samples it "
    "'missed' were edge cases near a decision boundary, where the AE has the least leverage. "
    "Under true LOO, the supervised model has zero exposure to the held-out class.\n\n"
)
lines.append(
    f"With true LOO, the AE flagged ≥ {H2_THRESHOLD:.0%} of LOO-E7-missed samples on "
    f"**{n_strict_pass_p95}/5** targets at p95 (and **{n_strict_pass_p90}/5** at p90). "
    f"The relaxed criterion (AE recall on *all* target samples, not just those E7 calls "
    f"benign) passes on **{n_relaxed_p95}/5** at p95. The fused binary detector — "
    f"the practical IDS metric — raised an alert (Cases 1+2+3) on **{n_binary_p95}/5** "
    f"targets.\n\n"
)
lines.append(
    "Why the gap between strict and binary criteria: when the LOO-E7 misclassifies a held-out "
    "attack as a different known attack (Case 1 or Case 3), the IDS still triggers a response — "
    "the operator sees an alert, even if the assigned class is wrong. Pure 'zero-day warnings' "
    "(Case 2: E7 says benign, AE flags anomalous) are only the subset where the supervised "
    "model entirely missed the sample. That subset is precisely what the H2 strict criterion "
    "stresses, and where the AE has to carry the full burden alone.\n\n"
)
lines.append(
    "The LOO-E7 prediction distribution (Section 5) tells us how 'detectable as some attack' "
    "the held-out class is to a model that has never seen it: classes that get mapped to "
    "neighbouring attacks (e.g. Recon family → other Recon variants) keep binary recall high "
    "even when zero-day warnings are rare. Classes that get mapped to Benign place the entire "
    "detection load on the AE.\n\n"
)

lines.append("## 7. Implications for IoMT deployment\n\n")
lines.append(
    "- The fused binary alert (any of Cases 1, 2, 3) is the metric an IoMT operator actually "
    "consumes. A misclassified-but-flagged attack still triggers triage; only Case 4 silently "
    "passes through.\n"
)
lines.append(
    "- For classes the LOO-E7 confidently mislabels as a sibling attack, the system degrades "
    "gracefully — coverage stays high without the AE doing heavy lifting.\n"
)
lines.append(
    "- For classes the LOO-E7 routes to Benign, the AE is the sole line of defence. Operators "
    "should treat the AE recall on the LOO-missed subset as the conservative lower bound on "
    "true zero-day coverage.\n\n"
)

lines.append("## 8. Limitations\n\n")
lines.append("- Single random seed for the LOO XGBoost models; per-fold variance not estimated.\n")
lines.append("- The 5 targets cover Recon, MQTT, and ARP families but exclude DDoS/DoS, where "
             "many sibling labels remain in training — those would test a harder LOO scenario.\n")
lines.append("- AE thresholds are fixed from the original benign validation; they are not "
             "re-tuned per fold, which is the conservative choice but may understate AE recall.\n")
lines.append("- The LOO-E7 still sees 18 of 19 attack classes during training, so its 'novel' "
             "decision is biased toward the closest known class. A field deployment in a new "
             "hospital would face many simultaneous unknowns.\n\n")

lines.append("## 9. Future work\n\n")
lines.append("- Repeat LOO with multiple seeds; report mean ± std on every recall figure.\n")
lines.append("- Sweep the AE threshold and plot precision–recall curves on each held-out class "
             "to characterise the operating point trade-off.\n")
lines.append("- Add a calibrated low-confidence floor on the LOO-E7 softmax (e.g. flag samples "
             "where max-prob < τ) to convert Case 3 into Case 2-style warnings — this would "
             "directly raise zero-day recall without retraining either model.\n")
lines.append("- Test a stricter LOO that holds out an entire attack family (all Recon, all "
             "MQTT) rather than a single class, to estimate cross-family generalisation.\n\n")

lines.append("## 10. Output index\n\n")
lines.append("- `metrics/loo_results.csv` — per-target metrics\n")
lines.append("- `metrics/loo_vs_phase6_comparison.csv` — side-by-side with Phase 6\n")
lines.append("- `metrics/loo_prediction_distribution.csv` — what LOO-E7 thinks held-out classes are\n")
lines.append("- `metrics/loo_case_distribution.csv` — fusion case breakdown per threshold\n")
lines.append("- `metrics/h2_loo_verdict.json` — H2 evaluation under all criteria\n")
lines.append("- `figures/loo_zero_day_results.png`\n")
lines.append("- `figures/loo_vs_phase6_comparison.png`\n")
lines.append("- `figures/loo_prediction_distribution.png`\n")
lines.append("- `figures/loo_case_distribution.png`\n")
lines.append("- `models/loo_xgb_without_*.pkl` — saved retrained models\n")
lines.append("- `predictions/loo_*_test_pred.npy`, `loo_*_test_proba.npy`\n")
lines.append("- `config.json` — full run configuration\n")

with open(OUTPUT_DIR / "summary.md", "w") as f:
    f.writelines(lines)
log(f"Saved {OUTPUT_DIR / 'summary.md'}")


# %% Final timing
total_min = (time.time() - t_start) / 60
log("\n" + "=" * 70)
log(f"PHASE 6B COMPLETE — total runtime: {total_min:.1f} min")
log("=" * 70)
log(f"Outputs at: {OUTPUT_DIR.absolute()}")
log(f"H2 strict (AE on LOO-missed @ p95) verdict: "
    f"{h2_eval['h2_strict_ae_on_loo_missed_p95']['verdict']} "
    f"({n_strict_pass_p95}/5)")
log(f"H2 binary (any-alert @ p95)      verdict: "
    f"{h2_eval['h2_binary_p95']['verdict']} "
    f"({n_binary_p95}/5)")