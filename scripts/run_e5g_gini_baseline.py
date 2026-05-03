"""
E5G — RF-Gini baseline (mirrors E5 except criterion='gini')
============================================================

Purpose
-------
The course-project report §5.5 ("Entropy vs Gini Criterion") presents an
RF-Gini vs RF-Entropy comparison framed as our own experiment. Investigation
on 2026-05-02 revealed that no Gini run was ever saved (git pickaxe for
"criterion='gini'" is empty). This script produces the missing Gini half so
the §5.5 numbers become real, sourced, and reproducible.

Mirrors E5 hyperparameters EXACTLY except `criterion`:
    E5  (entropy)  -> RF n_estimators=200, criterion='entropy', max_depth=30, ...
    E5G (gini)     -> RF n_estimators=200, criterion='gini',    max_depth=30, ...

Outputs (all under results/supervised/):
    metrics/E5G_multiclass.json                  # mirrors E5_multiclass.json schema
    metrics/E5G_classification_report_test.json  # per-class P/R/F1
    metrics/E5G_cm_19class_test.npy              # confusion matrix
    metrics/E5G_run.log                          # timestamped run log
    models/E5G_rf_full_gini_original.pkl         # for reproducibility (~100 MB)

Usage
-----
    cd ~/IoMT-Project
    source venv/bin/activate
    python scripts/run_e5g_gini_baseline.py

Expected runtime: ~3-5 minutes on M4 MacBook (E5 took 222s).
"""
from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(os.path.expanduser("~/IoMT-Project"))
PREPROCESSED = PROJECT_ROOT / "preprocessed" / "full_features"
ENC_PATH     = PROJECT_ROOT / "preprocessed" / "label_encoders.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "supervised"
METRICS_DIR  = RESULTS_DIR / "metrics"
MODELS_DIR   = RESULTS_DIR / "models"
LOG_PATH     = METRICS_DIR / "E5G_run.log"

# ----------------------------------------------------------------------
# Hyperparameters (mirror notebooks/supervised_training.py:RF_PARAMS exactly,
# except criterion is flipped to 'gini')
# ----------------------------------------------------------------------
RF_PARAMS = dict(
    n_estimators=200,
    criterion="gini",          # <-- ONLY DIFFERENCE FROM E5 (which uses 'entropy')
    max_depth=30,
    min_samples_split=20,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=8,                  # matches supervised_training.py:N_JOBS=8
    verbose=0,
)

EXPERIMENT_ID = "E5G"


# ----------------------------------------------------------------------
# Logger writing to both stdout and LOG_PATH
# ----------------------------------------------------------------------
def log(msg: str) -> None:
    """Timestamped print + append to LOG_PATH."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_class_names() -> list[str]:
    """Return ordered multiclass class names (index = encoded label int)."""
    with open(ENC_PATH, "r", encoding="utf-8") as f:
        encoders = json.load(f)
    multiclass_map = encoders["multiclass"]
    return [k for k, _ in sorted(multiclass_map.items(), key=lambda kv: kv[1])]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Eight overall metrics — same set/order as E5."""
    return {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "f1_macro":           float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "f1_weighted":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc":                float(matthews_corrcoef(y_true, y_pred)),
        "precision_macro":    float(precision_score(y_true, y_pred, average="macro",    zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro":       float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "recall_weighted":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Truncate prior log for a clean run record
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log(f"E5G — RF-Gini baseline (mirrors E5 except criterion='gini')")
    log(f"  Hyperparameters: {RF_PARAMS}")

    # ---------------- Load data ----------------
    log("Loading data ...")
    X_train = np.load(PREPROCESSED / "X_train.npy")
    X_val   = np.load(PREPROCESSED / "X_val.npy")
    X_test  = np.load(PREPROCESSED / "X_test.npy")
    y_train = pd.read_csv(PREPROCESSED / "y_train.csv")
    y_val   = pd.read_csv(PREPROCESSED / "y_val.csv")
    y_test  = pd.read_csv(PREPROCESSED / "y_test.csv")
    log(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    label_col = "multiclass_label"
    if label_col not in y_train.columns:
        raise KeyError(
            f"Column '{label_col}' missing from y_train.csv — "
            f"available columns: {list(y_train.columns)}"
        )
    y_train_mc = y_train[label_col].to_numpy()
    y_val_mc   = y_val[label_col].to_numpy()
    y_test_mc  = y_test[label_col].to_numpy()

    class_names = load_class_names()
    log(f"  {len(class_names)} multiclass labels loaded (label_col='{label_col}')")

    # ---------------- Train ----------------
    log("Training RandomForestClassifier(criterion='gini') ...")
    t0 = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train_mc)
    train_time = time.time() - t0
    log(f"  Train time: {train_time:.2f}s  ({train_time/60:.2f} min)")

    # ---------------- Predict ----------------
    log("Predicting on val + test ...")
    t0 = time.time()
    y_val_pred  = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    pred_time = time.time() - t0
    log(f"  Predict time (val+test): {pred_time:.2f}s")

    # ---------------- Evaluate ----------------
    log("Evaluating ...")
    val_metrics  = evaluate(y_val_mc,  y_val_pred)
    test_metrics = evaluate(y_test_mc, y_test_pred)

    log("Validation metrics:")
    for k, v in val_metrics.items():
        log(f"  val_{k}: {v:.4f}")
    log("Test metrics:")
    for k, v in test_metrics.items():
        log(f"  test_{k}: {v:.4f}")

    # ---------------- Save outputs ----------------
    out_json = {
        "experiment":          EXPERIMENT_ID,
        "model":               "RF",
        "data":                "Original",
        "feature_set":         "full",
        "n_features":          int(X_train.shape[1]),
        "task":                "multiclass",
        "n_classes":           int(len(class_names)),
        "training_time_sec":   round(float(train_time), 2),
        "prediction_time_sec": round(float(pred_time), 2),
        **{f"val_{k}":  v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "report_path":  f"{EXPERIMENT_ID}_classification_report_test.json",
        # Extra fields making the file self-documenting:
        "criterion":    "gini",
        "baseline_for": "E5",
        "generated_by": "scripts/run_e5g_gini_baseline.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = METRICS_DIR / f"{EXPERIMENT_ID}_multiclass.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    log(f"  Saved {out_path.relative_to(PROJECT_ROOT)}")

    report = classification_report(
        y_test_mc, y_test_pred,
        target_names=class_names,
        digits=4, zero_division=0,
        output_dict=True,
    )
    report_path = METRICS_DIR / f"{EXPERIMENT_ID}_classification_report_test.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"  Saved {report_path.relative_to(PROJECT_ROOT)}")

    cm = confusion_matrix(
        y_test_mc, y_test_pred,
        labels=list(range(len(class_names))),
    )
    cm_path = METRICS_DIR / f"{EXPERIMENT_ID}_cm_19class_test.npy"
    np.save(cm_path, cm)
    log(f"  Saved {cm_path.relative_to(PROJECT_ROOT)}")

    model_path = MODELS_DIR / f"{EXPERIMENT_ID}_rf_full_gini_original.pkl"
    joblib.dump(model, model_path, compress=3)
    log(f"  Saved {model_path.relative_to(PROJECT_ROOT)} "
        f"({model_path.stat().st_size / 1e6:.1f} MB)")

    # ---------------- Compare to E5 ----------------
    e5_path = METRICS_DIR / "E5_multiclass.json"
    with open(e5_path, "r", encoding="utf-8") as f:
        e5 = json.load(f)

    rows = [
        ("test_accuracy",    out_json["test_accuracy"],    e5["test_accuracy"]),
        ("test_f1_macro",    out_json["test_f1_macro"],    e5["test_f1_macro"]),
        ("test_f1_weighted", out_json["test_f1_weighted"], e5["test_f1_weighted"]),
        ("test_mcc",         out_json["test_mcc"],         e5["test_mcc"]),
    ]
    log("")
    log("=" * 64)
    log("COMPARISON: E5G (Gini) vs E5 (Entropy) — multiclass test set")
    log("=" * 64)
    log(f"  {'Metric':<20} {'Gini (E5G)':>12} {'Entropy (E5)':>14} {'Diff (pp)':>12}")
    log("  " + "-" * 60)
    for name, gini_v, entropy_v in rows:
        diff_pp = (entropy_v - gini_v) * 100.0
        log(f"  {name:<20} {gini_v:>12.4f} {entropy_v:>14.4f} {diff_pp:>+12.2f}")
    log("=" * 64)
    log(f"E5G done. Total wall time: {(train_time + pred_time):.1f}s")


if __name__ == "__main__":
    main()
