"""
Phase 4 — Supervised Model Training (RF + XGBoost)
====================================================

Trains 8 experiments (RF & XGBoost × reduced/full features × original/SMOTETomek)
and evaluates each at three classification granularities (binary, 6-class, 19-class).

This is Layer 1 of the Hybrid Supervised-Unsupervised IoMT IDS framework.
Outputs (predictions, probabilities, metrics, plots) feed into:
    Phase 5  — unsupervised layer (Autoencoder, Isolation Forest)
    Phase 6  — fusion engine (4-case decision logic)
    Phase 7  — per-class SHAP explainability

USAGE
-----
    cd ~/IoMT-Project
    source venv/bin/activate
    pip install xgboost joblib   # imbalanced-learn already installed in Phase 3
    python notebooks/supervised_training.py

EXPECTED RUNTIME
----------------
    ~1.5 – 2.5 hours on M4 MacBook (24 GB RAM).
    Disk usage of results/supervised/ ≈ 4 – 6 GB (mostly RF model pickles).

RESUME
------
    The script writes per-task metrics JSON immediately after each model
    finishes. If interrupted, set SKIP_IF_EXISTS=True and rerun: completed
    (experiment, task) pairs are detected and skipped automatically.

"""

# %% ============================================================
#                        IMPORTS
# ================================================================
from __future__ import annotations

import gc
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# %% ============================================================
#                        CONFIGURATION
# ================================================================
PREPROCESSED_DIR = Path("./preprocessed/")
OUTPUT_DIR       = Path("./results/supervised/")
RANDOM_STATE     = 42
N_JOBS           = 8            # use all cores
SKIP_IF_EXISTS   = True          # skip (exp, task) pairs whose metrics file exists
SAVE_FLOAT32_PROBA = True        # halve disk usage for predict_proba arrays

# RF hyperparameters (Yacoubi finding: entropy >> gini)
RF_PARAMS = dict(
    n_estimators=200,
    criterion="entropy",
    max_depth=30,
    min_samples_split=20,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbose=0,
)

# XGBoost hyperparameters
XGB_PARAMS_BASE = dict(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    tree_method="hist",          # ARM-optimized, fast on M-series
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbosity=0,
)

# Yacoubi et al. benchmarks for cross-comparison (raw, non-deduped data)
YACOUBI_BENCHMARKS = {
    "RF (entropy)": 0.9987,
    "XGBoost":      0.9980,
    "CatBoost":     0.9936,
    "Stacking":     0.9939,
}

# Hardest classification boundaries known from EDA — flagged in CM analysis
HARD_BOUNDARIES = [
    ("DDoS_SYN",  "DoS_SYN"),
    ("DDoS_TCP",  "DoS_TCP"),
    ("DDoS_ICMP", "DoS_ICMP"),
    ("DDoS_UDP",  "DoS_UDP"),
    ("Recon_OS_Scan", "Recon_VulScan"),
]


# %% ============================================================
#                      LOGGING / UTILITIES
# ================================================================
def log(msg: str) -> None:
    """Timestamped print, flushed immediately."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n {title}\n{bar}", flush=True)


def make_dirs() -> None:
    for sub in ("models", "predictions", "metrics", "figures"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


# %% ============================================================
#                  METADATA & FEATURE NAMES
# ================================================================
def load_metadata() -> dict:
    """Load config.json and label_encoders.json and build class-name lists."""
    with open(PREPROCESSED_DIR / "config.json") as f:
        cfg = json.load(f)
    with open(PREPROCESSED_DIR / "label_encoders.json") as f:
        enc = json.load(f)

    def _names(mapping: dict[str, int]) -> list[str]:
        return [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]

    meta = {
        "config":       cfg,
        "encoders":     enc,
        "binary_names":     _names(enc["binary"])     if "binary"     in enc else ["Benign", "Attack"],
        "category_names":   _names(enc["category"])   if "category"   in enc else None,
        "multiclass_names": _names(enc["multiclass"]) if "multiclass" in enc else None,
        "feature_names_reduced": cfg.get("feature_names_reduced"),
        "feature_names_full":    cfg.get("feature_names_full"),
    }

    # Defensive fallback if config.json doesn't have the names list
    if meta["feature_names_reduced"] is None:
        meta["feature_names_reduced"] = [f"f{i}" for i in range(28)]
    if meta["feature_names_full"] is None:
        meta["feature_names_full"] = [f"f{i}" for i in range(44)]

    return meta


# %% ============================================================
#                        DATA LOADING
# ================================================================
def load_data(feature_set: str, smote: bool):
    """Load X/y for one experiment configuration. Returns dict."""
    base = PREPROCESSED_DIR / f"{feature_set}_features"
    X_train_file = "X_train_smote.npy" if smote else "X_train.npy"
    y_train_file = "y_train_smote.csv" if smote else "y_train.csv"

    log(f"  Loading {feature_set}/{X_train_file} ...")
    X_train = np.load(base / X_train_file)
    log(f"    X_train shape: {X_train.shape}")

    X_val   = np.load(base / "X_val.npy")
    X_test  = np.load(base / "X_test.npy")
    y_train = pd.read_csv(base / y_train_file)
    y_val   = pd.read_csv(base / "y_val.csv")
    y_test  = pd.read_csv(base / "y_test.csv")

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )


# %% ============================================================
#                     MODEL FACTORY
# ================================================================
def get_rf(task: str, n_classes: int) -> RandomForestClassifier:
    """Random Forest factory. class_weight='balanced' helps even on SMOTE data."""
    return RandomForestClassifier(**RF_PARAMS)


def get_xgb(task: str, n_classes: int) -> XGBClassifier:
    """XGBoost factory — chooses objective per task."""
    params = dict(XGB_PARAMS_BASE)
    if task == "binary":
        params["objective"]   = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"]   = "multi:softprob"
        params["num_class"]   = n_classes
        params["eval_metric"] = "mlogloss"
    return XGBClassifier(**params)


# %% ============================================================
#                    EVALUATION METRICS
# ================================================================
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all overall metrics. zero_division=0 silences warnings on absent classes."""
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


# %% ============================================================
#                    CONFUSION MATRIX PLOTTING
# ================================================================
def plot_cm(cm: np.ndarray, class_names: list[str], title: str, save_path: Path,
            normalize: bool = True) -> None:
    """Pure-matplotlib confusion-matrix heatmap. Highlights known-hard boundary cells."""
    n = len(class_names)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        mat = cm.astype(float) / row_sums
    else:
        mat = cm.astype(float)

    figsize = (16, 14) if n > 6 else (9, 7)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else mat.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Annotate cells
    fontsize = 7 if n > 10 else 9
    fmt = ".2f" if normalize else "d"
    threshold = mat.max() * 0.6
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if normalize and val < 0.005:
                continue  # skip clutter from near-zero off-diagonals
            color = "white" if val > threshold else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    color=color, fontsize=fontsize)

    # Highlight the hardest known boundaries with red rectangles
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    for a, b in HARD_BOUNDARIES:
        if a in name_to_idx and b in name_to_idx:
            i, j = name_to_idx[a], name_to_idx[b]
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, edgecolor="red", lw=1.2))
            ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1,
                                        fill=False, edgecolor="red", lw=1.2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# %% ============================================================
#                    EXPERIMENT DEFINITIONS
# ================================================================
EXPERIMENTS = [
    {"id": "E1", "model": "rf",  "feature_set": "reduced", "smote": False},
    {"id": "E2", "model": "rf",  "feature_set": "reduced", "smote": True},
    {"id": "E3", "model": "xgb", "feature_set": "reduced", "smote": False},
    {"id": "E4", "model": "xgb", "feature_set": "reduced", "smote": True},
    {"id": "E5", "model": "rf",  "feature_set": "full",    "smote": False},
    {"id": "E6", "model": "rf",  "feature_set": "full",    "smote": True},
    {"id": "E7", "model": "xgb", "feature_set": "full",    "smote": False},
    {"id": "E8", "model": "xgb", "feature_set": "full",    "smote": True},
]

# Set during init() once metadata is loaded
TASKS: list[dict] = []


def init_tasks(meta: dict) -> None:
    global TASKS
    TASKS = [
        {"name": "binary",     "label_col": "binary_label",     "n_classes": 2,
         "names": meta["binary_names"]},
        {"name": "category",   "label_col": "category_label",   "n_classes": 6,
         "names": meta["category_names"]},
        {"name": "multiclass", "label_col": "multiclass_label", "n_classes": 19,
         "names": meta["multiclass_names"]},
    ]


def exp_filename(exp: dict) -> str:
    """E1_rf_reduced_original."""
    data_tag = "smote" if exp["smote"] else "original"
    return f"{exp['id']}_{exp['model']}_{exp['feature_set']}_{data_tag}"


# %% ============================================================
#                     SAVE PREDICTIONS
# ================================================================
def save_predictions(exp: dict, task: dict, y_val_pred, y_test_pred,
                     y_val_proba, y_test_proba) -> None:
    """
    Save predictions and probabilities.

    The 19-class (multiclass) outputs follow the spec naming exactly so the
    Phase 6 fusion engine can find them. Other tasks get a task suffix.
    """
    base = OUTPUT_DIR / "predictions"
    eid  = exp["id"]
    suffix = "" if task["name"] == "multiclass" else f"_{task['name']}"

    if SAVE_FLOAT32_PROBA:
        y_val_proba  = y_val_proba.astype(np.float32)
        y_test_proba = y_test_proba.astype(np.float32)

    np.save(base / f"{eid}_val_pred{suffix}.npy",   y_val_pred.astype(np.int32))
    np.save(base / f"{eid}_test_pred{suffix}.npy",  y_test_pred.astype(np.int32))
    np.save(base / f"{eid}_val_proba{suffix}.npy",  y_val_proba)
    np.save(base / f"{eid}_test_proba{suffix}.npy", y_test_proba)


# %% ============================================================
#                  SINGLE (EXPERIMENT, TASK) RUNNER
# ================================================================
def run_one(exp: dict, task: dict, data: dict, meta: dict) -> dict | None:
    """Run training + evaluation for a single (experiment, task) pair."""
    eid       = exp["id"]
    fname     = exp_filename(exp)
    task_name = task["name"]
    n_classes = task["n_classes"]

    metrics_path = OUTPUT_DIR / "metrics" / f"{eid}_{task_name}.json"
    if SKIP_IF_EXISTS and metrics_path.exists():
        log(f"  ↳ {eid}/{task_name}: cached metrics found — skipping training")
        with open(metrics_path) as f:
            return json.load(f)

    log(f"  ↳ {eid}/{task_name}: training {exp['model'].upper()} "
        f"({n_classes}-class) on {exp['feature_set']} features "
        f"({'SMOTE' if exp['smote'] else 'original'})")

    # ----- labels for this task -----
    label_col = task["label_col"]
    if label_col not in data["y_train"].columns:
        log(f"     ⚠ Column '{label_col}' missing in y_train — skipping task")
        return None

    y_train_t = data["y_train"][label_col].to_numpy()
    y_val_t   = data["y_val"][label_col].to_numpy()
    y_test_t  = data["y_test"][label_col].to_numpy()

    # ----- model -----
    model = (get_rf  if exp["model"] == "rf"  else get_xgb)(task_name, n_classes)

    # ----- train -----
    t0 = time.time()
    model.fit(data["X_train"], y_train_t)
    train_time = time.time() - t0
    log(f"     trained in {train_time:>7.1f}s")

    # ----- predict on val and test -----
    t0 = time.time()
    y_val_pred   = model.predict(data["X_val"])
    y_val_proba  = model.predict_proba(data["X_val"])
    y_test_pred  = model.predict(data["X_test"])
    y_test_proba = model.predict_proba(data["X_test"])
    pred_time = time.time() - t0
    log(f"     predicted val+test in {pred_time:.1f}s")

    # ----- save predictions (needed for Phase 6 fusion) -----
    save_predictions(exp, task, y_val_pred, y_test_pred, y_val_proba, y_test_proba)

    # ----- evaluate -----
    val_metrics  = evaluate(y_val_t,  y_val_pred)
    test_metrics = evaluate(y_test_t, y_test_pred)

    log(f"     VAL  acc={val_metrics['accuracy']:.4f} "
        f"F1_macro={val_metrics['f1_macro']:.4f} MCC={val_metrics['mcc']:.4f}")
    log(f"     TEST acc={test_metrics['accuracy']:.4f} "
        f"F1_macro={test_metrics['f1_macro']:.4f} MCC={test_metrics['mcc']:.4f}")

    # ----- save the multiclass model + classification report (per spec) -----
    extra = {}
    if task_name == "multiclass":
        # save model with the spec-mandated filename
        model_path = OUTPUT_DIR / "models" / f"{fname}.pkl"
        joblib.dump(model, model_path, compress=3)
        log(f"     model saved → {model_path.name}")

        # full per-class classification report on TEST set
        labels  = list(range(n_classes))
        report  = classification_report(
            y_test_t, y_test_pred,
            labels=labels, target_names=task["names"],
            output_dict=True, zero_division=0,
        )
        report_path = OUTPUT_DIR / "metrics" / f"{eid}_classification_report_test.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # confusion matrices (test set)
        cm_test = confusion_matrix(y_test_t, y_test_pred, labels=labels)
        np.save(OUTPUT_DIR / "metrics" / f"{eid}_cm_19class_test.npy", cm_test)
        plot_cm(
            cm_test, task["names"],
            f"{eid} {exp['model'].upper()} {exp['feature_set']} "
            f"{'SMOTE' if exp['smote'] else 'original'} — 19-class (test)",
            OUTPUT_DIR / "figures" / f"cm_{eid}_19class.png",
        )

        # feature importance (RF only — TreeSHAP-quality proxy for Phase 7 preview)
        if exp["model"] == "rf":
            feat_names = (meta["feature_names_full"] if exp["feature_set"] == "full"
                          else meta["feature_names_reduced"])
            feat_names = feat_names[:model.n_features_in_]
            importances = pd.Series(model.feature_importances_, index=feat_names)
            importances = importances.sort_values(ascending=False)
            importances.to_csv(
                OUTPUT_DIR / "metrics" / f"{eid}_feature_importance.csv",
                header=["importance"], index_label="feature",
            )
        extra["report_path"] = str(report_path.name)

    elif task_name == "category":
        labels = list(range(n_classes))
        cm_test = confusion_matrix(y_test_t, y_test_pred, labels=labels)
        np.save(OUTPUT_DIR / "metrics" / f"{eid}_cm_6class_test.npy", cm_test)
        plot_cm(
            cm_test, task["names"],
            f"{eid} {exp['model'].upper()} {exp['feature_set']} "
            f"{'SMOTE' if exp['smote'] else 'original'} — 6-class (test)",
            OUTPUT_DIR / "figures" / f"cm_{eid}_6class.png",
        )

    # ----- assemble row for the master comparison table -----
    row = {
        "experiment":  eid,
        "model":       exp["model"].upper(),
        "data":        "SMOTE" if exp["smote"] else "Original",
        "feature_set": exp["feature_set"],
        "n_features":  data["X_train"].shape[1],
        "task":        task_name,
        "n_classes":   n_classes,
        "training_time_sec": round(train_time, 2),
        "prediction_time_sec": round(pred_time, 2),
        **{f"val_{k}":  v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **extra,
    }

    # ----- persist immediately so a crash doesn't lose this result -----
    with open(metrics_path, "w") as f:
        json.dump(row, f, indent=2)

    # ----- cleanup -----
    del model, y_val_pred, y_val_proba, y_test_pred, y_test_proba
    gc.collect()

    return row


# %% ============================================================
#                          MAIN LOOP
# ================================================================
def run_all_experiments(meta: dict) -> list[dict]:
    """Iterate over the 8 experiments and 3 tasks; collect all result rows."""
    all_results: list[dict] = []
    overall_t0 = time.time()
    total_pairs = len(EXPERIMENTS) * len(TASKS)
    completed = 0

    for exp in EXPERIMENTS:
        section(f"EXPERIMENT {exp['id']} — {exp['model'].upper()} | "
                f"{exp['feature_set']} features | "
                f"{'SMOTE' if exp['smote'] else 'Original'}")
        try:
            data = load_data(exp["feature_set"], exp["smote"])
        except Exception as e:
            log(f"  ✗ Failed to load data: {e}")
            continue

        for task in TASKS:
            try:
                row = run_one(exp, task, data, meta)
                if row is not None:
                    all_results.append(row)
                    # write incremental comparison CSV
                    pd.DataFrame(all_results).to_csv(
                        OUTPUT_DIR / "metrics" / "overall_comparison.csv",
                        index=False,
                    )
            except Exception as e:
                log(f"  ✗ {exp['id']}/{task['name']} FAILED: {type(e).__name__}: {e}")
                # continue to next task — one failure shouldn't kill the run

            completed += 1
            elapsed = time.time() - overall_t0
            avg = elapsed / max(completed, 1)
            eta_min = (total_pairs - completed) * avg / 60
            log(f"  Progress: {completed}/{total_pairs} | "
                f"elapsed {elapsed/60:.1f} min | ETA {eta_min:.1f} min")

        # release this experiment's data before the next one
        del data
        gc.collect()

    return all_results


# %% ============================================================
#                  POST-RUN: COMPARISON TABLES
# ================================================================
def build_comparison_tables(meta: dict) -> dict:
    """Build the 4 comparison tables described in spec Section 6."""
    section("BUILDING COMPARISON TABLES")

    overall_csv = OUTPUT_DIR / "metrics" / "overall_comparison.csv"
    if not overall_csv.exists():
        log("No overall_comparison.csv to summarize.")
        return {}
    df = pd.read_csv(overall_csv)
    log(f"  Loaded {len(df)} result rows.")

    # ---- Table 1 — Overall metrics (already saved, just confirm) ----
    log("  Table 1: overall_comparison.csv  ✓")

    # ---- Table 2 — Per-class F1 for the BEST 19-class experiment ----
    best_per_class = None
    df_mc = df[df["task"] == "multiclass"]
    if not df_mc.empty:
        best_idx  = df_mc["test_f1_macro"].idxmax()
        best_row  = df_mc.loc[best_idx]
        best_eid  = best_row["experiment"]
        log(f"  Table 2: best 19-class experiment is {best_eid} "
            f"(test F1_macro={best_row['test_f1_macro']:.4f})")

        report_path = OUTPUT_DIR / "metrics" / f"{best_eid}_classification_report_test.json"
        if report_path.exists():
            with open(report_path) as f:
                rep = json.load(f)
            class_rows = []
            for cls in (meta["multiclass_names"] or []):
                if cls in rep:
                    class_rows.append({
                        "class":     cls,
                        "precision": rep[cls]["precision"],
                        "recall":    rep[cls]["recall"],
                        "f1-score":  rep[cls]["f1-score"],
                        "support":   int(rep[cls]["support"]),
                    })
            best_per_class = pd.DataFrame(class_rows).sort_values("support")
            best_per_class.to_csv(
                OUTPUT_DIR / "metrics" / "best_classification_report.csv",
                index=False,
            )
            log("    best_classification_report.csv  ✓")

    # ---- Table 3 — Original vs SMOTETomek comparison ----
    rows = []
    for model in df["model"].unique():
        for fs in df["feature_set"].unique():
            for task in df["task"].unique():
                orig = df[(df["model"] == model) & (df["feature_set"] == fs)
                          & (df["data"] == "Original") & (df["task"] == task)]
                smot = df[(df["model"] == model) & (df["feature_set"] == fs)
                          & (df["data"] == "SMOTE")    & (df["task"] == task)]
                if orig.empty or smot.empty:
                    continue
                for metric in ("test_f1_macro", "test_mcc", "test_recall_macro"):
                    o = float(orig.iloc[0][metric])
                    s = float(smot.iloc[0][metric])
                    rows.append({
                        "model":       model,
                        "feature_set": fs,
                        "task":        task,
                        "metric":      metric,
                        "original":    round(o, 4),
                        "smote":       round(s, 4),
                        "delta":       round(s - o, 4),
                    })
    smote_cmp = pd.DataFrame(rows)
    smote_cmp.to_csv(OUTPUT_DIR / "metrics" / "smote_comparison.csv", index=False)
    log("  Table 3: smote_comparison.csv  ✓")

    # ---- Table 4 — Minority class focus (5 rarest) ----
    minority_focus = None
    if best_per_class is not None and not df_mc.empty:
        # The 5 rarest classes by support in the best classification report
        rare5 = best_per_class.nsmallest(5, "support")["class"].tolist()
        log(f"    5 rarest classes: {rare5}")

        # Compare: best-original-RF vs best-SMOTE-RF (same feature set)
        rows = []
        for fs in ("reduced", "full"):
            for model in ("RF", "XGB"):
                orig = df[(df["model"] == model) & (df["feature_set"] == fs)
                          & (df["data"] == "Original") & (df["task"] == "multiclass")]
                smot = df[(df["model"] == model) & (df["feature_set"] == fs)
                          & (df["data"] == "SMOTE")    & (df["task"] == "multiclass")]
                if orig.empty or smot.empty:
                    continue
                eid_o = orig.iloc[0]["experiment"]
                eid_s = smot.iloc[0]["experiment"]
                rep_o = OUTPUT_DIR / "metrics" / f"{eid_o}_classification_report_test.json"
                rep_s = OUTPUT_DIR / "metrics" / f"{eid_s}_classification_report_test.json"
                if not rep_o.exists() or not rep_s.exists():
                    continue
                with open(rep_o) as f: rep_o_data = json.load(f)
                with open(rep_s) as f: rep_s_data = json.load(f)
                for cls in rare5:
                    if cls in rep_o_data and cls in rep_s_data:
                        f1_o = rep_o_data[cls]["f1-score"]
                        f1_s = rep_s_data[cls]["f1-score"]
                        rows.append({
                            "model":         model,
                            "feature_set":   fs,
                            "class":         cls,
                            "support":       int(rep_o_data[cls]["support"]),
                            "f1_original":   round(f1_o, 4),
                            "f1_smote":      round(f1_s, 4),
                            "improvement":   round(f1_s - f1_o, 4),
                        })
        minority_focus = pd.DataFrame(rows)
        minority_focus.to_csv(OUTPUT_DIR / "metrics" / "minority_focus.csv", index=False)
        log("  Table 4: minority_focus.csv  ✓")

    return {
        "overall":          df,
        "best_per_class":   best_per_class,
        "smote_comparison": smote_cmp,
        "minority_focus":   minority_focus,
    }


# %% ============================================================
#               POST-RUN: SUMMARY PLOTS
# ================================================================
def make_summary_plots(tables: dict, meta: dict) -> None:
    section("MAKING SUMMARY PLOTS")

    df = tables.get("overall")
    if df is None or df.empty:
        return

    # ---- Plot: macro-F1 across all experiments × all tasks ----
    fig, ax = plt.subplots(figsize=(13, 6))
    pivot = df.pivot_table(
        index="experiment", columns="task",
        values="test_f1_macro", aggfunc="first",
    ).reindex([e["id"] for e in EXPERIMENTS])

    # Order columns explicitly
    col_order = [c for c in ("binary", "category", "multiclass") if c in pivot.columns]
    pivot = pivot[col_order]
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Test macro-F1 across experiments and tasks")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.legend(title="Task")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "overall_comparison_bar.png", dpi=140)
    plt.close(fig)
    log("  overall_comparison_bar.png  ✓")

    # ---- Plot: feature importance (RF) — pick the best RF run ----
    df_mc_rf = df[(df["task"] == "multiclass") & (df["model"] == "RF")]
    if not df_mc_rf.empty:
        best_eid = df_mc_rf.loc[df_mc_rf["test_f1_macro"].idxmax(), "experiment"]
        fi_path  = OUTPUT_DIR / "metrics" / f"{best_eid}_feature_importance.csv"
        if fi_path.exists():
            fi = pd.read_csv(fi_path).set_index("feature")
            top = fi.head(20)
            fig, ax = plt.subplots(figsize=(9, 8))
            top["importance"].iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("Mean decrease in entropy (RF importance)")
            ax.set_title(f"Top 20 features — {best_eid}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "figures" / "feature_importance_rf.png", dpi=140)
            plt.close(fig)
            log("  feature_importance_rf.png  ✓")

    # ---- Plot: SMOTE effect on per-class F1 (rarest classes) ----
    mf = tables.get("minority_focus")
    if mf is not None and not mf.empty:
        # Group by (model, feature_set) and plot a small grid
        groups = list(mf.groupby(["model", "feature_set"]))
        n_groups = len(groups)
        if n_groups > 0:
            cols = 2
            rows = (n_groups + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows), squeeze=False)
            for idx, ((model, fs), grp) in enumerate(groups):
                ax = axes[idx // cols][idx % cols]
                x = np.arange(len(grp))
                width = 0.4
                ax.bar(x - width/2, grp["f1_original"], width, label="Original", color="#888")
                ax.bar(x + width/2, grp["f1_smote"],    width, label="SMOTE",    color="#3a86ff")
                ax.set_xticks(x)
                ax.set_xticklabels(grp["class"], rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("F1")
                ax.set_ylim(0, 1.0)
                ax.set_title(f"{model} {fs} — minority-class F1")
                ax.grid(axis="y", alpha=0.3)
                ax.legend(fontsize=8)
            # Hide any unused subplots
            for idx in range(n_groups, rows * cols):
                axes[idx // cols][idx % cols].axis("off")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "figures" / "smote_effect.png", dpi=140)
            plt.close(fig)
            log("  smote_effect.png  ✓")


# %% ============================================================
#                POST-RUN: SUMMARY MARKDOWN
# ================================================================
def write_summary_markdown(tables: dict, meta: dict) -> None:
    section("WRITING summary.md")

    df = tables.get("overall")
    if df is None or df.empty:
        log("No data to summarize.")
        return

    lines = ["# Phase 4 Summary — Supervised Layer\n"]
    lines.append(f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_\n")
    lines.append("")

    # ---- Best overall ----
    lines.append("## 1. Best Experiment Overall (19-class task)\n")
    df_mc = df[df["task"] == "multiclass"].copy()
    if not df_mc.empty:
        best_f1  = df_mc.loc[df_mc["test_f1_macro"].idxmax()]
        best_mcc = df_mc.loc[df_mc["test_mcc"].idxmax()]
        lines.append(f"- **By macro-F1**: `{best_f1['experiment']}` "
                     f"({best_f1['model']} / {best_f1['feature_set']} / {best_f1['data']}) "
                     f"— test F1_macro = **{best_f1['test_f1_macro']:.4f}**, "
                     f"MCC = **{best_f1['test_mcc']:.4f}**, "
                     f"acc = {best_f1['test_accuracy']:.4f}")
        lines.append(f"- **By MCC**: `{best_mcc['experiment']}` "
                     f"— test MCC = **{best_mcc['test_mcc']:.4f}**")
    lines.append("")

    # ---- Best per task ----
    lines.append("## 2. Best Model per Classification Task\n")
    for task in ("binary", "category", "multiclass"):
        sub = df[df["task"] == task]
        if sub.empty:
            continue
        best = sub.loc[sub["test_f1_macro"].idxmax()]
        lines.append(f"- **{task}**: `{best['experiment']}` "
                     f"— F1_macro={best['test_f1_macro']:.4f}, "
                     f"MCC={best['test_mcc']:.4f}, "
                     f"acc={best['test_accuracy']:.4f}")
    lines.append("")

    # ---- SMOTE impact ----
    lines.append("## 3. SMOTETomek Impact\n")
    sc = tables.get("smote_comparison")
    if sc is not None and not sc.empty:
        sub = sc[(sc["task"] == "multiclass") & (sc["metric"] == "test_f1_macro")]
        for _, r in sub.iterrows():
            arrow = "↑" if r["delta"] > 0 else "↓"
            lines.append(f"- {r['model']} / {r['feature_set']} (19-class F1_macro): "
                         f"{r['original']:.4f} → {r['smote']:.4f} ({arrow} {r['delta']:+.4f})")
        # net direction
        net_pos = (sub["delta"] > 0).sum()
        lines.append("")
        lines.append(f"**Net effect on 19-class macro-F1:** "
                     f"{net_pos}/{len(sub)} configurations improved with SMOTETomek.")
    lines.append("")

    # ---- Hardest boundaries ----
    lines.append("## 4. Hardest Classification Boundaries\n")
    lines.append("Off-diagonal cells highlighted in red on the 19-class confusion matrices:")
    for a, b in HARD_BOUNDARIES:
        lines.append(f"- `{a}` ↔ `{b}`")
    lines.append("")
    lines.append("Inspect `figures/cm_<EID>_19class.png` for confusion magnitudes.")
    lines.append("")

    # ---- Top features ----
    lines.append("## 5. Feature Importance — Top 10 (RF)\n")
    if not df_mc.empty:
        df_mc_rf = df_mc[df_mc["model"] == "RF"]
        if not df_mc_rf.empty:
            best_rf = df_mc_rf.loc[df_mc_rf["test_f1_macro"].idxmax()]
            fi_path = OUTPUT_DIR / "metrics" / f"{best_rf['experiment']}_feature_importance.csv"
            if fi_path.exists():
                fi = pd.read_csv(fi_path)
                lines.append(f"From **{best_rf['experiment']}** "
                             f"({best_rf['feature_set']} / {best_rf['data']}):\n")
                lines.append("| Rank | Feature | Importance |")
                lines.append("|------|---------|-----------|")
                for i, r in fi.head(10).iterrows():
                    lines.append(f"| {i+1} | `{r['feature']}` | {r['importance']:.4f} |")
                lines.append("")
                # Compare with Yacoubi's SHAP top features (from literature review)
                yacoubi_top = ["IAT", "Rate", "Header_Length", "Srate"]
                top10 = fi.head(10)["feature"].tolist()
                overlap = [f for f in yacoubi_top if f in top10]
                lines.append(f"**Overlap with Yacoubi et al. SHAP top-4** "
                             f"({yacoubi_top}): {overlap if overlap else 'none'}")
    lines.append("")

    # ---- Yacoubi comparison ----
    lines.append("## 6. Comparison with Yacoubi et al. Benchmarks\n")
    lines.append("> Yacoubi reported on **raw (non-deduplicated)** data; our metrics are on "
                 "**deduplicated** data so we expect lower headline accuracy. The gap is the "
                 "duplicate-leakage correction — a methodological contribution, not a regression.\n")
    lines.append("| Model | Yacoubi Acc. | Our Best Acc. (19-class) |")
    lines.append("|-------|--------------|--------------------------|")
    if not df_mc.empty:
        for label, ours_filter in [
            ("RF (entropy)",  df_mc[df_mc["model"] == "RF"]),
            ("XGBoost",       df_mc[df_mc["model"] == "XGB"]),
        ]:
            if ours_filter.empty:
                continue
            ours_best = ours_filter.loc[ours_filter["test_accuracy"].idxmax()]
            yac = YACOUBI_BENCHMARKS.get(label, float("nan"))
            lines.append(f"| {label} | {yac:.4f} | {ours_best['test_accuracy']:.4f} "
                         f"({ours_best['experiment']}) |")
    lines.append("")

    # ---- Recommendation for Phase 6 fusion ----
    lines.append("## 7. Recommendation for Phase 6 Fusion Engine\n")
    if not df_mc.empty:
        best_f1 = df_mc.loc[df_mc["test_f1_macro"].idxmax()]
        lines.append(
            f"Use **`{best_f1['experiment']}`** "
            f"({best_f1['model']} / {best_f1['feature_set']} / {best_f1['data']}) "
            f"as the supervised input to the 4-case fusion engine.\n"
        )
        lines.append(
            "- Probability vectors are saved as "
            f"`predictions/{best_f1['experiment']}_val_proba.npy` and "
            f"`predictions/{best_f1['experiment']}_test_proba.npy` "
            "(shape: N × 19)."
        )
        lines.append(
            f"- The trained model is at "
            f"`models/{exp_filename(next(e for e in EXPERIMENTS if e['id'] == best_f1['experiment']))}.pkl`."
        )
    lines.append("")

    # ---- Files generated ----
    lines.append("## 8. Generated Artifacts\n")
    lines.append("```")
    lines.append("results/supervised/")
    lines.append("├── metrics/")
    lines.append("│   ├── overall_comparison.csv")
    lines.append("│   ├── best_classification_report.csv")
    lines.append("│   ├── smote_comparison.csv")
    lines.append("│   ├── minority_focus.csv")
    lines.append("│   ├── E*_feature_importance.csv")
    lines.append("│   └── E*_classification_report_test.json")
    lines.append("├── figures/")
    lines.append("│   ├── cm_E*_19class.png")
    lines.append("│   ├── cm_E*_6class.png")
    lines.append("│   ├── feature_importance_rf.png")
    lines.append("│   ├── overall_comparison_bar.png")
    lines.append("│   └── smote_effect.png")
    lines.append("├── models/                  (8 × .pkl — 19-class)")
    lines.append("├── predictions/             (val/test pred + proba per task)")
    lines.append("└── summary.md               (this file)")
    lines.append("```")
    lines.append("")

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines))
    log(f"  summary.md written ({len(lines)} lines)")


# %% ============================================================
#                            ENTRY POINT
# ================================================================
def save_run_config(meta: dict) -> None:
    """Save experiment configuration for reproducibility."""
    cfg = {
        "timestamp":         datetime.now().isoformat(),
        "random_state":      RANDOM_STATE,
        "n_jobs":            N_JOBS,
        "skip_if_exists":    SKIP_IF_EXISTS,
        "save_float32_proba": SAVE_FLOAT32_PROBA,
        "rf_params":         RF_PARAMS,
        "xgb_params_base":   XGB_PARAMS_BASE,
        "experiments":       EXPERIMENTS,
        "tasks":             [{"name": t["name"], "n_classes": t["n_classes"],
                                "label_col": t["label_col"]} for t in TASKS],
        "yacoubi_benchmarks": YACOUBI_BENCHMARKS,
        "hard_boundaries":   HARD_BOUNDARIES,
        "feature_names_reduced": meta["feature_names_reduced"],
        "feature_names_full":    meta["feature_names_full"],
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def main() -> None:
    section("PHASE 4 — SUPERVISED MODEL TRAINING")
    log(f"Preprocessed dir: {PREPROCESSED_DIR.resolve()}")
    log(f"Output dir:       {OUTPUT_DIR.resolve()}")

    make_dirs()
    meta = load_metadata()
    init_tasks(meta)
    save_run_config(meta)

    log(f"Loaded metadata: "
        f"{len(meta['multiclass_names'] or [])} multiclass, "
        f"{len(meta['category_names'] or [])} category, "
        f"{len(meta['binary_names'] or [])} binary classes")

    # ---- run all 8 × 3 = 24 (experiment, task) pairs ----
    all_results = run_all_experiments(meta)
    log(f"All experiments complete. Total result rows: {len(all_results)}")

    # ---- post-processing ----
    tables = build_comparison_tables(meta)
    make_summary_plots(tables, meta)
    write_summary_markdown(tables, meta)

    section("DONE")
    log("Bring results/supervised/summary.md and the metrics/ folder back to the project chat.")


if __name__ == "__main__":
    main()
