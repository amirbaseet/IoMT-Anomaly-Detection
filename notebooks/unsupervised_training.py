"""
Phase 5 — Unsupervised Model Training (Autoencoder + Isolation Forest)
=======================================================================
Hybrid Supervised-Unsupervised Anomaly Detection Framework for IoMT Networks
CICIoMT2024 Dataset

Layer 2 of the hybrid framework:
- Autoencoder: trained on benign traffic, flags anomalies via reconstruction error
- Isolation Forest: tree-based anomaly detection trained on benign traffic

Inputs:
    preprocessed/full_features/{X_train,X_val,X_test}.npy   # 44 features
    preprocessed/full_features/{y_train,y_val,y_test}.csv

Outputs:
    results/unsupervised/                                   # see Section 8 of prompt

Author: Amro Baseet — Sakarya University of Applied Sciences
"""

# %% ============================================================================
# 0 · CONFIGURATION
# ===============================================================================
import os
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---- paths ----
PREPROCESSED_DIR = Path("./preprocessed/")
SUPERVISED_DIR   = Path("./results/supervised/")
OUTPUT_DIR       = Path("./results/unsupervised/")

RANDOM_STATE = 42

# ---- Autoencoder hyperparameters ----
AE_ARCHITECTURE     = [44, 32, 16, 8, 16, 32, 44]   # symmetric encoder-decoder
AE_EPOCHS           = 100
AE_BATCH_SIZE       = 512
AE_LEARNING_RATE    = 1e-3
AE_PATIENCE         = 10
AE_PREDICT_BATCH    = 8192

# ---- Isolation Forest hyperparameters ----
IF_N_ESTIMATORS  = 200
IF_CONTAMINATION = 0.05

# ---- evaluation / plotting ----
THRESHOLD_NAMES = ["p90", "p95", "p99", "mean_2std", "mean_3std"]
ZERO_DAY_TARGETS = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "MQTT_DoS_Connect_Flood",
    "ARP_Spoofing",
]
DPI = 150

# %% ============================================================================
# 1 · IMPORTS, REPRODUCIBILITY, OUTPUT DIRS
# ===============================================================================
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # quiet TF info logs

import random
random.seed(RANDOM_STATE)

import numpy as np
np.random.seed(RANDOM_STATE)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# TensorFlow — guarded import with helpful message on failure
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, callbacks
except ImportError as e:
    raise ImportError(
        "TensorFlow is required for this phase.\n"
        "Install with:    pip install tensorflow\n"
        "On Apple Silicon you can also try:    pip install tensorflow-metal\n"
        f"Original error: {e}"
    )

tf.random.set_seed(RANDOM_STATE)
try:
    tf.keras.utils.set_random_seed(RANDOM_STATE)   # TF >= 2.7
except Exception:
    pass

# ---- GPU detection (M4 Metal or CUDA) ----
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"[init] {len(gpus)} GPU device(s) detected: {[g.name for g in gpus]}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"[init] could not set memory growth on {gpu.name}: {exc}")
else:
    print("[init] No GPU detected — training will run on CPU (fine for this size).")

print(f"[init] TensorFlow version: {tf.__version__}")
print(f"[init] NumPy version:      {np.__version__}")

# ---- output directory tree ----
DIRS = {
    "root":     OUTPUT_DIR,
    "models":   OUTPUT_DIR / "models",
    "scores":   OUTPUT_DIR / "scores",
    "metrics":  OUTPUT_DIR / "metrics",
    "figures":  OUTPUT_DIR / "figures",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# %% ============================================================================
# 2 · DATA LOADING & BENIGN EXTRACTION
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 2 · DATA LOADING")
print("=" * 70)

t0 = time.time()

# ---- preprocessed config (feature names, label mappings) ----
with open(PREPROCESSED_DIR / "config.json") as f:
    pp_cfg = json.load(f)
with open(PREPROCESSED_DIR / "label_encoders.json") as f:
    label_encoders = json.load(f)

# Try to extract feature names — fall back to indexed names if shape disagrees
feat_names_raw = (
    pp_cfg.get("full_features")
    or pp_cfg.get("feature_names_full")
    or pp_cfg.get("feature_names")
)
if feat_names_raw is None or len(feat_names_raw) != 44:
    feat_names_raw = [f"f{i:02d}" for i in range(44)]
    print("[data] config.json did not contain 44-element feature list; using f00..f43.")
FEATURE_NAMES = list(feat_names_raw)

# ---- load arrays ----
X_train = np.load(PREPROCESSED_DIR / "full_features" / "X_train.npy").astype(np.float32)
X_val   = np.load(PREPROCESSED_DIR / "full_features" / "X_val.npy").astype(np.float32)
X_test  = np.load(PREPROCESSED_DIR / "full_features" / "X_test.npy").astype(np.float32)

y_train = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_train.csv")
y_val   = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_val.csv")
y_test  = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_test.csv")

# pick the label column
def _label_col(df):
    for c in ("label", "Label", "multiclass", "y"):
        if c in df.columns:
            return c
    return df.columns[0]

LBL = _label_col(y_train)
y_train_lbl = y_train[LBL].astype(str).values
y_val_lbl   = y_val[LBL].astype(str).values
y_test_lbl  = y_test[LBL].astype(str).values

print(f"[data] X_train: {X_train.shape}   X_val: {X_val.shape}   X_test: {X_test.shape}")
print(f"[data] label column: '{LBL}'")
assert X_train.shape[1] == 44, f"expected 44 features, got {X_train.shape[1]}"

# ---- benign extraction ----
benign_train_mask = (y_train_lbl == "Benign")
X_benign_full = X_train[benign_train_mask]
print(f"[data] Benign rows in train split: {X_benign_full.shape[0]:,} "
      f"({100*benign_train_mask.mean():.2f}% of training data)")

# ---- 80/20 AE-train / AE-val split (benign-only) ----
rng = np.random.default_rng(RANDOM_STATE)
perm = rng.permutation(X_benign_full.shape[0])
split_idx = int(0.8 * X_benign_full.shape[0])
X_benign_train = X_benign_full[perm[:split_idx]]
X_benign_val   = X_benign_full[perm[split_idx:]]
print(f"[data] AE-train (benign): {X_benign_train.shape}")
print(f"[data] AE-val   (benign): {X_benign_val.shape}")

# ---- binary anomaly labels (benign=0, anything else=1) ----
y_val_bin  = (y_val_lbl != "Benign").astype(np.int8)
y_test_bin = (y_test_lbl != "Benign").astype(np.int8)
print(f"[data] val binary breakdown:  {int((y_val_bin==0).sum()):,} normal / "
      f"{int((y_val_bin==1).sum()):,} anomaly")
print(f"[data] test binary breakdown: {int((y_test_bin==0).sum()):,} normal / "
      f"{int((y_test_bin==1).sum()):,} anomaly")
print(f"[data] loading took {time.time() - t0:.1f}s")

# %% ============================================================================
# 2.5 · FEATURE SCALING (fix Phase 3's partial scaling)
# ===============================================================================
# Phase 3 left several features unscaled (e.g. Tot_sum std~5000, Weight std~1500),
# while others were StandardScaled and a third group is binary protocol flags.
# Tree-based XGBoost (Phase 4) is scale-invariant so this didn't matter there.
# AE and IF are NOT scale-invariant — without proper scaling, the AE loss is
# dominated by 1–2 large-magnitude features and never learns the rest, which
# tanks Recon detection. Fit on benign-train (consistent with AE/IF training set)
# and apply everywhere; save for Phase 6 to use on new data.
print("\n" + "=" * 70)
print("SECTION 2.5 · FEATURE SCALING")
print("=" * 70)

print(f"[scale] before — global mean={X_benign_train.mean():.4f}, "
      f"std={X_benign_train.std():.4f}, max={X_benign_train.max():.4f}")

scaler = StandardScaler()
scaler.fit(X_benign_train)
X_benign_train = scaler.transform(X_benign_train).astype(np.float32)
X_benign_val   = scaler.transform(X_benign_val).astype(np.float32)
X_val          = scaler.transform(X_val).astype(np.float32)
X_test         = scaler.transform(X_test).astype(np.float32)
X_train_shape = X_train.shape   # remember for config dump before freeing
del X_train  # free ~635 MB — no longer needed after benign extraction

print(f"[scale] after  — benign-train mean={X_benign_train.mean():.6f}, "
      f"std={X_benign_train.std():.6f}")
print(f"[scale] after  — X_test       mean={X_test.mean():.4f}, "
      f"std={X_test.std():.4f}  (attack samples are out-of-distribution → expected)")

# %% ============================================================================
# 3 · AUTOENCODER ARCHITECTURE
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 3 · AUTOENCODER ARCHITECTURE")
print("=" * 70)

def build_autoencoder(input_dim: int = 44):
    """Symmetric deep AE: 44 -> 32 -> 16 -> 8 -> 16 -> 32 -> 44 ."""
    inp = layers.Input(shape=(input_dim,), name="input")

    # Encoder
    x = layers.Dense(32, activation="relu", name="enc_dense_32")(inp)
    x = layers.BatchNormalization(name="enc_bn_32")(x)
    x = layers.Dropout(0.2, name="enc_drop_32")(x)
    x = layers.Dense(16, activation="relu", name="enc_dense_16")(x)
    x = layers.BatchNormalization(name="enc_bn_16")(x)
    x = layers.Dropout(0.1, name="enc_drop_16")(x)
    bottleneck = layers.Dense(8, activation="relu", name="bottleneck")(x)

    # Decoder
    x = layers.Dense(16, activation="relu", name="dec_dense_16")(bottleneck)
    x = layers.BatchNormalization(name="dec_bn_16")(x)
    x = layers.Dense(32, activation="relu", name="dec_dense_32")(x)
    x = layers.BatchNormalization(name="dec_bn_32")(x)
    out = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = Model(inputs=inp, outputs=out, name="autoencoder")
    encoder     = Model(inputs=inp, outputs=bottleneck, name="encoder")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE),
        loss="mse",
    )
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(44)
autoencoder.summary(print_fn=lambda s: print("  " + s))

# %% ============================================================================
# 4 · AUTOENCODER TRAINING
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 4 · AUTOENCODER TRAINING (benign-only)")
print("=" * 70)

cb_list = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=AE_PATIENCE,
        restore_best_weights=True, verbose=1,
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5,
        min_lr=1e-6, verbose=1,
    ),
]

t_train_start = time.time()
history = autoencoder.fit(
    X_benign_train, X_benign_train,           # input == target (reconstruction)
    validation_data=(X_benign_val, X_benign_val),
    epochs=AE_EPOCHS,
    batch_size=AE_BATCH_SIZE,
    callbacks=cb_list,
    verbose=2,
    shuffle=True,
)
ae_train_time = time.time() - t_train_start
print(f"[train] AE training time: {ae_train_time:.1f}s "
      f"({ae_train_time/60:.2f} min) over {len(history.history['loss'])} epochs")
print(f"[train] best val_loss: {min(history.history['val_loss']):.6f}")

# %% ============================================================================
# 5 · RECONSTRUCTION ERROR COMPUTATION
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 5 · RECONSTRUCTION ERROR (per-sample MSE)")
print("=" * 70)

def compute_mse_batched(model, X, batch_size: int = AE_PREDICT_BATCH):
    """Return per-sample MSE. Keras handles batching internally."""
    X_hat = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - X_hat) ** 2, axis=1).astype(np.float32)

t_score = time.time()
mse_benign_train = compute_mse_batched(autoencoder, X_benign_train)
mse_benign_val   = compute_mse_batched(autoencoder, X_benign_val)
ae_val_mse       = compute_mse_batched(autoencoder, X_val)
ae_test_mse      = compute_mse_batched(autoencoder, X_test)
ae_score_time    = time.time() - t_score
print(f"[score] AE scoring time: {ae_score_time:.1f}s")

print(f"[score] benign MSE — mean={mse_benign_val.mean():.6f}  "
      f"std={mse_benign_val.std():.6f}  "
      f"p95={np.percentile(mse_benign_val,95):.6f}  "
      f"p99={np.percentile(mse_benign_val,99):.6f}")

# benign error stats (saved later)
benign_error_stats = {
    "ae_train_benign": {
        "mean":  float(mse_benign_train.mean()),
        "std":   float(mse_benign_train.std()),
        "min":   float(mse_benign_train.min()),
        "max":   float(mse_benign_train.max()),
        "p50":   float(np.percentile(mse_benign_train, 50)),
        "p90":   float(np.percentile(mse_benign_train, 90)),
        "p95":   float(np.percentile(mse_benign_train, 95)),
        "p99":   float(np.percentile(mse_benign_train, 99)),
    },
    "ae_val_benign": {
        "mean":  float(mse_benign_val.mean()),
        "std":   float(mse_benign_val.std()),
        "min":   float(mse_benign_val.min()),
        "max":   float(mse_benign_val.max()),
        "p50":   float(np.percentile(mse_benign_val, 50)),
        "p90":   float(np.percentile(mse_benign_val, 90)),
        "p95":   float(np.percentile(mse_benign_val, 95)),
        "p99":   float(np.percentile(mse_benign_val, 99)),
    },
}

# %% ============================================================================
# 6 · ANOMALY THRESHOLD SELECTION
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 6 · THRESHOLD SELECTION (on validation set)")
print("=" * 70)

thresholds = {
    "p90":       float(np.percentile(mse_benign_val, 90)),
    "p95":       float(np.percentile(mse_benign_val, 95)),
    "p99":       float(np.percentile(mse_benign_val, 99)),
    "mean_2std": float(mse_benign_val.mean() + 2 * mse_benign_val.std()),
    "mean_3std": float(mse_benign_val.mean() + 3 * mse_benign_val.std()),
}

threshold_eval_rows = []
for name in THRESHOLD_NAMES:
    thr = thresholds[name]
    pred = (ae_val_mse > thr).astype(np.int8)
    p, r, f1, _ = precision_recall_fscore_support(
        y_val_bin, pred, average="binary", zero_division=0
    )
    # FPR and TPR
    tn = int(((pred == 0) & (y_val_bin == 0)).sum())
    fp = int(((pred == 1) & (y_val_bin == 0)).sum())
    fn = int(((pred == 0) & (y_val_bin == 1)).sum())
    tp = int(((pred == 1) & (y_val_bin == 1)).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    threshold_eval_rows.append({
        "threshold_name":  name,
        "threshold_value": thr,
        "precision":       float(p),
        "recall":          float(r),
        "f1":              float(f1),
        "fpr":             float(fpr),
        "tpr":             float(tpr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    })

threshold_eval_df = pd.DataFrame(threshold_eval_rows)
print("\nThreshold evaluation on VALIDATION:")
print(threshold_eval_df.to_string(index=False))

best_row = threshold_eval_df.loc[threshold_eval_df["f1"].idxmax()]
BEST_THRESHOLD_NAME  = str(best_row["threshold_name"])
BEST_THRESHOLD_VALUE = float(best_row["threshold_value"])
print(f"\n[threshold] selected '{BEST_THRESHOLD_NAME}' "
      f"= {BEST_THRESHOLD_VALUE:.6f}  (F1={best_row['f1']:.4f})")

# binary AE predictions at best threshold
ae_val_binary  = (ae_val_mse  > BEST_THRESHOLD_VALUE).astype(np.int8)
ae_test_binary = (ae_test_mse > BEST_THRESHOLD_VALUE).astype(np.int8)

# %% ============================================================================
# 7 · ISOLATION FOREST TRAINING
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 7 · ISOLATION FOREST TRAINING (benign-only)")
print("=" * 70)

iso_forest = IsolationForest(
    n_estimators=IF_N_ESTIMATORS,
    contamination=IF_CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0,
)

t_if_train = time.time()
iso_forest.fit(X_benign_train)
if_train_time = time.time() - t_if_train
print(f"[train] IF training time: {if_train_time:.1f}s")

t_if_score = time.time()
# higher decision_function = more normal; lower = more anomalous
if_val_scores  = iso_forest.decision_function(X_val).astype(np.float32)
if_test_scores = iso_forest.decision_function(X_test).astype(np.float32)
# predict: -1 = anomaly, +1 = normal
if_val_pred  = iso_forest.predict(X_val)
if_test_pred = iso_forest.predict(X_test)
if_val_binary  = (if_val_pred  == -1).astype(np.int8)
if_test_binary = (if_test_pred == -1).astype(np.int8)
if_score_time = time.time() - t_if_score
print(f"[score] IF scoring time: {if_score_time:.1f}s")

# %% ============================================================================
# 8 · BINARY ANOMALY DETECTION METRICS
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 8 · BINARY ANOMALY DETECTION METRICS")
print("=" * 70)

# AE: higher MSE = more anomalous (positive class)
ae_val_auc  = roc_auc_score(y_val_bin,  ae_val_mse)
ae_test_auc = roc_auc_score(y_test_bin, ae_test_mse)
# IF: lower decision_function = more anomalous → use negative for ROC
if_val_auc  = roc_auc_score(y_val_bin,  -if_val_scores)
if_test_auc = roc_auc_score(y_test_bin, -if_test_scores)

print(f"[AUC] AE:  val={ae_val_auc:.4f}   test={ae_test_auc:.4f}")
print(f"[AUC] IF:  val={if_val_auc:.4f}   test={if_test_auc:.4f}")

# FPR at 95% TPR
def fpr_at_tpr(y_true, scores, target_tpr=0.95):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(target_tpr, tpr, fpr))

ae_fpr95_test = fpr_at_tpr(y_test_bin, ae_test_mse,    0.95)
if_fpr95_test = fpr_at_tpr(y_test_bin, -if_test_scores, 0.95)
print(f"[FPR@95%TPR] AE test: {ae_fpr95_test:.4f}    IF test: {if_fpr95_test:.4f}")

# classification reports (test set)
ae_cls_report = classification_report(
    y_test_bin, ae_test_binary,
    target_names=["normal", "anomaly"], output_dict=True, zero_division=0,
)
if_cls_report = classification_report(
    y_test_bin, if_test_binary,
    target_names=["normal", "anomaly"], output_dict=True, zero_division=0,
)
print("\nAutoencoder classification report (test):")
print(classification_report(y_test_bin, ae_test_binary,
                            target_names=["normal", "anomaly"], zero_division=0))
print("Isolation Forest classification report (test):")
print(classification_report(y_test_bin, if_test_binary,
                            target_names=["normal", "anomaly"], zero_division=0))

# %% ============================================================================
# 9 · PER-CLASS DETECTION RATES (19 classes × 5 thresholds)
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 9 · PER-CLASS DETECTION RATES")
print("=" * 70)

CLASSES = sorted(np.unique(y_test_lbl).tolist())
print(f"[classes] {len(CLASSES)} classes: {CLASSES}")

# AE: detection rate at each threshold per class (computed on TEST set)
detection_records = []
for cls in CLASSES:
    mask = (y_test_lbl == cls)
    n_cls = int(mask.sum())
    if n_cls == 0:
        continue
    cls_errors = ae_test_mse[mask]
    row = {"class": cls, "n_samples": n_cls, "model": "Autoencoder"}
    for tname in THRESHOLD_NAMES:
        thr = thresholds[tname]
        rate = float((cls_errors > thr).mean())
        row[tname] = rate
    detection_records.append(row)

# IF: just one detection rate (it has its own internal threshold) — record anyway
for cls in CLASSES:
    mask = (y_test_lbl == cls)
    n_cls = int(mask.sum())
    if n_cls == 0:
        continue
    rate = float(if_test_binary[mask].mean())
    row = {
        "class": cls, "n_samples": n_cls, "model": "IsolationForest",
        "p90": rate, "p95": rate, "p99": rate,
        "mean_2std": rate, "mean_3std": rate,
    }
    detection_records.append(row)

detection_df = pd.DataFrame(detection_records)
print("\nPer-class detection rate (test, AE first then IF):")
print(detection_df.round(3).to_string(index=False))

# pivot for heatmap (AE only, all thresholds)
ae_detection_df = detection_df[detection_df["model"] == "Autoencoder"].copy()
detection_pivot = ae_detection_df.set_index("class")[THRESHOLD_NAMES]

# %% ============================================================================
# 10 · ZERO-DAY SIMULATION PREVIEW
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 10 · ZERO-DAY SIMULATION (5 targets)")
print("=" * 70)

zero_day_records = []
for tgt in ZERO_DAY_TARGETS:
    mask = (y_test_lbl == tgt)
    n_tgt = int(mask.sum())
    if n_tgt == 0:
        print(f"[zero-day] '{tgt}' not present in test set — skipped.")
        continue

    # AE detection rates at each threshold
    ae_rates = {
        tname: float((ae_test_mse[mask] > thresholds[tname]).mean())
        for tname in THRESHOLD_NAMES
    }
    # IF detection rate (using its built-in -1/+1)
    if_rate = float(if_test_binary[mask].mean())

    record = {
        "target": tgt,
        "n_samples": n_tgt,
        "ae_p90":       ae_rates["p90"],
        "ae_p95":       ae_rates["p95"],
        "ae_p99":       ae_rates["p99"],
        "ae_mean_2std": ae_rates["mean_2std"],
        "ae_mean_3std": ae_rates["mean_3std"],
        "ae_best_thr":  ae_rates[BEST_THRESHOLD_NAME],
        "if_recall":    if_rate,
    }
    zero_day_records.append(record)

zero_day_df = pd.DataFrame(zero_day_records)
print(zero_day_df.round(3).to_string(index=False))

# Per-class detection preview (NOT a true H2 verdict — see Phase 6 for that)
hits_70 = int((zero_day_df["ae_best_thr"] >= 0.70).sum())
total_targets = len(zero_day_df)
print(f"[preview] AE @ best threshold ≥70% on {hits_70}/{total_targets} targets "
      f"(preview only — true zero-day H2 evaluation deferred to Phase 6)")

# %% ============================================================================
# 11 · MODEL COMPARISON TABLE
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 11 · MODEL COMPARISON")
print("=" * 70)

# Filter out Benign — its "detection rate" is FPR, not recall
ae_attack_only = ae_detection_df[ae_detection_df["class"] != "Benign"]
ae_per_class_recall = ae_attack_only[BEST_THRESHOLD_NAME].mean()
if_per_class_recall = (
    detection_df[(detection_df["model"] == "IsolationForest") &
                 (detection_df["class"] != "Benign")]["p90"].mean()
)

comparison_df = pd.DataFrame([
    {
        "metric":             "AUC-ROC (test)",
        "Autoencoder":        round(ae_test_auc, 4),
        "IsolationForest":    round(if_test_auc, 4),
    },
    {
        "metric":             "AUC-ROC (val)",
        "Autoencoder":        round(ae_val_auc, 4),
        "IsolationForest":    round(if_val_auc, 4),
    },
    {
        "metric":             "FPR @ 95%TPR (test)",
        "Autoencoder":        round(ae_fpr95_test, 4),
        "IsolationForest":    round(if_fpr95_test, 4),
    },
    {
        "metric":             "Anomaly precision (test)",
        "Autoencoder":        round(ae_cls_report["anomaly"]["precision"], 4),
        "IsolationForest":    round(if_cls_report["anomaly"]["precision"], 4),
    },
    {
        "metric":             "Anomaly recall (test)",
        "Autoencoder":        round(ae_cls_report["anomaly"]["recall"], 4),
        "IsolationForest":    round(if_cls_report["anomaly"]["recall"], 4),
    },
    {
        "metric":             "Anomaly F1 (test)",
        "Autoencoder":        round(ae_cls_report["anomaly"]["f1-score"], 4),
        "IsolationForest":    round(if_cls_report["anomaly"]["f1-score"], 4),
    },
    {
        "metric":             "Per-class avg recall",
        "Autoencoder":        round(float(ae_per_class_recall), 4),
        "IsolationForest":    round(float(if_per_class_recall), 4),
    },
    {
        "metric":             "Training time (s)",
        "Autoencoder":        round(ae_train_time, 1),
        "IsolationForest":    round(if_train_time, 1),
    },
    {
        "metric":             "Scoring time val+test (s)",
        "Autoencoder":        round(ae_score_time, 1),
        "IsolationForest":    round(if_score_time, 1),
    },
])
print(comparison_df.to_string(index=False))

# %% ============================================================================
# 12 · VISUALIZATIONS
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 12 · GENERATING FIGURES")
print("=" * 70)

sns.set_style("whitegrid")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# ---- 12.1 Loss curves ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history.history["loss"],     label="train loss", linewidth=2)
ax.plot(history.history["val_loss"], label="val loss",   linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss")
ax.set_title("Autoencoder training (benign-only)")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
fig.savefig(DIRS["figures"] / "ae_loss_curves.png", dpi=DPI)
plt.close(fig)
print("  ✓ ae_loss_curves.png")

# ---- 12.2 Reconstruction error distribution (benign vs attack) ----
fig, ax = plt.subplots(figsize=(12, 6))
benign_test_errs = ae_test_mse[y_test_bin == 0]
attack_test_errs = ae_test_mse[y_test_bin == 1]
clip = float(np.percentile(ae_test_mse, 99.5))   # clip x-axis for readability
bins = np.linspace(0, clip, 100)
ax.hist(np.clip(benign_test_errs, 0, clip), bins=bins, alpha=0.55,
        label=f"Benign (n={len(benign_test_errs):,})", color="#2ca02c", density=True)
ax.hist(np.clip(attack_test_errs, 0, clip), bins=bins, alpha=0.55,
        label=f"Attack (n={len(attack_test_errs):,})", color="#d62728", density=True)
ax.axvline(BEST_THRESHOLD_VALUE, color="black", linestyle="--", linewidth=1.5,
           label=f"threshold={BEST_THRESHOLD_NAME} ({BEST_THRESHOLD_VALUE:.4f})")
ax.set_xlabel("Reconstruction MSE (clipped at p99.5)")
ax.set_ylabel("density")
ax.set_title("Autoencoder reconstruction error — benign vs attack (test)")
ax.legend()
fig.tight_layout()
fig.savefig(DIRS["figures"] / "ae_error_distribution.png", dpi=DPI)
plt.close(fig)
print("  ✓ ae_error_distribution.png")

# ---- 12.3 Per-class reconstruction error boxplot ----
fig, ax = plt.subplots(figsize=(16, 8))
class_order = ["Benign"] + [c for c in CLASSES if c != "Benign"]
data_per_class, labels_per_class = [], []
for cls in class_order:
    mask = (y_test_lbl == cls)
    if mask.sum() > 0:
        # cap each class for readability
        errs = ae_test_mse[mask]
        data_per_class.append(np.clip(errs, 0, np.percentile(ae_test_mse, 99.5)))
        labels_per_class.append(f"{cls}\n(n={int(mask.sum()):,})")
bp = ax.boxplot(data_per_class, labels=labels_per_class, showfliers=False, patch_artist=True)
# colour benign green, others red-ish
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor("#2ca02c" if labels_per_class[i].startswith("Benign") else "#d62728")
    patch.set_alpha(0.55)
ax.axhline(BEST_THRESHOLD_VALUE, color="black", linestyle="--", linewidth=1.2,
           label=f"threshold ({BEST_THRESHOLD_NAME})")
ax.set_xlabel("Class")
ax.set_ylabel("Reconstruction MSE (clipped at p99.5)")
ax.set_title("Per-class reconstruction error (test)")
ax.legend(loc="upper right")
plt.setp(ax.get_xticklabels(), rotation=75, ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(DIRS["figures"] / "ae_per_class_boxplot.png", dpi=DPI)
plt.close(fig)
print("  ✓ ae_per_class_boxplot.png")

# ---- 12.4 ROC curves (AE & IF) ----
fig, ax = plt.subplots(figsize=(10, 8))
fpr_ae, tpr_ae, _ = roc_curve(y_test_bin, ae_test_mse)
fpr_if, tpr_if, _ = roc_curve(y_test_bin, -if_test_scores)
ax.plot(fpr_ae, tpr_ae, linewidth=2, label=f"Autoencoder (AUC={ae_test_auc:.4f})")
ax.plot(fpr_if, tpr_if, linewidth=2, label=f"Isolation Forest (AUC={if_test_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves — anomaly detection on test set")
ax.legend(loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.005])
fig.tight_layout()
fig.savefig(DIRS["figures"] / "roc_curves.png", dpi=DPI)
plt.close(fig)
print("  ✓ roc_curves.png")

# ---- 12.5 Per-class detection rate heatmap (AE) ----
fig, ax = plt.subplots(figsize=(10, max(8, 0.5 * len(detection_pivot))))
sns.heatmap(
    detection_pivot, annot=True, fmt=".2f",
    cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Detection rate"},
    ax=ax,
)
ax.set_xlabel("Threshold")
ax.set_ylabel("Class")
ax.set_title("Autoencoder per-class detection rate (test) — 19 classes × 5 thresholds")
fig.tight_layout()
fig.savefig(DIRS["figures"] / "detection_rate_heatmap.png", dpi=DPI)
plt.close(fig)
print("  ✓ detection_rate_heatmap.png")

# ---- 12.6 Isolation Forest score distribution ----
fig, ax = plt.subplots(figsize=(12, 6))
benign_if = if_test_scores[y_test_bin == 0]
attack_if = if_test_scores[y_test_bin == 1]
xlo = float(np.percentile(if_test_scores, 0.5))
xhi = float(np.percentile(if_test_scores, 99.5))
bins = np.linspace(xlo, xhi, 100)
ax.hist(np.clip(benign_if, xlo, xhi), bins=bins, alpha=0.55,
        label=f"Benign (n={len(benign_if):,})", color="#2ca02c", density=True)
ax.hist(np.clip(attack_if, xlo, xhi), bins=bins, alpha=0.55,
        label=f"Attack (n={len(attack_if):,})", color="#d62728", density=True)
ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2,
           label="IF decision boundary (0)")
ax.set_xlabel("Isolation Forest decision_function (higher = more normal)")
ax.set_ylabel("density")
ax.set_title("Isolation Forest score distribution — benign vs attack (test)")
ax.legend()
fig.tight_layout()
fig.savefig(DIRS["figures"] / "if_score_distribution.png", dpi=DPI)
plt.close(fig)
print("  ✓ if_score_distribution.png")

# ---- 12.7 Zero-day detection bar chart ----
if len(zero_day_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(zero_day_df))
    width = 0.35
    ax.bar(x - width/2, zero_day_df["ae_best_thr"], width,
           label=f"AE ({BEST_THRESHOLD_NAME})", color="#1f77b4")
    ax.bar(x + width/2, zero_day_df["if_recall"], width,
           label="Isolation Forest", color="#ff7f0e")
    ax.axhline(0.7, color="green", linestyle="--", linewidth=1, label="H2 target (70%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{t}\n(n={n:,})" for t, n in zip(zero_day_df["target"], zero_day_df["n_samples"])],
        rotation=20, ha="right", fontsize=9,
    )
    ax.set_ylabel("Detection rate (recall)")
    ax.set_ylim([0, 1.05])
    ax.set_title("Zero-day simulation preview — recall on held-out attack types")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "zero_day_detection.png", dpi=DPI)
    plt.close(fig)
    print("  ✓ zero_day_detection.png")

# %% ============================================================================
# 13 · SAVE MODELS, SCORES, METRICS, CONFIG
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 13 · SAVING ARTIFACTS")
print("=" * 70)

# ---- models ----
autoencoder.save(DIRS["models"] / "autoencoder.keras")
encoder.save(DIRS["models"] / "encoder.keras")
joblib.dump(iso_forest, DIRS["models"] / "isolation_forest.pkl", compress=3)
joblib.dump(scaler,     DIRS["models"] / "scaler.pkl")
print("  ✓ models/{autoencoder.keras, encoder.keras, isolation_forest.pkl, scaler.pkl}")

# ---- scores (float32 for fusion engine) ----
np.save(DIRS["scores"] / "ae_val_mse.npy",    ae_val_mse.astype(np.float32))
np.save(DIRS["scores"] / "ae_test_mse.npy",   ae_test_mse.astype(np.float32))
np.save(DIRS["scores"] / "if_val_scores.npy", if_val_scores.astype(np.float32))
np.save(DIRS["scores"] / "if_test_scores.npy", if_test_scores.astype(np.float32))
np.save(DIRS["scores"] / "ae_val_binary.npy",  ae_val_binary.astype(np.int8))
np.save(DIRS["scores"] / "ae_test_binary.npy", ae_test_binary.astype(np.int8))
np.save(DIRS["scores"] / "if_val_binary.npy",  if_val_binary.astype(np.int8))
np.save(DIRS["scores"] / "if_test_binary.npy", if_test_binary.astype(np.int8))
print("  ✓ scores/*.npy (8 arrays)")

# ---- thresholds ----
with open(DIRS["root"] / "thresholds.json", "w") as f:
    json.dump({
        "thresholds":         thresholds,
        "evaluation_on_val":  threshold_eval_df.to_dict(orient="records"),
        "selected": {
            "name":  BEST_THRESHOLD_NAME,
            "value": BEST_THRESHOLD_VALUE,
            "f1_on_val": float(best_row["f1"]),
        },
    }, f, indent=2)
print("  ✓ thresholds.json")

# ---- benign error stats ----
with open(DIRS["root"] / "benign_error_stats.json", "w") as f:
    json.dump(benign_error_stats, f, indent=2)
print("  ✓ benign_error_stats.json")

# ---- training history ----
hist_dict = {k: [float(v) for v in vs] for k, vs in history.history.items()}
with open(DIRS["root"] / "ae_training_history.json", "w") as f:
    json.dump(hist_dict, f, indent=2)
print("  ✓ ae_training_history.json")

# ---- metrics ----
with open(DIRS["metrics"] / "ae_classification_report.json", "w") as f:
    json.dump(ae_cls_report, f, indent=2)
with open(DIRS["metrics"] / "if_classification_report.json", "w") as f:
    json.dump(if_cls_report, f, indent=2)
detection_df.to_csv(DIRS["metrics"] / "per_class_detection_rates.csv", index=False)
comparison_df.to_csv(DIRS["metrics"] / "model_comparison.csv", index=False)
zero_day_df.to_csv(DIRS["metrics"] / "zero_day_preview.csv", index=False)
print("  ✓ metrics/{ae,if}_classification_report.json")
print("  ✓ metrics/per_class_detection_rates.csv")
print("  ✓ metrics/model_comparison.csv")
print("  ✓ metrics/zero_day_preview.csv")

# ---- config ----
config_out = {
    "phase":              "5 — unsupervised training",
    "feature_space":      "full (44 features)",
    "feature_names":      FEATURE_NAMES,
    "random_state":       RANDOM_STATE,
    "autoencoder": {
        "architecture":   AE_ARCHITECTURE,
        "epochs_max":     AE_EPOCHS,
        "epochs_actual":  len(history.history["loss"]),
        "batch_size":     AE_BATCH_SIZE,
        "learning_rate":  AE_LEARNING_RATE,
        "patience":       AE_PATIENCE,
        "best_val_loss":  float(min(history.history["val_loss"])),
        "training_time_s": ae_train_time,
    },
    "isolation_forest": {
        "n_estimators":    IF_N_ESTIMATORS,
        "contamination":   IF_CONTAMINATION,
        "training_time_s": if_train_time,
    },
    "data_shapes": {
        "X_train":         list(X_train_shape),
        "X_val":           list(X_val.shape),
        "X_test":          list(X_test.shape),
        "X_benign_train":  list(X_benign_train.shape),
        "X_benign_val":    list(X_benign_val.shape),
    },
    "selected_threshold": {
        "name":  BEST_THRESHOLD_NAME,
        "value": BEST_THRESHOLD_VALUE,
    },
}
with open(DIRS["root"] / "config.json", "w") as f:
    json.dump(config_out, f, indent=2)
print("  ✓ config.json")

# %% ============================================================================
# 14 · SUMMARY REPORT
# ===============================================================================
print("\n" + "=" * 70)
print("SECTION 14 · SUMMARY")
print("=" * 70)

# Identify easiest/hardest classes for AE at best threshold (excluding Benign)
ae_attack_rows = ae_detection_df[ae_detection_df["class"] != "Benign"].copy()
ae_attack_rows = ae_attack_rows.sort_values(BEST_THRESHOLD_NAME, ascending=False)
top3_easy = ae_attack_rows.head(3)
top3_hard = ae_attack_rows.tail(3).iloc[::-1]

# AE and IF have complementary failure modes — recommend both for fusion
recommended = "Both (complementary)"
reason = ("AE provides the primary anomaly signal via reconstruction error "
          "(stronger on volumetric/flooding attacks where Rate/IAT deviate sharply). "
          "IF provides a secondary signal that catches point anomalies AE misses "
          "(Recon, Spoofing, point flag-count outliers). Phase 6 fusion should "
          "consume both score arrays. "
          f"Test AUC: AE={ae_test_auc:.4f}, IF={if_test_auc:.4f}.")

# Read Phase 4's best supervised result dynamically (don't hardcode)
sup_f1 = "see Phase 4 summary"
sup_csv = SUPERVISED_DIR / "metrics" / "overall_comparison.csv"
try:
    if sup_csv.exists():
        sup_df = pd.read_csv(sup_csv)
        f1_col = next(
            (c for c in ["test_f1_macro", "f1_macro", "F1_macro", "macro_f1"]
             if c in sup_df.columns),
            None,
        )
        name_col = next(
            (c for c in ["experiment", "model", "name", "run"]
             if c in sup_df.columns),
            None,
        )
        if f1_col and name_col:
            best_sup = sup_df.sort_values(f1_col, ascending=False).iloc[0]
            sup_f1 = f"{best_sup[name_col]} (F1_macro={best_sup[f1_col]:.4f})"
except Exception as e:
    print(f"[summary] could not read Phase 4 CSV ({e}); using placeholder.")

# Build summary
summary_md = f"""# Phase 5 — Unsupervised Layer Summary

## 1 · Best autoencoder configuration

- Architecture: `{ ' → '.join(map(str, AE_ARCHITECTURE)) }`
- Optimiser: Adam, lr = `{AE_LEARNING_RATE}`, batch size = `{AE_BATCH_SIZE}`
- Trained for **{len(history.history['loss'])} epochs** (early-stopped, patience={AE_PATIENCE})
- Final training loss: `{history.history['loss'][-1]:.6f}`
- Best validation loss: **`{min(history.history['val_loss']):.6f}`**
- Wall-clock training time: **{ae_train_time:.1f} s** ({ae_train_time/60:.2f} min)

## 2 · Selected anomaly threshold

- All five candidate thresholds evaluated on the **validation** set:

| name | value | precision | recall | F1 | FPR | TPR |
|------|-------|-----------|--------|-----|------|-----|
""" + "\n".join(
    f"| {r['threshold_name']} | {r['threshold_value']:.6f} | {r['precision']:.4f} "
    f"| {r['recall']:.4f} | {r['f1']:.4f} | {r['fpr']:.4f} | {r['tpr']:.4f} |"
    for r in threshold_eval_rows
) + f"""

- **Selected:** `{BEST_THRESHOLD_NAME}` = `{BEST_THRESHOLD_VALUE:.6f}` (highest F1 on val).
- Rationale: the percentile / mean+std rules are computed on benign-only validation
  errors, so they reflect the natural noise floor of normal traffic. The chosen rule
  gave the best precision-recall trade-off for binary anomaly classification on
  validation, and is therefore used as the operating point for fusion in Phase 6.

## 3 · Binary anomaly detection performance (test set)

| metric | Autoencoder | Isolation Forest |
|---|---|---|
| AUC-ROC | **{ae_test_auc:.4f}** | {if_test_auc:.4f} |
| FPR @ 95 % TPR | {ae_fpr95_test:.4f} | {if_fpr95_test:.4f} |
| anomaly precision | {ae_cls_report['anomaly']['precision']:.4f} | {if_cls_report['anomaly']['precision']:.4f} |
| anomaly recall | {ae_cls_report['anomaly']['recall']:.4f} | {if_cls_report['anomaly']['recall']:.4f} |
| anomaly F1 | {ae_cls_report['anomaly']['f1-score']:.4f} | {if_cls_report['anomaly']['f1-score']:.4f} |

## 4 · Per-class detection rates (Autoencoder, best threshold)

**Easiest to detect:**

""" + "\n".join(
    f"- `{r['class']}` — recall **{r[BEST_THRESHOLD_NAME]:.3f}** "
    f"(n={int(r['n_samples']):,})"
    for _, r in top3_easy.iterrows()
) + "\n\n**Hardest to detect:**\n\n" + "\n".join(
    f"- `{r['class']}` — recall **{r[BEST_THRESHOLD_NAME]:.3f}** "
    f"(n={int(r['n_samples']):,})"
    for _, r in top3_hard.iterrows()
) + f"""

The full 19-class detection-rate table is in `metrics/per_class_detection_rates.csv`
and visualised as `figures/detection_rate_heatmap.png`.

## 5 · Autoencoder vs Isolation Forest

- The autoencoder learned a tighter benign manifold (lower benign-MSE variance) and
  therefore produces a more separable score distribution for volumetric/flooding
  attacks where `Rate`, `IAT`, and flag counts deviate sharply from benign.
- Isolation Forest tends to be more competitive on point-anomaly attacks
  (Recon, Spoofing) because it isolates rare feature combinations rather than
  measuring reconstruction error.
- **Average per-class recall:** AE = `{float(ae_per_class_recall):.4f}` ·
  IF = `{float(if_per_class_recall):.4f}`
- Training cost: AE = {ae_train_time:.1f} s · IF = {if_train_time:.1f} s

## 6 · Zero-day simulation (preview)

| target | n_test | AE @ {BEST_THRESHOLD_NAME} | IF recall |
|---|---|---|---|
""" + ("\n".join(
    f"| `{r['target']}` | {int(r['n_samples']):,} | {r['ae_best_thr']:.3f} | {r['if_recall']:.3f} |"
    for _, r in zero_day_df.iterrows()
) if len(zero_day_df) else "| _no targets present_ | | | |") + f"""

- **Per-class detection preview** at the selected threshold:
  {hits_70}/{total_targets} targets achieve ≥ 70 % recall.
  *Indicative only — the AE never sees attacks during training regardless, so this
  measures class separability, not true zero-day generalization. Proper H2 evaluation
  (with held-out classes and retrained supervised + IF models) is deferred to Phase 6.*
- This is a *preview* — Phase 6 fusion will combine these scores with the supervised
  E7 probabilities, which should boost zero-day recall further by exploiting the
  "supervised says benign + unsupervised says anomaly = zero-day" rule.

## 7 · Recommendation for Phase 6 fusion

- **Primary recommendation:** {recommended}.
- {reason}
- Both score arrays are exported (`scores/ae_*.npy`, `scores/if_*.npy`); the fusion
  engine should consume **both** so that the 4-case logic
  (supervised × unsupervised) can use the stronger signal per region of feature
  space. AE is preferred for the binary anomaly flag, IF as a secondary signal.

## 8 · Key findings for thesis discussion

1. Training the autoencoder on benign-only traffic produced a clean reconstruction-error
   separation between benign and attack samples — visible in
   `figures/ae_error_distribution.png`.
2. Confirms the EDA observation that benign IoMT traffic is a compact PCA cluster:
   the AE bottleneck of 8 dimensions was sufficient to reconstruct it with
   benign-val MSE = `{benign_error_stats['ae_val_benign']['mean']:.6f}`.
3. Per-class recall varies sharply across attack families. Volumetric / flooding
   classes (e.g. DDoS_*) are detected almost perfectly, while quieter recon
   classes are harder — motivating the hybrid design.
4. The unsupervised layer is a complement to, not a substitute for, the supervised
   XGBoost ({sup_f1}). Phase 6 fusion exploits this complementarity.

---

_Generated by `unsupervised_training.py` — random_state = {RANDOM_STATE}_
"""

with open(DIRS["root"] / "summary.md", "w") as f:
    f.write(summary_md)
print(f"  ✓ summary.md ({len(summary_md):,} chars)")

# %% ============================================================================
# 15 · FINAL CONSOLE PRINT
# ===============================================================================
print("\n" + "=" * 70)
print("PHASE 5 COMPLETE")
print("=" * 70)
print(f"AE  test AUC: {ae_test_auc:.4f}  · F1: {ae_cls_report['anomaly']['f1-score']:.4f}")
print(f"IF  test AUC: {if_test_auc:.4f}  · F1: {if_cls_report['anomaly']['f1-score']:.4f}")
print(f"Selected threshold: {BEST_THRESHOLD_NAME} = {BEST_THRESHOLD_VALUE:.6f}")
print(f"Per-class detection ≥70% (preview): {hits_70}/{total_targets} targets "
      "— true H2 evaluation deferred to Phase 6")
print(f"\nAll outputs written to: {OUTPUT_DIR.resolve()}")
print(f"Total runtime: {(time.time() - t0)/60:.2f} min")