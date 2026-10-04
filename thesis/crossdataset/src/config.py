"""
Central configuration for the ShieldMind cross-dataset generalization test.

Train: CICIoMT2024 (frozen thesis dataset).  Test: CICIoT2023 (63 merged CSVs).

Every path, constant and mapping lives here so the run scripts contain no
hard-coded values (project coding-style rule).  Nothing in this module mutates
state; callers treat the exported structures as read-only.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (confirmed with the user before the run)
# Repo-internal paths derive from this file's location so a repo move can
# never go stale again; CICIoT2023 lives outside the repo and stays absolute.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[3]

CICIOMT2024_TRAIN_DIR = _REPO_ROOT / "data" / "train"
CICIOMT2024_TEST_DIR = _REPO_ROOT / "data" / "test"
CICIOT2023_MERGED_DIR = Path("/Users/amoorabaseet/Downloads/Iot/MERGED_CSV")

OUT_DIR = _REPO_ROOT / "thesis" / "crossdataset"
ARTIFACT_DIR = OUT_DIR / "artifacts"          # models, scalers, cached counts

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
TRAIN_SEED = 42                # model training is deterministic on this seed
SAMPLING_SEEDS = (42, 43, 44)  # TRAP 4 — 3 independent CICIoT2023 samples

# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #
# The six the brief lists as absent in CICIoT2023 ...
BRIEF_ABSENT_IN_CICIOT2023 = (
    "Srate", "Drate", "Magnitue", "Radius", "Covariance", "Weight",
)
# ... plus 'Duration', which the brief missed but the headers prove is also
# absent in CICIoT2023.  Reported honestly in results_summary.md, never
# reconstructed (same rule as the other six).
EXTRA_ABSENT_IN_CICIOT2023 = ("Duration",)

# CICIoT2023 carries one column the CICIoMT2024 model never used.
CICIOT2023_ONLY = ("Time_To_Live",)

# The label column names differ between the two datasets' CSVs.
CICIOT2023_LABEL_COL = "Label"

# --------------------------------------------------------------------------- #
# Sampling protocol (design doc §6) — target ~400k flows
# --------------------------------------------------------------------------- #
BENIGN_CAP = 40_000     # keep a large benign pool for a stable FPR estimate
ATTACK_CAP = 20_000     # cap each dominant flooding class; rares kept whole
SAMPLE_CHUNK_ROWS = 250_000

# --------------------------------------------------------------------------- #
# XGBoost hyperparameters — copied verbatim from the frozen thesis pipeline
# (notebooks/supervised_training.py XGB_PARAMS_BASE) so the shared-feature
# model differs from E7 ONLY in its feature set, nothing else.
# --------------------------------------------------------------------------- #
XGB_PARAMS_BASE = dict(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    tree_method="hist",
    random_state=TRAIN_SEED,
    n_jobs=-1,
    verbosity=0,
)

# --------------------------------------------------------------------------- #
# Flow autoencoder hyperparameters — verbatim from the frozen thesis pipeline
# (notebooks/unsupervised_training.py).  Architecture: 38→32→16→8→16→32→38.
# --------------------------------------------------------------------------- #
AE_EPOCHS = 100
AE_BATCH_SIZE = 512
AE_LEARNING_RATE = 1e-3
AE_PREDICT_BATCH = 8192
AE_PATIENCE = 12
# Thresholds are percentiles of the CICIoMT2024 benign-val reconstruction
# error (never the test set).  p90 = F1-optimal, p99 = low-FPR (fusion).
AE_THRESHOLD_PERCENTILES = (90, 99)

# --------------------------------------------------------------------------- #
# In-dataset baselines from the frozen thesis README (44-feature models).
# Used only for the paired "frozen thesis reference" column in the report.
# The matched in-dataset baseline (shared-feature model on CICIoMT2024 test)
# is computed at run time; these are the headline numbers to compare against.
# --------------------------------------------------------------------------- #
README_BASELINE = {
    "design_a_xgb_binary_f1_macro": 0.9880,      # §12.4 best binary (E5 RF)
    "design_a_ae_binary_f1": 0.9853,             # §13.4 AE anomaly F1
    "design_a_ae_binary_auc": 0.9892,            # §13.4 AE AUC-ROC
    "design_b_category_macro_f1_6class": 0.9363,  # §12.4 E7 6-class (incl. MQTT)
}
