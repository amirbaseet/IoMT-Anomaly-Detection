"""
================================================================================
preprocessing_pipeline.py
--------------------------------------------------------------------------------
Phase 3 — Preprocessing & Feature Engineering for the
Hybrid Supervised-Unsupervised Anomaly Detection Framework
on the CICIoMT2024 dataset.

Author : Amro (M.Sc., Sakarya University)
Stage  : Phase 3 (Preprocessing) — input from Phase 2 EDA, output to Phases 4-6
Inputs : eda_output/train_cleaned.csv, eda_output/test_cleaned.csv
Outputs: ./preprocessed/ (config, scalers, encoded labels, scaled features,
         SMOTETomek-resampled training set, benign-only autoencoder set,
         five leave-one-attack-out zero-day scenarios)

Run:
    pip install pandas numpy scikit-learn imbalanced-learn joblib
    python preprocessing_pipeline.py

Notes on design choices (justified against the EDA findings):
    * float32 everywhere   — 4.5M rows × 44 cols ≈ 760 MB (vs ~1.5 GB float64)
    * RobustScaler         — heavy-tailed flow features (Rate spans 0.5–101k)
    * StandardScaler       — TCP-flag ratios already roughly bounded in [0, 1]
    * MinMaxScaler         — protocol indicators are already binary-ish
    * SMOTETomek with fallback — full 3.6M-row oversampling is borderline on
      24 GB; we ship a `targeted` sampling_strategy that only lifts classes
      below a threshold so the pipeline finishes deterministically.
    * Zero-day datasets    — built from the *un-resampled* train set, because
      the unsupervised layer should be evaluated on real flows only.
================================================================================
"""

# %% ===========================================================================
# SECTION 0 — Imports & configuration
# ==============================================================================
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----- paths & determinism -----
TRAIN_INPUT = Path("./eda_output/train_cleaned.csv")
TEST_INPUT = Path("./eda_output/test_cleaned.csv")
OUTPUT_DIR = Path("./preprocessed")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----- expected shapes (sanity guards from Phase 2 EDA) -----
EXPECTED_TRAIN_ROWS = 4_515_080
EXPECTED_TEST_ROWS = 892_268
SHAPE_TOLERANCE = 0.01  # ±1 % accommodates re-runs of EDA dedup

# ----- SMOTETomek strategy -----
# If the full-population SMOTETomek runs out of memory, we fall back to lifting
# only the classes below this threshold up to it. 50 000 rows is the largest
# round number that keeps every minority class >1 % of the resampled training
# pool while leaving the majority classes untouched.
SMOTE_TARGETED_THRESHOLD = 50_000
SMOTE_K_NEIGHBORS = 5

# ==============================================================================
# Feature lists (verified against Phase 2 EDA — see project README §10.3, 10.6)
# ==============================================================================
ALL_45_FEATURES = [
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT",
    "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]

# Variant A — drop only Drate (constant at 0.0 per EDA): 44 features
FEATURES_FULL = [c for c in ALL_45_FEATURES if c != "Drate"]

# Variant B — drop Drate + 11 redundant + 5 noise: 28 features (README §10.6)
FEATURES_REDUCED = [
    "Header_Length", "Protocol Type", "Duration", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "TCP", "DHCP", "ARP", "ICMP",
    "Tot sum", "Min", "Max", "IAT", "Covariance", "Variance",
]

# ----- scaler groupings (heavy-tailed → Robust, flag-ratios → Standard,
# binary indicators → MinMax). Derived from EDA distribution checks.
HEAVY_TAILED = [
    "IAT", "Rate", "Header_Length", "Tot sum", "Min", "Max",
    "Covariance", "Variance", "Duration",
    "ack_count", "syn_count", "fin_count", "rst_count",
]
FLAG_FEATURES = [
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
]
BINARY_FEATURES = [
    "HTTP", "HTTPS", "DNS", "TCP", "DHCP", "ARP", "ICMP", "Protocol Type",
]
# Extra columns that exist in the FULL variant only — also placed into a
# scaler bucket so nothing is left passthrough by accident.
HEAVY_TAILED_FULL_EXTRA = [
    "Srate", "AVG", "Std", "Tot size", "Number", "Magnitue", "Radius", "Weight",
]
BINARY_FULL_EXTRA = [
    "UDP", "IPv", "LLC", "Telnet", "SMTP", "SSH", "IRC", "IGMP",
]


# ==============================================================================
# Logging helper
# ==============================================================================
def log(msg: str) -> None:
    """Timestamped progress logger so long sections show their pulse."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


# %% ===========================================================================
# SECTION 1 — Loading
# ==============================================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two cleaned CSVs with float32 dtypes for memory efficiency."""
    log("SECTION 1 — Loading cleaned CSVs")

    if not TRAIN_INPUT.exists() or not TEST_INPUT.exists():
        sys.exit(
            f"❌ Input not found.\n"
            f"   Expected: {TRAIN_INPUT}\n             {TEST_INPUT}\n"
            f"   Run Phase 2 EDA first to produce these files."
        )

    # All 45 features → float32. The 3 metadata columns are strings.
    feat_dtypes = {c: np.float32 for c in ALL_45_FEATURES}
    str_dtypes = {"label": "string", "category": "string", "split": "string"}
    dtypes = {**feat_dtypes, **str_dtypes}

    log(f"  reading {TRAIN_INPUT}")
    train = pd.read_csv(TRAIN_INPUT, dtype=dtypes)
    log(f"    train: {train.shape[0]:,} rows × {train.shape[1]} cols "
        f"({memory_mb(train):.1f} MB)")

    log(f"  reading {TEST_INPUT}")
    test = pd.read_csv(TEST_INPUT, dtype=dtypes)
    log(f"    test:  {test.shape[0]:,} rows × {test.shape[1]} cols "
        f"({memory_mb(test):.1f} MB)")

    # Sanity check shapes (Phase 2 produced these exact counts)
    def _within(actual: int, expected: int) -> bool:
        return abs(actual - expected) / expected <= SHAPE_TOLERANCE

    if not _within(len(train), EXPECTED_TRAIN_ROWS):
        log(f"  ⚠ train row count {len(train):,} differs from expected "
            f"{EXPECTED_TRAIN_ROWS:,} by >{SHAPE_TOLERANCE*100:.0f}%")
    if not _within(len(test), EXPECTED_TEST_ROWS):
        log(f"  ⚠ test row count {len(test):,} differs from expected "
            f"{EXPECTED_TEST_ROWS:,} by >{SHAPE_TOLERANCE*100:.0f}%")

    # Quick health check
    for name, df in [("train", train), ("test", test)]:
        n_nan = df[ALL_45_FEATURES].isna().sum().sum()
        n_inf = np.isinf(df[ALL_45_FEATURES].to_numpy()).sum()
        log(f"    {name} health: NaN={n_nan}, inf={n_inf}")

    return train, test


# %% ===========================================================================
# SECTION 2 — Label encoding (binary, 6-class category, 19-class multiclass)
# ==============================================================================
def encode_labels(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Build three label encodings and return a dict containing:
      y_train, y_test (DataFrames with binary_label, category_label, multiclass_label)
      mappings        (str→int dicts for each scheme, for later decoding)
      encoders        (fitted LabelEncoder objects)
    """
    log("SECTION 2 — Label encoding (binary / 6-cat / multiclass)")

    # --- multiclass: fit on train so all 19 classes are seen ------------------
    le_multi = LabelEncoder()
    train_multi = le_multi.fit_transform(train["label"].astype(str))
    test_multi = le_multi.transform(test["label"].astype(str))
    multi_map = {cls: int(i) for i, cls in enumerate(le_multi.classes_)}
    log(f"  multiclass: {len(multi_map)} classes — "
        f"{sorted(multi_map.keys())[:3]}…{sorted(multi_map.keys())[-3:]}")

    # --- 6-class category -----------------------------------------------------
    le_cat = LabelEncoder()
    train_cat = le_cat.fit_transform(train["category"].astype(str))
    test_cat = le_cat.transform(test["category"].astype(str))
    cat_map = {cls: int(i) for i, cls in enumerate(le_cat.classes_)}
    log(f"  category:   {len(cat_map)} classes — {list(cat_map.keys())}")

    # --- binary: Benign=0, anything else=1 ------------------------------------
    train_bin = (train["label"].astype(str) != "Benign").astype(np.int8).to_numpy()
    test_bin = (test["label"].astype(str) != "Benign").astype(np.int8).to_numpy()
    bin_map = {"Benign": 0, "Attack": 1}
    log(f"  binary:     train attack rate = {train_bin.mean():.4f}, "
        f"test attack rate = {test_bin.mean():.4f}")

    y_train = pd.DataFrame({
        "binary_label": train_bin,
        "category_label": train_cat.astype(np.int16),
        "multiclass_label": train_multi.astype(np.int16),
        "label": train["label"].to_numpy(),
        "category": train["category"].to_numpy(),
    })
    y_test = pd.DataFrame({
        "binary_label": test_bin,
        "category_label": test_cat.astype(np.int16),
        "multiclass_label": test_multi.astype(np.int16),
        "label": test["label"].to_numpy(),
        "category": test["category"].to_numpy(),
    })

    return {
        "y_train": y_train,
        "y_test": y_test,
        "mappings": {
            "binary": bin_map,
            "category": cat_map,
            "multiclass": multi_map,
        },
        "encoders": {"category": le_cat, "multiclass": le_multi},
    }


# %% ===========================================================================
# SECTION 3 — Feature selection (two variants)
# ==============================================================================
def select_features(
    train: pd.DataFrame, test: pd.DataFrame, feature_list: list[str], variant: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice both DataFrames to the requested feature subset, with sanity checks."""
    log(f"SECTION 3 — Feature selection (variant={variant}, n={len(feature_list)})")
    missing = [c for c in feature_list if c not in train.columns]
    if missing:
        sys.exit(f"❌ Missing features in train: {missing}")

    Xtr = train[feature_list].copy()
    Xte = test[feature_list].copy()

    # Replace any sneaky inf with NaN, then NaN with 0.0 (Phase 2 confirmed
    # zero true NaN/inf, but we defend in depth).
    for X in (Xtr, Xte):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X.isna().any().any():
            X.fillna(0.0, inplace=True)

    log(f"  {variant}: X_train {Xtr.shape}, X_test {Xte.shape}")
    return Xtr, Xte


# %% ===========================================================================
# SECTION 4 — Scaling pipeline (ColumnTransformer)
# ==============================================================================
def build_scaler(feature_list: list[str], variant: str) -> ColumnTransformer:
    """
    Build a ColumnTransformer that routes each feature into the appropriate
    scaler. Any feature not explicitly assigned is passed through unchanged
    (this should not happen — the assertion below proves coverage).
    """
    heavy = [c for c in HEAVY_TAILED if c in feature_list]
    flags = [c for c in FLAG_FEATURES if c in feature_list]
    binary = [c for c in BINARY_FEATURES if c in feature_list]

    if variant == "full":
        heavy += [c for c in HEAVY_TAILED_FULL_EXTRA if c in feature_list]
        binary += [c for c in BINARY_FULL_EXTRA if c in feature_list]

    assigned = set(heavy) | set(flags) | set(binary)
    leftover = [c for c in feature_list if c not in assigned]
    if leftover:
        log(f"  ⚠ {len(leftover)} unassigned features → passthrough: {leftover}")

    log(f"  scaler groups [{variant}]: "
        f"robust={len(heavy)}, standard={len(flags)}, minmax={len(binary)}, "
        f"passthrough={len(leftover)}")

    return ColumnTransformer(
        transformers=[
            ("robust", RobustScaler(), heavy),
            ("standard", StandardScaler(), flags),
            ("minmax", MinMaxScaler(), binary),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def fit_scale(
    X_train: pd.DataFrame, X_test: pd.DataFrame, variant: str, feature_list: list[str]
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """Fit the ColumnTransformer on TRAIN ONLY, transform both."""
    log(f"SECTION 4 — Scaling [{variant}]")
    pre = build_scaler(feature_list, variant)
    log("  fitting on train…")
    Xtr = pre.fit_transform(X_train).astype(np.float32, copy=False)
    log("  transforming test…")
    Xte = pre.transform(X_test).astype(np.float32, copy=False)
    log(f"  done: X_train {Xtr.shape}, X_test {Xte.shape}")
    return Xtr, Xte, pre


# %% ===========================================================================
# SECTION 5 — Stratified 80/20 train/validation split
# ==============================================================================
def split_train_val(
    X: np.ndarray, y: pd.DataFrame, variant: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    log(f"SECTION 5 — Train/val split [{variant}] (stratified on multiclass)")
    Xtr, Xva, ytr, yva = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y["multiclass_label"],
    )
    log(f"  train: {Xtr.shape[0]:,} rows | val: {Xva.shape[0]:,} rows")

    # Show a couple of minority counts as a smell test
    rare = ["Recon_Ping_Sweep", "Recon_VulScan", "ARP_Spoofing"]
    for cls in rare:
        n_tr = int((ytr["label"] == cls).sum())
        n_va = int((yva["label"] == cls).sum())
        log(f"    {cls}: train={n_tr}, val={n_va}")
    return Xtr, Xva, ytr.reset_index(drop=True), yva.reset_index(drop=True)


# %% ===========================================================================
# SECTION 6 — SMOTETomek resampling (training split only)
# ==============================================================================
def apply_smote_tomek(
    X_train: np.ndarray, y_train: pd.DataFrame, multi_map: dict
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Apply SMOTETomek to the training split only. We ALWAYS use the targeted
    fallback (oversample minorities to SMOTE_TARGETED_THRESHOLD only) because
    full-population SMOTETomek on ~3.6M × 28 features is prohibitively slow on
    a 24 GB workstation. The targeted strategy gives the same downstream
    benefit (better minority-class recall) without the runtime explosion.
    """
    log(f"SECTION 6 — SMOTETomek resampling on training split only")
    y_multi = y_train["multiclass_label"].to_numpy()

    # Build a per-class target dict: classes already above the threshold are
    # left untouched; classes below it are lifted up to the threshold.
    counts = pd.Series(y_multi).value_counts().to_dict()
    sampling_strategy = {
        int(cls_id): max(int(n), SMOTE_TARGETED_THRESHOLD)
        for cls_id, n in counts.items()
        if n < SMOTE_TARGETED_THRESHOLD
    }
    if not sampling_strategy:
        log("  no minority classes below threshold — skipping SMOTETomek")
        return X_train.copy(), y_train.copy()

    # k_neighbors must be < smallest class size; ensure it's safe.
    smallest = min(counts.values())
    k = min(SMOTE_K_NEIGHBORS, max(1, smallest - 1))
    log(f"  targets (cls_id → target_count): "
        + ", ".join(f"{c}→{n:,}" for c, n in sampling_strategy.items()))
    log(f"  k_neighbors = {k} (smallest minority = {smallest})")

    inv_multi = {v: k for k, v in multi_map.items()}
    log("  before:")
    for cls_id, n in sorted(counts.items(), key=lambda x: x[1]):
        if cls_id in sampling_strategy:
            log(f"    {inv_multi[cls_id]:30s} {n:>8,}")

    # Note: newer imbalanced-learn dropped `n_jobs` from SMOTE.__init__ but
    # SMOTETomek itself still accepts it; we set it where supported.
    def _build_smote_tomek(strategy: dict) -> SMOTETomek:
        smote = SMOTE(
            sampling_strategy=strategy,
            k_neighbors=k,
            random_state=RANDOM_STATE,
        )
        try:
            return SMOTETomek(smote=smote, random_state=RANDOM_STATE, n_jobs=-1)
        except TypeError:
            return SMOTETomek(smote=smote, random_state=RANDOM_STATE)

    smt = _build_smote_tomek(sampling_strategy)
    try:
        t0 = time.time()
        X_res, y_res = smt.fit_resample(X_train, y_multi)
        dt = time.time() - t0
        log(f"  ✓ SMOTETomek finished in {dt:.1f}s — {X_res.shape[0]:,} rows")
    except MemoryError:
        log("  ✗ MemoryError. Re-attempting with smaller threshold (25 000)…")
        sampling_strategy = {
            int(cls_id): max(int(n), 25_000)
            for cls_id, n in counts.items()
            if n < 25_000
        }
        smt = _build_smote_tomek(sampling_strategy)
        X_res, y_res = smt.fit_resample(X_train, y_multi)
        log(f"  ✓ retry succeeded — {X_res.shape[0]:,} rows")

    X_res = X_res.astype(np.float32, copy=False)

    # Re-derive binary / category labels deterministically from multiclass.
    label_strs = np.array([inv_multi[int(c)] for c in y_res])
    binary = (label_strs != "Benign").astype(np.int8)
    # Reconstruct category from label by lookup against y_train
    cat_lookup = (
        y_train.drop_duplicates("label").set_index("label")["category"].to_dict()
    )
    category = np.array([cat_lookup[lbl] for lbl in label_strs])
    cat_le = LabelEncoder().fit(y_train["category"])
    category_id = cat_le.transform(category).astype(np.int16)

    y_res_df = pd.DataFrame({
        "binary_label": binary,
        "category_label": category_id,
        "multiclass_label": y_res.astype(np.int16),
        "label": label_strs,
        "category": category,
    })

    log("  after:")
    new_counts = y_res_df["label"].value_counts()
    for cls_id, _ in sorted(counts.items(), key=lambda x: x[1])[:8]:
        cls = inv_multi[cls_id]
        log(f"    {cls:30s} {int(new_counts.get(cls, 0)):>8,}")

    return X_res, y_res_df


# %% ===========================================================================
# SECTION 7 — Benign-only dataset for the Autoencoder
# ==============================================================================
def build_autoencoder_set(
    X_full_train: np.ndarray, y_full_train: pd.DataFrame
) -> dict:
    """
    Extract benign-only rows from the FULL train set (before train/val split)
    and split 80/20 for autoencoder training and reconstruction-error
    monitoring.
    """
    log("SECTION 7 — Benign-only dataset for the Autoencoder")
    mask = (y_full_train["label"] == "Benign").to_numpy()
    X_benign = X_full_train[mask]
    log(f"  benign rows: {X_benign.shape[0]:,}")

    X_b_tr, X_b_va = train_test_split(
        X_benign, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )
    log(f"  AE train: {X_b_tr.shape[0]:,} | AE val: {X_b_va.shape[0]:,}")

    benign_stats = {
        "mean": X_benign.mean(axis=0).astype(float).tolist(),
        "std": X_benign.std(axis=0).astype(float).tolist(),
        "p95": np.percentile(X_benign, 95, axis=0).astype(float).tolist(),
        "p99": np.percentile(X_benign, 99, axis=0).astype(float).tolist(),
    }
    return {
        "X_benign_train": X_b_tr.astype(np.float32),
        "X_benign_val": X_b_va.astype(np.float32),
        "benign_stats": benign_stats,
    }


# %% ===========================================================================
# SECTION 8 — Zero-day simulation datasets (leave-one-attack-out)
# ==============================================================================
ZERO_DAY_TARGETS = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "MQTT_DoS_Connect_Flood",
    "ARP_Spoofing",
]


def build_zero_day_sets(
    X_train: np.ndarray, y_train: pd.DataFrame,
    X_test: np.ndarray, y_test: pd.DataFrame,
) -> dict:
    """
    For each rare attack class, build:
      - X_train_without / y_train_without: training set with that class removed
      - X_held_out / y_held_out: only that class's test rows
    The training set used here is the un-resampled one — the unsupervised layer
    must be trained on real distributions for honest evaluation.
    """
    log("SECTION 8 — Zero-day leave-one-attack-out datasets (×5)")
    out = {}
    for target in ZERO_DAY_TARGETS:
        if target not in y_train["label"].unique():
            log(f"  ⚠ {target} not present in train — skipped")
            continue
        keep_train = (y_train["label"] != target).to_numpy()
        keep_held = (y_test["label"] == target).to_numpy()
        n_train_kept = int(keep_train.sum())
        n_held = int(keep_held.sum())
        log(f"  {target}: train_without={n_train_kept:,} | held_out={n_held:,}")
        out[target] = {
            "X_train_without": X_train[keep_train].astype(np.float32),
            "y_train_without": y_train.loc[keep_train].reset_index(drop=True),
            "X_held_out": X_test[keep_held].astype(np.float32),
            "y_held_out": y_test.loc[keep_held].reset_index(drop=True),
        }
    return out


# %% ===========================================================================
# SECTION 9 — Saving outputs
# ==============================================================================
def _save_variant(
    base: Path,
    X_train: np.ndarray, y_train: pd.DataFrame,
    X_val: np.ndarray, y_val: pd.DataFrame,
    X_test: np.ndarray, y_test: pd.DataFrame,
    X_train_smote: np.ndarray, y_train_smote: pd.DataFrame,
) -> None:
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "X_train.npy", X_train)
    np.save(base / "X_val.npy", X_val)
    np.save(base / "X_test.npy", X_test)
    np.save(base / "X_train_smote.npy", X_train_smote)
    y_train.to_csv(base / "y_train.csv", index=False)
    y_val.to_csv(base / "y_val.csv", index=False)
    y_test.to_csv(base / "y_test.csv", index=False)
    y_train_smote.to_csv(base / "y_train_smote.csv", index=False)


def save_outputs(
    config: dict,
    label_encoders_dump: dict,
    full: dict,
    reduced: dict,
    autoencoder: dict,
    zero_day: dict,
) -> None:
    log("SECTION 9 — Saving outputs to ./preprocessed/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # config + label mappings
    with (OUTPUT_DIR / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
    with (OUTPUT_DIR / "label_encoders.json").open("w") as f:
        json.dump(label_encoders_dump, f, indent=2)

    # scalers
    joblib.dump(full["scaler"], OUTPUT_DIR / "scaler_full.pkl")
    joblib.dump(reduced["scaler"], OUTPUT_DIR / "scaler_reduced.pkl")

    # variants
    _save_variant(
        OUTPUT_DIR / "full_features",
        full["X_train"], full["y_train"],
        full["X_val"], full["y_val"],
        full["X_test"], full["y_test"],
        full["X_train_smote"], full["y_train_smote"],
    )
    _save_variant(
        OUTPUT_DIR / "reduced_features",
        reduced["X_train"], reduced["y_train"],
        reduced["X_val"], reduced["y_val"],
        reduced["X_test"], reduced["y_test"],
        reduced["X_train_smote"], reduced["y_train_smote"],
    )

    # autoencoder
    ae_dir = OUTPUT_DIR / "autoencoder"
    ae_dir.mkdir(exist_ok=True)
    np.save(ae_dir / "X_benign_train.npy", autoencoder["X_benign_train"])
    np.save(ae_dir / "X_benign_val.npy", autoencoder["X_benign_val"])
    with (ae_dir / "benign_stats.json").open("w") as f:
        json.dump(autoencoder["benign_stats"], f, indent=2)

    # zero-day
    zd_dir = OUTPUT_DIR / "zero_day"
    zd_dir.mkdir(exist_ok=True)
    for target, payload in zero_day.items():
        sub = zd_dir / target
        sub.mkdir(exist_ok=True)
        np.save(sub / "X_train_without.npy", payload["X_train_without"])
        np.save(sub / "X_held_out.npy", payload["X_held_out"])
        payload["y_train_without"].to_csv(sub / "y_train_without.csv", index=False)
        payload["y_held_out"].to_csv(sub / "y_held_out.csv", index=False)

    log("  ✓ all artefacts written")


# %% ===========================================================================
# SECTION 10 — Verification & summary report
# ==============================================================================
def verify_outputs(zero_day_targets: list[str]) -> None:
    log("SECTION 10 — Verifying saved artefacts")
    issues = 0

    def _check_arr(path: Path) -> np.ndarray:
        nonlocal issues
        arr = np.load(path, mmap_mode="r")
        if np.isnan(arr).any() or np.isinf(arr).any():
            log(f"  ✗ NaN/inf found in {path}")
            issues += 1
        return arr

    for variant in ("full_features", "reduced_features"):
        base = OUTPUT_DIR / variant
        for npy in ("X_train.npy", "X_val.npy", "X_test.npy", "X_train_smote.npy"):
            arr = _check_arr(base / npy)
            csv = base / npy.replace("X_", "y_").replace(".npy", ".csv")
            n_y = sum(1 for _ in csv.open()) - 1  # minus header
            if n_y != arr.shape[0]:
                log(f"  ✗ shape mismatch {npy} ({arr.shape[0]}) vs "
                    f"{csv.name} ({n_y})")
                issues += 1
        log(f"  ✓ {variant} integrity OK")

    # Benign-only must contain no attack labels — we trust the split mask, but
    # the file should at least be loadable and nonzero.
    ae = np.load(OUTPUT_DIR / "autoencoder" / "X_benign_train.npy", mmap_mode="r")
    if ae.shape[0] == 0:
        log("  ✗ benign autoencoder set is empty"); issues += 1
    else:
        log(f"  ✓ autoencoder set OK ({ae.shape[0]:,} rows)")

    # Zero-day held-out sets should each contain ONLY the target label
    for target in zero_day_targets:
        held_csv = OUTPUT_DIR / "zero_day" / target / "y_held_out.csv"
        if not held_csv.exists():
            continue
        labels_in = pd.read_csv(held_csv)["label"].unique()
        if list(labels_in) == [target]:
            log(f"  ✓ zero-day {target}: held_out contains only {target}")
        else:
            log(f"  ✗ zero-day {target}: held_out contains {labels_in.tolist()}")
            issues += 1

    if issues == 0:
        log("  ✓ verification passed with no issues")
    else:
        log(f"  ⚠ verification found {issues} issue(s) — inspect logs above")


# %% ===========================================================================
# MAIN
# ==============================================================================
def main() -> None:
    t_start = time.time()
    log("=" * 78)
    log("Phase 3 — Preprocessing & Feature Engineering")
    log("=" * 78)

    # ---------- 1. Load
    train, test = load_data()

    # ---------- 2. Encode labels
    enc = encode_labels(train, test)
    y_train_full, y_test_full = enc["y_train"], enc["y_test"]
    multi_map = enc["mappings"]["multiclass"]

    # ---------- 3. Two feature variants
    Xtr_full, Xte_full = select_features(train, test, FEATURES_FULL, "full")
    Xtr_red, Xte_red = select_features(train, test, FEATURES_REDUCED, "reduced")
    # Free the parent DataFrames now that we've sliced them
    del train, test; gc.collect()

    # ---------- 4. Fit scalers, transform train+test
    Xtr_full_sc, Xte_full_sc, scaler_full = fit_scale(
        Xtr_full, Xte_full, "full", FEATURES_FULL
    )
    Xtr_red_sc, Xte_red_sc, scaler_red = fit_scale(
        Xtr_red, Xte_red, "reduced", FEATURES_REDUCED
    )
    del Xtr_full, Xte_full, Xtr_red, Xte_red; gc.collect()

    # ---------- 5. 80/20 train/val split (each variant uses the same indices,
    # but we recompute with the same random_state so they line up)
    Xtr_F, Xva_F, ytr_F, yva_F = split_train_val(Xtr_full_sc, y_train_full, "full")
    Xtr_R, Xva_R, ytr_R, yva_R = split_train_val(Xtr_red_sc, y_train_full, "reduced")

    # ---------- 6. SMOTETomek (training split only)
    Xtr_F_sm, ytr_F_sm = apply_smote_tomek(Xtr_F, ytr_F, multi_map)
    Xtr_R_sm, ytr_R_sm = apply_smote_tomek(Xtr_R, ytr_R, multi_map)

    # ---------- 7. Benign-only set (from FULL pre-split train, REDUCED features
    # — Layer 2 uses the reduced feature space to match the supervised layer.
    autoencoder = build_autoencoder_set(Xtr_R, ytr_R)

    # ---------- 8. Zero-day sets (REDUCED features, un-resampled train)
    zero_day = build_zero_day_sets(
        Xtr_red_sc, y_train_full, Xte_red_sc, y_test_full,
    )

    # ---------- 9. Save everything
    config = {
        "random_state": RANDOM_STATE,
        "smote_targeted_threshold": SMOTE_TARGETED_THRESHOLD,
        "smote_k_neighbors": SMOTE_K_NEIGHBORS,
        "features_full": FEATURES_FULL,
        "features_reduced": FEATURES_REDUCED,
        "scaler_groups": {
            "heavy_tailed_reduced": HEAVY_TAILED,
            "flag_features": FLAG_FEATURES,
            "binary_features": BINARY_FEATURES,
            "heavy_tailed_full_extra": HEAVY_TAILED_FULL_EXTRA,
            "binary_full_extra": BINARY_FULL_EXTRA,
        },
        "shapes": {
            "full":    {"X_train": list(Xtr_F.shape), "X_val": list(Xva_F.shape),
                        "X_test": list(Xte_full_sc.shape),
                        "X_train_smote": list(Xtr_F_sm.shape)},
            "reduced": {"X_train": list(Xtr_R.shape), "X_val": list(Xva_R.shape),
                        "X_test": list(Xte_red_sc.shape),
                        "X_train_smote": list(Xtr_R_sm.shape)},
        },
        "zero_day_targets": ZERO_DAY_TARGETS,
        "autoencoder": {
            "n_train": int(autoencoder["X_benign_train"].shape[0]),
            "n_val": int(autoencoder["X_benign_val"].shape[0]),
            "feature_space": "reduced",
        },
    }
    label_encoders_dump = {
        "binary": enc["mappings"]["binary"],
        "category": enc["mappings"]["category"],
        "multiclass": enc["mappings"]["multiclass"],
    }
    config["feature_names_full"] = scaler_full.get_feature_names_out().tolist()
    config["feature_names_reduced"] = scaler_red.get_feature_names_out().tolist()

    save_outputs(
        config=config,
        label_encoders_dump=label_encoders_dump,
        full={
            "scaler": scaler_full,
            "X_train": Xtr_F, "y_train": ytr_F,
            "X_val": Xva_F,  "y_val": yva_F,
            "X_test": Xte_full_sc, "y_test": y_test_full,
            "X_train_smote": Xtr_F_sm, "y_train_smote": ytr_F_sm,
        },
        reduced={
            "scaler": scaler_red,
            "X_train": Xtr_R, "y_train": ytr_R,
            "X_val": Xva_R,  "y_val": yva_R,
            "X_test": Xte_red_sc, "y_test": y_test_full,
            "X_train_smote": Xtr_R_sm, "y_train_smote": ytr_R_sm,
        },
        autoencoder=autoencoder,
        zero_day=zero_day,
    )

    # ---------- 10. Verify
    verify_outputs(ZERO_DAY_TARGETS)

    # ---------- summary
    dt = time.time() - t_start
    print()
    print("=" * 78)
    print("=== PREPROCESSING COMPLETE ===")
    print("=" * 78)
    print(f"Full features:    {len(FEATURES_FULL)} features")
    print(f"Reduced features: {len(FEATURES_REDUCED)} features")
    print()
    print(f"Train (pre-split): {y_train_full.shape[0]:,}")
    print(f"  → after 80/20:   train={Xtr_R.shape[0]:,} / val={Xva_R.shape[0]:,}")
    print(f"Test (untouched):  {y_test_full.shape[0]:,}")
    print(f"SMOTETomek (full):    {Xtr_F.shape[0]:,} → {Xtr_F_sm.shape[0]:,}")
    print(f"SMOTETomek (reduced): {Xtr_R.shape[0]:,} → {Xtr_R_sm.shape[0]:,}")
    print(f"Benign-only:       AE_train={autoencoder['X_benign_train'].shape[0]:,} "
          f"/ AE_val={autoencoder['X_benign_val'].shape[0]:,}")
    print(f"Zero-day:          {len(zero_day)} scenarios "
          f"× (train_without + held_out)")
    print()
    print(f"All files saved to {OUTPUT_DIR.resolve()}/")
    print(f"Total runtime: {dt/60:.1f} min")
    print("Ready for Phase 4 (Supervised Training).")
    print("=" * 78)


if __name__ == "__main__":
    main()
