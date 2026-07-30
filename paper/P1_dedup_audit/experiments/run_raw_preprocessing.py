"""
C1 step 2 — preprocess the NON-deduplicated arm through the FROZEN pipeline.

This script does NOT reimplement preprocessing. It imports
notebooks/preprocessing_pipeline.py, repoints its input/output constants at the
raw arm, and calls its own functions — so the deduplicated arm (preprocessed/)
and the raw arm (preprocessed_raw/) are produced by identical code. Forking the
module would let the arms drift, which is the exact confound C1 exists to remove.

Only what the C1 ablation needs is produced: the FULL-44 tree variant.
SMOTETomek variants, the autoencoder set and the zero-day sets are skipped —
C1 is a tree-only, no-resampling comparison (EXPERIMENT_C1.md).

Inputs  -> eda_output_raw/{train,test}_cleaned.csv   (build_raw_cleaned.py)
Outputs -> preprocessed_raw/full_features/*.npy|csv, preprocessed_raw/config.json
Nothing under preprocessed/ is read or written.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import notebooks.preprocessing_pipeline as pp  # noqa: E402  (path set above)

RAW_IN = REPO / "eda_output_raw"
RAW_OUT = REPO / "preprocessed_raw"
DEDUP_CONFIG = REPO / "preprocessed" / "config.json"

RAW_ROWS = {"train": 7_160_831, "test": 1_614_182}


def repoint_pipeline() -> None:
    """Point the frozen module at the raw arm. Constants only — no code changes."""
    pp.TRAIN_INPUT = RAW_IN / "train_cleaned.csv"
    pp.TEST_INPUT = RAW_IN / "test_cleaned.csv"
    pp.OUTPUT_DIR = RAW_OUT
    pp.EXPECTED_TRAIN_ROWS = RAW_ROWS["train"]
    pp.EXPECTED_TEST_ROWS = RAW_ROWS["test"]
    for p in (pp.TRAIN_INPUT, pp.TEST_INPUT):
        if not p.exists():
            sys.exit(f"Missing {p} — run build_raw_cleaned.py first.")
    print(f"  pipeline repointed: {pp.TRAIN_INPUT.name}/{pp.TEST_INPUT.name} -> {RAW_OUT}")
    print(f"  RANDOM_STATE = {pp.RANDOM_STATE} (must be 42)")
    assert pp.RANDOM_STATE == 42, "RANDOM_STATE is not 42 — C1 comparison invalid."


def assert_label_space_matches(multi_map: dict) -> None:
    """INV-01 gate: the raw arm must live in the dedup arm's label space."""
    if not DEDUP_CONFIG.exists():
        print(f"  WARNING: {DEDUP_CONFIG} absent — cannot verify label space now; "
              "run_c1_matrix.py re-checks before any cross-evaluation.")
        return
    dedup_cfg = json.loads(DEDUP_CONFIG.read_text())
    dedup_map = (dedup_cfg.get("label_mappings") or {}).get("multiclass")
    if dedup_map is None:
        # Fall back to the sibling dump the pipeline writes.
        enc_path = REPO / "preprocessed" / "label_encoders.json"
        if enc_path.exists():
            dedup_map = (json.loads(enc_path.read_text()) or {}).get("multiclass")
    if dedup_map is None:
        print("  WARNING: no multiclass mapping found in the dedup artifacts — "
              "deferring the INV-01 check to run_c1_matrix.py.")
        return
    if {k: int(v) for k, v in dedup_map.items()} != {k: int(v) for k, v in multi_map.items()}:
        sys.exit(
            "INV-01 VIOLATION: raw-arm label mapping differs from the deduplicated arm.\n"
            f"  raw:   {multi_map}\n  dedup: {dedup_map}\n"
            "Cross-evaluation across different label spaces is meaningless — aborting."
        )
    print(f"  INV-01 OK: label space identical to the dedup arm ({len(multi_map)} classes)")


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("C1 step 2 — preprocessing the raw arm through the frozen pipeline")
    print("=" * 70)
    repoint_pipeline()

    train, test = pp.load_data()
    for name, df, expected in (("train", train, RAW_ROWS["train"]),
                               ("test", test, RAW_ROWS["test"])):
        if len(df) != expected:
            sys.exit(f"{name}: {len(df):,} rows, expected {expected:,} — wrong substrate.")
    print(f"  loaded raw: train {len(train):,} / test {len(test):,}")

    enc = pp.encode_labels(train, test)
    y_train_full, y_test_full = enc["y_train"], enc["y_test"]
    multi_map = enc["mappings"]["multiclass"]
    assert_label_space_matches(multi_map)

    # FULL-44 variant only (E7's feature set).
    Xtr_full, Xte_full = pp.select_features(train, test, pp.FEATURES_FULL, "full")
    del train, test
    gc.collect()

    # DN-03: the scaler is fitted on TRAIN ONLY, inside the frozen function.
    Xtr_sc, Xte_sc, scaler = pp.fit_scale(Xtr_full, Xte_full, "full", pp.FEATURES_FULL)
    del Xtr_full, Xte_full
    gc.collect()

    Xtr, Xva, ytr, yva = pp.split_train_val(Xtr_sc, y_train_full, "full")
    del Xtr_sc
    gc.collect()

    base = RAW_OUT / "full_features"
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "X_train.npy", Xtr)
    np.save(base / "X_val.npy", Xva)
    np.save(base / "X_test.npy", Xte_sc)
    ytr.to_csv(base / "y_train.csv", index=False)
    yva.to_csv(base / "y_val.csv", index=False)
    y_test_full.to_csv(base / "y_test.csv", index=False)
    joblib.dump(scaler, RAW_OUT / "scaler_full.pkl")

    config = {
        "arm": "raw (non-deduplicated)",
        "produced_by": "paper/P1_dedup_audit/experiments/run_raw_preprocessing.py",
        "pipeline_module": "notebooks/preprocessing_pipeline.py (imported, not forked)",
        "random_state": pp.RANDOM_STATE,
        "features_full": pp.FEATURES_FULL,
        "n_features": len(pp.FEATURES_FULL),
        "label_mappings": {"multiclass": {k: int(v) for k, v in multi_map.items()}},
        "shapes": {
            "X_train": list(Xtr.shape), "X_val": list(Xva.shape),
            "X_test": list(Xte_sc.shape),
        },
        "row_counts_raw": RAW_ROWS,
        "skipped_on_purpose": ["smote variants", "autoencoder set", "zero-day sets"],
    }
    (RAW_OUT / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\n  X_train {Xtr.shape} | X_val {Xva.shape} | X_test {Xte_sc.shape}")
    print(f"  saved -> {base}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
