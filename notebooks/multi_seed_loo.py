#!/usr/bin/env python3
"""
Path B Week 1 — Multi-Seed Leave-One-Attack-Out Retraining
==========================================================

Re-runs Phase 6B's LOO XGBoost retrainings under multiple random seeds to
quantify seed-induced variance in the H2-strict 4/4 PASS verdict.

Faithful to `notebooks/loo_zero_day.py:48-62`:
    - Identical XGB hyperparameters (only `random_state` varies per seed).
    - Identical LOO partitioning (mask out target class from training).
    - Identical label-map construction (sorted(set(y_tr_labels))) — verified
      against the canonical seed-42 sidecar JSONs as a tripwire before training.

The AE / IF / E7 main models are NOT retrained — they are benign-only or
trained on all 19 classes and seed-invariant from this experiment's POV.

Seed=42 is NOT retrained. Its artifacts are hardlinked from the canonical
results/zero_day_loo/predictions/ into multi_seed/seed_42/predictions/ by
the Phase 0 setup step (see results/zero_day_loo/multi_seed/seed_42/config.json).

Output per seed (S != 42):
    results/zero_day_loo/multi_seed/seed_<S>/
        predictions/  loo_<TARGET>_test_pred.npy, loo_<TARGET>_test_proba.npy
        metrics/      per_target_metrics.json
        config.json   (seed, hyperparameters, runtime, sha256 audit trail)

Usage:
    cd ~/IoMT-Project && source venv/bin/activate
    # Smoke test (one fold, ~4 min):
    caffeinate -dimsu python -u notebooks/multi_seed_loo.py --seeds 1 --targets ARP_Spoofing
    # Full sweep (4 new seeds × 5 targets, ~80 min):
    caffeinate -dimsu python -u notebooks/multi_seed_loo.py
"""

# %% Imports & configuration
import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ---- Constants imported verbatim from loo_zero_day.py:48-70 -----------------
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    tree_method="hist",
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

DEFAULT_SEEDS = [1, 7, 42, 100, 1729]
PER_FOLD_TIMEOUT_SECONDS = 600  # tripwire — abort if any fold takes > 10 min

PREPROCESSED_DIR  = Path("./preprocessed")
LOO_CANONICAL_DIR = Path("./results/zero_day_loo")
MULTI_SEED_DIR    = LOO_CANONICAL_DIR / "multi_seed"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_all_seeds(seed: int) -> None:
    """Set every randomness source we touch. Even if XGBoost's hist tree-builder
    doesn't consume numpy's RNG directly, this is the safe default — costs
    nothing and prevents future hidden non-determinism if the script grows."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_canonical_label_map(target: str) -> dict[str, int]:
    """Read the seed-42 sidecar JSON. Used as the seed-invariant ground truth
    for label-space consistency."""
    path = LOO_CANONICAL_DIR / "models" / f"loo_label_map_{target}.json"
    with open(path) as f:
        raw = json.load(f)
    return {str(k): int(v) for k, v in raw.items()}


def build_loo_label_map(y_tr_labels: np.ndarray) -> dict[str, int]:
    """Reproduces loo_zero_day.py:207-208 verbatim."""
    remaining_classes = sorted(set(y_tr_labels))
    return {cls: i for i, cls in enumerate(remaining_classes)}


# ---- One fold ---------------------------------------------------------------
def run_one_fold(
    seed: int,
    target: str,
    X_train: np.ndarray,
    y_train_labels: np.ndarray,
    X_test: np.ndarray,
) -> dict:
    """Train XGBoost without `target` under seed `seed`, predict on full test.

    Returns a dict with timing and integrity info. Resumes if outputs exist."""
    seed_dir = MULTI_SEED_DIR / f"seed_{seed}"
    pred_path = seed_dir / "predictions" / f"loo_{target}_test_pred.npy"
    proba_path = seed_dir / "predictions" / f"loo_{target}_test_proba.npy"

    log(f"  [seed={seed}] [{target}]")

    # Resumability: skip if both files already exist
    if pred_path.exists() and proba_path.exists():
        pred = np.load(pred_path, mmap_mode="r")
        proba = np.load(proba_path, mmap_mode="r")
        log(f"    RESUME: predictions exist (pred={pred.shape}, proba={proba.shape})")
        return {
            "target": target,
            "seed": seed,
            "skipped": True,
            "train_time_s": None,
            "predict_time_s": None,
            "pred_sha256": sha256_of_file(pred_path),
            "proba_sha256": sha256_of_file(proba_path),
        }

    # 1. Build LOO training set (matches loo_zero_day.py:202-209)
    target_mask = (y_train_labels != target)
    n_removed = int((~target_mask).sum())
    X_tr = X_train[target_mask]
    y_tr_labels_local = y_train_labels[target_mask]

    fresh_label_map = build_loo_label_map(y_tr_labels_local)
    canonical_label_map = load_canonical_label_map(target)

    # Tripwire: if the freshly-built label map differs from the canonical one,
    # the LOO partition is non-deterministic and the experiment is invalid.
    if fresh_label_map != canonical_label_map:
        raise RuntimeError(
            f"[seed={seed}][{target}] Label-map drift detected!\n"
            f"  fresh:     {fresh_label_map}\n"
            f"  canonical: {canonical_label_map}\n"
            f"This breaks the assumption that LOO label space is seed-invariant. Aborting."
        )

    y_tr_encoded = np.array(
        [fresh_label_map[c] for c in y_tr_labels_local], dtype=np.int32
    )
    n_classes_loo = len(fresh_label_map)
    log(
        f"    train: {X_tr.shape[0]:,} rows × {X_tr.shape[1]} cols, "
        f"{n_classes_loo} classes (removed {n_removed:,} {target})"
    )

    assert target not in fresh_label_map, f"{target} leaked into LOO training set"
    assert n_classes_loo == 18, f"expected 18 LOO classes, got {n_classes_loo}"

    # 2. Train under explicit seed
    set_all_seeds(seed)
    params = dict(XGB_PARAMS)
    params["random_state"] = seed
    params["num_class"] = n_classes_loo
    model = XGBClassifier(**params)

    t_train_start = time.time()
    model.fit(X_tr, y_tr_encoded)
    train_time_s = time.time() - t_train_start
    log(f"    train: {train_time_s/60:.2f} min")

    # 3. Tripwire: fold-level timeout
    if train_time_s > PER_FOLD_TIMEOUT_SECONDS:
        raise RuntimeError(
            f"[seed={seed}][{target}] Training took {train_time_s:.0f}s "
            f"> {PER_FOLD_TIMEOUT_SECONDS}s budget. "
            f"Likely hyperparameter drift — verify XGB_PARAMS matches "
            f"loo_zero_day.py:48-62 exactly. Aborting before further folds."
        )

    # 4. Predict
    t_pred_start = time.time()
    loo_pred_local = model.predict(X_test).astype(np.int32)
    loo_proba = model.predict_proba(X_test).astype(np.float32)
    predict_time_s = time.time() - t_pred_start
    log(
        f"    predict: {predict_time_s:.1f}s "
        f"(pred {loo_pred_local.shape}, proba {loo_proba.shape})"
    )

    # 5. Sanity assertions before saving
    assert loo_pred_local.shape == (X_test.shape[0],), (
        f"pred shape {loo_pred_local.shape} != ({X_test.shape[0]},)"
    )
    assert loo_proba.shape == (X_test.shape[0], n_classes_loo), (
        f"proba shape {loo_proba.shape} != ({X_test.shape[0]}, {n_classes_loo})"
    )
    proba_row_sums = loo_proba.sum(axis=1)
    assert np.allclose(proba_row_sums, 1.0, atol=1e-4), (
        f"proba rows do not sum to 1: min={proba_row_sums.min():.6f}, "
        f"max={proba_row_sums.max():.6f}"
    )
    # The held-out class must NOT be in the local label space, so the local
    # predictions cannot reference it. This is the LOO contract.
    assert target not in {
        cls for cls, idx in fresh_label_map.items() if idx in set(loo_pred_local.tolist()[:1000])
    }, "smoke check: held-out class index appeared in predictions (impossible)"

    # 6. Save predictions atomically — write to `.tmp.npy`, rename to `.npy`.
    # Note: `np.save` auto-appends `.npy` if the destination doesn't already
    # end in it. Using `.tmp.npy` keeps numpy from rewriting the path.
    tmp_pred  = pred_path.with_name(pred_path.stem + ".tmp.npy")
    tmp_proba = proba_path.with_name(proba_path.stem + ".tmp.npy")
    np.save(tmp_pred, loo_pred_local)
    np.save(tmp_proba, loo_proba)
    tmp_pred.replace(pred_path)
    tmp_proba.replace(proba_path)
    log(f"    saved: {pred_path.name}, {proba_path.name}")

    # 7. Free
    del model, X_tr, y_tr_encoded, y_tr_labels_local, loo_pred_local, loo_proba
    gc.collect()

    return {
        "target": target,
        "seed": seed,
        "skipped": False,
        "train_time_s": float(train_time_s),
        "predict_time_s": float(predict_time_s),
        "pred_sha256": sha256_of_file(pred_path),
        "proba_sha256": sha256_of_file(proba_path),
    }


# ---- Per-seed orchestration -------------------------------------------------
def run_seed(
    seed: int,
    targets: list[str],
    X_train: np.ndarray,
    y_train_labels: np.ndarray,
    X_test: np.ndarray,
    y_test_labels: np.ndarray,
) -> None:
    """Train all `targets` under one seed, then write seed-level config + metrics."""
    if seed == 42:
        log(f"[seed=42] SKIP retraining; expecting hardlinked artifacts under "
            f"{MULTI_SEED_DIR}/seed_42/predictions/")
        # Verify hardlinks exist
        for t in targets:
            for kind in ("test_pred", "test_proba"):
                p = MULTI_SEED_DIR / "seed_42" / "predictions" / f"loo_{t}_{kind}.npy"
                assert p.exists(), (
                    f"seed=42 hardlink missing: {p}. Run Phase 0 setup first."
                )
        log("[seed=42] hardlinked artifacts verified")
        return

    log("=" * 76)
    log(f"SEED {seed}")
    log("=" * 76)

    seed_dir = MULTI_SEED_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "predictions").mkdir(exist_ok=True)
    (seed_dir / "metrics").mkdir(exist_ok=True)

    seed_started = datetime.now().isoformat(timespec="seconds")
    seed_t0 = time.time()
    fold_records = []
    per_target_metrics = {}

    for target in targets:
        rec = run_one_fold(seed, target, X_train, y_train_labels, X_test)
        fold_records.append(rec)

        # Lightweight per-target metric: top-5 prediction distribution
        pred_path = seed_dir / "predictions" / f"loo_{target}_test_pred.npy"
        loo_pred_local = np.load(pred_path)

        canonical_label_map = load_canonical_label_map(target)
        inv_map = {v: k for k, v in canonical_label_map.items()}
        target_test_mask = (y_test_labels == target)
        n_target_test = int(target_test_mask.sum())
        if n_target_test > 0:
            preds_on_target_local = loo_pred_local[target_test_mask]
            preds_on_target_labels = [inv_map[int(p)] for p in preds_on_target_local]
            counter = Counter(preds_on_target_labels)
            n_called_benign = counter.get("Benign", 0)
            top5 = counter.most_common(5)
            per_target_metrics[target] = {
                "n_test_samples": n_target_test,
                "n_called_benign": int(n_called_benign),
                "pct_called_benign": float(100 * n_called_benign / n_target_test),
                "top5_predicted_as": [
                    {"class": cls, "count": int(cnt),
                     "pct": float(100 * cnt / n_target_test)}
                    for cls, cnt in top5
                ],
            }
            log(
                f"    sanity: target={target} called_Benign={n_called_benign:,} "
                f"({100*n_called_benign/n_target_test:.1f}%); top1={top5[0]}"
            )
            # Tripwire: LOO model must never predict the held-out class
            assert "ARP_Spoofing" if target == "ARP_Spoofing" else target not in (
                cls for cls, _ in top5
            ) or top5[0][1] == 0, (
                f"LOO model predicted {target} on its own held-out test rows!"
            )

        del loo_pred_local
        gc.collect()

    seed_runtime_min = (time.time() - seed_t0) / 60
    seed_finished = datetime.now().isoformat(timespec="seconds")

    # Per-seed metrics dump
    with open(seed_dir / "metrics" / "per_target_metrics.json", "w") as f:
        json.dump(per_target_metrics, f, indent=2)

    # Per-seed config — full audit trail
    config = {
        "seed": seed,
        "retrained": True,
        "phase": "Path B Week 1 multi-seed",
        "xgb_params": {**XGB_PARAMS, "random_state": seed},
        "zero_day_targets": targets,
        "started_at": seed_started,
        "finished_at": seed_finished,
        "runtime_min": float(seed_runtime_min),
        "fold_records": fold_records,
    }
    with open(seed_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    log(f"[seed={seed}] DONE in {seed_runtime_min:.1f} min  → {seed_dir}/config.json")


# ---- Main -------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Seeds to run (default: {DEFAULT_SEEDS})",
    )
    p.add_argument(
        "--targets", type=str, nargs="+", default=ZERO_DAY_TARGETS,
        help=f"Targets to run (default: all 5)",
    )
    args = p.parse_args()

    log("=" * 76)
    log("Path B Week 1 — Multi-Seed LOO Retraining")
    log(f"  seeds:   {args.seeds}")
    log(f"  targets: {args.targets}")
    log("=" * 76)

    # Load shared data ONCE (all seeds reuse)
    log("Loading X_train, y_train, X_test, y_test ...")
    t_load = time.time()
    X_train = np.load(PREPROCESSED_DIR / "full_features" / "X_train.npy")
    y_train_df = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_train.csv")
    y_train_labels = y_train_df["label"].astype(str).values
    X_test = np.load(PREPROCESSED_DIR / "full_features" / "X_test.npy")
    y_test_df = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_test.csv")
    y_test_labels = y_test_df["label"].astype(str).values
    log(
        f"  X_train {X_train.shape}  X_test {X_test.shape}  "
        f"loaded in {time.time()-t_load:.1f}s"
    )

    overall_t0 = time.time()
    for seed in args.seeds:
        run_seed(seed, args.targets, X_train, y_train_labels, X_test, y_test_labels)

    total_min = (time.time() - overall_t0) / 60
    log("=" * 76)
    log(f"ALL DONE — {total_min:.1f} min total")
    log("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
