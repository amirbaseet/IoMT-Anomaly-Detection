#!/usr/bin/env python3
"""
Inverse cross-dataset direction on the FULL data: TRAIN on CICIoT2023, TEST on CICIoMT2024.

Supersedes the preliminary ``inverse_run.py`` (kept unchanged so its number stays
reproducible).  Differences from the preliminary run, all deliberate:
  * TRAIN = every row of the 63 CICIoT2023 merged CSVs (streamed out of the
    MERGED_CSV.zip), minus the held-out file below and the empty-label rows —
    not the 578,659-row class-capped seed-42 sample.
  * Model = ``config.XGB_PARAMS_BASE`` (the forward run's frozen params), unweighted,
    median-imputed with medians fit on the TRAIN data only (TRAP 1) — as in the forward run.
  * TEST  = ALL rows of CICIoMT2024 ``data/test`` (no per-file cap).

Known-positive control: ``Merged63.csv`` is held out of training and scored as a
within-CICIoT2023 sanity cell.  A near-0.5 AUC there means the loader is broken,
not that the domains differ.

One model seed (config.TRAIN_SEED): sampling-seed σ (TRAP 4) does not exist when
the whole dataset is used, so no σ is reported.
Device/MAC identity is never a feature (the shared schema is 38 behavioral fields).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ciciomt_data  # noqa: E402
import config        # noqa: E402
import features      # noqa: E402
import labels        # noqa: E402

DEFAULT_ZIP = Path("/Volumes/PortableSSD/CICIOT23/csv/MERGED_CSV.zip")
HOLDOUT_MEMBER = "MERGED_CSV/Merged63.csv"
SHARED_JSON = config.OUT_DIR / "shared_features.json"
DEFAULT_OUT = config.OUT_DIR / "inverse_full_results.json"
THRESHOLD = 0.5
N_SHARED = 38


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_member(zf: zipfile.ZipFile, member: str, shared: list[str]) -> dict:
    """Stream one merged CSV out of the zip -> float32 X, int8 y (1=attack)."""
    xs, ys = [], []
    n_raw = n_unknown = 0
    dtypes = {f: "float32" for f in shared}
    dtypes[config.CICIOT2023_LABEL_COL] = str
    with zf.open(member) as fh:
        for chunk in pd.read_csv(fh, usecols=shared + [config.CICIOT2023_LABEL_COL],
                                 dtype=dtypes, chunksize=config.SAMPLE_CHUNK_ROWS):
            n_raw += len(chunk)
            norm = chunk[config.CICIOT2023_LABEL_COL].map(labels.normalize_ciciot2023)  # TRAP 2
            known_set = {v for v in norm.unique() if labels.is_known_ciciot2023(v)}
            keep = norm.isin(known_set).to_numpy()
            n_unknown += int((~keep).sum())
            x = features.align_to_schema(chunk, shared).to_numpy(dtype=np.float32)  # TRAP 3
            xs.append(x[keep])
            ys.append((norm.to_numpy()[keep] != "BENIGN").astype(np.int8))
    x = np.concatenate(xs)
    x[np.isinf(x)] = np.nan
    return {"X": x, "y": np.concatenate(ys), "n_raw": n_raw, "n_unknown": n_unknown}


def column_medians(xs: list[np.ndarray]) -> np.ndarray:
    """Per-feature median over the TRAIN arrays only (TRAP 1), one column at a time."""
    n_feat = xs[0].shape[1]
    return np.array([np.nanmedian(np.concatenate([a[:, j] for a in xs]))
                     for j in range(n_feat)], dtype=np.float32)


def impute(x: np.ndarray, med: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a NaN-free copy of ``x`` filled with the train medians, plus the NaN count."""
    mask = np.isnan(x)
    n_nan = int(mask.sum())
    if n_nan == 0:
        return x, 0
    return np.where(mask, med[np.newaxis, :], x).astype(np.float32), n_nan


class _Batches(xgb.DataIter):
    """Feed the per-file arrays to QuantileDMatrix without concatenating them."""

    def __init__(self, xs: list[np.ndarray], ys: list[np.ndarray]) -> None:
        self._xs, self._ys, self._i = xs, ys, 0
        super().__init__()

    def next(self, input_data) -> bool:
        if self._i == len(self._xs):
            return False
        input_data(data=self._xs[self._i], label=self._ys[self._i])
        self._i += 1
        return True

    def reset(self) -> None:
        self._i = 0


def booster_params() -> tuple[dict, int]:
    """Translate the forward run's sklearn-style XGB_PARAMS_BASE for xgb.train."""
    p = dict(config.XGB_PARAMS_BASE)
    n_rounds = p.pop("n_estimators")
    p["seed"] = p.pop("random_state")
    n_jobs = p.pop("n_jobs")
    p["nthread"] = os.cpu_count() if n_jobs == -1 else n_jobs
    p.update(objective="binary:logistic", eval_metric="logloss")
    return p, n_rounds


def score(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "n": int(len(y)), "n_benign": int((y == 0).sum()), "n_attack": int((y == 1).sum()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "benign_recall": float(recall_score(y, pred, pos_label=0)),
        "attack_recall": float(recall_score(y, pred, pos_label=1)),
    }


def require_both_classes(name: str, y: np.ndarray) -> None:
    if len(np.unique(y)) != 2:
        raise AssertionError(f"{name}: needs both benign and attack rows, got {np.unique(y)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--max-files", type=int, default=None,
                    help="smoke run: only the first N training files (holdout always scored)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    t0 = time.time()

    shared = json.loads(SHARED_JSON.read_text())["shared_features"]
    assert len(shared) == N_SHARED, f"expected {N_SHARED} shared features, got {len(shared)}"

    # ---- TRAIN + HOLDOUT: CICIoT2023, streamed from the zip ----
    zf = zipfile.ZipFile(args.zip)
    members = sorted(m for m in zf.namelist() if m.endswith(".csv"))
    train_members = [m for m in members if m != HOLDOUT_MEMBER]
    if HOLDOUT_MEMBER not in members:
        raise FileNotFoundError(f"{HOLDOUT_MEMBER} not in {args.zip}")
    if args.max_files:
        train_members = train_members[:args.max_files]

    xs, ys, n_raw, n_unknown = [], [], 0, 0
    for i, m in enumerate(train_members, 1):
        part = read_member(zf, m, shared)
        xs.append(part["X"])
        ys.append(part["y"])
        n_raw += part["n_raw"]
        n_unknown += part["n_unknown"]
        _log(f"train {i}/{len(train_members)} {m}: {len(part['y'])} rows")
    hold = read_member(zf, HOLDOUT_MEMBER, shared)
    t_read = time.time() - t0
    y_tr = np.concatenate(ys)
    require_both_classes("train", y_tr)
    require_both_classes("holdout", hold["y"])
    _log(f"TRAIN rows {len(y_tr)} (benign {int((y_tr == 0).sum())}) | "
         f"holdout {len(hold['y'])} | dropped unknown-label {n_unknown + hold['n_unknown']} | "
         f"read {t_read:.0f}s")

    # ---- impute with TRAIN medians (TRAP 1), then quantize ----
    med = column_medians(xs)
    imputed = [impute(a, med) for a in xs]
    xs = [a for a, _ in imputed]
    n_nan_train = sum(n for _, n in imputed)
    x_hold, n_nan_hold = impute(hold["X"], med)

    t1 = time.time()
    dtrain = xgb.QuantileDMatrix(_Batches(xs, ys))
    del xs, imputed
    params, n_rounds = booster_params()
    booster = xgb.train(params, dtrain, num_boost_round=n_rounds)
    t_train = time.time() - t1
    _log(f"trained {n_rounds} rounds in {t_train:.0f}s")

    # ---- known-positive control: held-out CICIoT2023 file ----
    holdout = score(hold["y"], booster.predict(xgb.DMatrix(x_hold)))
    _log(f"HOLDOUT (within CICIoT2023) AUC {holdout['roc_auc']:.4f} "
         f"bal-acc {holdout['balanced_accuracy']:.4f}")

    # ---- TEST: all CICIoMT2024 test rows, projected to the same 38 columns ----
    mt = ciciomt_data.load_split(config.CICIOMT2024_TEST_DIR, shared)
    x_mt = features.align_to_schema(mt["X"], shared).to_numpy(dtype=np.float32)
    x_mt[np.isinf(x_mt)] = np.nan
    x_mt, n_nan_test = impute(x_mt, med)
    y_mt = (mt["y_binary"] != labels.BENIGN.lower()).astype(np.int8)
    require_both_classes("test", y_mt)
    inverse = score(y_mt, booster.predict(xgb.DMatrix(x_mt)))
    _log(f"INVERSE IoT2023 -> MT  AUC {inverse['roc_auc']:.4f} "
         f"bal-acc {inverse['balanced_accuracy']:.4f}")

    result = {
        "status": "Measured",
        "direction": "train CICIoT2023 (full merged CSVs) -> test CICIoMT2024 (data/test, all rows)",
        "smoke": bool(args.max_files),
        "source_zip": str(args.zip),
        "train_files": len(train_members),
        "holdout_file": HOLDOUT_MEMBER,
        "n_shared_features": len(shared),
        "train": {"n_raw_rows": n_raw, "n_unknown_label_dropped": n_unknown,
                  "n": int(len(y_tr)), "n_benign": int((y_tr == 0).sum()),
                  "n_attack": int((y_tr == 1).sum()), "n_nan_cells_imputed": n_nan_train},
        "holdout_within_ciciot2023": {**holdout, "n_nan_cells_imputed": n_nan_hold},
        "inverse_test_ciciomt2024": {**inverse, "n_nan_cells_imputed": n_nan_test,
                                     "n_test_files": len(list(config.CICIOMT2024_TEST_DIR.glob("*.csv")))},
        "model": {"xgb_params": config.XGB_PARAMS_BASE, "api": "xgb.train + QuantileDMatrix",
                  "threshold": THRESHOLD, "class_weighting": None,
                  "imputation": "per-feature median, fit on TRAIN only",
                  "model_seed": config.TRAIN_SEED, "sigma": "not reported (single model seed)"},
        "wall_seconds": {"read": round(t_read), "train": round(t_train),
                         "total": round(time.time() - t0)},
        "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
        "xgboost_version": xgb.__version__,
    }
    args.out.write_text(json.dumps(result, indent=2))
    _log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
