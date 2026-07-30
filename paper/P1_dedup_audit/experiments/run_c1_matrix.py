"""
C1 step 3 — the 2x2 controlled ablation.

                      tested on RAW      tested on DEDUP
   trained on RAW        C1-a                C1-b
   trained on DEDUP      C1-c                C1-d   (= the published E7 arm)

C1-a - C1-d is the pairing the literature effectively makes (raw everywhere vs
deduplicated everywhere) and it moves TWO factors at once. The single-factor
contrasts are what actually decompose the leakage:
  test-set effect      C1-a - C1-b  and  C1-c - C1-d   (training held fixed)
  training-set effect  C1-c - C1-a  and  C1-d - C1-b   (test held fixed)

Both models are E7 exactly: XGBClassifier with XGB_PARAMS_BASE from
notebooks/supervised_training.py, FULL-44 features, NO resampling,
random_state=42, 19-class multiclass. The model factory and the metric function
are IMPORTED from that module so neither arm can drift from the published one.

Single seed (42): a like-for-like comparison against E7, not a stability claim.
This repo has NO multi-seed sigma for supervised macro-F1, so no noise band is
asserted here; see the seed caveat emitted into the results JSON.

CRITICAL — why the test views are rebuilt rather than loaded (fixed 2026-07-30):
the two arms do NOT share a feature space. Each fits its own ColumnTransformer on
its own training rows, and the duplicate mass moves the fitted statistics a long
way (RobustScaler center 11,884 on dedup-train vs 108 on raw-train for one
feature). Scoring a model on the OTHER arm's saved X_test therefore feeds it
mis-scaled data; the first run of this script did exactly that and produced two
nonsense off-diagonal cells (accuracy ~0.63). Every test view is now transformed
with the SCORING MODEL'S OWN scaler, so all four cells are in-space.

That scaler shift is itself a P1 result: deduplication changes the fitted
preprocessing statistics, not merely the row count.

Inputs  -> preprocessed/full_features/      (dedup arm, existing)
           preprocessed_raw/full_features/  (raw arm, run_raw_preprocessing.py)
           eda_output/test_cleaned.csv      (dedup test, unscaled)
           eda_output_raw/test_cleaned.csv  (raw test, unscaled)
           preprocessed{,_raw}/scaler_full.pkl
Outputs -> results/c1_dedup_ablation/{c1_matrix.json, c1_per_class_f1.csv,
           models/*.ubj}
Nothing existing is overwritten.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib   # loads only this repo's own scaler artifacts, written by
                # notebooks/preprocessing_pipeline.py — local, first-party, not untrusted input
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import notebooks.supervised_training as st  # noqa: E402  (path set above)

DEDUP_DIR = REPO / "preprocessed" / "full_features"
RAW_DIR = REPO / "preprocessed_raw" / "full_features"
DEDUP_ENCODERS = REPO / "preprocessed" / "label_encoders.json"
RAW_CONFIG = REPO / "preprocessed_raw" / "config.json"
RAW_TEST_CSV = REPO / "eda_output_raw" / "test_cleaned.csv"
DEDUP_TEST_CSV = REPO / "eda_output" / "test_cleaned.csv"
OUT_DIR = REPO / "results" / "c1_dedup_ablation"

# Published E7 anchors (deliverables/numbers_map.md Sections 1 & 4) — cell C1-d
# should land on these; any drift is reported, never silently absorbed (DN-04).
E7_PUBLISHED = {"f1_macro": 0.907626622882394,
                "accuracy": 0.9926557939991124,
                "mcc": 0.9906169668153739}
# No multi-seed sigma exists for E7 macro-F1 in this repo. numbers_map.md's 0.023 is
# the H2-strict FUSION rescue recall sigma (README 15B.3) and is NOT a valid noise
# band for supervised macro-F1 — deliberately not used as one here.


def load_arm(name: str, base: Path) -> dict:
    if not base.exists():
        sys.exit(f"Missing {base} — run the earlier C1 steps first.")
    X_train = np.load(base / "X_train.npy", mmap_mode="r")
    y_train = pd.read_csv(base / "y_train.csv")["multiclass_label"].to_numpy()
    print(f"  [{name}] X_train {X_train.shape} | classes {len(np.unique(y_train))}")
    return {"name": name, "X_train": X_train, "y_train": y_train}


def load_unscaled_test(csv_path: Path, label_map: dict, features: list[str]) -> tuple:
    """Read a cleaned test CSV and return (feature frame, encoded labels)."""
    if not csv_path.exists():
        sys.exit(f"Missing {csv_path} — cannot build an in-space test view.")
    df = pd.read_csv(csv_path, dtype={c: np.float32 for c in features})
    missing = [c for c in features if c not in df.columns]
    if missing:
        sys.exit(f"{csv_path}: missing features {missing}")
    X = df[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isna().any().any():
        X.fillna(0.0, inplace=True)          # identical to pp.select_features
    unknown = set(df["label"].unique()) - set(label_map)
    if unknown:
        sys.exit(f"{csv_path}: labels outside the shared space: {sorted(unknown)}")
    y = df["label"].map(label_map).to_numpy()
    print(f"    {csv_path.name}: {len(X):,} rows")
    return X, y


def assert_same_label_space() -> dict:
    """INV-01 gate — refuse to cross-evaluate across different label spaces."""
    dedup_map = {k: int(v) for k, v in json.loads(
        DEDUP_ENCODERS.read_text())["multiclass"].items()}
    raw_map = {k: int(v) for k, v in json.loads(
        RAW_CONFIG.read_text())["label_mappings"]["multiclass"].items()}
    if dedup_map != raw_map:
        only_raw = set(raw_map) - set(dedup_map)
        only_dedup = set(dedup_map) - set(raw_map)
        moved = {k: (dedup_map[k], raw_map[k]) for k in set(raw_map) & set(dedup_map)
                 if dedup_map[k] != raw_map[k]}
        sys.exit("INV-01 VIOLATION — label spaces differ; cross-evaluation aborted.\n"
                 f"  only in raw: {sorted(only_raw)}\n"
                 f"  only in dedup: {sorted(only_dedup)}\n"
                 f"  moved (dedup -> raw): {moved}")
    print(f"  INV-01 OK: identical 19-class label space ({len(dedup_map)} classes)")
    return dedup_map


def train_arm(arm: dict, n_classes: int):
    """Train, or reuse a previously saved arm so re-scoring is cheap."""
    cached = OUT_DIR / "models" / f"e7_{arm['name']}_arm.ubj"
    if cached.exists():
        model = st.get_xgb("multiclass", n_classes)
        model.load_model(cached)
        print(f"  reusing cached {arm['name']} model -> {cached.name}")
        return model
    model = st.get_xgb("multiclass", n_classes)
    print(f"  training on {arm['name']} ({arm['X_train'].shape[0]:,} rows)...")
    t0 = time.time()
    model.fit(np.asarray(arm["X_train"]), arm["y_train"])
    print(f"    done in {(time.time() - t0) / 60:.1f} min")
    return model


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("C1 step 3 — 2x2 controlled raw-vs-deduplicated ablation")
    print("=" * 70)
    label_map = assert_same_label_space()
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    raw = load_arm("raw", RAW_DIR)
    dedup = load_arm("dedup", DEDUP_DIR)
    arms = {"raw": raw, "dedup": dedup}
    n_classes = len(label_map)

    models = {"raw": train_arm(raw, n_classes), "dedup": train_arm(dedup, n_classes)}
    model_dir = OUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        m.save_model(model_dir / f"e7_{name}_arm.ubj")   # re-scoring never needs retraining
    print(f"  models saved -> {model_dir}")

    # Unscaled test frames, shared by both scoring spaces.
    print("  loading unscaled test sets:")
    features = json.loads(RAW_CONFIG.read_text())["features_full"]
    X_test_unscaled = {
        "raw": load_unscaled_test(RAW_TEST_CSV, label_map, features),
        "dedup": load_unscaled_test(DEDUP_TEST_CSV, label_map, features),
    }
    scalers = {"raw": joblib.load(REPO / "preprocessed_raw" / "scaler_full.pkl"),
               "dedup": joblib.load(REPO / "preprocessed" / "scaler_full.pkl")}

    cell_ids = {("raw", "raw"): "C1-a", ("raw", "dedup"): "C1-b",
                ("dedup", "raw"): "C1-c", ("dedup", "dedup"): "C1-d"}

    cells, per_class_rows = {}, []
    for trained_on, model in models.items():
        for tested_on in ("raw", "dedup"):
            cid = cell_ids[(trained_on, tested_on)]
            X_un, y_true = X_test_unscaled[tested_on]
            # IN-SPACE RULE: transform with the SCORING MODEL'S scaler.
            X_scored = scalers[trained_on].transform(X_un).astype(np.float32, copy=False)
            print(f"  scoring {cid}: trained={trained_on}, tested={tested_on} "
                  f"({len(X_scored):,} rows, {trained_on}-arm scaler)")
            y_pred = model.predict(X_scored)
            metrics = st.evaluate(y_true, y_pred)
            ts = {"y_test": y_true, "X_test": X_scored}
            cells[cid] = {"cell": cid, "trained_on": trained_on, "tested_on": tested_on,
                          "scaler_used": f"{trained_on}-arm (in-space)",
                          "n_train_rows": int(arms[trained_on]["X_train"].shape[0]),
                          "n_test_rows": int(ts["X_test"].shape[0]), **metrics}
            rep = classification_report(ts["y_test"], y_pred, output_dict=True,
                                        zero_division=0,
                                        labels=list(range(n_classes)),
                                        target_names=class_names)
            for cls in class_names:
                per_class_rows.append({"cell": cid, "trained_on": trained_on,
                                       "tested_on": tested_on, "class": cls,
                                       "f1": rep[cls]["f1-score"],
                                       "precision": rep[cls]["precision"],
                                       "recall": rep[cls]["recall"],
                                       "support": rep[cls]["support"]})
            print(f"    macro-F1 {metrics['f1_macro']:.4f} | acc {metrics['accuracy']:.6f} "
                  f"| MCC {metrics['mcc']:.4f}")

    # --- the headline contrasts -------------------------------------------------
    def delta(a: str, b: str, key: str) -> float:
        return cells[a][key] - cells[b][key]

    # Each contrast must vary EXACTLY ONE factor. (An earlier version mislabelled
    # a-b as "training-side": a-b holds training fixed at raw and varies the TEST
    # set, so it measures the test-set effect. Training-side effects are c-a and d-b.)
    KEYS = ("accuracy", "f1_macro", "mcc")
    contrasts = {
        # vary the TEST set, hold training fixed -> metric inflation from a duplicated test set
        "test_set_effect | raw-trained (C1-a - C1-b)": {k: delta("C1-a", "C1-b", k) for k in KEYS},
        "test_set_effect | dedup-trained (C1-c - C1-d)": {k: delta("C1-c", "C1-d", k) for k in KEYS},
        # vary the TRAINING set, hold test fixed -> what duplicated training does to the model
        "training_set_effect | on raw test (C1-c - C1-a)": {k: delta("C1-c", "C1-a", k) for k in KEYS},
        "training_set_effect | on dedup test (C1-d - C1-b)": {k: delta("C1-d", "C1-b", k) for k in KEYS},
        # both factors at once: the comparison the literature actually makes
        "naive_literature_pairing (C1-a - C1-d)": {k: delta("C1-a", "C1-d", k) for k in KEYS},
    }

    # --- honesty checks ---------------------------------------------------------
    d = cells["C1-d"]
    drift = {k: d[k] - v for k, v in E7_PUBLISHED.items()}
    notes = []
    for k, v in drift.items():
        if abs(v) > 1e-6:
            notes.append(
                f"C1-d {k} = {d[k]:.6f} vs published E7 {E7_PUBLISHED[k]:.6f} "
                f"(delta {v:+.6f}) — expected: xgboost 3.2.0 is installed while the "
                "manifest pins <3.0 (known stack drift, CLAUDE.md). All four cells "
                "share this library, so the contrasts remain internally valid.")
    notes.append(
        "SEED CAVEAT (DN-05): every cell is single-seed (42). The repo has NO multi-seed "
        "sigma for E7 macro-F1 — numbers_map.md's 0.023 is the H2-strict FUSION rescue "
        "recall sigma (README 15B.3), a different metric, so it must NOT be used as a "
        "noise band for these deltas. Before P1 cites any delta as an effect, run both "
        "arms over the INV-04 seed set [1, 7, 42, 100, 1729] and report mean +- sigma.")

    # --- side result: how far deduplication moves the fitted scaler statistics ---
    def scaler_shift() -> dict:
        out = {}
        for kind in ("robust", "standard", "minmax"):
            try:
                r = scalers["raw"].named_transformers_[kind]
                d = scalers["dedup"].named_transformers_[kind]
            except (KeyError, AttributeError):
                continue
            pairs = [("center_", "center_"), ("scale_", "scale_"),
                     ("mean_", "mean_"), ("data_min_", "data_min_"),
                     ("data_max_", "data_max_")]
            for attr, _ in pairs:
                if not (hasattr(r, attr) and hasattr(d, attr)):
                    continue
                rv, dv = np.asarray(getattr(r, attr), float), np.asarray(getattr(d, attr), float)
                if rv.shape != dv.shape:
                    continue
                denom = np.where(np.abs(dv) > 0, np.abs(dv), np.nan)
                rel = np.abs(rv - dv) / denom
                out[f"{kind}.{attr}"] = {
                    "n_params": int(rv.size),
                    "n_differing": int(np.sum(~np.isclose(rv, dv))),
                    "max_rel_diff": (None if np.all(np.isnan(rel))
                                     else float(np.nanmax(rel))),
                    "median_rel_diff": (None if np.all(np.isnan(rel))
                                        else float(np.nanmedian(rel))),
                }
        return out

    shift = scaler_shift()
    n_diff = sum(v["n_differing"] for v in shift.values())
    if n_diff:
        notes.append(
            f"Deduplication moves the fitted preprocessing statistics themselves: "
            f"{n_diff} scaler parameters differ between the arms. Leakage is not only "
            "about rows seen twice — the duplicate mass distorts every statistic fitted "
            "on the training data. This is why each cell must be scored in its own space.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "C1 — controlled raw-vs-deduplicated ablation",
        "authorized_by": "Amendment 1, .claude/plans/2026-07-30-p1-dedup-paper-brief.md",
        "model": "E7 (XGBClassifier, FULL-44, no resampling)",
        "xgb_params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                       for k, v in st.XGB_PARAMS_BASE.items()},
        "seed": 42,
        "seed_sigma_reference": None,  # see notes: no E7 macro-F1 seed sigma exists
        "cells": cells, "contrasts": contrasts,
        "scaler_shift_raw_vs_dedup": shift,
        "c1d_vs_published_e7": {"published": E7_PUBLISHED, "drift": drift},
        "notes": notes,
        "runtime_min": round((time.time() - t0) / 60, 2),
    }
    (OUT_DIR / "c1_matrix.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(per_class_rows).to_csv(OUT_DIR / "c1_per_class_f1.csv", index=False)

    print("\n" + "=" * 70)
    print(f"{'cell':6s}{'trained':>10}{'tested':>10}{'macro-F1':>12}{'accuracy':>12}{'MCC':>10}")
    for cid in ("C1-a", "C1-b", "C1-c", "C1-d"):
        c = cells[cid]
        print(f"{cid:6s}{c['trained_on']:>10}{c['tested_on']:>10}"
              f"{c['f1_macro']:>12.4f}{c['accuracy']:>12.6f}{c['mcc']:>10.4f}")
    print("\ncontrasts:")
    for k, v in contrasts.items():
        print(f"  {k}: macro-F1 {v['f1_macro']:+.4f} | acc {v['accuracy']:+.6f} "
              f"| MCC {v['mcc']:+.4f}")
    for n in notes:
        print(f"\nNOTE: {n}")
    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
