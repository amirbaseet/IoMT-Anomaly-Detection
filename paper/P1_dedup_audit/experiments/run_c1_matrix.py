"""
C1 step 3 — the 2x2 controlled ablation.

                      tested on RAW      tested on DEDUP
   trained on RAW        C1-a                C1-b
   trained on DEDUP      C1-c                C1-d   (= the published E7 arm)

C1-a - C1-d is the matched replacement for the cross-paper 99.80 -> 99.27
comparison. C1-b isolates training-side memorization (leaky training, honest
test); C1-c isolates test-side redundancy inflation (clean training, leaky test).

Both models are E7 exactly: XGBClassifier with XGB_PARAMS_BASE from
notebooks/supervised_training.py, FULL-44 features, NO resampling,
random_state=42, 19-class multiclass. The model factory and the metric function
are IMPORTED from that module so neither arm can drift from the published one.

Single seed (42) on purpose: this is a like-for-like comparison against E7, not
a stability claim. Deltas are reported against the known 5-seed sigma = 0.023
(numbers_map.md) and a delta inside that band is reported as such.

Inputs  -> preprocessed/full_features/      (dedup arm, existing)
           preprocessed_raw/full_features/  (raw arm, run_raw_preprocessing.py)
Outputs -> results/c1_dedup_ablation/{c1_matrix.json, c1_per_class_f1.csv}
Nothing existing is overwritten.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
OUT_DIR = REPO / "results" / "c1_dedup_ablation"

# Published E7 anchors (deliverables/numbers_map.md Sections 1 & 4) — cell C1-d
# should land on these; any drift is reported, never silently absorbed (DN-04).
E7_PUBLISHED = {"f1_macro": 0.907626622882394,
                "accuracy": 0.9926557939991124,
                "mcc": 0.9906169668153739}
SEED_SIGMA = 0.023  # 5-seed sigma, numbers_map.md


def load_arm(name: str, base: Path) -> dict:
    if not base.exists():
        sys.exit(f"Missing {base} — run the earlier C1 steps first.")
    X_train = np.load(base / "X_train.npy", mmap_mode="r")
    X_test = np.load(base / "X_test.npy", mmap_mode="r")
    y_train = pd.read_csv(base / "y_train.csv")["multiclass_label"].to_numpy()
    y_test = pd.read_csv(base / "y_test.csv")["multiclass_label"].to_numpy()
    print(f"  [{name}] X_train {X_train.shape} | X_test {X_test.shape} | "
          f"classes {len(np.unique(y_train))}")
    return {"name": name, "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test}


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
    n_classes = len(label_map)

    models = {"raw": train_arm(raw, n_classes), "dedup": train_arm(dedup, n_classes)}
    test_sets = {"raw": raw, "dedup": dedup}
    cell_ids = {("raw", "raw"): "C1-a", ("raw", "dedup"): "C1-b",
                ("dedup", "raw"): "C1-c", ("dedup", "dedup"): "C1-d"}

    cells, per_class_rows = {}, []
    for trained_on, model in models.items():
        for tested_on, ts in test_sets.items():
            cid = cell_ids[(trained_on, tested_on)]
            print(f"  scoring {cid}: trained={trained_on}, tested={tested_on} "
                  f"({ts['X_test'].shape[0]:,} rows)")
            y_pred = model.predict(np.asarray(ts["X_test"]))
            metrics = st.evaluate(ts["y_test"], y_pred)
            cells[cid] = {"cell": cid, "trained_on": trained_on, "tested_on": tested_on,
                          "n_train_rows": int(test_sets[trained_on]["X_train"].shape[0]),
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

    contrasts = {
        "headline_raw_vs_dedup (C1-a - C1-d)": {
            k: delta("C1-a", "C1-d", k) for k in ("accuracy", "f1_macro", "mcc")},
        "training_side_memorization (C1-a - C1-b)": {
            k: delta("C1-a", "C1-b", k) for k in ("accuracy", "f1_macro", "mcc")},
        "test_side_inflation (C1-c - C1-d)": {
            k: delta("C1-c", "C1-d", k) for k in ("accuracy", "f1_macro", "mcc")},
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
    hl = contrasts["headline_raw_vs_dedup (C1-a - C1-d)"]["f1_macro"]
    if abs(hl) < SEED_SIGMA:
        notes.append(
            f"headline macro-F1 delta {hl:+.4f} is INSIDE the 5-seed sigma band "
            f"({SEED_SIGMA}) — report as 'not separable from seed noise at one seed', "
            "not as a measured effect (DN-05 spirit).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "C1 — controlled raw-vs-deduplicated ablation",
        "authorized_by": "Amendment 1, .claude/plans/2026-07-30-p1-dedup-paper-brief.md",
        "model": "E7 (XGBClassifier, FULL-44, no resampling)",
        "xgb_params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                       for k, v in st.XGB_PARAMS_BASE.items()},
        "seed": 42, "seed_sigma_reference": SEED_SIGMA,
        "cells": cells, "contrasts": contrasts,
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
