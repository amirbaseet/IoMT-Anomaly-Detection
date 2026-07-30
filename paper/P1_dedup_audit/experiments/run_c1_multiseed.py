"""
C1 step 4 — multi-seed the 2x2 ablation over the INV-04 seed set.

run_c1_matrix.py established the four cells at seed 42. DN-05 forbids presenting a
single-seed result as robust, and this repo has no multi-seed sigma for supervised
macro-F1, so the C1 deltas had no noise band. This script supplies one: both arms
are retrained at every seed in INV-04's set [1, 7, 42, 100, 1729] and every cell
and contrast is reported as mean +- sigma.

Only the MODEL seed varies. The preprocessing (scalers, 80/20 train/val split) is
fixed at RANDOM_STATE=42 in both arms, exactly as the published pipeline does it,
so the seed sweep measures classifier stochasticity (subsample=0.8,
colsample_bytree=0.8) and not split churn. Seed 42's cells must reproduce
run_c1_matrix.py's numbers exactly — asserted, not assumed.

Reuses run_c1_matrix's loaders and INV-01 gate by import (that module is guarded
by __main__, so importing it runs no work).

Outputs -> results/c1_dedup_ablation/c1_multiseed.json
Per-seed models are NOT persisted (5 x 2 x ~100 MB of disk for no analytical gain);
seed 42's two models already sit in results/c1_dedup_ablation/models/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib   # first-party scaler artifacts only (see run_c1_matrix.py)
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import notebooks.supervised_training as st          # noqa: E402
import run_c1_matrix as c1                          # noqa: E402

SEEDS = [1, 7, 42, 100, 1729]                       # INV-04 main-pipeline seed set
METRICS = ("accuracy", "f1_macro", "mcc")
CELL_IDS = {("raw", "raw"): "C1-a", ("raw", "dedup"): "C1-b",
            ("dedup", "raw"): "C1-c", ("dedup", "dedup"): "C1-d"}
OUT_DIR = REPO / "results" / "c1_dedup_ablation"

# Seed-42 anchors from run_c1_matrix.py — this sweep must reproduce them.
SEED42_ANCHORS = {
    "C1-a": {"f1_macro": 0.8842, "accuracy": 0.993791},
    "C1-b": {"f1_macro": 0.8760, "accuracy": 0.989221},
    "C1-c": {"f1_macro": 0.9158, "accuracy": 0.995867},
    "C1-d": {"f1_macro": 0.9076, "accuracy": 0.992656},
}

CONTRASTS = {
    "test_set_effect | raw-trained": ("C1-a", "C1-b"),
    "test_set_effect | dedup-trained": ("C1-c", "C1-d"),
    "training_set_effect | on raw test": ("C1-c", "C1-a"),
    "training_set_effect | on dedup test": ("C1-d", "C1-b"),
    "naive_literature_pairing": ("C1-a", "C1-d"),
}


def mean_sd(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print(f"C1 step 4 — multi-seed 2x2 over seeds {SEEDS}")
    print("=" * 70)

    label_map = c1.assert_same_label_space()
    n_classes = len(label_map)
    arms = {"raw": c1.load_arm("raw", c1.RAW_DIR),
            "dedup": c1.load_arm("dedup", c1.DEDUP_DIR)}

    # Test views are seed-independent — build them once (this is the big time saving).
    print("  building in-space test views once:")
    features = json.loads(c1.RAW_CONFIG.read_text())["features_full"]
    unscaled = {"raw": c1.load_unscaled_test(c1.RAW_TEST_CSV, label_map, features),
                "dedup": c1.load_unscaled_test(c1.DEDUP_TEST_CSV, label_map, features)}
    scalers = {"raw": joblib.load(REPO / "preprocessed_raw" / "scaler_full.pkl"),
               "dedup": joblib.load(REPO / "preprocessed" / "scaler_full.pkl")}
    views = {}
    for trained_on in ("raw", "dedup"):
        for tested_on in ("raw", "dedup"):
            X_un, y_true = unscaled[tested_on]
            views[(trained_on, tested_on)] = (
                scalers[trained_on].transform(X_un).astype(np.float32, copy=False), y_true)
    del unscaled

    per_seed: dict[int, dict] = {}
    for seed in SEEDS:
        print(f"\n--- seed {seed} " + "-" * 50)
        cells = {}
        for trained_on, arm in arms.items():
            model = st.get_xgb("multiclass", n_classes)
            model.set_params(random_state=seed)
            assert model.get_params()["random_state"] == seed
            print(f"  training {trained_on} ({arm['X_train'].shape[0]:,} rows, seed {seed})...")
            ts = time.time()
            model.fit(np.asarray(arm["X_train"]), arm["y_train"])
            print(f"    {(time.time() - ts) / 60:.1f} min")
            for tested_on in ("raw", "dedup"):
                X_scored, y_true = views[(trained_on, tested_on)]
                cid = CELL_IDS[(trained_on, tested_on)]
                cells[cid] = st.evaluate(y_true, model.predict(X_scored))
                print(f"    {cid} macro-F1 {cells[cid]['f1_macro']:.4f} "
                      f"| acc {cells[cid]['accuracy']:.6f} | MCC {cells[cid]['mcc']:.4f}")
            del model
        per_seed[seed] = cells

    # Reproduction gate: seed 42 must match run_c1_matrix.py.
    drift = {}
    for cid, anchors in SEED42_ANCHORS.items():
        for k, expected in anchors.items():
            got = per_seed[42][cid][k]
            if abs(got - expected) > 5e-5:
                drift[f"{cid}.{k}"] = {"expected": expected, "got": got}
    if drift:
        print(f"\n  WARNING: seed-42 cells did not reproduce run_c1_matrix.py: {drift}")
    else:
        print("\n  seed-42 cells reproduce run_c1_matrix.py exactly.")

    agg_cells = {cid: {k: mean_sd([per_seed[s][cid][k] for s in SEEDS]) for k in METRICS}
                 for cid in CELL_IDS.values()}
    agg_contrasts = {
        name: {k: mean_sd([per_seed[s][a][k] - per_seed[s][b][k] for s in SEEDS])
               for k in METRICS}
        for name, (a, b) in CONTRASTS.items()}

    # A contrast is separable from seed noise if its mean exceeds ~2 sigma AND every
    # seed agrees on the sign. Both conditions are reported, never just one.
    verdicts = {}
    for name, (a, b) in CONTRASTS.items():
        v = {}
        for k in METRICS:
            st_ = agg_contrasts[name][k]
            per = [per_seed[s][a][k] - per_seed[s][b][k] for s in SEEDS]
            same_sign = all(x > 0 for x in per) or all(x < 0 for x in per)
            v[k] = {"mean": st_["mean"], "sd": st_["sd"],
                    "sign_consistent_across_seeds": same_sign,
                    "abs_mean_over_2sd": (None if st_["sd"] == 0
                                          else abs(st_["mean"]) / (2 * st_["sd"])),
                    "separable": bool(same_sign and st_["sd"] > 0
                                      and abs(st_["mean"]) > 2 * st_["sd"])}
        verdicts[name] = v

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "C1 step 4 — multi-seed 2x2 (INV-04 seed set)",
        "seeds": SEEDS,
        "what_varies": "XGBoost random_state only; preprocessing fixed at RANDOM_STATE=42",
        "model": "E7 (XGBClassifier, FULL-44, no resampling)",
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "cells_mean_sd": agg_cells,
        "contrasts_mean_sd": agg_contrasts,
        "separability": verdicts,
        "seed42_reproduction_drift": drift,
        "runtime_min": round((time.time() - t0) / 60, 2),
    }
    (OUT_DIR / "c1_multiseed.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 70)
    print(f"{'cell':6s}{'macro-F1 mean+-sd':>26}{'accuracy mean+-sd':>28}")
    for cid in ("C1-a", "C1-b", "C1-c", "C1-d"):
        f, a = agg_cells[cid]["f1_macro"], agg_cells[cid]["accuracy"]
        print(f"{cid:6s}{f['mean']:>16.4f} +-{f['sd']:.4f}"
              f"{a['mean']:>18.6f} +-{a['sd']:.6f}")
    print("\ncontrasts (mean +- sd over 5 seeds):")
    for name in CONTRASTS:
        f = agg_contrasts[name]["f1_macro"]
        a = agg_contrasts[name]["accuracy"]
        sep = verdicts[name]["f1_macro"]
        print(f"  {name}")
        print(f"    macro-F1 {f['mean']:+.4f} +-{f['sd']:.4f} | acc {a['mean']:+.6f} "
              f"+-{a['sd']:.6f} | sign-consistent={sep['sign_consistent_across_seeds']} "
              f"| separable={sep['separable']}")
    print(f"\nsaved -> {OUT_DIR / 'c1_multiseed.json'}  ({out['runtime_min']} min)")


if __name__ == "__main__":
    main()
