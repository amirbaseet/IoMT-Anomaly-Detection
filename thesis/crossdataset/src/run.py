"""
Orchestrator for the ShieldMind cross-dataset generalization test.

    python run.py            # full run (7M-row train + 8.5GB test, multi-seed)
    python run.py --smoke    # end-to-end validation on a tiny slice (minutes)

XGBoost (Layer 1) trains in this process; the Flow AE (Layer 2) trains + scores
in an isolated ``ae_worker.py`` subprocess, because TensorFlow and XGBoost
deadlock when co-resident (duelling OpenMP runtimes).  Data crosses the process
boundary as .npy matrices in the artifact dir.

Flow:
  1. Shared-feature intersection (+ provenance JSON).
  2. Load CICIoMT2024 train; fit imputer (train only, TRAP 1).
  3. Train shared-feature XGBoost (binary + 5-family).
  4. Impute + persist: benign-train, CICIoMT2024 test, each CICIoT2023 seed.
  5. AE subprocess: train on benign, score every persisted matrix -> MSE npy.
  6. Evaluate Designs A & B (in-dataset + per seed); aggregate mean ± σ.
  7. Emit results_summary.md, confusion_A/B.csv, sampled_test_set.parquet,
     shared_features.json, raw_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config          # noqa: E402
import ciciomt_data    # noqa: E402
import ciciot_sampler  # noqa: E402
import features        # noqa: E402
import labels          # noqa: E402
import metrics         # noqa: E402
import models          # noqa: E402
import report          # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Evaluation core (XGBoost in-process; AE reconstruction error precomputed)
# --------------------------------------------------------------------------- #
def _eval(x_imp, raw_labels, xgb_bin, xgb_fam, ae_mse, ae_thresholds) -> dict:
    # ---- Design A: XGBoost -------------------------------------------------
    y_bin = np.array(
        [0 if labels.ciciot2023_binary(r) == "benign" else 1 for r in raw_labels],
        dtype=np.int32)
    proba = xgb_bin.predict_proba(x_imp)[:, 1]
    a_xgb = metrics.binary_metrics(y_bin, (proba >= 0.5).astype(int), proba)

    # ---- Design A: Flow AE (thresholds set on CICIoMT2024 benign-val) ------
    a_ae = {}
    for tname, thr in ae_thresholds.items():
        a_ae[tname] = metrics.binary_metrics(y_bin, (ae_mse > thr).astype(int), ae_mse)
    a_ae["roc_auc"] = a_ae[next(iter(ae_thresholds))]["roc_auc"]

    # ---- Design B: 5 families (drop out-of-scope) --------------------------
    fam = np.array([labels.ciciot2023_family(r) for r in raw_labels], dtype=object)
    keep = np.array([f is not None for f in fam])
    y_fam = np.array([models.FAMILY_TO_INT[f] for f in fam[keep]], dtype=np.int32)
    pred_fam = xgb_fam.predict(x_imp[keep]).astype(np.int32)
    b = metrics.family_metrics(y_fam, pred_fam)
    return {"a_xgb": a_xgb, "a_ae": a_ae, "b": b}


def _raw_from_ciciomt(y_binary, y_family) -> np.ndarray:
    """Represent CICIoMT2024 labels as raw strings the ciciot mappers accept,
    so the in-dataset baseline runs through the identical eval core."""
    proxy = {labels.SPOOFING: "MITM-ARPSPOOFING", labels.RECON: "RECON-OSSCAN",
             labels.DOS: "DOS-SYN_FLOOD", labels.DDOS: "DDOS-SYN_FLOOD"}
    raw = []
    for b, f in zip(y_binary, y_family):
        if b == "benign":
            raw.append("BENIGN")
        else:
            raw.append(proxy.get(f, "MIRAI-UDPPLAIN"))  # MQTT -> attack, B-dropped
    return np.array(raw, dtype=object)


def _aggregate(seed_results: list[dict]) -> dict:
    a_xgb = {m: metrics.aggregate_scalar([r["a_xgb"] for r in seed_results], m)
             for m in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    thr_names = [k for k in seed_results[0]["a_ae"] if k != "roc_auc"]
    a_ae = {
        thr: {m: metrics.aggregate_scalar([r["a_ae"][thr] for r in seed_results], m)
              for m in ("f1", "recall", "roc_auc")}
        for thr in thr_names
    }
    b = {
        "macro_f1": metrics.aggregate_scalar([r["b"] for r in seed_results], "macro_f1"),
        "accuracy": metrics.aggregate_scalar([r["b"] for r in seed_results], "accuracy"),
        "per_family": {
            fam: {
                "f1": metrics.aggregate_scalar(
                    [r["b"]["per_family"][fam] for r in seed_results], "f1"),
                "recall": metrics.aggregate_scalar(
                    [r["b"]["per_family"][fam] for r in seed_results], "recall"),
            }
            for fam in labels.FAMILIES
        },
    }
    return {"design_a_xgb": a_xgb, "design_a_ae": a_ae, "design_b": b}


def _run_ae_worker(job_dir: Path, benign_npy: Path, score_npy: dict) -> dict:
    """Train + score the Flow AE in an isolated subprocess; return thresholds."""
    spec = {"benign_train_npy": str(benign_npy),
            "score_npy": {k: str(v) for k, v in score_npy.items()},
            "out_dir": str(job_dir), "epochs": config.AE_EPOCHS}
    spec_path = job_dir / "jobspec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONUNBUFFERED": "1"}
    log("Launching isolated AE worker (TensorFlow) ...")
    subprocess.run(
        [sys.executable, "-u", str(Path(__file__).parent / "ae_worker.py"),
         str(spec_path)],
        check=True, cwd=str(Path(__file__).parent), env=env,
    )
    meta = json.loads((job_dir / "ae_meta.json").read_text())
    return meta["thresholds"]


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end validation run")
    args = ap.parse_args()

    out, art = config.OUT_DIR, config.ARTIFACT_DIR
    job_dir = art / "aejob"
    for d in (out, art, job_dir):
        d.mkdir(parents=True, exist_ok=True)

    nrows = 40_000 if args.smoke else None
    seeds = (42, 43) if args.smoke else config.SAMPLING_SEEDS
    canonical_seed = seeds[0]
    if args.smoke:
        config.AE_EPOCHS = 3
        config.BENIGN_CAP, config.ATTACK_CAP = 3_000, 1_500

    # ---- 1. Feature intersection ------------------------------------------
    log("Computing shared-feature intersection ...")
    ciciomt_header = features.read_header(
        sorted(config.CICIOMT2024_TRAIN_DIR.glob("*.csv"))[0])
    ciciot_header = features.read_header(
        sorted(config.CICIOT2023_MERGED_DIR.glob("*.csv"))[0])
    shared = features.compute_shared_features(
        ciciomt_header, ciciot_header, config.CICIOT2023_LABEL_COL)
    dropped = [c for c in ciciomt_header if c not in shared]
    features.save_shared_features(
        shared, ciciomt_header, ciciot_header, dropped,
        list(config.CICIOT2023_ONLY), out / "shared_features.json")
    log(f"  {len(shared)} shared features; dropped {len(dropped)}: {dropped}")

    # ---- 2. Load CICIoMT2024 train + fit imputer --------------------------
    log("Loading CICIoMT2024 train ...")
    train = ciciomt_data.load_split(config.CICIOMT2024_TRAIN_DIR, shared, nrows=nrows)
    log(f"  train rows: {len(train['X']):,}")
    imp = models.fit_imputer(train["X"])

    def impute(df):
        return models.apply_imputer(imp, df).astype("float32")

    # ---- 3. Train shared-feature XGBoost ----------------------------------
    log("Training XGBoost (binary) ...")
    xgb_bin = models.train_xgb_binary(impute(train["X"]), train["y_binary"])
    log("Training XGBoost (5-family, MQTT dropped) ...")
    train_b = ciciomt_data.design_b_subset(train)
    xgb_fam = models.train_xgb_family(impute(train_b["X"]), train_b["y_family"])

    # ---- 4. Persist benign-train (for AE) ---------------------------------
    benign_mask = train["y_binary"] == "benign"
    benign_npy = job_dir / "benign_train.npy"
    np.save(benign_npy, impute(train["X"].loc[benign_mask].reset_index(drop=True)))
    del train, train_b

    # In-dataset baseline features (CICIoMT2024 test).
    log("Loading CICIoMT2024 test (in-dataset baseline) ...")
    test = ciciomt_data.load_split(config.CICIOMT2024_TEST_DIR, shared, nrows=nrows)
    ind_raw = _raw_from_ciciomt(test["y_binary"], test["y_family"])
    ind_x = impute(test["X"])
    np.save(job_dir / "indataset.npy", ind_x)
    del test

    # ---- 5. Sample CICIoT2023 (3 seeds, single feature pass) --------------
    files = ciciot_sampler.merged_files(config.CICIOT2023_MERGED_DIR)
    if args.smoke:
        files = files[:1]
    cache = art / ("label_counts_smoke.json" if args.smoke else "label_counts.json")
    log(f"Counting CICIoT2023 labels over {len(files)} file(s) ...")
    counts = ciciot_sampler.count_labels(
        files, config.CICIOT2023_LABEL_COL, config.SAMPLE_CHUNK_ROWS, cache)
    caps = ciciot_sampler.build_caps(counts, config.BENIGN_CAP, config.ATTACK_CAP)
    unknown = {lab: n for lab, n in counts.items()
               if not labels.is_known_ciciot2023(lab)}
    log(f"  {len(counts)} distinct labels; {sum(counts.values()):,} total rows")
    if unknown:
        log(f"  excluded {sum(unknown.values()):,} rows with unrecognised/"
            f"malformed labels: { {k or '<empty>': v for k, v in unknown.items()} }")

    log("Sampling CICIoT2023 (single feature-read pass) ...")
    samples = ciciot_sampler.sample_all_seeds(
        files, shared, config.CICIOT2023_LABEL_COL, counts, caps, seeds,
        config.SAMPLE_CHUNK_ROWS)
    samples[canonical_seed].to_parquet(out / "sampled_test_set.parquet", index=False)

    seed_x, seed_raw, score_npy = {}, {}, {"indataset": job_dir / "indataset.npy"}
    for s in seeds:
        df = samples[s]
        seed_x[s] = impute(features.align_to_schema(df, shared))
        seed_raw[s] = df["raw_label"].to_numpy()
        p = job_dir / f"seed_{s}.npy"
        np.save(p, seed_x[s])
        score_npy[f"seed_{s}"] = p
        log(f"  seed {s}: {len(df):,} sampled flows")
    del samples

    # ---- 6. AE subprocess: train on benign, score everything --------------
    thresholds = _run_ae_worker(job_dir, benign_npy, score_npy)
    log(f"  AE thresholds: {thresholds}")
    ind_mse = np.load(job_dir / "mse_indataset.npy")
    seed_mse = {s: np.load(job_dir / f"mse_seed_{s}.npy") for s in seeds}

    # ---- 7. Evaluate ------------------------------------------------------
    ind_eval = _eval(ind_x, ind_raw, xgb_bin, xgb_fam, ind_mse, thresholds)
    in_dataset = {
        "design_a_xgb": ind_eval["a_xgb"],
        "design_a_ae": {**{t: ind_eval["a_ae"][t] for t in thresholds},
                        "roc_auc": ind_eval["a_ae"]["roc_auc"]},
        "design_b": ind_eval["b"],
    }

    seed_results, canonical = [], None
    for s in seeds:
        res = _eval(seed_x[s], seed_raw[s], xgb_bin, xgb_fam, seed_mse[s], thresholds)
        seed_results.append(res)
        if s == canonical_seed:
            canonical = res
        log(f"  seed {s}: A-XGB F1={res['a_xgb']['f1']:.4f} "
            f"AUC={res['a_xgb']['roc_auc']:.4f} | B macro-F1={res['b']['macro_f1']:.4f}")
    cross = _aggregate(seed_results)

    # ---- 8. Deliverables --------------------------------------------------
    report.write_confusion_binary(canonical["a_xgb"]["confusion"], out / "confusion_A.csv")
    report.write_confusion_family(canonical["b"]["confusion"], out / "confusion_B.csv")
    ctx = {
        "n_shared": len(shared), "n_dropped": len(dropped),
        "dropped_features": dropped, "canonical_seed": canonical_seed,
        "seeds": list(seeds),
        "b_support": {fam: canonical["b"]["per_family"][fam]["support"]
                      for fam in labels.FAMILIES},
        "in_dataset": in_dataset, "cross": cross,
    }
    report.write_summary(ctx, out / "results_summary.md")
    (out / "raw_results.json").write_text(json.dumps(
        {"in_dataset": in_dataset, "cross": cross, "per_seed": seed_results,
         "counts": counts, "caps": caps, "seeds": list(seeds),
         "shared_features": shared, "ae_thresholds": thresholds},
        indent=2, default=str))
    log(f"DONE. Deliverables in {out}")


if __name__ == "__main__":
    main()
