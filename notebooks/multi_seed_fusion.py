#!/usr/bin/env python3
"""
Path B Week 1 — Per-Seed Enhanced Fusion Ablation
==================================================

For each seed in {1, 7, 42, 100, 1729}, re-runs the Phase 6C fusion ablation
against that seed's LOO predictions. The 11 fusion variants, the H2-strict
criterion (threshold 0.70, 4 eligible targets), the entropy/ensemble
calibration, and all formulas are IDENTICAL to notebooks/enhanced_fusion.py.

The only thing that varies across seeds is the LOO prediction array. The AE,
IF, E7, label maps, and entropy/ensemble thresholds are all seed-invariant.

Inputs (per seed):
    results/zero_day_loo/multi_seed/seed_<S>/predictions/loo_<TARGET>_test_*.npy

Inputs (shared, seed-invariant):
    results/supervised/predictions/E7_{val,test}_proba.npy   (entropy/conf calibration)
    results/unsupervised/scores/{ae,if}_{val,test}_*.npy
    results/unsupervised/thresholds.json
    results/zero_day_loo/models/loo_label_map_<TARGET>.json
    preprocessed/full_features/y_{val,test}.csv
    preprocessed/label_encoders.json

Outputs:
    results/enhanced_fusion/multi_seed/seed_<S>/metrics/ablation_table.csv
    results/enhanced_fusion/multi_seed/seed_<S>/metrics/per_target_results.csv

Tripwire: after seed=42 completes, asserts that entropy_benign_p95
strict_avg matches the canonical Phase 6C run (0.8035264623662012 ± 1e-9).
If this fails, the bug is in this script, not in XGBoost — abort immediately.

Usage:
    cd ~/IoMT-Project && source venv/bin/activate
    python -u notebooks/multi_seed_fusion.py
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Constants — duplicated VERBATIM from enhanced_fusion.py ----------------
SUPERVISED_DIR    = Path("./results/supervised/")
UNSUPERVISED_DIR  = Path("./results/unsupervised/")
LOO_CANONICAL_DIR = Path("./results/zero_day_loo/")
PREPROCESSED_DIR  = Path("./preprocessed/")
OUTPUT_DIR        = Path("./results/enhanced_fusion/multi_seed/")

ZERO_DAY_TARGETS = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "MQTT_DoS_Connect_Flood",
    "ARP_Spoofing",
]

H2_STRICT_ELIGIBLE = [
    "Recon_Ping_Sweep",
    "Recon_VulScan",
    "MQTT_Malformed_Data",
    "ARP_Spoofing",
]
H2_STRICT_MIN_BENIGN_N = 30
H2_PASS_THRESHOLD      = 0.70

CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
ENTROPY_PERCENTILES   = [90, 95, 97, 99]
ENSEMBLE_PERCENTILES  = [90, 95, 99]

SEEDS = [1, 7, 42, 100, 1729]

# Seed-42 reference values from results/enhanced_fusion/metrics/ablation_table.csv
SEED42_REFERENCE_STRICT_AVG = 0.8035264623662012
SEED42_REFERENCE_TOLERANCE  = 1e-9


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- Helpers (copied from enhanced_fusion.py) -------------------------------
def load_labels_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).values
    for col in ("label", "class", "y", "target", "Label"):
        if col in df.columns:
            return df[col].astype(str).values
    return df.iloc[:, 0].astype(str).values


def extract_global_class_map(encoders: dict) -> Dict[str, int]:
    if all(isinstance(v, int) for v in encoders.values()):
        return {str(k): int(v) for k, v in encoders.items()}
    for key in ("label", "y", "class", "multiclass", "target"):
        if key in encoders and isinstance(encoders[key], dict):
            sub = encoders[key]
            if all(isinstance(v, int) for v in sub.values()):
                return {str(k): int(v) for k, v in sub.items()}
            if "classes_" in sub:
                return {str(c): i for i, c in enumerate(sub["classes_"])}
            if "mapping" in sub:
                return {str(k): int(v) for k, v in sub["mapping"].items()}
    if "classes_" in encoders:
        return {str(c): i for i, c in enumerate(encoders["classes_"])}
    raise ValueError(f"Could not extract global class map; keys={list(encoders.keys())[:10]}")


def extract_fold_classes(target: str) -> List[str]:
    """Read the canonical seed-42 sidecar JSON. Seed-invariant."""
    sidecar = LOO_CANONICAL_DIR / "models" / f"loo_label_map_{target}.json"
    with open(sidecar) as f:
        m = json.load(f)
    return [c for c, _ in sorted(m.items(), key=lambda kv: int(kv[1]))]


def get_ae_threshold(d: dict, pct: int) -> float:
    for key in (f"p{pct}", f"P{pct}", str(pct), f"q{pct}", f"percentile_{pct}"):
        if key in d:
            return float(d[key])
    for sub in d.values():
        if isinstance(sub, dict):
            for key in (f"p{pct}", str(pct)):
                if key in sub:
                    return float(sub[key])
    raise KeyError(f"AE p{pct} threshold not found")


def compute_entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-10, 1.0)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)


# ---- Fusion variant functions (copied verbatim) -----------------------------
def baseline_fusion(sup_pred, ae_binary, benign_id):
    sup_attack = (sup_pred != benign_id)
    return np.where(sup_attack &  ae_binary, 1,
           np.where(~sup_attack &  ae_binary, 2,
           np.where(sup_attack & ~ae_binary, 3, 4)))


def confidence_fusion(sup_pred, ae_binary, confidence, conf_threshold, benign_id):
    sup_attack = (sup_pred != benign_id)
    uncertain  = (confidence < conf_threshold)
    return np.where( sup_attack &  ae_binary & ~uncertain, 1,
           np.where(~sup_attack &  ae_binary,              2,
           np.where( uncertain  &  ae_binary,              2,
           np.where( sup_attack & ~ae_binary & ~uncertain, 3,
           np.where( uncertain  & ~ae_binary,              5,
                                                            4)))))


def entropy_fusion(sup_pred, ae_binary, entropy, ent_threshold, benign_id):
    sup_attack    = (sup_pred != benign_id)
    high_entropy  = (entropy > ent_threshold)
    return np.where( sup_attack &  ae_binary & ~high_entropy, 1,
           np.where(~sup_attack &  ae_binary,                 2,
           np.where( high_entropy &  ae_binary,               2,
           np.where( high_entropy & ~ae_binary,               5,
           np.where( sup_attack & ~ae_binary & ~high_entropy, 3,
                                                              4)))))


def full_enhanced_fusion(sup_pred, ae_binary, confidence, entropy,
                         conf_threshold, ent_threshold, benign_id):
    sup_attack = (sup_pred != benign_id)
    suspicious = (confidence < conf_threshold) | (entropy > ent_threshold)
    return np.where( sup_attack &  ae_binary & ~suspicious, 1,
           np.where((~sup_attack | suspicious) &  ae_binary, 2,
           np.where( suspicious & ~ae_binary,                5,
           np.where( sup_attack & ~ae_binary & ~suspicious,  3,
                                                              4))))


# ---- The 11 variants — copied verbatim from enhanced_fusion.py:542-565 ------
VARIANTS: List[Tuple[str, str, dict]] = [
    ("baseline_ae_p90", "Baseline (Phase 6, AE p90)",
        dict(family="baseline", ae="p90")),
    ("baseline_ae_p95", "Baseline (AE p95)",
        dict(family="baseline", ae="p95")),
    ("confidence_0.6", "Confidence floor (τ=0.6)",
        dict(family="confidence", tau=0.6, ae="p90")),
    ("confidence_0.7", "Confidence floor (τ=0.7)",
        dict(family="confidence", tau=0.7, ae="p90")),
    ("entropy_benign_p90", "Entropy (benign-val p90)",
        dict(family="entropy", ent="ent_p90", ae="p90")),
    ("entropy_benign_p95", "Entropy (benign-val p95)",
        dict(family="entropy", ent="ent_p95", ae="p90")),
    ("entropy_benign_p99", "Entropy (benign-val p99)",
        dict(family="entropy", ent="ent_p99", ae="p90")),
    ("ensemble_p90", "Ensemble AE+IF (p90)",
        dict(family="baseline", ae="ens_p90")),
    ("ensemble_p95", "Ensemble AE+IF (p95)",
        dict(family="baseline", ae="ens_p95")),
    ("conf07_ent_p95", "Confidence + Entropy (τ=0.7, benign p95)",
        dict(family="full", tau=0.7, ent="ent_p95", ae="p90")),
    ("full_enhanced", "Full enhanced (conf+ent+ensemble)",
        dict(family="full", tau=0.7, ent="ent_p95", ae="ens_p90")),
]

DETECTED_CASES = (1, 2, 3, 5)


def case_distribution(cases: np.ndarray) -> Dict[str, float]:
    return {f"case{c}_pct": float((cases == c).mean()) for c in (1, 2, 3, 4, 5)}


# ---- Shared (seed-invariant) state — loaded once ----------------------------
def load_shared_state() -> dict:
    log("Loading shared (seed-invariant) state ...")
    with open(PREPROCESSED_DIR / "label_encoders.json") as f:
        encoders = json.load(f)
    global_class_map = extract_global_class_map(encoders)
    benign_id = global_class_map["Benign"]
    n_classes_global = len(global_class_map)

    e7_test_proba = np.load(SUPERVISED_DIR / "predictions" / "E7_test_proba.npy")
    e7_val_proba  = np.load(SUPERVISED_DIR / "predictions" / "E7_val_proba.npy")
    e7_val_pred   = np.load(SUPERVISED_DIR / "predictions" / "E7_val_pred.npy")

    ae_test_mse    = np.load(UNSUPERVISED_DIR / "scores" / "ae_test_mse.npy")
    if_test_scores = np.load(UNSUPERVISED_DIR / "scores" / "if_test_scores.npy")
    ae_val_mse     = np.load(UNSUPERVISED_DIR / "scores" / "ae_val_mse.npy")
    if_val_scores  = np.load(UNSUPERVISED_DIR / "scores" / "if_val_scores.npy")

    with open(UNSUPERVISED_DIR / "thresholds.json") as f:
        ae_thresholds_blob = json.load(f)
    ae_t_p90 = get_ae_threshold(ae_thresholds_blob, 90)
    ae_t_p95 = get_ae_threshold(ae_thresholds_blob, 95)

    fold_classes = {t: extract_fold_classes(t) for t in ZERO_DAY_TARGETS}
    for t, classes in fold_classes.items():
        assert "Benign" in classes
        assert t not in classes
        assert len(classes) == n_classes_global - 1

    y_test_labels = load_labels_csv(PREPROCESSED_DIR / "full_features" / "y_test.csv")
    y_val_labels  = load_labels_csv(PREPROCESSED_DIR / "full_features" / "y_val.csv")
    y_val_encoded = np.array([global_class_map[s] for s in y_val_labels])

    # Entropy thresholds — calibrated on benign-val (seed-invariant: E7 not retrained)
    e7_val_entropy = compute_entropy(e7_val_proba)
    benign_val_mask = (y_val_labels == "Benign")
    entropy_benign  = e7_val_entropy[benign_val_mask]
    entropy_thresholds = {
        f"ent_p{pct}": float(np.percentile(entropy_benign, pct))
        for pct in ENTROPY_PERCENTILES
    }
    log(f"  entropy_thresholds: {entropy_thresholds}")

    # Ensemble thresholds — same fitting as enhanced_fusion.py
    ae_scaler = MinMaxScaler().fit(ae_val_mse.reshape(-1, 1))
    if_scaler = MinMaxScaler().fit((-if_val_scores).reshape(-1, 1))
    ae_norm_val  = np.clip(ae_scaler.transform(ae_val_mse.reshape(-1, 1)).flatten(), 0, 1)
    ae_norm_test = np.clip(ae_scaler.transform(ae_test_mse.reshape(-1, 1)).flatten(), 0, 1)
    if_norm_val  = np.clip(if_scaler.transform((-if_val_scores).reshape(-1, 1)).flatten(), 0, 1)
    if_norm_test = np.clip(if_scaler.transform((-if_test_scores).reshape(-1, 1)).flatten(), 0, 1)
    ensemble_val  = np.maximum(ae_norm_val,  if_norm_val).astype(np.float32)
    ensemble_test = np.maximum(ae_norm_test, if_norm_test).astype(np.float32)
    ensemble_thresholds = {
        f"ens_p{pct}": float(np.percentile(ensemble_val[benign_val_mask], pct))
        for pct in ENSEMBLE_PERCENTILES
    }
    log(f"  ensemble_thresholds: {ensemble_thresholds}")

    # AE binary lookup — shared across seeds
    ae_binaries = {
        "p90":     (ae_test_mse > ae_t_p90),
        "p95":     (ae_test_mse > ae_t_p95),
        "ens_p90": (ensemble_test > ensemble_thresholds["ens_p90"]),
        "ens_p95": (ensemble_test > ensemble_thresholds["ens_p95"]),
    }

    return {
        "global_class_map": global_class_map,
        "benign_id": benign_id,
        "n_classes_global": n_classes_global,
        "fold_classes": fold_classes,
        "y_test_labels": y_test_labels,
        "ae_binaries": ae_binaries,
        "entropy_thresholds": entropy_thresholds,
        "ensemble_thresholds": ensemble_thresholds,
    }


def make_local_to_global(target: str, fold_classes: dict, global_class_map: dict) -> np.ndarray:
    classes = fold_classes[target]
    return np.array([global_class_map[c] for c in classes], dtype=np.int32)


def apply_variant(spec: dict, sup_pred, confidence, entropy, ae_binaries, entropy_thresholds, benign_id):
    ae_binary = ae_binaries[spec["ae"]]
    fam = spec["family"]
    if fam == "baseline":
        return baseline_fusion(sup_pred, ae_binary, benign_id)
    if fam == "confidence":
        return confidence_fusion(sup_pred, ae_binary, confidence, spec["tau"], benign_id)
    if fam == "entropy":
        return entropy_fusion(sup_pred, ae_binary, entropy,
                              entropy_thresholds[spec["ent"]], benign_id)
    if fam == "full":
        return full_enhanced_fusion(sup_pred, ae_binary, confidence, entropy,
                                    spec["tau"], entropy_thresholds[spec["ent"]],
                                    benign_id)
    raise ValueError(f"Unknown family: {fam}")


# ---- Per-seed driver --------------------------------------------------------
def run_seed_fusion(seed: int, shared: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log(f"--- seed {seed} ---")
    seed_loo_dir = LOO_CANONICAL_DIR / "multi_seed" / f"seed_{seed}" / "predictions"

    # Load LOO predictions for this seed; compute entropy/confidence/global-mapped pred
    loo_test_entropy = {}
    loo_test_confidence = {}
    loo_test_pred_global = {}
    for t in ZERO_DAY_TARGETS:
        proba_path = seed_loo_dir / f"loo_{t}_test_proba.npy"
        pred_path  = seed_loo_dir / f"loo_{t}_test_pred.npy"
        if not (proba_path.exists() and pred_path.exists()):
            raise FileNotFoundError(f"[seed={seed}] missing {proba_path} or {pred_path}")
        proba = np.load(proba_path)
        pred_local = np.load(pred_path)
        if proba.shape[1] != len(shared["fold_classes"][t]):
            raise AssertionError(
                f"[seed={seed}][{t}] proba cols {proba.shape[1]} != "
                f"{len(shared['fold_classes'][t])}"
            )
        loo_test_entropy[t] = compute_entropy(proba)
        loo_test_confidence[t] = proba.max(axis=1).astype(np.float32)
        l2g = make_local_to_global(t, shared["fold_classes"], shared["global_class_map"])
        loo_test_pred_global[t] = l2g[pred_local]
        del proba, pred_local

    # Per-target evaluation — copied from enhanced_fusion.py:600-667
    per_target_rows = []
    benign_test_mask = (shared["y_test_labels"] == "Benign")

    for t in ZERO_DAY_TARGETS:
        target_mask = (shared["y_test_labels"] == t)
        n_target = int(target_mask.sum())
        loo_pred_global = loo_test_pred_global[t]
        loo_benign_target_mask = target_mask & (loo_pred_global == shared["benign_id"])
        n_loo_benign = int(loo_benign_target_mask.sum())

        ae_p90_binary = shared["ae_binaries"]["p90"]
        if n_loo_benign > 0:
            ae_only_rescue = float(ae_p90_binary[loo_benign_target_mask].mean())
        else:
            ae_only_rescue = float("nan")

        for variant_id, variant_name, spec in VARIANTS:
            cases = apply_variant(
                spec, loo_pred_global, loo_test_confidence[t], loo_test_entropy[t],
                shared["ae_binaries"], shared["entropy_thresholds"], shared["benign_id"],
            )
            target_cases = cases[target_mask]
            cd = case_distribution(target_cases)
            h2_binary_recall = float(np.isin(target_cases, DETECTED_CASES).mean())

            if n_loo_benign >= H2_STRICT_MIN_BENIGN_N:
                sub_cases = cases[loo_benign_target_mask]
                h2_strict_rescue = float(np.isin(sub_cases, DETECTED_CASES).mean())
            else:
                h2_strict_rescue = float("nan")

            flag_rate_all = float(np.isin(cases, DETECTED_CASES).mean())
            false_alert_rate = float(np.isin(cases[benign_test_mask], DETECTED_CASES).mean())

            per_target_rows.append({
                "seed": seed,
                "target": t,
                "variant": variant_id,
                "variant_name": variant_name,
                "n_target": n_target,
                "n_loo_benign": n_loo_benign,
                "h2_strict_rescue_recall": h2_strict_rescue,
                "h2_binary_recall": h2_binary_recall,
                "ae_only_rescue_recall": ae_only_rescue,
                "flag_rate_all": flag_rate_all,
                "false_alert_rate_benign": false_alert_rate,
                **cd,
            })

    per_target_df = pd.DataFrame(per_target_rows)

    # Ablation table — copied from enhanced_fusion.py:679-712
    ablation_rows = []
    for variant_id, variant_name, _ in VARIANTS:
        sub = per_target_df[per_target_df["variant"] == variant_id]
        strict_sub = sub[
            sub["target"].isin(H2_STRICT_ELIGIBLE)
            & (sub["n_loo_benign"] >= H2_STRICT_MIN_BENIGN_N)
            & sub["h2_strict_rescue_recall"].notna()
        ]
        n_strict_evaluated = len(strict_sub)
        n_strict_pass = int((strict_sub["h2_strict_rescue_recall"] >= H2_PASS_THRESHOLD).sum())
        avg_strict = float(strict_sub["h2_strict_rescue_recall"].mean()) if len(strict_sub) else float("nan")
        n_binary_pass = int((sub["h2_binary_recall"] >= H2_PASS_THRESHOLD).sum())
        avg_binary = float(sub["h2_binary_recall"].mean())
        ablation_rows.append({
            "seed": seed,
            "variant": variant_id,
            "variant_name": variant_name,
            "h2_strict_pass": f"{n_strict_pass}/4",
            "h2_strict_pass_int": n_strict_pass,
            "h2_strict_avg": avg_strict,
            "h2_strict_evaluated": n_strict_evaluated,
            "h2_binary_pass": f"{n_binary_pass}/5",
            "h2_binary_pass_int": n_binary_pass,
            "h2_binary_avg": avg_binary,
            "avg_flag_rate": float(sub["flag_rate_all"].mean()),
            "avg_false_alert_rate": float(sub["false_alert_rate_benign"].mean()),
        })
    ablation_df = pd.DataFrame(ablation_rows)

    return ablation_df, per_target_df


# ---- Main -------------------------------------------------------------------
def main() -> int:
    log("=" * 76)
    log("Path B Week 1 — Per-Seed Fusion Ablation")
    log("=" * 76)

    shared = load_shared_state()

    for seed in SEEDS:
        ablation_df, per_target_df = run_seed_fusion(seed, shared)

        out_dir = OUTPUT_DIR / f"seed_{seed}" / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        ablation_df.to_csv(out_dir / "ablation_table.csv", index=False)
        per_target_df.to_csv(out_dir / "per_target_results.csv", index=False)
        log(f"  saved → {out_dir}/ablation_table.csv  ({len(ablation_df)} rows)")
        log(f"  saved → {out_dir}/per_target_results.csv  ({len(per_target_df)} rows)")

        # Show this seed's strict_avg for entropy_benign_p95
        ebp95 = ablation_df[ablation_df["variant"] == "entropy_benign_p95"].iloc[0]
        log(
            f"  [seed={seed}] entropy_benign_p95 → "
            f"strict={ebp95['h2_strict_pass']} avg={ebp95['h2_strict_avg']:.6f}  "
            f"binary={ebp95['h2_binary_pass']} avg={ebp95['h2_binary_avg']:.4f}  "
            f"FPR={ebp95['avg_false_alert_rate']:.4f}"
        )

        # Hard tripwire after seed=42
        if seed == 42:
            actual = float(ebp95["h2_strict_avg"])
            diff = abs(actual - SEED42_REFERENCE_STRICT_AVG)
            if diff > SEED42_REFERENCE_TOLERANCE:
                raise RuntimeError(
                    f"[seed=42] entropy_benign_p95 strict_avg drift!\n"
                    f"  actual:    {actual!r}\n"
                    f"  reference: {SEED42_REFERENCE_STRICT_AVG!r}\n"
                    f"  diff:      {diff:.3e}  (tolerance {SEED42_REFERENCE_TOLERANCE:.0e})\n"
                    f"This means the multi-seed fusion driver computes Phase 6C\n"
                    f"differently from the canonical run. Aborting before saving\n"
                    f"the other seeds' outputs would be misleading."
                )
            log(
                f"  ✓ seed-42 reference check PASSED "
                f"(actual={actual!r}, diff={diff:.3e})"
            )

    log("=" * 76)
    log("ALL SEEDS DONE")
    log("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
