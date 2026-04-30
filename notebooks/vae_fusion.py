#!/usr/bin/env python3
"""
Phase 6D Task 2 — VAE log-likelihood as fusion signal
======================================================

Re-applies the Phase 6C 5-case fusion ablation logic with the VAE channel
substituting for the deterministic AE channel. Computes 8 VAE-conditioned
variants per beta, plus 2 AE-only reference rows (the §15C and §15D anchors)
that double as reproducibility tripwires.

For each beta in {0.1, 0.5, 1.0, 4.0}:
  - Derive VAE percentile thresholds (p90/p95/p99) on benign-val VAE score
  - Run 8 VAE-conditioned variants × 5 LOO targets
  - Run 2 AE-only reference variants (entropy_benign_p93/p95 + ae_p90)
  - Tripwire 1 (HARD): entropy_benign_p95_ae_p90 strict_avg == 0.8035264623662012 ± 1e-9
  - Tripwire 2 (SOFT): entropy_benign_p93_ae_p90 strict_avg matches sweep_table.csv ± 1e-9
  - Compute VAE-only AUC for the REPLACE-AE-ONLY decision branch
  - Write per-beta ablation_table.csv + per_target_results.csv + vae_thresholds.json

Then concatenate all betas into all_betas_ablation.csv with an extra `beta` column.

Inputs (read-only):
  preprocessed/full_features/{y_val, y_test}.csv
  preprocessed/label_encoders.json
  results/supervised/predictions/E7_val_proba.npy
  results/zero_day_loo/predictions/loo_<TGT>_test_proba.npy
  results/zero_day_loo/predictions/loo_<TGT>_test_pred.npy
  results/zero_day_loo/models/loo_label_map_<TGT>.json
  results/unsupervised/scores/ae_test_mse.npy
  results/unsupervised/thresholds.json
  results/enhanced_fusion/threshold_sweep/sweep_table.csv
  results/unsupervised/vae/beta_<beta>/{val_loglik, test_loglik}.npy

Outputs:
  results/enhanced_fusion/vae_ablation/beta_<beta>/
    vae_thresholds.json
    ablation_table.csv          (10 rows: 8 VAE variants + 2 AE references)
    per_target_results.csv      (10 × 5 = 50 rows)
  results/enhanced_fusion/vae_ablation/all_betas_ablation.csv
  results/enhanced_fusion/vae_ablation/run.log
"""

# %% SECTION 0 — Imports
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# %% SECTION 1 — Configuration

ROOT             = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = ROOT / "preprocessed"
SUPERVISED_DIR   = ROOT / "results" / "supervised"
LOO_DIR          = ROOT / "results" / "zero_day_loo"
UNSUPERVISED_DIR = ROOT / "results" / "unsupervised"
VAE_DIR          = UNSUPERVISED_DIR / "vae"
SWEEP_TABLE_CSV  = ROOT / "results" / "enhanced_fusion" / "threshold_sweep" / "sweep_table.csv"
OUTPUT_DIR       = ROOT / "results" / "enhanced_fusion" / "vae_ablation"

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
DETECTED_CASES         = (1, 2, 3, 5)
OPERATIONAL_FPR_BUDGET = 0.25

# β grid — must match the directory names produced by vae_train.py.
BETA_GRID = [0.1, 0.5, 1.0, 4.0]

# Reproducibility tripwires (BOTH active per user instruction).
TRIPWIRE_HARD_VARIANT = "entropy_benign_p95_ae_p90"
TRIPWIRE_HARD_VALUE   = 0.8035264623662012   # Week 2A canonical (threshold_sweep.py:106)
TRIPWIRE_HARD_TOL     = 1e-9
TRIPWIRE_SOFT_VARIANT = "entropy_benign_p93_ae_p90"
TRIPWIRE_SOFT_TOL     = 1e-9
TRIPWIRE_SOFT_PCT     = 93.0   # row percentile in sweep_table.csv to load

# Decision-rule constants (per user spec).
DECISION_BASELINE_VARIANT = TRIPWIRE_SOFT_VARIANT       # the §15D anchor
SHIP_DELTA                = 0.02                         # +2pp on strict_avg
PHASE5_AE_TEST_AUC        = 0.9892                       # for REPLACE-AE-ONLY check

SMOKE = bool(int(os.environ.get("SMOKE", "0")))


def log(msg: str = "", t0: float | None = None) -> None:
    elapsed = f" [+{time.time() - t0:6.1f}s]" if t0 is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)


# %% SECTION 2 — Pure functions copied verbatim from existing scripts
#
# We deliberately copy rather than import because both source modules execute
# their full pipelines at import time. The two reproducibility tripwires below
# catch any silent drift between this copy and enhanced_fusion.py / threshold_sweep.py.


# Copied verbatim from enhanced_fusion.py:338-341 (also in threshold_sweep.py:136-139).
def compute_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy of per-row probability vector. Higher = more uncertain."""
    p = np.clip(proba, 1e-10, 1.0)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)


# Copied verbatim from enhanced_fusion.py:473-480.
def baseline_fusion(sup_pred: np.ndarray,
                    ae_binary: np.ndarray,
                    benign_id: int) -> np.ndarray:
    sup_attack = (sup_pred != benign_id)
    return np.where(sup_attack &  ae_binary, 1,
           np.where(~sup_attack &  ae_binary, 2,
           np.where(sup_attack & ~ae_binary, 3, 4)))


# Copied verbatim from enhanced_fusion.py:483-496.
def confidence_fusion(sup_pred: np.ndarray,
                      ae_binary: np.ndarray,
                      confidence: np.ndarray,
                      conf_threshold: float,
                      benign_id: int) -> np.ndarray:
    sup_attack = (sup_pred != benign_id)
    uncertain  = (confidence < conf_threshold)
    return np.where( sup_attack &  ae_binary & ~uncertain, 1,
           np.where(~sup_attack &  ae_binary,              2,
           np.where( uncertain  &  ae_binary,              2,
           np.where( sup_attack & ~ae_binary & ~uncertain, 3,
           np.where( uncertain  & ~ae_binary,              5,
                                                            4)))))


# Copied verbatim from enhanced_fusion.py:499-512 (also in threshold_sweep.py:143-158).
def entropy_fusion(sup_pred: np.ndarray,
                   ae_binary: np.ndarray,
                   entropy: np.ndarray,
                   ent_threshold: float,
                   benign_id: int) -> np.ndarray:
    sup_attack    = (sup_pred != benign_id)
    high_entropy  = (entropy > ent_threshold)
    return np.where( sup_attack &  ae_binary & ~high_entropy, 1,
           np.where(~sup_attack &  ae_binary,                 2,
           np.where( high_entropy &  ae_binary,               2,
           np.where( high_entropy & ~ae_binary,               5,
           np.where( sup_attack & ~ae_binary & ~high_entropy, 3,
                                                              4)))))


# Copied verbatim from enhanced_fusion.py:515-529.
def full_enhanced_fusion(sup_pred: np.ndarray,
                         ae_binary: np.ndarray,
                         confidence: np.ndarray,
                         entropy: np.ndarray,
                         conf_threshold: float,
                         ent_threshold: float,
                         benign_id: int) -> np.ndarray:
    sup_attack = (sup_pred != benign_id)
    suspicious = (confidence < conf_threshold) | (entropy > ent_threshold)
    return np.where( sup_attack &  ae_binary & ~suspicious, 1,
           np.where((~sup_attack | suspicious) &  ae_binary, 2,
           np.where( suspicious & ~ae_binary,                5,
           np.where( sup_attack & ~ae_binary & ~suspicious,  3,
                                                              4))))


# Copied verbatim from threshold_sweep.py:182-198.
def _extract_global_class_map(encoders: dict) -> Dict[str, int]:
    if all(isinstance(v, int) for v in encoders.values()):
        return {str(k): int(v) for k, v in encoders.items()}
    for key in ("label", "y", "class", "multiclass", "target"):
        if key in encoders and isinstance(encoders[key], dict):
            sub = encoders[key]
            if all(isinstance(v, int) for v in sub.values()):
                return {str(k): int(v) for k, v in sub.items()}
            if "classes_" in sub:
                return {str(c): i for i, c in enumerate(sub["classes_"])}
    if "classes_" in encoders:
        return {str(c): i for i, c in enumerate(encoders["classes_"])}
    raise ValueError(
        f"Cannot extract global class map from encoders.json — keys = "
        f"{list(encoders.keys())[:10]}"
    )


# %% SECTION 3 — Variant table (8 VAE-conditioned + 2 AE-only references)

# Spec keys mirror enhanced_fusion.py: family ∈ {baseline,confidence,entropy,full},
# ae key looks up a binary mask in the (per-β) BINARIES dict, ent key indexes
# the entropy_thresholds dict, tau is the confidence floor.
VARIANTS: List[Tuple[str, str, dict]] = [
    # --- 8 VAE-conditioned variants ---
    ("vae_p90", "VAE-only baseline (p90)",
        dict(family="baseline", ae="vae_p90")),
    ("vae_p95", "VAE-only baseline (p95)",
        dict(family="baseline", ae="vae_p95")),
    ("vae_p99", "VAE-only baseline (p99)",
        dict(family="baseline", ae="vae_p99")),
    ("entropy_benign_p93_vae_p90", "Entropy p93 + VAE p90",
        dict(family="entropy", ent="ent_p93", ae="vae_p90")),
    ("entropy_benign_p95_vae_p90", "Entropy p95 + VAE p90",
        dict(family="entropy", ent="ent_p95", ae="vae_p90")),
    ("entropy_benign_p95_vae_p95", "Entropy p95 + VAE p95",
        dict(family="entropy", ent="ent_p95", ae="vae_p95")),
    ("entropy_benign_p93_vae_p95", "Entropy p93 + VAE p95",
        dict(family="entropy", ent="ent_p93", ae="vae_p95")),
    ("conf07_ent_p95_vae_p90", "Confidence + Entropy p95 + VAE p90 (τ=0.7)",
        dict(family="full", tau=0.7, ent="ent_p95", ae="vae_p90")),

    # --- 2 AE-only reference variants (reproducibility tripwires) ---
    ("entropy_benign_p93_ae_p90", "Entropy p93 + AE p90 (§15D baseline / SOFT tripwire)",
        dict(family="entropy", ent="ent_p93", ae="ae_p90")),
    ("entropy_benign_p95_ae_p90", "Entropy p95 + AE p90 (§15C anchor / HARD tripwire)",
        dict(family="entropy", ent="ent_p95", ae="ae_p90")),
]


def case_distribution(cases: np.ndarray) -> Dict[str, float]:
    return {f"case{c}_pct": float((cases == c).mean()) for c in (1, 2, 3, 4, 5)}


# %% SECTION 4 — Load baseline signals (E7 entropy, AE binaries, LOO predictions+confidence)

def load_baseline_signals() -> dict:
    """Load all signals needed by the fusion ablation (β-independent)."""
    t0 = time.time()
    log("Loading β-independent baseline signals ...")

    # Label encoders → global class map.
    with open(PREPROCESSED_DIR / "label_encoders.json") as f:
        encoders = json.load(f)
    global_class_map = _extract_global_class_map(encoders)
    if "Benign" not in global_class_map:
        raise RuntimeError("Expected 'Benign' in global class map.")
    global_benign_id = int(global_class_map["Benign"])
    log(f"  global label space: {len(global_class_map)} classes; benign_id={global_benign_id}")

    # y_val / y_test as string labels.
    y_val_df  = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_val.csv")
    y_test_df = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_test.csv")
    label_col = "label" if "label" in y_val_df.columns else y_val_df.columns[0]
    y_val_labels  = y_val_df[label_col].astype(str).values
    y_test_labels = y_test_df[label_col].astype(str).values
    benign_val_mask  = (y_val_labels  == "Benign")
    benign_test_mask = (y_test_labels == "Benign")
    y_test_bin       = (y_test_labels != "Benign").astype(np.int8)
    log(
        f"  y_val={len(y_val_labels):,}  benign_val={int(benign_val_mask.sum()):,}  "
        f"y_test={len(y_test_labels):,}  benign_test={int(benign_test_mask.sum()):,}"
    )

    # E7 val entropy → drives entropy threshold derivation (p93, p95).
    e7_val_proba    = np.load(SUPERVISED_DIR / "predictions" / "E7_val_proba.npy")
    e7_val_entropy  = compute_entropy(e7_val_proba)
    benign_val_ent  = e7_val_entropy[benign_val_mask]
    entropy_thresholds = {
        "ent_p93": float(np.percentile(benign_val_ent, 93.0)),
        "ent_p95": float(np.percentile(benign_val_ent, 95.0)),
    }
    log(f"  entropy thresholds (benign-val): "
        f"p93={entropy_thresholds['ent_p93']:.6f}  p95={entropy_thresholds['ent_p95']:.6f}")
    del e7_val_proba

    # AE test scores + p90 threshold (the published Phase 5 anchor).
    ae_test_mse = np.load(UNSUPERVISED_DIR / "scores" / "ae_test_mse.npy")
    with open(UNSUPERVISED_DIR / "thresholds.json") as f:
        ae_thr_doc = json.load(f)
    ae_t_p90 = float(ae_thr_doc["thresholds"]["p90"])
    ae_p90_binary = (ae_test_mse > ae_t_p90)
    log(f"  AE p90 threshold = {ae_t_p90:.6f}  flag-rate(test)={float(ae_p90_binary.mean()):.4f}")
    log(f"  AE test AUC (Phase 5 published) = {PHASE5_AE_TEST_AUC:.4f}")

    # Per-target LOO test predictions, entropy, AND max-prob confidence.
    loo_test_pred_global: Dict[str, np.ndarray] = {}
    loo_test_entropy:     Dict[str, np.ndarray] = {}
    loo_test_confidence:  Dict[str, np.ndarray] = {}
    for tgt in ZERO_DAY_TARGETS:
        with open(LOO_DIR / "models" / f"loo_label_map_{tgt}.json") as f:
            label_map = json.load(f)
        local_to_name = sorted(label_map.items(), key=lambda kv: kv[1])
        local_to_global = np.array(
            [global_class_map[name] for name, _ in local_to_name], dtype=np.int32,
        )

        proba      = np.load(LOO_DIR / "predictions" / f"loo_{tgt}_test_proba.npy")
        pred_local = np.load(LOO_DIR / "predictions" / f"loo_{tgt}_test_pred.npy")
        if proba.shape[1] != len(local_to_global):
            raise RuntimeError(
                f"{tgt}: proba has {proba.shape[1]} cols but label_map has "
                f"{len(local_to_global)} entries"
            )
        loo_test_entropy[tgt]    = compute_entropy(proba)
        loo_test_confidence[tgt] = proba.max(axis=1).astype(np.float32)
        loo_test_pred_global[tgt] = local_to_global[pred_local]
        del proba, pred_local

    log(f"signal loading: {time.time() - t0:.1f}s")
    return {
        "entropy_thresholds":   entropy_thresholds,
        "y_test_labels":        y_test_labels,
        "y_test_bin":           y_test_bin,
        "benign_test_mask":     benign_test_mask,
        "ae_test_mse":          ae_test_mse,           # for VAE AUC comparison context
        "ae_p90_binary":        ae_p90_binary,
        "loo_test_pred_global": loo_test_pred_global,
        "loo_test_entropy":     loo_test_entropy,
        "loo_test_confidence":  loo_test_confidence,
        "global_benign_id":     global_benign_id,
    }


# %% SECTION 5 — Load soft-tripwire baseline value from sweep_table.csv

def load_soft_tripwire_value() -> float:
    """Read entropy_p93 + ae_p90 strict_avg from sweep_table.csv (§15D anchor)."""
    if not SWEEP_TABLE_CSV.exists():
        raise FileNotFoundError(
            f"sweep_table.csv not found at {SWEEP_TABLE_CSV}; cannot load soft tripwire."
        )
    df = pd.read_csv(SWEEP_TABLE_CSV)
    row = df[np.isclose(df["percentile"], TRIPWIRE_SOFT_PCT)]
    if len(row) == 0:
        raise RuntimeError(
            f"sweep_table.csv has no row at percentile={TRIPWIRE_SOFT_PCT}"
        )
    val = float(row.iloc[0]["h2_strict_avg"])
    log(f"  soft tripwire: §15D entropy_p{TRIPWIRE_SOFT_PCT} + ae_p90 strict_avg = {val:.10f}")
    return val


# %% SECTION 6 — Per-(target, variant) evaluation

def apply_variant(spec: dict, target: str,
                  binaries: Dict[str, np.ndarray],
                  signals: dict) -> np.ndarray:
    """Return Case-array (1..5) of length N_test for one (target, variant)."""
    sup_pred  = signals["loo_test_pred_global"][target]
    confidence = signals["loo_test_confidence"][target]
    entropy   = signals["loo_test_entropy"][target]
    ae_binary = binaries[spec["ae"]]
    fam = spec["family"]
    if fam == "baseline":
        return baseline_fusion(sup_pred, ae_binary, signals["global_benign_id"])
    if fam == "confidence":
        return confidence_fusion(
            sup_pred, ae_binary, confidence, spec["tau"], signals["global_benign_id"],
        )
    if fam == "entropy":
        return entropy_fusion(
            sup_pred, ae_binary, entropy,
            signals["entropy_thresholds"][spec["ent"]],
            signals["global_benign_id"],
        )
    if fam == "full":
        return full_enhanced_fusion(
            sup_pred, ae_binary, confidence, entropy,
            spec["tau"],
            signals["entropy_thresholds"][spec["ent"]],
            signals["global_benign_id"],
        )
    raise ValueError(f"Unknown variant family: {fam}")


def evaluate_per_target(binaries: Dict[str, np.ndarray],
                        signals: dict) -> List[dict]:
    """Return per-(target, variant) rows for all VARIANTS × ZERO_DAY_TARGETS."""
    y_test_labels    = signals["y_test_labels"]
    benign_test_mask = signals["benign_test_mask"]
    benign_id        = signals["global_benign_id"]
    ae_p90_binary    = signals["ae_p90_binary"]

    rows: List[dict] = []
    for t in ZERO_DAY_TARGETS:
        target_mask  = (y_test_labels == t)
        n_target     = int(target_mask.sum())
        loo_pred_g   = signals["loo_test_pred_global"][t]
        loo_benign_target_mask = target_mask & (loo_pred_g == benign_id)
        n_loo_benign = int(loo_benign_target_mask.sum())

        # AE-only rescue recall on LOO→Benign subset (β-independent reference).
        ae_only_rescue = (
            float(ae_p90_binary[loo_benign_target_mask].mean())
            if n_loo_benign > 0 else float("nan")
        )

        for variant_id, variant_name, spec in VARIANTS:
            cases = apply_variant(spec, t, binaries, signals)
            target_cases = cases[target_mask]
            cd = case_distribution(target_cases)

            h2_binary_recall = float(np.isin(target_cases, DETECTED_CASES).mean())
            if n_loo_benign >= H2_STRICT_MIN_BENIGN_N:
                sub_cases = cases[loo_benign_target_mask]
                h2_strict_rescue = float(np.isin(sub_cases, DETECTED_CASES).mean())
            else:
                h2_strict_rescue = float("nan")

            flag_rate_all = float(np.isin(cases, DETECTED_CASES).mean())
            false_alert_rate = float(
                np.isin(cases[benign_test_mask], DETECTED_CASES).mean()
            )

            rows.append({
                "target":                  t,
                "variant":                 variant_id,
                "variant_name":            variant_name,
                "n_target":                n_target,
                "n_loo_benign":            n_loo_benign,
                "h2_strict_rescue_recall": h2_strict_rescue,
                "h2_binary_recall":        h2_binary_recall,
                "ae_only_rescue_recall":   ae_only_rescue,
                "flag_rate_all":           flag_rate_all,
                "false_alert_rate_benign": false_alert_rate,
                **cd,
            })
    return rows


def aggregate_ablation(per_target_rows: List[dict]) -> pd.DataFrame:
    """Build the per-variant ablation_table from per-(variant, target) rows."""
    pdf = pd.DataFrame(per_target_rows)
    out_rows: List[dict] = []
    for variant_id, variant_name, _ in VARIANTS:
        sub = pdf[pdf["variant"] == variant_id]
        strict_sub = sub[
            sub["target"].isin(H2_STRICT_ELIGIBLE)
            & (sub["n_loo_benign"] >= H2_STRICT_MIN_BENIGN_N)
            & sub["h2_strict_rescue_recall"].notna()
        ]
        n_strict_evaluated = len(strict_sub)
        n_strict_pass = int((strict_sub["h2_strict_rescue_recall"] >= H2_PASS_THRESHOLD).sum())
        avg_strict = (
            float(strict_sub["h2_strict_rescue_recall"].mean())
            if len(strict_sub) else float("nan")
        )
        n_binary_pass = int((sub["h2_binary_recall"] >= H2_PASS_THRESHOLD).sum())
        avg_binary = float(sub["h2_binary_recall"].mean())

        out_rows.append({
            "variant":              variant_id,
            "variant_name":         variant_name,
            "h2_strict_pass":       f"{n_strict_pass}/4",
            "h2_strict_pass_int":   n_strict_pass,
            "h2_strict_avg":        avg_strict,
            "h2_strict_evaluated":  n_strict_evaluated,
            "h2_binary_pass":       f"{n_binary_pass}/5",
            "h2_binary_pass_int":   n_binary_pass,
            "h2_binary_avg":        avg_binary,
            "avg_flag_rate":        float(sub["flag_rate_all"].mean()),
            "avg_false_alert_rate": float(sub["false_alert_rate_benign"].mean()),
        })
    return pd.DataFrame(out_rows)


# %% SECTION 7 — Reproducibility tripwires

def assert_tripwires_pass(ablation_df: pd.DataFrame, soft_value: float) -> None:
    """Both tripwires fire on the AE-only reference rows (β-independent)."""
    # HARD: Week 2A canonical 0.8035264623662012
    hard_actual = float(
        ablation_df.loc[ablation_df["variant"] == TRIPWIRE_HARD_VARIANT, "h2_strict_avg"].iloc[0]
    )
    hard_diff = abs(hard_actual - TRIPWIRE_HARD_VALUE)
    if hard_diff > TRIPWIRE_HARD_TOL:
        raise RuntimeError(
            f"[HARD tripwire] {TRIPWIRE_HARD_VARIANT} drift!\n"
            f"  actual:    {hard_actual!r}\n"
            f"  reference: {TRIPWIRE_HARD_VALUE!r}\n"
            f"  diff:      {hard_diff:.3e}  (tol {TRIPWIRE_HARD_TOL:.0e})\n"
            f"Copied fusion functions have drifted from enhanced_fusion.py.\n"
            f"Aborting before generating any VAE-conditioned numbers."
        )
    log(f"  ✓ HARD tripwire PASS: {TRIPWIRE_HARD_VARIANT} = {hard_actual!r} "
        f"(diff {hard_diff:.3e})")

    # SOFT: §15D anchor read from sweep_table.csv
    soft_actual = float(
        ablation_df.loc[ablation_df["variant"] == TRIPWIRE_SOFT_VARIANT, "h2_strict_avg"].iloc[0]
    )
    soft_diff = abs(soft_actual - soft_value)
    if soft_diff > TRIPWIRE_SOFT_TOL:
        raise RuntimeError(
            f"[SOFT tripwire] {TRIPWIRE_SOFT_VARIANT} drift!\n"
            f"  actual:        {soft_actual!r}\n"
            f"  sweep_table:   {soft_value!r}\n"
            f"  diff:          {soft_diff:.3e}  (tol {TRIPWIRE_SOFT_TOL:.0e})\n"
            f"sweep_table.csv may be stale or fusion logic drifted. Investigate before proceeding."
        )
    log(f"  ✓ SOFT tripwire PASS: {TRIPWIRE_SOFT_VARIANT} = {soft_actual!r} "
        f"(diff {soft_diff:.3e})")


# %% SECTION 8 — Per-β orchestration

def run_one_beta(beta: float, signals: dict, soft_value: float) -> Tuple[pd.DataFrame, dict]:
    """Train-free fusion ablation for one β; return (ablation_df, summary_dict)."""
    beta_str = f"{beta:.1f}"
    out_dir = OUTPUT_DIR / f"beta_{beta_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n--- β={beta} ---")

    # Load VAE val/test scores from Phase 2 artifacts.
    vae_dir = VAE_DIR / f"beta_{beta_str}"
    vae_val  = np.load(vae_dir / "val_loglik.npy")
    vae_test = np.load(vae_dir / "test_loglik.npy")
    log(f"  loaded VAE scores from {vae_dir}: val={vae_val.shape}, test={vae_test.shape}")

    # Derive VAE thresholds on benign-val (mirroring AE p90/p95/p99 convention).
    vae_thresholds = {
        "p90": float(np.percentile(vae_val, 90)),
        "p95": float(np.percentile(vae_val, 95)),
        "p99": float(np.percentile(vae_val, 99)),
    }
    log(f"  VAE benign-val thresholds: "
        f"p90={vae_thresholds['p90']:.4f}  "
        f"p95={vae_thresholds['p95']:.4f}  "
        f"p99={vae_thresholds['p99']:.4f}")
    with open(out_dir / "vae_thresholds.json", "w") as f:
        json.dump({
            "beta": beta,
            "thresholds": vae_thresholds,
            "n_benign_val_used": int(len(vae_val)),
        }, f, indent=2)

    # Build per-β BINARIES dict (AE keys plus VAE keys).
    binaries: Dict[str, np.ndarray] = {
        "ae_p90":  signals["ae_p90_binary"],                          # for tripwires
        "vae_p90": (vae_test > vae_thresholds["p90"]),
        "vae_p95": (vae_test > vae_thresholds["p95"]),
        "vae_p99": (vae_test > vae_thresholds["p99"]),
    }
    for k, b in binaries.items():
        log(f"  binary[{k:8s}] flag-rate(test) = {float(b.mean()):.4f}")

    # Direction sanity check: VAE score on benign-test should be lower than on attacks.
    benign_score_mean = float(vae_test[signals["benign_test_mask"]].mean())
    attack_score_mean = float(vae_test[~signals["benign_test_mask"]].mean())
    if attack_score_mean <= benign_score_mean:
        raise RuntimeError(
            f"β={beta}: VAE score direction inverted! "
            f"benign mean={benign_score_mean:.3f} >= attack mean={attack_score_mean:.3f}. "
            "Expected higher score => more anomalous. Investigate vae_train.py before proceeding."
        )
    log(f"  ✓ direction OK: benign mean={benign_score_mean:.3f} < "
        f"attack mean={attack_score_mean:.3f} (separation {attack_score_mean/max(benign_score_mean, 1e-6):.1f}×)")

    # VAE-only AUC (for REPLACE-AE-ONLY decision branch).
    vae_test_auc = float(roc_auc_score(signals["y_test_bin"], vae_test))
    log(f"  VAE test AUC = {vae_test_auc:.4f}    "
        f"(AE Phase 5 = {PHASE5_AE_TEST_AUC:.4f}; "
        f"Δ = {vae_test_auc - PHASE5_AE_TEST_AUC:+.4f})")

    # Run all 10 variants × 5 targets.
    per_target_rows = evaluate_per_target(binaries, signals)
    pdf = pd.DataFrame(per_target_rows)
    pdf.insert(0, "beta", beta)
    pdf.to_csv(out_dir / "per_target_results.csv", index=False)

    # Build per-β ablation table.
    ablation_df = aggregate_ablation(per_target_rows)
    ablation_df.insert(0, "beta", beta)
    ablation_df.to_csv(out_dir / "ablation_table.csv", index=False)
    log(f"  wrote ablation_table.csv ({len(ablation_df)} rows) and per_target_results.csv "
        f"({len(pdf)} rows)")

    # Reproducibility tripwires (BOTH must pass before reporting any VAE numbers).
    assert_tripwires_pass(ablation_df, soft_value)

    # Brief per-β log of the ablation table.
    log(f"\n  β={beta} ablation table (sorted by h2_strict_avg desc):")
    show = ablation_df[[
        "variant", "h2_strict_pass", "h2_strict_avg", "h2_binary_avg",
        "avg_false_alert_rate",
    ]].copy()
    show = show.sort_values("h2_strict_avg", ascending=False, na_position="last")
    log("\n" + show.to_string(index=False))

    summary = {
        "beta":              beta,
        "vae_test_auc":      vae_test_auc,
        "vae_thresholds":    vae_thresholds,
        "vae_score_benign_mean": benign_score_mean,
        "vae_score_attack_mean": attack_score_mean,
    }
    return ablation_df, summary


# %% SECTION 9 — Main

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("=" * 76)
    log("Path B Phase 6D Task 2 — VAE log-likelihood as fusion signal")
    log(f"  SMOKE={int(SMOKE)}  output_dir={OUTPUT_DIR}")
    log(f"  β grid: {BETA_GRID}")
    log("=" * 76)

    # Pre-flight: all 4 β model artifacts must exist.
    missing = []
    for b in BETA_GRID:
        for fname in ("val_loglik.npy", "test_loglik.npy"):
            if not (VAE_DIR / f"beta_{b:.1f}" / fname).exists():
                missing.append(f"{VAE_DIR / f'beta_{b:.1f}' / fname}")
    if missing:
        raise FileNotFoundError(
            f"VAE artifacts missing for some β:\n  " + "\n  ".join(missing) +
            "\nRun vae_train.py (full sweep) first."
        )

    # Load β-independent signals + soft-tripwire baseline.
    signals = load_baseline_signals()
    soft_value = load_soft_tripwire_value()

    # Per-β fusion runs.
    all_ablations: List[pd.DataFrame] = []
    summaries: List[dict] = []
    for b in BETA_GRID:
        ablation_df, summary = run_one_beta(b, signals, soft_value)
        all_ablations.append(ablation_df)
        summaries.append(summary)

    # Concatenated all_betas table.
    all_df = pd.concat(all_ablations, ignore_index=True)
    all_csv = OUTPUT_DIR / "all_betas_ablation.csv"
    all_df.to_csv(all_csv, index=False)
    log(f"\nwrote {all_csv} ({len(all_df)} rows)")

    # Brief β-summary stash for Phase 4 use.
    with open(OUTPUT_DIR / "per_beta_summary.json", "w") as f:
        json.dump({
            "betas":            BETA_GRID,
            "phase5_ae_test_auc": PHASE5_AE_TEST_AUC,
            "summaries":        summaries,
            "tripwire_hard_value":  TRIPWIRE_HARD_VALUE,
            "tripwire_soft_value":  soft_value,
            "operational_fpr_budget": OPERATIONAL_FPR_BUDGET,
            "ship_delta_threshold": SHIP_DELTA,
        }, f, indent=2)
    log(f"wrote {OUTPUT_DIR / 'per_beta_summary.json'}")

    log(f"\nDONE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
