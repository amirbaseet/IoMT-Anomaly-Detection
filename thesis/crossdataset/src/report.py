"""
Write the deliverables: results_summary.md (paired baseline vs cross-dataset,
Designs A & B, σ over seeds), confusion_A.csv, confusion_B.csv.

Confusion CSVs use the canonical seed (the one saved as the reproducible
parquet sample); metric tables report mean ± σ over all seeds.
"""
from __future__ import annotations

import csv
from pathlib import Path

import config
from labels import FAMILIES


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def write_confusion_binary(confusion: list[list[int]], out_path: Path) -> None:
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred", "benign", "attack"])
        for name, row in zip(("benign", "attack"), confusion):
            w.writerow([name, *row])


def write_confusion_family(confusion: list[list[int]], out_path: Path) -> None:
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred", *FAMILIES])
        for name, row in zip(FAMILIES, confusion):
            w.writerow([name, *row])


def write_summary(ctx: dict, out_path: Path) -> None:
    """``ctx`` is assembled by run.py; see its docstring for the shape."""
    b = config.README_BASELINE
    a_xgb = ctx["cross"]["design_a_xgb"]
    a_ae = ctx["cross"]["design_a_ae"]          # dict[threshold_name] -> agg
    b_fam = ctx["cross"]["design_b"]
    ind = ctx["in_dataset"]

    lines: list[str] = []
    add = lines.append

    add("# Cross-Dataset Generalization — Results Summary")
    add("")
    add("**ShieldMind** · Train: CICIoMT2024 · Test: CICIoT2023 · "
        "Status: **Measured**")
    add("")
    add(f"- Shared-feature schema: **{ctx['n_shared']} features** "
        f"(intersection; {ctx['n_dropped']} CICIoMT2024-only features dropped, "
        "not reconstructed).")
    seeds = tuple(ctx["seeds"])
    add(f"- Multi-seed: **{len(seeds)} seeds** "
        f"{seeds}; cross-dataset numbers are mean ± σ.")
    add(f"- Canonical reproducible sample (seed {ctx['canonical_seed']}) saved "
        "to `sampled_test_set.parquet`.")
    add("")
    add("> Caveat (per design doc §7): cross-dataset used a "
        f"{ctx['n_shared']}-feature shared-schema retrain; the "
        f"{ctx['n_dropped']} CICIoMT2024-only features "
        f"({', '.join(ctx['dropped_features'])}) were unavailable in "
        "CICIoT2023. A drop below the in-dataset baseline is the expected, "
        "honest generalization result — not a defect.")
    add("")

    # ---- Design A ---------------------------------------------------------- #
    add("## Design A — Binary attack/benign generalization")
    add("")
    add("### Supervised XGBoost (Layer 1)")
    add("")
    add("| Metric | In-dataset (shared-feat, CICIoMT2024 test) | "
        "Cross-dataset (CICIoT2023) | Frozen thesis (README, 44-feat) |")
    add("|---|---|---|---|")
    add(f"| F1 (attack) | {ind['design_a_xgb']['f1']:.4f} | "
        f"{_fmt(a_xgb['f1']['mean'], a_xgb['f1']['std'])} | "
        f"{b['design_a_xgb_binary_f1_macro']:.4f} (best binary, macro) |")
    for m in ("accuracy", "precision", "recall", "roc_auc"):
        add(f"| {m} | {ind['design_a_xgb'][m]:.4f} | "
            f"{_fmt(a_xgb[m]['mean'], a_xgb[m]['std'])} | — |")
    add("")
    add("### Unsupervised Flow AE (Layer 2) — threshold set on CICIoMT2024 "
        "benign-val")
    add("")
    add("| Threshold | Metric | In-dataset | Cross-dataset (CICIoT2023) | "
        "Frozen thesis (README) |")
    add("|---|---|---|---|---|")
    for thr in a_ae:
        ind_thr = ind["design_a_ae"][thr]
        add(f"| {thr} | F1 (attack) | {ind_thr['f1']:.4f} | "
            f"{_fmt(a_ae[thr]['f1']['mean'], a_ae[thr]['f1']['std'])} | "
            f"{b['design_a_ae_binary_f1']:.4f} |")
        add(f"| {thr} | recall | {ind_thr['recall']:.4f} | "
            f"{_fmt(a_ae[thr]['recall']['mean'], a_ae[thr]['recall']['std'])} "
            "| — |")
    add(f"| — | ROC-AUC | {ind['design_a_ae']['roc_auc']:.4f} | "
        f"{_fmt(a_ae[list(a_ae)[0]]['roc_auc']['mean'], a_ae[list(a_ae)[0]]['roc_auc']['std'])} "
        f"| {b['design_a_ae_binary_auc']:.4f} |")
    add("")

    # ---- Design B ---------------------------------------------------------- #
    add("## Design B — 5 shared families (Benign, Spoofing, Recon, DoS, DDoS)")
    add("")
    add("Out-of-scope CICIoT2023 classes (all MIRAI-*, SQLINJECTION, XSS, "
        "BACKDOOR_MALWARE, BROWSERHIJACKING, COMMANDINJECTION, "
        "UPLOADING_ATTACK, DICTIONARYBRUTEFORCE) were dropped from the B "
        "test set. MQTT was dropped from the CICIoMT2024 training set "
        "(no CICIoT2023 counterpart).")
    add("")
    add(f"| Metric | In-dataset (shared-feat) | Cross-dataset (CICIoT2023) | "
        "Frozen thesis (README 6-class incl. MQTT) |")
    add("|---|---|---|---|")
    add(f"| macro-F1 (5 families) | {ind['design_b']['macro_f1']:.4f} | "
        f"{_fmt(b_fam['macro_f1']['mean'], b_fam['macro_f1']['std'])} | "
        f"{b['design_b_category_macro_f1_6class']:.4f} |")
    add(f"| accuracy | {ind['design_b']['accuracy']:.4f} | "
        f"{_fmt(b_fam['accuracy']['mean'], b_fam['accuracy']['std'])} | — |")
    add("")
    add("### Per-family F1 (cross-dataset, mean ± σ)")
    add("")
    add("| Family | In-dataset F1 | Cross-dataset F1 | "
        "Cross-dataset recall | support (seed " f"{ctx['canonical_seed']}) |")
    add("|---|---|---|---|---|")
    for fam in FAMILIES:
        cd = b_fam["per_family"][fam]
        ind_f = ind["design_b"]["per_family"][fam]
        add(f"| {fam} | {ind_f['f1']:.4f} | "
            f"{_fmt(cd['f1']['mean'], cd['f1']['std'])} | "
            f"{_fmt(cd['recall']['mean'], cd['recall']['std'])} | "
            f"{ctx['b_support'][fam]} |")
    add("")
    add("Confusion matrices: `confusion_A.csv` (binary, XGBoost), "
        "`confusion_B.csv` (5 families) — both for the canonical seed.")
    add("")
    add("_Interpretation (honest generalization vs pipeline artifact) is left "
        "to the planning chat, per the brief._")

    out_path.write_text("\n".join(lines) + "\n")
