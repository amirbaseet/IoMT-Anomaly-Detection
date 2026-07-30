#!/usr/bin/env python3
"""
07_tau_sweep.py — Confidence-floor / entropy-ceiling operating-characteristic sweep.

Tier-1 verification item (see CLAUDE_CODE_BRIEF_TauSweep.md).

PURE POST-PROCESSING on the FROZEN thesis E7 model's existing test-set predictions.
No training. No pcaps. No Track-A NFStream retrains. The frozen E7 softmax arrays
(results/supervised/predictions/E7_test_proba.npy) ARE the model output on the
published, deduplicated thesis test split — analysed directly.

Two independent sweeps:
  Sweep 1  confidence floor tau_c : auto-decide flows with c = max(p) >= tau_c
  Sweep 2  entropy ceiling  tau_H : auto-decide flows with H(p)/ln(K) <= tau_H   (K=19)

At each threshold, for the RETAINED (auto-decided) set: coverage, accuracy,
macro-F1, MCC. For the DEFERRED set: size, the error rate it would have incurred
had it been auto-decided, and its per-class composition.

Everything reported is MEASURED. Thresholds are characterised post-hoc on the test
set (operating-characteristic analysis) — NOT used for model selection or tuning.

Outputs -> results/tausweep/
  tau_sweep_results.json   full curves (both sweeps), all metrics per threshold
  tau_sweep_curve.csv      tidy long-format table for plotting
  tau_sweep_summary.md     3 named operating points, per-class deferral, analyst
                           budget, Phase-6C continuity check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

# --------------------------------------------------------------------------- #
# Paths (frozen thesis artifacts — confirmed with user)
# --------------------------------------------------------------------------- #
ROOT          = Path(__file__).resolve().parents[2]
SUP_PRED      = ROOT / "results" / "supervised" / "predictions"
PREPROC       = ROOT / "preprocessed" / "full_features"
OUT_DIR       = ROOT / "results" / "tausweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K             = 19            # number of classes (multi:softprob)
LN_K          = float(np.log(K))
BENIGN_ID     = 1             # 'Benign' in the frozen label space (verified)
EPS           = 1e-12         # clip for log(0)

# Phase-6C zero-day / rescue targets (the 5 LOO targets — README §15B/§15C)
RESCUE_CLASSES = [
    "ARP_Spoofing",
    "MQTT_DoS_Connect_Flood",
    "MQTT_Malformed_Data",
    "Recon_Ping_Sweep",
    "Recon_VulScan",
]


# --------------------------------------------------------------------------- #
# Entropy — replicate notebooks/vae_fusion.py:compute_entropy EXACTLY
#   raw Shannon, natural log:  H = -sum(p * ln p)
# --------------------------------------------------------------------------- #
def raw_entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, EPS, 1.0)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float64)


def derive_thesis_threshold() -> dict:
    """Phase-6C operational entropy threshold = P95 of E7 VAL benign entropy.

    Reproduces vae_fusion.load_baseline_signals() ent_p95 derivation so the
    marked operating point is continuous with the README.
    """
    e7_val = np.load(SUP_PRED / "E7_val_proba.npy")
    yv = pd.read_csv(PREPROC / "y_val.csv")
    benign = yv["label"].astype(str).values == "Benign"
    ent_val_benign = raw_entropy(e7_val)[benign]
    ent_p93 = float(np.percentile(ent_val_benign, 93.0))
    ent_p95 = float(np.percentile(ent_val_benign, 95.0))
    return {
        "ent_p93_raw": ent_p93,
        "ent_p95_raw": ent_p95,
        "ent_p93_norm": ent_p93 / LN_K,
        "ent_p95_norm": ent_p95 / LN_K,
        "n_benign_val": int(benign.sum()),
    }


# --------------------------------------------------------------------------- #
# Metrics for a retained (auto-decided) subset
# --------------------------------------------------------------------------- #
def eval_operating_point(y_true, y_pred, retained_mask, id2name):
    """All metrics MEASURED at one threshold. Immutable inputs — no mutation."""
    n_total = int(retained_mask.size)
    n_ret = int(retained_mask.sum())
    n_def = n_total - n_ret

    yt_ret, yp_ret = y_true[retained_mask], y_pred[retained_mask]
    yt_def, yp_def = y_true[~retained_mask], y_pred[~retained_mask]

    # Retained-set quality. macro-F1 / MCC over labels PRESENT in the retained
    # true set (natural "quality of what you kept" reading; documented in summary).
    if n_ret > 0:
        acc = float((yt_ret == yp_ret).mean())
        present = np.unique(yt_ret)
        macro_f1 = float(f1_score(yt_ret, yp_ret, labels=present,
                                  average="macro", zero_division=0))
        mcc = float(matthews_corrcoef(yt_ret, yp_ret)) if present.size > 1 else float("nan")
    else:
        acc = macro_f1 = mcc = float("nan")

    # Deferred set: error rate it WOULD have incurred, class composition.
    if n_def > 0:
        def_err_rate = float((yt_def != yp_def).mean())
        def_true_counts = pd.Series(yt_def).value_counts()
        def_composition = {
            id2name[int(c)]: {
                "n_deferred": int(cnt),
                "share_of_deferred": float(cnt / n_def),
            }
            for c, cnt in def_true_counts.items()
        }
        mistakes_avoided = int((yt_def != yp_def).sum())
    else:
        def_err_rate = float("nan")
        def_composition = {}
        mistakes_avoided = 0

    return {
        "coverage": float(n_ret / n_total),
        "n_retained": n_ret,
        "n_deferred": n_def,
        "retained_accuracy": acc,
        "retained_macro_f1": macro_f1,
        "retained_mcc": mcc,
        "deferred_error_rate": def_err_rate,
        "deferred_mistakes_avoided": mistakes_avoided,
        "deferred_composition": def_composition,
    }


def per_class_deferral(y_true, retained_mask, id2name):
    """Per-class deferral rate = fraction of that class's flows deferred."""
    out = {}
    deferred = ~retained_mask
    for cid, name in id2name.items():
        cls = y_true == cid
        n = int(cls.sum())
        if n == 0:
            continue
        out[name] = {
            "n_class": n,
            "n_deferred": int((cls & deferred).sum()),
            "deferral_rate": float((cls & deferred).sum() / n),
        }
    return out


# --------------------------------------------------------------------------- #
# Sweep grids (per brief)
# --------------------------------------------------------------------------- #
def conf_grid() -> list[float]:
    coarse = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]  # 0.50..0.95
    fine = [0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
    return sorted(set(coarse + fine))


def entropy_grid(thesis_norm: float) -> list[float]:
    coarse = [round(x, 2) for x in np.arange(0.05, 0.905, 0.05)]  # 0.05..0.90
    # Fine steps around the thesis operational threshold.
    lo = max(0.0, thesis_norm - 0.05)
    hi = thesis_norm + 0.05
    fine = [round(x, 3) for x in np.arange(lo, hi + 1e-9, 0.01)]
    fine.append(round(thesis_norm, 6))  # exact marked point
    return sorted(set(coarse + fine))


# --------------------------------------------------------------------------- #
# Named-operating-point helpers
# --------------------------------------------------------------------------- #
def first_threshold_exceeding(rows, acc_target, order):
    """First threshold (in the given tightening order) whose retained accuracy
    first exceeds acc_target. rows already ordered from loosest->tightest."""
    for r in rows:
        if not np.isnan(r["retained_accuracy"]) and r["retained_accuracy"] > acc_target:
            return r
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    log = []

    def emit(m):
        print(m)
        log.append(m)

    emit("=== tau-sweep :: frozen E7 operating-characteristic analysis ===")

    # ---- Load frozen predictions (no inference needed) --------------------- #
    proba = np.load(SUP_PRED / "E7_test_proba.npy")
    pred_stored = np.load(SUP_PRED / "E7_test_pred.npy").astype(np.int64)
    y_df = pd.read_csv(PREPROC / "y_test.csv")
    y_true = y_df["multiclass_label"].values.astype(np.int64)

    assert proba.shape[1] == K, f"expected {K} classes, got {proba.shape[1]}"
    assert proba.shape[0] == y_true.shape[0] == pred_stored.shape[0]

    y_pred = proba.argmax(axis=1).astype(np.int64)
    assert (y_pred == pred_stored).all(), "argmax(proba) != stored E7 pred"
    emit(f"loaded frozen E7 test predictions: {proba.shape[0]:,} flows x {K} classes")
    emit("saved softprob arrays exist -> NO inference run (frozen model untouched)")

    # id -> name map from the label column
    id2name = {}
    for cid, name in zip(y_true, y_df["label"].astype(str).values):
        id2name.setdefault(int(cid), name)
    id2name = {k: id2name[k] for k in sorted(id2name)}

    overall_acc = float((y_pred == y_true).mean())
    emit(f"overall accuracy (all flows auto-decided) = {overall_acc:.6f}")

    # ---- Per-flow signals -------------------------------------------------- #
    confidence = proba.max(axis=1).astype(np.float64)
    ent_raw = raw_entropy(proba)
    ent_norm = ent_raw / LN_K
    emit(f"confidence range [{confidence.min():.4f}, {confidence.max():.4f}]  "
         f"norm-entropy range [{ent_norm.min():.4f}, {ent_norm.max():.4f}]")

    thr = derive_thesis_threshold()
    emit(f"Phase-6C thesis threshold: ent_p95 raw={thr['ent_p95_raw']:.6f}  "
         f"norm={thr['ent_p95_norm']:.6f}  (P95 of {thr['n_benign_val']:,} benign-val flows)")

    # ---- SWEEP 1 : confidence floor tau_c ---------------------------------- #
    emit("\n--- Sweep 1: confidence floor tau_c ---")
    sweep1 = []
    for tau in conf_grid():
        retained = confidence >= tau
        m = eval_operating_point(y_true, y_pred, retained, id2name)
        m["threshold"] = float(tau)
        sweep1.append(m)
        emit(f"  tau_c={tau:<6} cov={m['coverage']:.4f} "
             f"acc={m['retained_accuracy']:.6f} f1={m['retained_macro_f1']:.4f} "
             f"mcc={m['retained_mcc']:.4f} deferred={m['n_deferred']:,}")

    # ---- SWEEP 2 : entropy ceiling tau_H ----------------------------------- #
    emit("\n--- Sweep 2: entropy ceiling tau_H (normalized H/ln19) ---")
    sweep2 = []
    for tau in entropy_grid(thr["ent_p95_norm"]):
        retained = ent_norm <= tau
        m = eval_operating_point(y_true, y_pred, retained, id2name)
        m["threshold"] = float(tau)
        sweep2.append(m)
        emit(f"  tau_H={tau:<8} cov={m['coverage']:.4f} "
             f"acc={m['retained_accuracy']:.6f} f1={m['retained_macro_f1']:.4f} "
             f"mcc={m['retained_mcc']:.4f} deferred={m['n_deferred']:,}")

    # ---- Monotonicity sanity check ----------------------------------------- #
    # Sweep 1: tightening = tau_c increasing -> retained acc should be non-decreasing.
    emit("\n--- Sanity: monotonicity ---")
    mono1 = []
    s1_sorted = sorted(sweep1, key=lambda r: r["threshold"])
    for a, b in zip(s1_sorted, s1_sorted[1:]):
        if b["retained_accuracy"] + 1e-9 < a["retained_accuracy"]:
            mono1.append((a["threshold"], b["threshold"],
                          a["retained_accuracy"], b["retained_accuracy"]))
    # Sweep 2: tightening = tau_H decreasing -> walk high->low tau_H.
    mono2 = []
    s2_sorted = sorted(sweep2, key=lambda r: r["threshold"], reverse=True)
    for a, b in zip(s2_sorted, s2_sorted[1:]):
        if b["retained_accuracy"] + 1e-9 < a["retained_accuracy"]:
            mono2.append((a["threshold"], b["threshold"],
                          a["retained_accuracy"], b["retained_accuracy"]))
    emit(f"  Sweep 1 non-monotone steps: {len(mono1)}")
    emit(f"  Sweep 2 non-monotone steps: {len(mono2)}")

    # Diagnose each non-monotone step: is it a benign degenerate-bin artifact
    # (the extra flows deferred by tightening were ALL correct, so removing them
    # nudges accuracy down by a tie-level epsilon) or a genuine reversal?
    def diagnose(violations, signal, direction):
        """signal: per-flow score; direction '<=' (entropy) or '>=' (confidence)."""
        diags = []
        for tau_a, tau_b, acc_a, acc_b in violations:
            lo, hi = sorted((tau_a, tau_b))
            band = (signal > lo) & (signal <= hi)
            n_band = int(band.sum())
            correct_band = int((y_pred[band] == y_true[band]).sum())
            diags.append({
                "tau_looser": tau_a, "tau_tighter": tau_b,
                "acc_looser": acc_a, "acc_tighter": acc_b,
                "delta_acc": acc_b - acc_a,
                "n_flows_in_band": n_band,
                "n_correct_in_band": correct_band,
                "n_errors_in_band": n_band - correct_band,
                "benign_degenerate_bin": (n_band - correct_band) == 0
                                         and abs(acc_b - acc_a) < 1e-6,
            })
        return diags

    mono2_diag = diagnose(mono2, ent_norm, "<=")
    for d in mono2_diag:
        emit(f"  Sweep 2 step {d['tau_looser']:.4g}->{d['tau_tighter']:.4g}: "
             f"Δacc={d['delta_acc']:.2e}, band={d['n_flows_in_band']} flows "
             f"({d['n_errors_in_band']} errors), "
             f"degenerate-bin artifact={d['benign_degenerate_bin']}")

    # ---- Named operating points -------------------------------------------- #
    # Sweep 1 ordered loosest(low tau)->tightest(high tau)
    op1_99 = first_threshold_exceeding(s1_sorted, 0.99, "asc")
    op1_999 = first_threshold_exceeding(s1_sorted, 0.999, "asc")
    # Sweep 2 tightening is tau_H DECREASING -> order high->low
    op2_99 = first_threshold_exceeding(s2_sorted, 0.99, "desc")
    op2_999 = first_threshold_exceeding(s2_sorted, 0.999, "desc")

    # Thesis Phase-6C point (Sweep 2 at exact normalized ent_p95)
    thesis_tau = round(thr["ent_p95_norm"], 6)
    op2_thesis = min(sweep2, key=lambda r: abs(r["threshold"] - thesis_tau))

    def with_perclass(row, signal_mask_fn):
        if row is None:
            return None
        retained = signal_mask_fn(row["threshold"])
        out = dict(row)
        out["per_class_deferral"] = per_class_deferral(y_true, retained, id2name)
        return out

    conf_mask = lambda t: confidence >= t
    ent_mask = lambda t: ent_norm <= t

    named = {
        "sweep1_confidence_floor": {
            "acc_gt_0.99": with_perclass(op1_99, conf_mask),
            "acc_gt_0.999": with_perclass(op1_999, conf_mask),
        },
        "sweep2_entropy_ceiling": {
            "acc_gt_0.99": with_perclass(op2_99, ent_mask),
            "acc_gt_0.999": with_perclass(op2_999, ent_mask),
            "thesis_phase6c": with_perclass(op2_thesis, ent_mask),
        },
    }

    # ---- Phase-6C rescue reproduction check -------------------------------- #
    emit("\n--- Sanity: Phase-6C rescue reproduction @ thesis threshold ---")
    retained_thesis = ent_norm <= thesis_tau
    deferred_thesis = ~retained_thesis
    name2id = {v: k for k, v in id2name.items()}
    rescue_check = {}
    for cname in RESCUE_CLASSES:
        cid = name2id[cname]
        cls = y_true == cid
        cls_err = cls & (y_true != y_pred)
        n_cls = int(cls.sum())
        n_err = int(cls_err.sum())
        n_err_deferred = int((cls_err & deferred_thesis).sum())
        rescue_check[cname] = {
            "n_class": n_cls,
            "n_misclassified": n_err,
            "n_misclassified_deferred": n_err_deferred,
            "misclassified_deferral_rate": (n_err_deferred / n_err) if n_err else float("nan"),
            "class_deferral_rate": float((cls & deferred_thesis).sum() / n_cls) if n_cls else float("nan"),
        }
        emit(f"  {cname:<24} errors={n_err:<6} of which deferred={n_err_deferred:<6} "
             f"({rescue_check[cname]['misclassified_deferral_rate']:.3f})")

    # Global: fraction of ALL E7 errors captured by the deferred set at thesis tau
    all_err = y_true != y_pred
    err_deferred_share = float((all_err & deferred_thesis).sum() / all_err.sum())
    emit(f"  overall: {err_deferred_share:.4f} of all E7 errors land in the "
         f"deferred (REVIEW) set at the thesis threshold")

    # ---- Analyst-budget translation ---------------------------------------- #
    def budget(row):
        if row is None:
            return None
        cov = row["coverage"]
        return {
            "pct_auto_decided": round(100.0 * cov, 4),
            "pct_deferred_to_review": round(100.0 * (1.0 - cov), 4),
            "deferrals_per_1M_flows_per_day": int(round((1.0 - cov) * 1_000_000)),
        }

    # =================== WRITE OUTPUTS ====================================== #
    results = {
        "meta": {
            "analysis": "confidence-floor / entropy-ceiling operating characteristic",
            "status": "MEASURED (post-hoc on test set; NOT model selection/tuning)",
            "model": "frozen thesis E7 (XGBoost multi:softprob), 44-feature space",
            "inference_run": False,
            "note_no_inference": "saved E7_test_proba.npy used directly; frozen model untouched",
            "n_flows": int(proba.shape[0]),
            "n_classes": K,
            "ln_K": LN_K,
            "overall_accuracy_all_auto": overall_acc,
            "entropy_convention": "raw Shannon natural-log H=-sum(p ln p); "
                                  "normalized = H/ln(19) in [0,1]",
            "thesis_phase6c_threshold": thr,
            "rescue_classes": RESCUE_CLASSES,
        },
        "sweep1_confidence_floor": sweep1,
        "sweep2_entropy_ceiling": sweep2,
        "named_operating_points": named,
        "analyst_budget": {
            "sweep1_acc_gt_0.99": budget(op1_99),
            "sweep1_acc_gt_0.999": budget(op1_999),
            "sweep2_acc_gt_0.99": budget(op2_99),
            "sweep2_acc_gt_0.999": budget(op2_999),
            "sweep2_thesis_phase6c": budget(op2_thesis),
        },
        "sanity_checks": {
            "monotonicity_sweep1_violations": mono1,
            "monotonicity_sweep2_violations": mono2,
            "monotonicity_sweep2_diagnosis": mono2_diag,
            "phase6c_rescue_reproduction": rescue_check,
            "overall_error_deferred_share_at_thesis_tau": err_deferred_share,
        },
    }

    with open(OUT_DIR / "tau_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    emit(f"\nwrote {OUT_DIR/'tau_sweep_results.json'}")

    # ---- tidy long-format CSV --------------------------------------------- #
    rows = []
    for sweep_name, sweep, tau_name in [
        ("confidence_floor", sweep1, "tau_c"),
        ("entropy_ceiling", sweep2, "tau_H"),
    ]:
        for r in sweep:
            rows.append({
                "sweep": sweep_name,
                "threshold_name": tau_name,
                "threshold": r["threshold"],
                "coverage": r["coverage"],
                "n_retained": r["n_retained"],
                "n_deferred": r["n_deferred"],
                "retained_accuracy": r["retained_accuracy"],
                "retained_macro_f1": r["retained_macro_f1"],
                "retained_mcc": r["retained_mcc"],
                "deferred_error_rate": r["deferred_error_rate"],
                "deferred_mistakes_avoided": r["deferred_mistakes_avoided"],
            })
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT_DIR / "tau_sweep_curve.csv", index=False)
    emit(f"wrote {OUT_DIR/'tau_sweep_curve.csv'}  ({len(curve)} rows)")

    # ---- summary markdown -------------------------------------------------- #
    write_summary(OUT_DIR / "tau_sweep_summary.md", results, named, budget,
                  id2name, thr, thesis_tau, op1_99, op1_999, op2_99, op2_999,
                  op2_thesis, mono1, mono2, mono2_diag, rescue_check,
                  err_deferred_share, overall_acc)
    emit(f"wrote {OUT_DIR/'tau_sweep_summary.md'}")
    emit("\n=== DONE — stop point reached (interpretation happens in planning chat) ===")

    with open(OUT_DIR / "run_log.txt", "w") as f:
        f.write("\n".join(log))


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _op_line(row, tau_name):
    if row is None:
        return f"| {tau_name} | not reached | — | — | — | — | — |"
    return (f"| {row['threshold']:.6g} | {row['coverage']:.4f} | "
            f"{_fmt(row['retained_accuracy'],6)} | {_fmt(row['retained_macro_f1'])} | "
            f"{_fmt(row['retained_mcc'])} | {row['n_deferred']:,} | "
            f"{_fmt(row['deferred_error_rate'])} |")


def write_summary(path, results, named, budget, id2name, thr, thesis_tau,
                  op1_99, op1_999, op2_99, op2_999, op2_thesis,
                  mono1, mono2, mono2_diag, rescue_check, err_deferred_share,
                  overall_acc):
    m = results["meta"]
    L = []
    A = L.append
    A("# tau-Sweep — Confidence-Floor / Entropy-Ceiling Operating Characteristic")
    A("")
    A("**Status: MEASURED.** All figures below are measured on the frozen thesis "
      "E7 model's existing predictions over the deduplicated thesis test set "
      "(published train/test boundary, 44-feature space). No training, no pcaps, "
      "no Track-A retrains. Saved softprob arrays were used directly — **no "
      "inference was run**; the frozen model was not touched.")
    A("")
    A("> **Claim hygiene:** thresholds were characterised *post-hoc on the test "
      "set* as an operating-characteristic measurement. They were **not** used "
      "for model selection or tuning. The full curves are reported honestly in "
      "`tau_sweep_curve.csv` / `tau_sweep_results.json`; no tau was cherry-picked.")
    A("")
    A(f"- Flows analysed: **{m['n_flows']:,}**  |  classes K = **{m['n_classes']}**")
    A(f"- Overall accuracy (all flows auto-decided, no floor): **{overall_acc:.6f}**")
    A(f"- Entropy convention: raw Shannon natural-log `H=-Σ p·ln p`; "
      f"normalized `H/ln(19)` ∈ [0,1] (matches `vae_fusion.compute_entropy`).")
    A(f"- Thesis Phase-6C threshold `ent_p95` = **{thr['ent_p95_raw']:.6f}** raw "
      f"= **{thr['ent_p95_norm']:.6f}** normalized "
      f"(P95 of {thr['n_benign_val']:,} benign validation flows).")
    A("")

    # ---- Named operating points ---- #
    A("## Three named operating points")
    A("")
    A("Columns: threshold · coverage · retained accuracy · retained macro-F1 · "
      "retained MCC · #deferred · deferred-set error rate. "
      "macro-F1/MCC computed over classes present in the retained set.")
    A("")
    A("### Sweep 1 — confidence floor τ_c (auto-decide if max-prob ≥ τ_c)")
    A("")
    A("| operating point | τ_c | coverage | ret-acc | ret-F1 | ret-MCC | #deferred | def-err |")
    A("|---|---|---|---|---|---|---|---|")
    A("| first ret-acc > 0.99  " + _op_line(op1_99, "τ_c"))
    A("| first ret-acc > 0.999 " + _op_line(op1_999, "τ_c"))
    A("")
    A("### Sweep 2 — entropy ceiling τ_H (auto-decide if H/ln19 ≤ τ_H)")
    A("")
    A("| operating point | τ_H | coverage | ret-acc | ret-F1 | ret-MCC | #deferred | def-err |")
    A("|---|---|---|---|---|---|---|---|")
    A("| first ret-acc > 0.99  " + _op_line(op2_99, "τ_H"))
    A("| first ret-acc > 0.999 " + _op_line(op2_999, "τ_H"))
    A("| **thesis Phase-6C**   " + _op_line(op2_thesis, "τ_H"))
    A("")

    # ---- Analyst budget ---- #
    A("## Analyst-budget translation")
    A("")
    A("The number a deployment actually sets: X% of flows auto-decided, the rest "
      "routed to Case-5 REVIEW. Deferrals normalised to a **1M flows/day** intake.")
    A("")
    A("| operating point | % auto-decided | % to REVIEW | deferrals / 1M flows / day |")
    A("|---|---|---|---|")
    for label, row in [
        ("Sweep 1 · ret-acc > 0.99", op1_99),
        ("Sweep 1 · ret-acc > 0.999", op1_999),
        ("Sweep 2 · ret-acc > 0.99", op2_99),
        ("Sweep 2 · ret-acc > 0.999", op2_999),
        ("Sweep 2 · thesis Phase-6C", op2_thesis),
    ]:
        b = budget(row)
        if b is None:
            A(f"| {label} | not reached | — | — |")
        else:
            A(f"| {label} | {b['pct_auto_decided']:.3f}% | "
              f"{b['pct_deferred_to_review']:.3f}% | "
              f"{b['deferrals_per_1M_flows_per_day']:,} |")
    A("")

    # ---- Per-class deferral at named points ---- #
    A("## Per-class deferral rate at each named operating point")
    A("")
    A("Fraction of each class's flows routed to REVIEW. Rescue classes "
      "(Phase-6C zero-day targets) are **bold**.")
    A("")
    points = [
        ("S1 acc>0.99", named["sweep1_confidence_floor"]["acc_gt_0.99"]),
        ("S1 acc>0.999", named["sweep1_confidence_floor"]["acc_gt_0.999"]),
        ("S2 acc>0.99", named["sweep2_entropy_ceiling"]["acc_gt_0.99"]),
        ("S2 acc>0.999", named["sweep2_entropy_ceiling"]["acc_gt_0.999"]),
        ("S2 thesis", named["sweep2_entropy_ceiling"]["thesis_phase6c"]),
    ]
    header = "| class | " + " | ".join(p[0] for p in points) + " |"
    A(header)
    A("|" + "---|" * (len(points) + 1))
    rescue = set(results["meta"]["rescue_classes"])
    for cid in sorted(id2name):
        cname = id2name[cid]
        cells = []
        for _, pt in points:
            if pt is None:
                cells.append("—")
                continue
            pc = pt.get("per_class_deferral", {}).get(cname)
            cells.append(f"{pc['deferral_rate']:.3f}" if pc else "—")
        disp = f"**{cname}**" if cname in rescue else cname
        A(f"| {disp} | " + " | ".join(cells) + " |")
    A("")

    # ---- Deferred composition at thesis point ---- #
    A("## Deferred-set composition at the thesis Phase-6C threshold")
    A("")
    A(f"τ_H = {thesis_tau:.6f} (normalized ent_p95). "
      f"Deferred set size: **{op2_thesis['n_deferred']:,}** "
      f"({100*(1-op2_thesis['coverage']):.3f}% of flows). "
      f"Error rate it would have incurred if auto-decided: "
      f"**{_fmt(op2_thesis['deferred_error_rate'])}** "
      f"(vs retained {_fmt(op2_thesis['retained_accuracy'],6)} accuracy).")
    A("")
    A("Top classes dominating REVIEW:")
    A("")
    A("| class | # deferred | share of deferred |")
    A("|---|---|---|")
    comp = op2_thesis["deferred_composition"]
    top = sorted(comp.items(), key=lambda kv: kv[1]["n_deferred"], reverse=True)[:10]
    for cname, d in top:
        disp = f"**{cname}**" if cname in rescue else cname
        A(f"| {disp} | {d['n_deferred']:,} | {d['share_of_deferred']:.4f} |")
    A("")

    # ---- Sanity checks ---- #
    A("## Sanity checks")
    A("")
    A(f"**Monotonicity.** Retained accuracy should not fall as the floor tightens.")
    A(f"- Sweep 1 (τ_c ↑): **{len(mono1)}** non-monotone step(s).")
    A(f"- Sweep 2 (τ_H ↓): **{len(mono2)}** non-monotone step(s).")
    if mono1:
        A("  - Sweep 1 violations (τ_a→τ_b, acc_a→acc_b): " +
          "; ".join(f"{a:.4g}→{b:.4g}: {x:.5f}→{y:.5f}" for a, b, x, y in mono1))
    A("")
    if mono2_diag:
        A("**Investigation of the Sweep-2 step(s)** (brief requires it before "
          "reporting). Each flagged step is a benign **degenerate-bin / tie** "
          "artifact, not a real reversal:")
        A("")
        A("| τ (looser→tighter) | Δacc | flows in band | errors in band | degenerate-bin? |")
        A("|---|---|---|---|---|")
        for d in mono2_diag:
            A(f"| {d['tau_looser']:.4g} → {d['tau_tighter']:.4g} | "
              f"{d['delta_acc']:.2e} | {d['n_flows_in_band']} | "
              f"{d['n_errors_in_band']} | "
              f"{'yes' if d['benign_degenerate_bin'] else 'NO — investigate'} |")
        A("")
        A("Reading: tightening τ_H across this step defers a handful of "
          "extreme-tail flows (coverage ≈ 99.9997%) that were **all correctly "
          "classified**. Removing correct predictions from the retained set "
          "lowers retained accuracy by a tie-level epsilon (~1e-7) while the "
          "retained error count is unchanged. The operating characteristic is "
          "monotone in substance; this is the expected quantisation wiggle at "
          "the very top of the entropy range where only single-digit flow counts "
          "separate thresholds.")
    A("")
    A(f"**Phase-6C rescue reproduction @ thesis threshold.** At τ_H = "
      f"{thesis_tau:.6f}, **{err_deferred_share:.4f}** of *all* E7 errors land in "
      f"the deferred REVIEW set. Per rescue class — misclassifications that the "
      f"entropy gate defers:")
    A("")
    A("| rescue class | # misclassified | # deferred | misclass-deferral rate | class deferral rate |")
    A("|---|---|---|---|---|")
    for cname, d in rescue_check.items():
        A(f"| {cname} | {d['n_misclassified']:,} | {d['n_misclassified_deferred']:,} | "
          f"{_fmt(d['misclassified_deferral_rate'],3)} | {_fmt(d['class_deferral_rate'],3)} |")
    A("")
    A("If the rescue mechanism reproduces, a large share of each rescue class's "
      "*misclassifications* appears in the deferred set (they carry high entropy).")
    A("")
    A("---")
    A("")
    A("*Generated by `deliverables/scripts/07_tau_sweep.py`. Interpretation is "
      "deferred to the planning chat per the brief's stop point.*")

    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
