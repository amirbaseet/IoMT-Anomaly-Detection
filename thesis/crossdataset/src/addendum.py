"""
Post-hoc addendum to the cross-dataset report, computed from the frozen
raw_results.json (no model re-run):

  1. Design A base-rate diagnostics — benign recall + balanced accuracy for
     XGBoost and the Flow AE, making explicit that the ~0.96 F1 is a base-rate
     artifact and ROC-AUC is the headline (incl. the AE inversion, AUC ~0.26).

  2. Design B robustness — DoS+DDoS collapsed into one "Flooding" family
     (4 families), the fairer transfer measure since the DoS/DDoS split is
     effectively per-flow-impossible.

Re-runnable and idempotent: it rewrites the section between the ADDENDUM
markers in results_summary.md each time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config

FAMILIES5 = ["Benign", "Spoofing", "Recon", "DoS", "DDoS"]
FAMILIES4 = ["Benign", "Spoofing", "Recon", "Flooding"]
MARK_START = "<!-- ADDENDUM:START -->"
MARK_END = "<!-- ADDENDUM:END -->"


def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=0))


def _fmt(vals: list[float]) -> str:
    m, s = _mean_std(vals)
    return f"{m:.4f} ± {s:.4f}"


# --------------------------------------------------------------------------- #
# Design A base-rate diagnostics from a 2x2 confusion [[bb,ba],[ab,aa]]
# --------------------------------------------------------------------------- #
def binary_rates(cm: list[list[int]]) -> dict:
    (bb, ba), (ab, aa) = cm
    benign_recall = bb / (bb + ba) if (bb + ba) else float("nan")
    attack_recall = aa / (ab + aa) if (ab + aa) else float("nan")
    return {"benign_recall": benign_recall, "attack_recall": attack_recall,
            "balanced_acc": (benign_recall + attack_recall) / 2}


# --------------------------------------------------------------------------- #
# Design B: collapse 5x5 (Benign,Spoofing,Recon,DoS,DDoS) -> 4x4 with Flooding
# --------------------------------------------------------------------------- #
def collapse_flooding(cm5: list[list[int]]) -> np.ndarray:
    m = np.asarray(cm5, dtype=float)          # rows=true, cols=pred, order FAMILIES5
    # merge indices 3 (DoS) and 4 (DDoS) in both axes
    rows = np.vstack([m[0:3], m[3:5].sum(axis=0, keepdims=True)])
    cm4 = np.hstack([rows[:, 0:3], rows[:, 3:5].sum(axis=1, keepdims=True)])
    return cm4                                 # order FAMILIES4


def per_class_f1(cm: np.ndarray) -> list[float]:
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return f1s


def macro_f1(cm: np.ndarray) -> float:
    return float(np.mean(per_class_f1(cm)))


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan")


# --------------------------------------------------------------------------- #
def build_addendum(raw: dict) -> str:
    seeds = raw["seeds"]
    per_seed = raw["per_seed"]
    ind = raw["in_dataset"]

    # ---- Design A base-rate diagnostics -----------------------------------
    xgb = [binary_rates(s["a_xgb"]["confusion"]) for s in per_seed]
    ae90 = [binary_rates(s["a_ae"]["p90"]["confusion"]) for s in per_seed]
    ae99 = [binary_rates(s["a_ae"]["p99"]["confusion"]) for s in per_seed]
    ind_xgb = binary_rates(ind["design_a_xgb"]["confusion"])
    xgb_auc = [s["a_xgb"]["roc_auc"] for s in per_seed]
    ae_auc = [s["a_ae"]["roc_auc"] for s in per_seed]

    # ---- Design B 4-family (Flooding) -------------------------------------
    cm4_seeds = [collapse_flooding(s["b"]["confusion"]) for s in per_seed]
    macro4 = [macro_f1(cm) for cm in cm4_seeds]
    acc4 = [accuracy(cm) for cm in cm4_seeds]
    f1_by_fam = {fam: [per_class_f1(cm)[i] for cm in cm4_seeds]
                 for i, fam in enumerate(FAMILIES4)}
    ind_cm4 = collapse_flooding(ind["design_b"]["confusion"])
    ind_macro4, ind_f1_4 = macro_f1(ind_cm4), per_class_f1(ind_cm4)

    L: list[str] = [MARK_START, ""]
    a = L.append
    a("## Addendum 1 — Design A base-rate diagnostics (Measured)")
    a("")
    a("The ~0.96 attack F1 is a **base-rate artifact**: CICIoT2023 is ~93% "
      "attack, so flagging nearly everything as attack scores high F1/recall. "
      "**ROC-AUC is the headline transfer metric**, and benign recall / "
      "balanced accuracy expose the collapsed benign side.")
    a("")
    a("| Detector | Benign recall | Attack recall | Balanced acc | ROC-AUC |")
    a("|---|---|---|---|---|")
    a(f"| XGBoost (cross) | {_fmt([r['benign_recall'] for r in xgb])} | "
      f"{_fmt([r['attack_recall'] for r in xgb])} | "
      f"{_fmt([r['balanced_acc'] for r in xgb])} | {_fmt(xgb_auc)} |")
    a(f"| XGBoost (in-dataset) | {ind_xgb['benign_recall']:.4f} | "
      f"{ind_xgb['attack_recall']:.4f} | {ind_xgb['balanced_acc']:.4f} | "
      f"{ind['design_a_xgb']['roc_auc']:.4f} |")
    a(f"| Flow AE p90 (cross) | {_fmt([r['benign_recall'] for r in ae90])} | "
      f"{_fmt([r['attack_recall'] for r in ae90])} | "
      f"{_fmt([r['balanced_acc'] for r in ae90])} | {_fmt(ae_auc)} |")
    a(f"| Flow AE p99 (cross) | {_fmt([r['benign_recall'] for r in ae99])} | "
      f"{_fmt([r['attack_recall'] for r in ae99])} | "
      f"{_fmt([r['balanced_acc'] for r in ae99])} | {_fmt(ae_auc)} |")
    a("")
    a(f"**Finding — AE inversion.** The Flow AE cross-dataset ROC-AUC is "
      f"**{_fmt(ae_auc)}**, i.e. **below 0.5**: under the CICIoMT2024-benign-"
      "trained autoencoder, CICIoT2023 *attacks* reconstruct with **lower** "
      "error than CICIoT2023 *benign*. The reconstruction-error signal is "
      "inverted across testbeds — flooding traffic sits closer to the "
      "CICIoMT2024 benign manifold than CICIoT2023's own benign does. Verified "
      "not a scaling artifact: the standardizer and imputer are fit on "
      "CICIoMT2024 only and applied transform-only to CICIoT2023 (no `.fit`/"
      "`.fit_transform` ever touches the test set).")
    a("")
    a("## Addendum 2 — Design B robustness: DoS+DDoS merged into 'Flooding' "
      "(4 families, Measured)")
    a("")
    a("The DoS vs DDoS distinction is effectively per-flow-impossible (it is a "
      "property of the campaign, not the individual flow), so collapsing them "
      "into one **Flooding** family is the fairer transfer measure. Computed "
      "post-hoc by merging DoS+DDoS in both truth and prediction of the "
      "existing 5-family model — no penalty for DoS↔DDoS confusion.")
    a("")
    a("| Metric | In-dataset (shared-feat) | Cross-dataset (CICIoT2023) |")
    a("|---|---|---|")
    a(f"| macro-F1 (4 families) | {ind_macro4:.4f} | {_fmt(macro4)} |")
    a(f"| accuracy | {accuracy(ind_cm4):.4f} | {_fmt(acc4)} |")
    a("")
    a("Per-family F1 (4-family):")
    a("")
    a("| Family | In-dataset F1 | Cross-dataset F1 |")
    a("|---|---|---|")
    for i, fam in enumerate(FAMILIES4):
        a(f"| {fam} | {ind_f1_4[i]:.4f} | {_fmt(f1_by_fam[fam])} |")
    a("")
    a("For reference, the 5-family macro-F1 was "
      f"{_fmt([s['b']['macro_f1'] for s in per_seed])}; merging DoS+DDoS moves "
      f"it to {_fmt(macro4)} — the delta quantifies how much of the Design-B "
      "degradation was DoS↔DDoS confusion versus genuine non-transfer.")
    a("")
    a(MARK_END)
    return "\n".join(L)


def main() -> None:
    out = config.OUT_DIR
    raw = json.loads((out / "raw_results.json").read_text())
    addendum = build_addendum(raw)

    summary_path = out / "results_summary.md"
    text = summary_path.read_text()
    if MARK_START in text and MARK_END in text:
        pre = text.split(MARK_START)[0].rstrip()
        post = text.split(MARK_END)[1]
        text = f"{pre}\n\n{addendum}\n{post}"
    else:
        text = text.rstrip() + "\n\n" + addendum + "\n"
    summary_path.write_text(text)
    print(addendum)
    print(f"\n[addendum] merged into {summary_path}")


if __name__ == "__main__":
    main()
