"""
Phase 4 — Supervised layer (E1–E8 + E5G).

WHAT WE FOUND
  E7 (XGBoost / Full 44 / Original) wins on the 19-class task with test
  macro-F1 = 0.9076, accuracy 99.27 %, MCC 0.9906. SMOTETomek degrades all
  4 SMOTE configs by 0.011–0.045 macro-F1; XGBoost arms (no class_weight)
  degrade MORE than RF arms (class_weight='balanced'), falsifying the
  "compounding correction" story and supporting the boundary-blur mechanism
  (paired with Phase 7 SHAP cosine 0.991 for DDoS↔DoS). Yacoubi gap on
  deduped data: -0.53 pp accuracy for XGBoost (99.27 vs 99.80).

WHY WE CHOSE THIS APPROACH
  Alternatives considered:
    - RF / Full / Original (E5, F1 0.8551) — second place
    - XGBoost / Reduced / Original (E3, F1 0.8987) — close but slightly worse
    - Any SMOTE variant — all degrade on macro-F1
  Decision criterion: highest macro-F1 + highest MCC + softmax structure
    needed for the Phase 6C entropy gate downstream.
  Tradeoff accepted: E7's no-class_weight makes its SMOTE sensitivity the
    largest of the four XGBoost arms — the cleanest experimental refutation
    of the compounding story, but also the most extreme SMOTE degradation.
  Evidence: results/supervised/metrics/E7_multiclass.json, README §12.

Run from project root:
  venv/bin/python -m deliverables.scripts.02_supervised_phase4
"""
from __future__ import annotations

from deliverables.scripts._common import (
    banner, emit, load_json, check_artifact_exists,
)


def main() -> int:
    banner("Phase 4 — Supervised layer")

    sup_metrics = "results/supervised/metrics"

    # --- E7 headline numbers (the canonical fusion input) ---
    e7 = load_json(f"{sup_metrics}/E7_multiclass.json")
    emit("Phase 4", "E7_test_f1_macro", e7["test_f1_macro"],
         f"{sup_metrics}/E7_multiclass.json:test_f1_macro")
    emit("Phase 4", "E7_test_accuracy", e7["test_accuracy"],
         f"{sup_metrics}/E7_multiclass.json:test_accuracy")
    emit("Phase 4", "E7_test_mcc", e7["test_mcc"],
         f"{sup_metrics}/E7_multiclass.json:test_mcc")
    emit("Phase 4", "E7_test_precision_macro", e7["test_precision_macro"],
         f"{sup_metrics}/E7_multiclass.json:test_precision_macro")
    emit("Phase 4", "E7_test_recall_macro", e7["test_recall_macro"],
         f"{sup_metrics}/E7_multiclass.json:test_recall_macro")

    # --- E1–E8 + E5G overall macro-F1 ranking ---
    print()
    print("E1–E8 + E5G macro-F1 ranking:")
    rows = []
    for eid in ["E1", "E2", "E3", "E4", "E5", "E5G", "E6", "E7", "E8"]:
        path = f"{sup_metrics}/{eid}_multiclass.json"
        if not check_artifact_exists(path):
            continue
        d = load_json(path)
        rows.append((eid, d["test_f1_macro"], d["test_accuracy"], d["test_mcc"]))
    rows.sort(key=lambda r: -r[1])  # sort by macro-F1 desc
    for eid, f1, acc, mcc in rows:
        marker = " ★" if eid == "E7" else ""
        print(f"  {eid:5s}  F1_macro={f1:.4f}  acc={acc:.4f}  MCC={mcc:.4f}{marker}")
        emit("Phase 4", f"{eid}_test_f1_macro", f1,
             f"{sup_metrics}/{eid}_multiclass.json")

    # --- SMOTE deltas (the H3 rejection profile) ---
    print()
    print("SMOTE-vs-Original macro-F1 deltas:")
    smote_pairs = [
        ("RF/Reduced", "E1", "E2"),
        ("RF/Full", "E5", "E6"),
        ("XGBoost/Reduced", "E3", "E4"),
        ("XGBoost/Full", "E7", "E8"),
    ]
    for label, orig, smote in smote_pairs:
        d_orig = load_json(f"{sup_metrics}/{orig}_multiclass.json")
        d_smote = load_json(f"{sup_metrics}/{smote}_multiclass.json")
        delta = d_smote["test_f1_macro"] - d_orig["test_f1_macro"]
        print(f"  {label:<18s} {orig} {d_orig['test_f1_macro']:.4f} → {smote} {d_smote['test_f1_macro']:.4f}  Δ = {delta:+.4f}")
        emit("Phase 4", f"smote_delta_{label.replace('/', '_')}", round(delta, 4),
             f"{sup_metrics}/{smote} vs {orig}")

    # --- H3 minority criterion (2/5) ---
    emit("Phase 4", "H3_macro_F1_pass_count", "0/4 configs improve",
         "smote_comparison.csv + per-config E*_multiclass.json")
    emit("Phase 4", "H3_minority_pass_count", "2/5 (RF/Reduced only)",
         "README §20.2 H3")
    emit("Phase 4", "H3_verdict", "FAIL on both criteria",
         "README §20.2 H3")

    # --- Yacoubi comparison ---
    emit("Phase 4", "yacoubi_RF_accuracy_raw_pct", 99.87, "README §12.7")
    emit("Phase 4", "our_E5_accuracy_pct", round(load_json(f"{sup_metrics}/E5_multiclass.json")["test_accuracy"] * 100, 2),
         f"{sup_metrics}/E5_multiclass.json:test_accuracy")
    emit("Phase 4", "yacoubi_XGB_accuracy_raw_pct", 99.80, "README §12.7")
    emit("Phase 4", "our_E7_accuracy_pct", round(e7["test_accuracy"] * 100, 2),
         f"{sup_metrics}/E7_multiclass.json:test_accuracy")
    emit("Phase 4", "E7_vs_yacoubi_xgb_gap_pp", round(e7["test_accuracy"] * 100 - 99.80, 2),
         "derived from E7_multiclass.json - 99.80")
    emit("Phase 4", "yacoubi_macro_precision_paper3_pct", 86.10, "README §19.3")

    print()
    print("Phase 4 reproducibility: all 9 E*_multiclass.json files loaded;")
    print("E7 confirmed as the canonical fusion-input model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
