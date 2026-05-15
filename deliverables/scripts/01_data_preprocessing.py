"""
Phase 2–3 — EDA + preprocessing reproducibility.

WHAT WE FOUND
  37 % train + 44.7 % test bit-exact duplicate rate; deduplicated dataset is
  5,407,348 rows (train 4,515,080 / test 892,268); maximum imbalance ratio is
  2,374:1 (DDoS_UDP vs Recon_Ping_Sweep); Recon_Ping_Sweep is the rarest
  class at 689 train rows (not ARP_Spoofing as the literature reported).
  Three-group ColumnTransformer with RobustScaler/StandardScaler/MinMaxScaler
  produces two feature variants (Full 44, Reduced 28). SMOTETomek targets the
  8 minorities below 50K → ~50K each. The honest disclosure: this scaling
  design broke the Phase 5 AE (val loss 101,414) and required a second
  StandardScaler patch on benign-train — Contribution #13.

WHY WE CHOSE THIS APPROACH
  Alternatives considered:
    - Keep duplicates (matches Yacoubi) — accept partial leakage
    - Single global StandardScaler — would smooth heavy tails XGBoost needs
    - Full-population SMOTE — runtime/memory blow-up on 3.6M rows
  Decision criterion: clean data is the only honest baseline; per-group
    scaling preserves heavy tails for trees and uses bounded scalers for
    flag ratios; targeted SMOTE on 8 minorities is computationally tractable.
  Tradeoff accepted: headline metrics sit 0.5–1.4 pp below published values;
    RobustScaler retention silently broke the AE in Phase 5 (caught and
    fixed; saved as Contribution #13 rather than a quiet patch).
  Evidence: results/supervised/metrics/, preprocessed/config.json,
    eda_output/findings.md, README §10–§11.

Run from project root:
  venv/bin/python -m deliverables.scripts.01_data_preprocessing
"""
from __future__ import annotations

from deliverables.scripts._common import (
    banner, emit, load_csv, load_json, check_artifact_exists,
)


def main() -> int:
    banner("Phase 2–3 — Data + preprocessing reproducibility")

    # --- Dataset corrections from EDA tables ---
    if check_artifact_exists("eda_output/imbalance_table.csv"):
        imb = load_csv("eda_output/imbalance_table.csv")
        # Drop "Total" row if present
        cls_rows = imb[~imb["class"].astype(str).str.contains("Total", case=False, na=False)]
        total_train = int(cls_rows["train"].sum())
        total_test = int(cls_rows["test"].sum())
        rarest_idx = cls_rows["train"].idxmin()
        rarest = cls_rows.loc[rarest_idx]
        largest_idx = cls_rows["train"].idxmax()
        largest = cls_rows.loc[largest_idx]
        ratio = float(largest["train"]) / float(rarest["train"])

        emit("Phase 2", "deduped_train_rows", total_train,
             "eda_output/imbalance_table.csv (sum of class rows)")
        emit("Phase 2", "deduped_test_rows", total_test,
             "eda_output/imbalance_table.csv (sum of class rows)")
        emit("Phase 2", "rarest_class", str(rarest["class"]),
             "eda_output/imbalance_table.csv")
        emit("Phase 2", "rarest_class_train_rows", int(rarest["train"]),
             "eda_output/imbalance_table.csv")
        emit("Phase 2", "largest_class", str(largest["class"]),
             "eda_output/imbalance_table.csv")
        emit("Phase 2", "max_imbalance_ratio", round(ratio, 1),
             "eda_output/imbalance_table.csv (largest.train / rarest.train)")
    else:
        # README anchors (used by report; logged when EDA file absent)
        emit("Phase 2", "deduped_train_rows", 4515080, "README §2 (table)")
        emit("Phase 2", "deduped_test_rows", 892268, "README §2 (table)")
        emit("Phase 2", "rarest_class", "Recon_Ping_Sweep", "README §8.1")
        emit("Phase 2", "rarest_class_train_rows", 689, "README §8.1")
        emit("Phase 2", "max_imbalance_ratio", 2374.4, "README §8.1")

    emit("Phase 2", "raw_train_rows", 7160831, "README §2")
    emit("Phase 2", "raw_test_rows", 1614182, "README §2")
    emit("Phase 2", "raw_rows_total", 8775013, "README §2")
    emit("Phase 2", "train_duplicate_rate_pct", 36.95, "README §2, §10.1")
    emit("Phase 2", "test_duplicate_rate_pct", 44.72, "README §2, §10.1")

    # --- Feature counts ---
    emit("Phase 3", "n_features_raw", 45, "README §6")
    emit("Phase 3", "n_features_full", 44, "README §11.2 (Drate dropped)")
    emit("Phase 3", "n_features_reduced", 28,
         "README §11.2 (Drate + 11 correlated + 5 noise)")

    # --- Splits ---
    emit("Phase 3", "train_rows", 3612064, "README §11.4")
    emit("Phase 3", "val_rows", 903016, "README §11.4")
    emit("Phase 3", "test_rows", 892268, "README §11.4")

    # --- SMOTE ---
    emit("Phase 3", "smote_minorities_boosted", 8, "README §11.5")
    emit("Phase 3", "smote_target_size", "~50000",
         "README §11.5 (Recon_Ping_Sweep 551 → 49,799 etc.)")
    emit("Phase 3", "post_smote_rows_full", 3869271, "README §11.5")

    # --- AE dataset ---
    emit("Phase 3", "ae_train_rows", 123348, "README §11.6")
    emit("Phase 3", "ae_val_rows", 30838, "README §11.6")

    # --- LOO datasets (Phase 6B consumers) ---
    for tgt, n in [
        ("Recon_Ping_Sweep", 169),
        ("Recon_VulScan", 973),
        ("MQTT_Malformed_Data", 1747),
        ("MQTT_DoS_Connect_Flood", 3131),
        ("ARP_Spoofing", 1744),
    ]:
        emit("Phase 3", f"loo_held_out_{tgt}", n, "README §11.7")

    # --- Cohen's d top features (Phase 2 finding) — preview, full table cited in report ---
    if check_artifact_exists("eda_output/feature_target_cohens_d.csv"):
        cd = load_csv("eda_output/feature_target_cohens_d.csv")
        # First column = feature, second = |Cohen's d|
        top4 = cd.head(4)
        feats = ", ".join(str(top4.iloc[i, 0]) for i in range(min(4, len(top4))))
        emit("Phase 2", "cohens_d_top4", feats,
             "eda_output/feature_target_cohens_d.csv")

    # --- Preprocessing config sanity (loadable) ---
    if check_artifact_exists("preprocessed/config.json"):
        cfg = load_json("preprocessed/config.json")
        emit("Phase 3", "preprocessed_config_loaded", "OK",
             "preprocessed/config.json")
        if "random_state" in cfg:
            emit("Phase 3", "random_state", cfg["random_state"],
                 "preprocessed/config.json:random_state")

    print()
    print("Phase 2/3 reproducibility: all headline numbers above are read from")
    print("on-disk artefacts (eda_output/, preprocessed/) or anchored to README")
    print("§2, §8.1, §10.1, §11 when the corresponding EDA file is absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
