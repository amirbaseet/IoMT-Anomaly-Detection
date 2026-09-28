#!/usr/bin/env python3
"""
Inverse cross-dataset direction: TRAIN on CICIoT2023, TEST on CICIoMT2024.

Complements the forward run (train CICIoMT2024 -> test CICIoT2023, bal-acc 0.4998).
This is the SAME direction as Domenech et al. (2025) (train CICIoT2023 -> test
CICIoMT2024, 5-class F1 0.93->0.27-0.41) -> replication + bidirectional robustness.

TRAIN data: thesis/crossdataset/sampled_test_set.parquet  (the cached CICIoT2023 sample
            in the 38-feature shared schema: 38 features + raw_label).
TEST data:  IoMT-Project/data/test/*.pcap.csv  (CICIoMT2024; project to the SAME 38
            feature columns by NAME).

Binary task (attack=1, benign=0), matching the forward Design A. XGBoost only here
(trees need no scaler); the AE inverse is a follow-up. fit-on-train (CICIoT2023) only.
Device/MAC identity is never a feature (the shared schema is 38 behavioral fields).
"""
import glob, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             accuracy_score, recall_score)
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "..", "sampled_test_set.parquet")   # CICIoT2023, 38-feat
MT_TEST_DIR = os.path.join(HERE, "..", "..", "..", "data", "test")  # CICIoMT2024
CAP_PER_MT_FILE = 20000
SEED = 42


def main() -> None:
    # ---- TRAIN: CICIoT2023 (parquet) ----
    tr = pd.read_parquet(PARQUET)
    feat = [c for c in tr.columns if c != "raw_label"]
    Xtr = tr[feat].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    ytr = (tr["raw_label"].str.upper() != "BENIGN").astype(int).values
    print(f"TRAIN CICIoT2023: {len(tr)} rows | {len(feat)} shared features")

    # ---- TEST: CICIoMT2024 (data/test) projected to the SAME 38 columns by name ----
    frames = []
    for f in sorted(glob.glob(os.path.join(MT_TEST_DIR, "*.pcap.csv"))):
        d = pd.read_csv(f, nrows=200000)
        if len(d) > CAP_PER_MT_FILE:
            d = d.sample(CAP_PER_MT_FILE, random_state=SEED)
        base = os.path.basename(f)
        d["_y"] = 0 if base.lower().startswith("benign") else 1
        frames.append(d)
    mt = pd.concat(frames, ignore_index=True)
    missing = [c for c in feat if c not in mt.columns]
    if missing:
        print(f"WARNING: {len(missing)} shared features missing in CICIoMT2024: {missing}")
    use = [c for c in feat if c in mt.columns]
    Xte = mt[use].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    Xtr_use = Xtr[use]
    yte = mt["_y"].values
    print(f"TEST CICIoMT2024: {len(mt)} rows | benign {int((yte==0).sum())} | "
          f"attack {int((yte==1).sum())} | features aligned {len(use)}/{len(feat)}")

    # ---- train XGBoost on CICIoT2023, test on CICIoMT2024 ----
    m = xgb.XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1,
                          random_state=SEED, eval_metric="logloss")
    m.fit(Xtr_use, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    p = (proba >= 0.5).astype(int)

    auc = roc_auc_score(yte, proba)
    ba = balanced_accuracy_score(yte, p)
    print("\n=== INVERSE: train CICIoT2023 -> test CICIoMT2024 (binary) ===")
    print(f"  ROC-AUC        : {auc:.4f}   (threshold-independent oracle)")
    print(f"  balanced acc   : {ba:.4f}")
    print(f"  accuracy       : {accuracy_score(yte, p):.4f}")
    print(f"  benign recall  : {recall_score(yte, p, pos_label=0):.4f}")
    print(f"  attack recall  : {recall_score(yte, p, pos_label=1):.4f}")

    print("\n=== 2x2 GENERALIZATION TABLE (ROC-AUC / balanced acc) ===")
    print("                     test: CICIoMT2024      test: CICIoT2023")
    print(f"  train: MT          ~0.99 (within)         0.6904 / 0.4998 (fwd)")
    print(f"  train: IoT2023     {auc:.4f} / {ba:.4f} (INVERSE)   0.974 (within, B2)")
    print("\n  Domenech 2025 (same inverse direction): 5-class F1 0.93 -> 0.27-0.41")
    print("DONE")


if __name__ == "__main__":
    main()
