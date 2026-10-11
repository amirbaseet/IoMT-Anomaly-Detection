"""Step 04 - origin control: can a classifier tell CICIoMT2024 benign flows from CICIoT2023 benign flows?

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/04_origin_control.py

Owner ruling (after the step-03 report): group by src_mac ONLY. Grouping by capture is impossible:
each dataset has ONE independent benign capture (step 03), so a capture group would be a whole dataset.

Rows: deduplicated benign flows, 59 model-input columns, label = dataset.
  Dedup = stage2c global method. Benign files sort AFTER only the two ARP_Spoofing_* files in CICIoMT2024
  and FIRST in CICIoT2023, so deduplicating (ARP files + benign files) in sorted order and keeping the
  benign survivors is the exact global result. CONTROL: must equal step 03's 10,540 / 108,841.
Folds: StratifiedGroupKFold(5, shuffle, seed) with group = src_mac string; a MAC string seen in both
  datasets is ONE group, so no group sits on both sides of any split.
Model (a choice, not tuned): XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
  tree_method=hist, random_state=seed), class-balanced sample weights from the training fold.
Reference (same model, same folds protocol): within CICIoMT2024 Benign_train vs Benign_test;
  within CICIoT2023 first vs second half of the benign capture (split at the midpoint of the flow
  first-seen time span). Same dedup rows.
Metrics per fold and seed: ROC AUC (NaN if the test fold has one class) and balanced accuracy at 0.5.
Seeds 42, 43, 44.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

C.bootstrap()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

N_SPLITS = 5
TS = "bidirectional_first_seen_ms"
XGB = dict(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=4)


def read(paths):
    frames = []
    for p in paths:
        head = pd.read_csv(p, nrows=0)
        cols = C.model_input_columns(head.columns)
        df = pd.read_csv(p, usecols=cols + ["src_mac", TS], dtype={c: "float64" for c in cols})
        df["file"] = os.path.basename(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True), cols


def dedup_benign(files, benign_names, cols):
    df, _ = read(files)
    keep = ~df.duplicated(subset=cols, keep="first")
    return df[keep & df["file"].isin(benign_names)].reset_index(drop=True)


def run_cv(X, y, groups, seed):
    folds = []
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        assert not (set(groups[tr]) & set(groups[te])), "group on both sides"
        rec = {"fold": k, "seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)),
               "train_class_sizes": [int((y[tr] == c).sum()) for c in (0, 1)],
               "test_class_sizes": [int((y[te] == c).sum()) for c in (0, 1)],
               "train_groups": int(len(set(groups[tr]))), "test_groups": int(len(set(groups[te])))}
        if len(set(y[tr])) < 2:
            rec.update(auc=None, balanced_accuracy=None, note="train fold has one class")
        else:
            m = XGBClassifier(**XGB, random_state=seed)
            m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            p = m.predict_proba(X[te])[:, 1]
            two = len(set(y[te])) == 2
            rec.update(auc=float(roc_auc_score(y[te], p)) if two else None,
                       balanced_accuracy=float(balanced_accuracy_score(y[te], p >= 0.5)) if two else None,
                       note="" if two else "test fold has one class: metrics undefined")
        folds.append(rec)
    return folds


def summarise(folds):
    out = {}
    for s in sorted({f["seed"] for f in folds}):
        a = [f["auc"] for f in folds if f["seed"] == s and f["auc"] is not None]
        b = [f["balanced_accuracy"] for f in folds if f["seed"] == s and f["balanced_accuracy"] is not None]
        out[str(s)] = {"folds_with_metrics": len(a), "auc_mean": float(np.mean(a)) if a else None,
                       "bal_acc_mean": float(np.mean(b)) if b else None}
    aucs = [v["auc_mean"] for v in out.values() if v["auc_mean"] is not None]
    bas = [v["bal_acc_mean"] for v in out.values() if v["bal_acc_mean"] is not None]
    out["over_seeds"] = {"auc_mean": float(np.mean(aucs)) if aucs else None, "auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
                         "bal_acc_mean": float(np.mean(bas)) if bas else None, "bal_acc_sd": float(np.std(bas, ddof=1)) if len(bas) > 1 else None}
    return out


def experiment(name, X, y, groups, what):
    folds = [f for s in C.SEEDS for f in run_cv(X, y, groups, s)]
    print(name, summarise(folds)["over_seeds"], flush=True)
    return {"what": what, "rows": int(len(y)), "class_sizes_total": [int((y == c).sum()) for c in (0, 1)],
            "groups_total": int(len(set(groups))), "per_fold": folds, "summary": summarise(folds)}


def main():
    ref = C.iot_manifest()
    iot_files = [os.path.join(C.FEATURES_IOT, m["file"] + ".csv") for m in ref if m["class"] == "Benign"]
    iot_files.sort(key=os.path.basename)
    from stage2_device_disjoint import EXCLUDE_SUBSTR, filename_to_class
    iomt_all = [os.path.join(C.FEATURES_IOMT, f) for f in sorted(os.listdir(C.FEATURES_IOMT))
                if f.endswith(".csv") and not any(s in f for s in EXCLUDE_SUBSTR) and filename_to_class(f)]
    before_benign = [p for p in iomt_all if os.path.basename(p) < "Benign_test.pcap.csv"]
    assert all(filename_to_class(p) == "ARP_Spoofing" for p in before_benign), [os.path.basename(p) for p in before_benign]
    iomt_files = before_benign + [p for p in iomt_all if filename_to_class(p) == "Benign"]
    iomt_files.sort(key=os.path.basename)
    iomt_benign_names = {os.path.basename(p) for p in iomt_all if filename_to_class(p) == "Benign"}

    _, cols = read(iot_files[:1])
    a = dedup_benign(iomt_files, iomt_benign_names, cols)
    b = dedup_benign(iot_files, {os.path.basename(p) for p in iot_files}, cols)
    s3 = __import__("json").load(open(os.path.join(C.RESULTS_DIR, "03_family_counts.json")))
    control = {"iomt_got": int(len(a)), "iomt_expected_step03": s3["benign_share_in_table"]["CICIoMT2024"]["dedup_global"]["benign"],
               "iot_got": int(len(b)), "iot_expected_step03": s3["benign_share_in_table"]["CICIoT2023"]["dedup_global"]["benign"]}
    control["passed"] = control["iomt_got"] == control["iomt_expected_step03"] and control["iot_got"] == control["iot_expected_step03"]
    print("CONTROL", control, flush=True)
    if not control["passed"]:
        raise SystemExit("dedup control failed")

    both = pd.concat([a.assign(y=0), b.assign(y=1)], ignore_index=True)
    shared_macs = sorted(set(a["src_mac"].dropna()) & set(b["src_mac"].dropna()))
    X, y, g = both[cols].to_numpy(), both["y"].to_numpy(), both["src_mac"].fillna("<NA>").to_numpy()
    results = {"origin": experiment("origin", X, y, g, "dataset origin: CICIoMT2024 benign (0) vs CICIoT2023 benign (1), group = src_mac")}

    ra = a.assign(y=(a["file"] == "Benign_test.pcap.csv").astype(int))
    results["reference_cicIoMT2024_train_vs_test"] = experiment(
        "ref_iomt", ra[cols].to_numpy(), ra["y"].to_numpy(), ra["src_mac"].fillna("<NA>").to_numpy(),
        "within CICIoMT2024: Benign_train (0) vs Benign_test (1), group = src_mac")
    mid = (b[TS].min() + b[TS].max()) / 2
    rb = b.assign(y=(b[TS] > mid).astype(int))
    results["reference_cicIoT2023_first_vs_second_half"] = experiment(
        "ref_iot", rb[cols].to_numpy(), rb["y"].to_numpy(), rb["src_mac"].fillna("<NA>").to_numpy(),
        "within CICIoT2023: first (0) vs second (1) half of the benign capture by flow first-seen midpoint, group = src_mac")

    inputs = iomt_files + iot_files
    payload = {
        "meta": C.provenance(__file__, inputs, seeds=C.SEEDS,
                             inputs_note="CICIoMT2024 nfstream version NOT RECORDED; CICIoT2023 6.6.0 (manifest.jsonl)"),
        "owner_ruling": "group by src_mac only; capture grouping impossible (1 independent benign capture per dataset, step 03)",
        "model": {"class": "xgboost.XGBClassifier", "params": {**XGB, "random_state": "seed"}, "weights": "balanced per training fold",
                  "xgboost": __import__("xgboost").__version__, "tuned": False},
        "folds": {"splitter": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state=seed)", "group": "src_mac string"},
        "model_input_columns": cols,
        "dedup_control_vs_step03": control,
        "src_mac_strings_in_both_datasets": shared_macs,
        "results": results,
    }
    print("wrote", C.write_json("04_origin_control.json", payload))


if __name__ == "__main__":
    main()
