"""Step 04b - follow-up to step 04: (1) why seeds changed folds in two tasks and not in one,
(2) re-run of the task whose folds ignored the seed, (3) group-level permutation null of the origin task.

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/04b_followup.py [--perms 200]

Does NOT modify 04_origin_control.py or its result (their sha256 stay valid); reuses its loaders/constants.

(1) Mechanism (sklearn 1.8 StratifiedGroupKFold._iter_test_indices): `shuffle=True` permutes the groups, then
    `np.argsort(-np.std(y_counts_per_group, axis=1), kind="mergesort")` re-orders them by the std of their class
    counts. The shuffled order survives ONLY among groups with exactly equal std (e.g. identical count vectors).
    If every group has a distinct std, the sort erases the shuffle and the greedy fold assignment is identical
    for every seed. This script counts, per task, the tie classes and compares the fold partitions across seeds.
    Separately: the XGBClassifier here has subsample=colsample=1.0 and hist trees, so random_state does not change
    the model; the seed only enters through the fold assignment.
(2) Fix for a task whose partitions are seed-invariant: GroupKFold(shuffle=True, random_state=seed), which assigns
    shuffled groups without the std sort. Folds lacking a class in the test set are reported, not hidden.
(3) Null: the dataset label is shuffled across the 90 src_mac groups (each MAC keeps ONE label; 23 vs 67 groups
    kept), rows inherit their group's label, everything else identical to step 04 (59 inputs, same XGB params,
    StratifiedGroupKFold(5, shuffle, random_state=42)). Statistic = mean AUC over the folds whose test set has both
    classes. One fold seed (42) per permutation (200 x 5 fits; 3 seeds would triple the cost), so the matching
    observed value is the seed-42 mean; the 3-seed mean is reported alongside.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

C.bootstrap()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

spec = importlib.util.spec_from_file_location("step04", os.path.join(C.SRC_DIR, "04_origin_control.py"))
S4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S4)   # defines read/dedup_benign/XGB/TS; main() is guarded


def load_tables():
    ref = C.iot_manifest()
    iot_files = sorted((os.path.join(C.FEATURES_IOT, m["file"] + ".csv") for m in ref if m["class"] == "Benign"), key=os.path.basename)
    from stage2_device_disjoint import EXCLUDE_SUBSTR, filename_to_class
    iomt_all = [os.path.join(C.FEATURES_IOMT, f) for f in sorted(os.listdir(C.FEATURES_IOMT))
                if f.endswith(".csv") and not any(s in f for s in EXCLUDE_SUBSTR) and filename_to_class(f)]
    iomt_files = sorted([p for p in iomt_all if os.path.basename(p) < "Benign_test.pcap.csv" or filename_to_class(p) == "Benign"], key=os.path.basename)
    benign_names = {os.path.basename(p) for p in iomt_all if filename_to_class(p) == "Benign"}
    _, cols = S4.read(iot_files[:1])
    a = S4.dedup_benign(iomt_files, benign_names, cols)
    b = S4.dedup_benign(iot_files, {os.path.basename(p) for p in iot_files}, cols)
    return a, b, cols, iomt_files + iot_files


def partition(splitter, X, y, g):
    """Fold id per row as a canonical frozenset of frozensets of group names (order-free)."""
    folds = [frozenset(g[te]) for _, te in splitter.split(X, y, g)]
    return frozenset(folds)


def tie_stats(y, g):
    _, gi, _ = np.unique(g, return_inverse=True, return_counts=True)
    counts = np.zeros((gi.max() + 1, 2))
    for cls, grp in zip(y, gi):
        counts[grp, cls] += 1
    std = np.round(np.std(counts, axis=1), 9)
    _, c = np.unique(std, return_counts=True)
    return {"groups": int(len(std)), "distinct_std_values": int(len(c)), "groups_in_a_tie": int(c[c > 1].sum())}


def fit_eval(X, y, tr, te, seed):
    m = XGBClassifier(**S4.XGB, random_state=seed)
    m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    p = m.predict_proba(X[te])[:, 1]
    if len(set(y[te])) < 2:
        return None, None
    return float(roc_auc_score(y[te], p)), float(balanced_accuracy_score(y[te], p >= 0.5))


def cv(X, y, g, splitter_factory, seed):
    out = []
    for k, (tr, te) in enumerate(splitter_factory(seed).split(X, y, g)):
        assert not (set(g[tr]) & set(g[te]))
        auc, ba = fit_eval(X, y, tr, te, seed)
        out.append({"fold": k, "seed": seed, "test_class_sizes": [int((y[te] == c).sum()) for c in (0, 1)],
                    "train_class_sizes": [int((y[tr] == c).sum()) for c in (0, 1)],
                    "test_groups": int(len(set(g[te]))), "auc": auc, "balanced_accuracy": ba})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    a, b, cols, inputs = load_tables()
    assert len(a) == 10540 and len(b) == 108841

    both = pd.concat([a.assign(y=0), b.assign(y=1)], ignore_index=True)
    tasks = {}
    tasks["origin"] = (both[cols].to_numpy(), both["y"].to_numpy(), both["src_mac"].fillna("<NA>").to_numpy())
    ra = a.assign(y=(a["file"] == "Benign_test.pcap.csv").astype(int))
    tasks["ref_iomt_train_vs_test"] = (ra[cols].to_numpy(), ra["y"].to_numpy(), ra["src_mac"].fillna("<NA>").to_numpy())
    mid = (b[S4.TS].min() + b[S4.TS].max()) / 2
    rb = b.assign(y=(b[S4.TS] > mid).astype(int))
    tasks["ref_iot_first_vs_second_half"] = (rb[cols].to_numpy(), rb["y"].to_numpy(), rb["src_mac"].fillna("<NA>").to_numpy())

    sgk = lambda s: StratifiedGroupKFold(n_splits=S4.N_SPLITS, shuffle=True, random_state=s)  # noqa: E731
    gk = lambda s: GroupKFold(n_splits=S4.N_SPLITS, shuffle=True, random_state=s)  # noqa: E731
    diag = {}
    for name, (X, y, g) in tasks.items():
        parts = {s: partition(sgk(s), X, y, g) for s in C.SEEDS}
        diag[name] = {"ties": tie_stats(y, g),
                      "partitions_identical_42_43": parts[42] == parts[43], "partitions_identical_42_44": parts[42] == parts[44],
                      "n_distinct_partitions_over_3_seeds": len(set(parts.values()))}
        print(name, diag[name], flush=True)

    # (2) re-run of the seed-invariant task with a splitter that honours the seed
    X, y, g = tasks["ref_iomt_train_vs_test"]
    gk_parts = {s: partition(gk(s), X, y, g) for s in C.SEEDS}
    rerun_folds = [f for s in C.SEEDS for f in cv(X, y, g, gk, s)]
    aucs = {s: [f["auc"] for f in rerun_folds if f["seed"] == s and f["auc"] is not None] for s in C.SEEDS}
    seed_means = [float(np.mean(v)) for v in aucs.values() if v]
    bas = {s: [f["balanced_accuracy"] for f in rerun_folds if f["seed"] == s and f["balanced_accuracy"] is not None] for s in C.SEEDS}
    rerun = {"splitter": "GroupKFold(n_splits=5, shuffle=True, random_state=seed); NOT stratified",
             "n_distinct_partitions_over_3_seeds": len(set(gk_parts.values())),
             "per_fold": rerun_folds,
             "folds_without_both_classes_in_test": sum(f["auc"] is None for f in rerun_folds),
             "auc_mean_per_seed": {str(s): (float(np.mean(v)) if v else None) for s, v in aucs.items()},
             "bal_acc_mean_per_seed": {str(s): (float(np.mean(v)) if v else None) for s, v in bas.items()},
             "auc_mean_over_seeds": float(np.mean(seed_means)) if seed_means else None,
             "auc_sd_over_seeds": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None,
             "bal_acc_mean_over_seeds": float(np.mean([np.mean(v) for v in bas.values() if v])),
             "bal_acc_sd_over_seeds": float(np.std([np.mean(v) for v in bas.values() if v], ddof=1))}
    print("rerun", {k: v for k, v in rerun.items() if k != "per_fold"}, flush=True)

    # (3) permutation null, origin task
    X, y, g = tasks["origin"]
    names, gi = np.unique(g, return_inverse=True)
    glabel = np.array([int(y[gi == k][0]) for k in range(len(names))])
    assert all(len(set(y[gi == k])) == 1 for k in range(len(names))), "label not constant within a MAC"
    obs_folds = cv(X, y, g, sgk, 42)
    obs42 = float(np.mean([f["auc"] for f in obs_folds if f["auc"] is not None]))
    s4 = json.load(open(os.path.join(C.RESULTS_DIR, "04_origin_control.json")))["results"]["origin"]["summary"]
    obs3 = s4["over_seeds"]["auc_mean"]
    null, invalid_folds, t0 = [], 0, time.time()
    for i in range(args.perms):
        rng = np.random.default_rng(10_000 + i)
        yp = rng.permutation(glabel)[gi]
        folds = cv(X, yp, g, sgk, 42)
        ok = [f["auc"] for f in folds if f["auc"] is not None]
        invalid_folds += len(folds) - len(ok)
        null.append(float(np.mean(ok)) if ok else float("nan"))
        if i % 10 == 0:
            print(f"  perm {i + 1}/{args.perms} null={null[-1]:.3f} elapsed={time.time() - t0:.0f}s", flush=True)
    nv = np.array([v for v in null if not np.isnan(v)])
    pctl = lambda q: float(np.percentile(nv, q))  # noqa: E731
    perm = {"n_permutations": args.perms, "n_valid": int(len(nv)), "folds_skipped_single_class_test": int(invalid_folds),
            "permutation_seeds": "np.random.default_rng(10000 + i), i = 0..n-1; fold seed fixed at 42",
            "label_counts_preserved": {"groups_label0": int((glabel == 0).sum()), "groups_label1": int((glabel == 1).sum())},
            "null_auc_mean": float(nv.mean()), "null_auc_sd": float(nv.std(ddof=1)),
            "null_p2_5": pctl(2.5), "null_p97_5": pctl(97.5), "null_max": float(nv.max()),
            "observed_auc_seed42_mean_of_folds": obs42, "observed_auc_3seed_mean": obs3,
            "n_null_ge_observed_seed42": int((nv >= obs42).sum()), "n_null_ge_observed_3seed": int((nv >= obs3).sum()),
            "p_one_sided_seed42": float((1 + (nv >= obs42).sum()) / (1 + len(nv))),
            "p_one_sided_3seed": float((1 + (nv >= obs3).sum()) / (1 + len(nv))),
            "null_aucs": null}
    print({k: v for k, v in perm.items() if k != "null_aucs"}, flush=True)

    payload = {"meta": C.provenance(__file__, inputs, seeds=C.SEEDS + (10000,),
                                    inputs_note="same deduplicated benign rows as step 04 (control 10,540 / 108,841 asserted); CICIoMT2024 nfstream version NOT RECORDED"),
               "step04_files_untouched": {"script": "04_origin_control.py", "result": "04_origin_control.json"},
               "seed_mechanism_diagnostics": diag,
               "xgb_seed_note": "XGB params have no subsampling and hist trees: random_state does not change the model; the seed acts only via folds",
               "rerun_ref_iomt_train_vs_test": rerun, "permutation_null_origin": perm}
    print("wrote", C.write_json("04b_followup.json", payload))


if __name__ == "__main__":
    main()
