"""Step 08 - two sensitivities on arms 1-4, declared in README "Changed after seeing arms 1-4", section C, before they were run.

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/08_sensitivities.py --scratch <dir>   (<dir> = the step-05 scratch with cache_iomt / cache_iot)

(a) arms 1-4, seeds 42/43/44: the flood families' attack TEST rows are limited to rows whose `protocol` matches the attack
    (ICMP=1, SYN/TCP=6, UDP=17). Benign test rows, scan/ARP test rows and the training data are unchanged; the models are re-fitted
    with the same seeds, and the UNRESTRICTED metrics must reproduce results/05_detection_arms.json (control).
(b) arms 2 and 4, seed 42: training = the four flood families + benign only. The capped draw is the arm's own draw with the scan/ARP
    rows removed afterwards, so the flood and benign rows are the same ones. The test set is unchanged.
Everything else (model, cap, embargo, splits, metrics) is imported from step 05.
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

spec = importlib.util.spec_from_file_location("step05", os.path.join(C.SRC_DIR, "05_detection_arms.py"))
S5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S5)
FAMS = S5.FAMS
FLOOD_PROTO = {"SYN": 6, "TCP": 6, "UDP": 17, "ICMP": 1}
FLOODS = list(FLOOD_PROTO)
ARMS = {"arm1_within_iomt": ("iomt", "iomt"), "arm2_within_iot": ("iot", "iot"), "arm3_iomt_to_iot": ("iomt", "iot"), "arm4_iot_to_iomt": ("iot", "iomt")}


def flood_macro(per_family):
    v = [per_family[f]["auc_vs_all_benign"] for f in FLOODS if per_family[f]["auc_vs_all_benign"] is not None]
    return float(np.mean(v)) if v else None


def arm_rows(arm, D, meta, seed, cap_a, cap_b):
    """(train idx, train dataset, test idx, test dataset) exactly as step 05 builds them."""
    a, b = ARMS[arm]
    if arm == "arm1_within_iomt":
        side, _ = S5.split_arm1(D["iomt"], meta["iomt"]["files"])
    elif arm == "arm2_within_iot":
        side, _ = S5.split_arm2(D["iot"], meta["iot"]["files"], seed)
    if a == b:
        tr_all, te = np.where(side == 0)[0], np.where(side == 1)[0]
    else:
        tr_all, te = np.arange(len(D[a]["fam"])), np.arange(len(D[b]["fam"]))
    fa = np.asarray(D[a]["fam"])
    return S5.capped_train_idx(fa, tr_all, seed, cap_a, cap_b), a, te, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()
    config, models = S5.load_config_and_models()
    cap_a, cap_b = int(config.ATTACK_CAP), int(config.BENIGN_CAP)
    ref = json.load(open(os.path.join(C.RESULTS_DIR, "05_detection_arms.json")))
    meta = {w: json.load(open(os.path.join(args.scratch, f"cache_{w}", "meta.json"))) for w in ("iomt", "iot")}
    assert all(m["control"]["passed"] for m in meta.values())
    D = {w: S5.open_cache(w, os.path.join(args.scratch, f"cache_{w}")) for w in meta}
    pidx = meta["iomt"]["cols"].index("protocol")
    assert meta["iot"]["cols"].index("protocol") == pidx
    totals = {w: {f: int((np.asarray(D[w]["fam"]) == k).sum()) for k, f in enumerate(FAMS)} for w in D}
    ben = FAMS.index("Benign")
    t0 = time.time()

    # ---- (a) protocol-matched flood test rows
    sa, control = {arm: {} for arm in ARMS}, {}
    for seed in C.SEEDS:
        for arm in ARMS:
            tr, a, te, b = arm_rows(arm, D, meta, seed, cap_a, cap_b)
            score, _ = S5.fit_predict(models, D[a], tr, D[b], te, seed)
            fb = np.asarray(D[b]["fam"])[te]
            proto = np.asarray(D[b]["X"][:, pidx])[te]
            full = S5.evaluate(fb, score, totals[b])
            drop = np.zeros(len(te), dtype=bool)
            n_by_fam = {}
            for f, want in FLOOD_PROTO.items():
                k = FAMS.index(f)
                m = fb == k
                drop |= m & (proto != want)
                n_by_fam[f] = {"n_test": int(m.sum()), "n_protocol_matching": int((m & (proto == want)).sum())}
            res = S5.evaluate(fb[~drop], score[~drop], totals[b])
            ref_e = ref["per_seed"][arm][str(seed)]
            diffs = {k: abs(full[k] - ref_e[k]) for k in ("pooled_auc", "macro_common", "macro_per_arm", "benign_recall", "balanced_accuracy")}
            control[f"{arm}/{seed}"] = {"max_abs_diff_vs_step05": max(diffs.values()), "diffs": diffs}
            sa[arm][str(seed)] = {
                "flood_rows": n_by_fam,
                "unrestricted": {"flood_macro": flood_macro(full["per_family"]), "macro_common": full["macro_common"], "pooled_auc": full["pooled_auc"],
                                 "flood_auc": {f: full["per_family"][f]["auc_vs_all_benign"] for f in FLOODS}},
                "restricted": {"flood_macro": flood_macro(res["per_family"]), "macro_common": res["macro_common"], "pooled_auc": res["pooled_auc"],
                               "benign_recall": res["benign_recall"], "balanced_accuracy": res["balanced_accuracy"], "n_test": res["n_test"],
                               "flood_auc": {f: res["per_family"][f]["auc_vs_all_benign"] for f in FLOODS},
                               "flood_small_support": {f: res["per_family"][f]["small_support"] for f in FLOODS}}}
            r = sa[arm][str(seed)]
            print("(a)", arm, seed, "flood macro", round(r["unrestricted"]["flood_macro"], 4), "->", round(r["restricted"]["flood_macro"], 4),
                  "control max diff", f"{control[f'{arm}/{seed}']['max_abs_diff_vs_step05']:.2e}", f"{time.time() - t0:.0f}s", flush=True)
    summary_a = {}
    for arm, per in sa.items():
        g = lambda path: [per[s][path[0]][path[1]] for s in per]  # noqa: E731
        def ms(v):
            return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
        summary_a[arm] = {"flood_macro_unrestricted": ms(g(("unrestricted", "flood_macro"))), "flood_macro_restricted": ms(g(("restricted", "flood_macro"))),
                          "macro_common_unrestricted": ms(g(("unrestricted", "macro_common"))), "macro_common_restricted": ms(g(("restricted", "macro_common"))),
                          "flood_auc_unrestricted": {f: ms([per[s]["unrestricted"]["flood_auc"][f] for s in per]) for f in FLOODS},
                          "flood_auc_restricted": {f: ms([per[s]["restricted"]["flood_auc"][f] for s in per]) for f in FLOODS}}

    # ---- (b) flood-only training, arms 2 and 4, seed 42
    sb = {}
    keep_fams = [FAMS.index(f) for f in FLOODS] + [ben]
    for arm in ("arm2_within_iot", "arm4_iot_to_iomt"):
        tr, a, te, b = arm_rows(arm, D, meta, 42, cap_a, cap_b)
        tr_f = tr[np.isin(np.asarray(D[a]["fam"])[tr], keep_fams)]
        score, _ = S5.fit_predict(models, D[a], tr_f, D[b], te, 42)
        fb = np.asarray(D[b]["fam"])[te]
        ev = S5.evaluate(fb, score, totals[b])
        cur = ref["per_seed"][arm]["42"]
        sb[arm] = {"train_rows_flood_only": int(len(tr_f)), "train_rows_current": int(len(tr)),
                   "flood_only": {k: ev[k] for k in ("benign_recall", "balanced_accuracy", "pooled_auc", "macro_common", "macro_per_arm", "attack_recall_pooled", "n_test", "n_benign")},
                   "current_step05": {k: cur[k] for k in ("benign_recall", "balanced_accuracy", "pooled_auc", "macro_common", "macro_per_arm", "attack_recall_pooled", "n_test", "n_benign")},
                   "recall_at_0.5_by_family_flood_only": {f: ev["per_family"][f]["recall_at_0.5"] for f in FAMS[:-1]},
                   "recall_at_0.5_by_family_current": {f: cur["per_family"][f]["recall_at_0.5"] for f in FAMS[:-1]},
                   "auc_by_family_flood_only": {f: ev["per_family"][f]["auc_vs_all_benign"] for f in FAMS[:-1]},
                   "auc_by_family_current": {f: cur["per_family"][f]["auc_vs_all_benign"] for f in FAMS[:-1]}}
        print("(b)", arm, "benign recall", round(cur["benign_recall"], 4), "->", round(ev["benign_recall"], 4), f"{time.time() - t0:.0f}s", flush=True)

    payload = {"meta": C.provenance(__file__, [os.path.join(C.RESULTS_DIR, "05_detection_arms.json")], seeds=C.SEEDS, inputs_note="step-05 row caches in scratch (see 05_detection_arms.json meta)", hash_inputs=True),
               "settings": "README.md 'Changed after seeing arms 1-4', section C", "flood_protocol_rule": FLOOD_PROTO,
               "control_unrestricted_refit_vs_step05": control, "control_max_abs_diff": max(v["max_abs_diff_vs_step05"] for v in control.values()),
               "a_protocol_matched_flood_test_rows": {"summary_over_seeds": summary_a, "per_seed": sa},
               "b_flood_only_training_seed42": sb}
    print("control max abs diff", payload["control_max_abs_diff"])
    print("wrote", C.write_json("08_sensitivities.json", payload), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
