"""Step 09 - overlap of arms 1-4 re-measured in the model's own float32 space (added after reviewing step 06b; NOT pre-declared).

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/09_float32_overlap_arms1to4.py --scratch <dir>   (<dir> = the step-05 scratch)

Why: step 05 hashed the float64 values, but XGBoost's histogram trainer reads float32. Two float64 vectors that differ only below float32
precision are one input for the model, and the float64 key calls them different. Step 06b found this for the CSV tables (0.85% / 3.3%
within-dataset after a float64 dedup). Here: the same 128-bit key over float32 values widened to float64, arms 1-4.
Arms 3-4: the models are re-fitted with the same seeds (control: unrestricted pooled AUC / macros equal results/05_detection_arms.json) and the
non-overlap pooled AUC and common macro are recomputed under the float32 key. Arms 1-2: overlap share only (test rows whose vector is on the train side).
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


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(C.SRC_DIR, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S5, S8 = _load("step05", "05_detection_arms.py"), _load("step08", "08_sensitivities.py")
FAMS = S5.FAMS


def keys32(X):
    parts = []
    for s in range(0, len(X), 500_000):
        parts.append(S5.row_hashes(np.asarray(X[s:s + 500_000], dtype=np.float32).astype(np.float64)))
    h = np.concatenate(parts)
    return pd.MultiIndex.from_arrays([h[:, 0], h[:, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()
    config, models = S5.load_config_and_models()
    cap_a, cap_b = int(config.ATTACK_CAP), int(config.BENIGN_CAP)
    ref = json.load(open(os.path.join(C.RESULTS_DIR, "05_detection_arms.json")))
    meta = {w: json.load(open(os.path.join(args.scratch, f"cache_{w}", "meta.json"))) for w in ("iomt", "iot")}
    D = {w: S5.open_cache(w, os.path.join(args.scratch, f"cache_{w}")) for w in meta}
    totals = {w: {f: int((np.asarray(D[w]["fam"]) == k).sum()) for k, f in enumerate(FAMS)} for w in D}
    t0 = time.time()
    K = {w: keys32(D[w]["X"]) for w in D}
    print("float32 keys built", f"{time.time() - t0:.0f}s", flush=True)
    out = {"within_dataset_share": {}, "cross_dataset": {}, "control_max_abs_diff": 0.0}
    for arm in ("arm1_within_iomt", "arm2_within_iot"):
        w = S8.ARMS[arm][0]
        per = {}
        for seed in (C.SEEDS if arm == "arm2_within_iot" else (42,)):
            tr, a, te, b = S8.arm_rows(arm, D, meta, seed, cap_a, cap_b)
            # full train side = every row on the train side of the split, not the capped sample
            if arm == "arm1_within_iomt":
                side, _ = S5.split_arm1(D["iomt"], meta["iomt"]["files"])
            else:
                side, _ = S5.split_arm2(D["iot"], meta["iot"]["files"], seed)
            tr_full, te_full = np.where(side == 0)[0], np.where(side == 1)[0]
            per[str(seed)] = {"share_test_rows_on_train_side": float(K[w][te_full].isin(K[w][tr_full]).mean()), "n_test": int(len(te_full))}
        out["within_dataset_share"][arm] = per
        print(arm, per, flush=True)
    for arm in ("arm3_iomt_to_iot", "arm4_iot_to_iomt"):
        a, b = S8.ARMS[arm]
        per = {}
        for seed in C.SEEDS:
            tr, _, te, _ = S8.arm_rows(arm, D, meta, seed, cap_a, cap_b)
            score, _ = S5.fit_predict(models, D[a], tr, D[b], te, seed)
            fb = np.asarray(D[b]["fam"])[te]
            full = S5.evaluate(fb, score, totals[b])
            ref_e = ref["per_seed"][arm][str(seed)]
            diff = max(abs(full[k] - ref_e[k]) for k in ("pooled_auc", "macro_common", "benign_recall"))
            out["control_max_abs_diff"] = max(out["control_max_abs_diff"], diff)
            ov = np.asarray(K[b][te].isin(K[a]))
            non = S5.evaluate(fb, score, totals[b], keep_mask=~ov)
            per[str(seed)] = {"overlap_share_float32_key": float(ov.mean()), "overlap_share_float64_key_step05": ref_e["overlap_full_table"]["share_all"],
                              "overlap_by_family_float32_key": {FAMS[k]: float(ov[fb == k].mean()) if (fb == k).any() else None for k in range(len(FAMS))},
                              "non_overlap_pooled_auc_float32_key": non["pooled_auc"], "non_overlap_macro_common_float32_key": non["macro_common"],
                              "non_overlap_pooled_auc_float64_key_step05": ref_e["non_overlap_vs_full_table"]["pooled_auc"],
                              "non_overlap_macro_common_float64_key_step05": ref_e["non_overlap_vs_full_table"]["macro_common"], "refit_control_max_abs_diff": diff}
            print(arm, seed, {k: per[str(seed)][k] for k in ("overlap_share_float32_key", "overlap_share_float64_key_step05", "non_overlap_pooled_auc_float32_key", "non_overlap_macro_common_float32_key")},
                  f"diff {diff:.1e} {time.time() - t0:.0f}s", flush=True)
        out["cross_dataset"][arm] = per
    payload = {"meta": C.provenance(__file__, [os.path.join(C.RESULTS_DIR, "05_detection_arms.json")], seeds=C.SEEDS, inputs_note="step-05 row caches in scratch", hash_inputs=True),
               "note": "added after review, not pre-declared; key = 128-bit hash of the cached values cast to float32 and widened to float64 (not a byte compare)", **out}
    print("control max abs diff", out["control_max_abs_diff"])
    print("wrote", C.write_json("09_float32_overlap_arms1to4.json", payload), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
