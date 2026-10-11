"""Step 10 - 4-family macro (SYN, TCP, UDP, OS scan; ICMP excluded) from existing results. NO model is fitted.

Label in every output: "added after the background audit" (README, 'ICMP decision').
Venv: ~/code/IoMT-Project/venv   Run: ~/code/IoMT-Project/venv/bin/python -B -I src/10_macro4.py
Reads results/05_detection_arms.json (arms 1-4, NFStream) and results/06_like_for_like.json (arm 5, CSV; both feature sets).
Per seed: mean of the per-family AUC (vs all benign) over the four families; then mean and sd (ddof=1) over seeds 42/43/44.
A family with auc None or small_support is NOT silently dropped: it is listed in `excluded` for that entry.
"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402

FOUR = ["SYN", "TCP", "UDP", "os_scan"]


def macro4(per_family):
    vals, excl = [], []
    for f in FOUR:
        e = per_family.get(f)
        if e is None or e["auc_vs_all_benign"] is None or e["small_support"]:
            excl.append(f)
        else:
            vals.append(e["auc_vs_all_benign"])
    return (float(np.mean(vals)) if vals else None), excl


def ms(v):
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}


def entry(per_seed, key):
    seeds = [str(s) for s in C.SEEDS if str(s) in per_seed]
    m, ex, fam = [], {}, {f: [] for f in FOUR}
    for s in seeds:
        v, e = macro4(per_seed[s]["per_family"])
        m.append(v); ex[s] = e
        for f in FOUR:
            fam[f].append(per_seed[s]["per_family"][f]["auc_vs_all_benign"])
    out = {"seeds": seeds, "macro4": ms(m) if None not in m else None, "macro4_by_seed": m, "excluded_by_seed": ex,
           "macro5_common_stored": ms([per_seed[s]["macro_common"] for s in seeds]),
           "per_family_auc": {f: (ms(v) if None not in v else None) for f, v in fam.items()}}
    return out


def main():
    a5 = json.load(open(os.path.join(C.RESULTS_DIR, "05_detection_arms.json")))
    a6 = json.load(open(os.path.join(C.RESULTS_DIR, "06_like_for_like.json")))
    res = {"nfstream": {arm: entry(a5["per_seed"][arm], arm) for arm in a5["per_seed"] if arm.startswith("arm")},
           "csv": {fs: {tk: entry(a6["per_seed"][fs][tk], tk) for tk in a6["per_seed"][fs]} for fs in a6["per_seed"]}}
    payload = {"meta": C.provenance(__file__, [os.path.join(C.RESULTS_DIR, "05_detection_arms.json"), os.path.join(C.RESULTS_DIR, "06_like_for_like.json")],
                                    seeds=C.SEEDS, inputs_note="existing result JSONs only; no model fitted", hash_inputs=True),
               "label": "added after the background audit", "families": FOUR, **res}
    for k, v in res["nfstream"].items():
        print("nfstream", k, v["macro4"], "vs 5-fam", v["macro5_common_stored"]["mean"])
    for fs, d in res["csv"].items():
        for tk, v in d.items():
            print(fs, tk, v["macro4"], "vs 5-fam", v["macro5_common_stored"]["mean"])
    print("wrote", C.write_json("10_macro4.json", payload))


if __name__ == "__main__":
    main()
