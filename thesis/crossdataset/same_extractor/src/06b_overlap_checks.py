"""Step 06b - checks added after reviewing step 06 (declared as added, not pre-declared): README "Results after the declaration".

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/06b_overlap_checks.py --scratch <dir>   (<dir> = scratch holding cache_csv_iomt / cache_csv_iot)

1. POSITIVE CONTROL for the overlap tool: test = 1,000 rows taken FROM the training table itself, must report 1.0 for both feature sets.
2. Within-dataset baselines, shared 31 columns: dedup used the 38-column key, so two rows that differ only in the 7 dropped columns
   are identical in the 31-column space. Share of test rows whose 31-column vector also occurs on the train side (full train side,
   not the capped sample). For shared 38 the same number is computed as a second control (must be 0 by construction).
Same hash key as step 06 (128-bit, two pandas row hashes) - not a byte compare.
"""
import argparse
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

C.bootstrap()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

spec = importlib.util.spec_from_file_location("step06", os.path.join(C.SRC_DIR, "06_like_for_like.py"))
S6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S6)


def keys(X64):
    h = S6.hashes(X64)
    return pd.MultiIndex.from_arrays([h[:, 0], h[:, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()
    sets, _ = S6.feature_sets()
    shared = sets["shared38"]
    D = {"iomt": S6.load_cache(os.path.join(args.scratch, "cache_csv_iomt"), ("X", "fam", "side")),
         "iot": S6.load_cache(os.path.join(args.scratch, "cache_csv_iot"), ("X", "fam", "fid"))}
    out = {"positive_control": {}, "within_dataset_overlap": {}}
    for fs, cols in sets.items():
        idx = np.array([shared.index(c) for c in cols])
        k = {w: keys(np.asarray(D[w]["X"], dtype=np.float32)[:, idx].astype(np.float64)) for w in D}
        for w in D:
            rng = np.random.default_rng(0)
            pick = np.sort(rng.choice(len(k[w]), size=1000, replace=False))
            share = float(k[w][pick].isin(k[w]).mean())
            out["positive_control"][f"{fs}/{w}"] = {"share_of_1000_training_rows_found": share}
            assert share == 1.0, f"overlap tool failed its positive control: {fs}/{w} {share}"
        # NOTE: the cached X is float32; step 06 hashes the same float32-derived values (np.asarray(..., float64) of float32 rows)
        side = np.asarray(D["iomt"]["side"])
        res = {"base_iomt": float(k["iomt"][side == 1].isin(k["iomt"][side == 0]).mean())}
        fid = np.asarray(D["iot"]["fid"])
        members = np.arange(int(fid.max()) + 1)
        for seed in C.SEEDS:
            test_m = np.random.default_rng(seed).permutation(members)[:round(S6.HOLDOUT_FRACTION * len(members))]
            is_te = np.isin(fid, test_m)
            res[f"base_iot_seed{seed}"] = float(k["iot"][is_te].isin(k["iot"][~is_te]).mean())
        out["within_dataset_overlap"][fs] = res
        print(fs, out["positive_control"], res, flush=True)
    payload = {"meta": C.provenance(__file__, [], seeds=C.SEEDS, inputs_note="step-06 row caches in scratch", hash_inputs=False),
               "note": "added after reviewing step 06; hash key as in step 06 (128-bit, not a byte compare); hashes taken over the cached float32 values widened to float64, "
                       "so a key here is not identical to the float64-parse key used for the step-06 dedup, but the cross-dataset overlap in step 06 used the same widened-float32 values",
               **out}
    print("wrote", C.write_json("06b_overlap_checks.json", payload))


if __name__ == "__main__":
    main()
