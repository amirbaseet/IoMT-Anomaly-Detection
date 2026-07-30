"""
Precision sweep — duplicate rate as a function of numeric precision (figure F1).

Sections 4 and 5 of the manuscript rest on two measured points: float64-as-printed
and float32-as-computed. Two points do not show whether the collapse between them
is a cliff or a slope, and that shape is the argument for making precision a
mandatory reporting parameter. This script fills in the curve by rounding every
feature to k significant digits for k = 1..8 and recounting.

Method matches dr6_float32_check.py exactly: 64-bit row hashing over the ordered
45-column tuple via pandas.util.hash_pandas_object(index=False); duplicates =
len(h) - len(unique(h)). Rows are read at float64 and rounded down to k
significant digits, so every point in the sweep is the same rows at a different
precision, never a different subset.

Anchors (asserted, not hoped): the float64 pass must reproduce 5,119 / 2,065 and
the float32 pass 2,645,751 / 721,914 (numbers_map.md Section 2).

Outputs -> paper/P1_dedup_audit/figures/precision_sweep.json
Reads   -> data/{train,test}/*.csv (never written to)
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "paper" / "P1_dedup_audit" / "figures"
SIG_DIGITS = [1, 2, 3, 4, 5, 6, 7, 8]

ANCHORS = {                      # numbers_map.md Section 2
    ("float64", "train"): 5_119, ("float64", "test"): 2_065,
    ("float32", "train"): 2_645_751, ("float32", "test"): 721_914,
}
ROWS = {"train": 7_160_831, "test": 1_614_182}


def round_sig(a: np.ndarray, k: int) -> np.ndarray:
    """Round to k significant digits, elementwise, preserving zeros and sign."""
    out = np.zeros_like(a)
    nz = a != 0
    if not nz.any():
        return out
    v = a[nz]
    mag = np.floor(np.log10(np.abs(v)))
    factor = np.power(10.0, k - 1 - mag)
    out[nz] = np.round(v * factor) / factor
    return out


def dup(h: np.ndarray) -> int:
    return int(len(h) - len(np.unique(h)))


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("Precision sweep — duplicate rate vs numeric precision")
    print("=" * 70)

    reps = [("float64", None), ("float32", None)] + [(f"sig{k}", k) for k in SIG_DIGITS]
    hashes: dict[tuple[str, str], list[np.ndarray]] = {
        (name, split): [] for name, _ in reps for split in ("train", "test")}

    for split in ("train", "test"):
        files = sorted(glob.glob(str(REPO / "data" / split / "*.csv")))
        if not files:
            sys.exit(f"No CSVs under {REPO/'data'/split}")
        print(f"[{split}] {len(files)} files")
        for i, f in enumerate(files, 1):
            df = pd.read_csv(f)                      # float64 as printed
            cols = df.columns
            arr = df.to_numpy(dtype=np.float64)
            for name, k in reps:
                if name == "float64":
                    frame = df
                elif name == "float32":
                    frame = df.astype(np.float32)
                else:
                    frame = pd.DataFrame(round_sig(arr, k), columns=cols)
                hashes[(name, split)].append(
                    pd.util.hash_pandas_object(frame, index=False).to_numpy())
            del df, arr
            if i % 15 == 0 or i == len(files):
                print(f"  [{i:02d}/{len(files)}] {(time.time()-t0)/60:.1f} min")

    result = {"sig_digits": SIG_DIGITS, "points": []}
    for name, k in reps:
        row = {"representation": name, "significant_digits": k}
        for split in ("train", "test"):
            h = np.concatenate(hashes[(name, split)])
            if len(h) != ROWS[split]:
                sys.exit(f"{name}/{split}: {len(h):,} rows, expected {ROWS[split]:,}")
            d = dup(h)
            row[split] = {"rows": int(len(h)), "dups": d, "rate": d / len(h)}
            anchor = ANCHORS.get((name, split))
            if anchor is not None:
                ok = d == anchor
                print(f"  anchor {name}/{split}: {d:,} vs {anchor:,} -> "
                      f"{'MATCH' if ok else 'MISMATCH'}")
                if not ok:
                    sys.exit(f"{name}/{split} failed its numbers_map anchor — aborting.")
                row[f"{split}_anchor_ok"] = True
        result["points"].append(row)
        print(f"  {name:>8}: train {row['train']['rate']*100:6.3f}% | "
              f"test {row['test']['rate']*100:6.3f}%")

    result["runtime_min"] = round((time.time() - t0) / 60, 2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "precision_sweep.json").write_text(json.dumps(result, indent=2))
    print(f"\nsaved -> {OUT/'precision_sweep.json'}  ({result['runtime_min']} min)")


if __name__ == "__main__":
    main()
