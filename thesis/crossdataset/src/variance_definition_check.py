#!/usr/bin/env python3
"""
Is ``Variance`` the same quantity in CICIoT2023 (MERGED release) and CICIoMT2024?

Follow-up to ``feature_scale_check.csv`` (benign medians 1.616e+05 vs 0.9).  Tests one
concrete hypothesis per dataset, row by row:
  * ``Variance == Std ** 2``   (a plain variance of the packet lengths in the window)
  * ``0 <= Variance <= 1``     (a bounded quantity; the extract of Doménech et al. 2025 Table A.9
                                pairs ``Variance`` with "Ratio of the variance")

CICIoT2023 side = the cached 578,659-row class-capped sample (``sampled_test_set.parquet``),
CICIoMT2024 side = ALL rows of ``data/test``.  Read-only on both; writes one small JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

SAMPLE_PARQUET = config.OUT_DIR / "sampled_test_set.parquet"
OUT = config.OUT_DIR / "variance_definition_check.json"
COLS = ["Variance", "Std"]
RTOL, ATOL = 1e-3, 1e-6


def describe(df: pd.DataFrame) -> dict:
    finite = np.isfinite(df["Variance"]) & np.isfinite(df["Std"])
    d = df[finite]
    var, std = d["Variance"].to_numpy(dtype=np.float64), d["Std"].to_numpy(dtype=np.float64)
    eq = np.isclose(var, std ** 2, rtol=RTOL, atol=ATOL)
    nonzero = std > 0  # rows where both are 0 match trivially, so report them apart
    above_one = std > 1  # here Std ** 2 > 1, so a Variance <= 1 cannot be Std ** 2
    return {
        "n_rows": int(len(df)),
        "n_finite_rows": int(finite.sum()),
        "share_variance_eq_std_squared": float(eq.mean()),
        "n_rows_std_gt_0": int(nonzero.sum()),
        "share_variance_eq_std_squared_where_std_gt_0": float(eq[nonzero].mean()),
        "n_rows_std_gt_1": int(above_one.sum()),
        "share_variance_le_1_where_std_gt_1": float((var[above_one] <= 1).mean()),
        "share_variance_in_0_1": float(((var >= 0) & (var <= 1)).mean()),
        "variance_min": float(var.min()),
        "variance_median": float(np.median(var)),
        "variance_max": float(var.max()),
    }


def main() -> None:
    iot = pd.read_parquet(SAMPLE_PARQUET, columns=COLS)
    files = sorted(config.CICIOMT2024_TEST_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs under {config.CICIOMT2024_TEST_DIR}")
    mt = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)

    result = {
        "status": "Measured",
        "question": "is Variance computed the same way in both releases?",
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "ciciot2023_merged_sample": {"source": SAMPLE_PARQUET.name, **describe(iot)},
        "ciciomt2024_test": {"source": "data/test (all rows)", "n_files": len(files), **describe(mt)},
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
