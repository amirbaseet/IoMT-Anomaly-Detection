"""
Load the CICIoMT2024 per-class feature CSVs (train / test) into shared-feature
matrices with binary and 5-family labels.

The class label is carried by the filename, not a column, so we attach it as we
read each per-class file.  Only the shared features are kept; the six/seven
CICIoMT2024-only columns are simply never selected.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from labels import ciciomt2024_binary, ciciomt2024_category, ciciomt2024_family


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric and turn ±inf into NaN (imputed later). New frame."""
    out = df.apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def load_split(split_dir: Path, shared_features: list[str],
               nrows: int | None = None) -> dict:
    """
    Read every per-class CSV in ``split_dir`` and return a dict with:
      X            : DataFrame [n, shared_features]  (numeric, may hold NaN)
      y_binary     : np.ndarray[str]   'benign' / 'attack'
      y_category   : np.ndarray[str]   6-class thesis category (incl. MQTT)
      y_family     : np.ndarray[object] 5-family name or None (MQTT -> None)

    ``nrows`` caps rows read per class file (smoke runs only).

    Nothing is scaled or imputed here — that happens in the fitted
    preprocessors so the fit-on-train-only discipline stays centralised.
    """
    files = sorted(p for p in split_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {split_dir}")

    frames: list[pd.DataFrame] = []
    binaries: list[np.ndarray] = []
    categories: list[np.ndarray] = []
    families: list[np.ndarray] = []

    for path in files:
        raw = pd.read_csv(path, low_memory=False, nrows=nrows)
        missing = [c for c in shared_features if c not in raw.columns]
        if missing:
            raise KeyError(f"{path.name} missing shared features: {missing}")
        x = _clean_numeric(raw.loc[:, shared_features])
        n = len(x)
        frames.append(x)
        binaries.append(np.full(n, ciciomt2024_binary(path.name), dtype=object))
        categories.append(np.full(n, ciciomt2024_category(path.name), dtype=object))
        families.append(np.full(n, ciciomt2024_family(path.name), dtype=object))

    return {
        "X": pd.concat(frames, ignore_index=True),
        "y_binary": np.concatenate(binaries),
        "y_category": np.concatenate(categories),
        "y_family": np.concatenate(families),
    }


def design_b_subset(data: dict) -> dict:
    """
    Drop MQTT (family is None) so the training/eval frame holds only the 5
    shared families.  Returns a NEW dict; the input is untouched.
    """
    mask = np.array([f is not None for f in data["y_family"]])
    return {
        "X": data["X"].loc[mask].reset_index(drop=True),
        "y_family": data["y_family"][mask].astype(str),
    }
