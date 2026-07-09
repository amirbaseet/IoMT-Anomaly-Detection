"""
Feature-schema handling: compute the CICIoMT2024 ∩ CICIoT2023 intersection and
align a CICIoT2023 frame to the exact column order the shared-feature model
expects (TRAP 3 — match by name, then reorder; never assume positions).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_header(csv_path: Path) -> list[str]:
    """Return the column names of a CSV without loading any rows."""
    return list(pd.read_csv(csv_path, nrows=0).columns)


def compute_shared_features(
    ciciomt_header: list[str],
    ciciot_header: list[str],
    ciciot_label_col: str,
) -> list[str]:
    """
    Intersection of the two feature schemas, preserving CICIoMT2024 order so
    the retrained model has a stable, documented column layout.

    The label column is excluded.  Order follows CICIoMT2024 because that is
    the training frame; CICIoT2023 is reordered to match at load time.
    """
    ciciot_set = {c for c in ciciot_header if c != ciciot_label_col}
    shared = [c for c in ciciomt_header if c in ciciot_set]
    if not shared:
        raise ValueError("No shared features between the two datasets — "
                         "check the header sources.")
    return shared


def align_to_schema(df: pd.DataFrame, shared_features: list[str]) -> pd.DataFrame:
    """
    Reorder/select ``df`` to exactly ``shared_features`` (TRAP 3).

    Raises if any expected feature is missing, so a silent positional
    misalignment can never happen.  Returns a NEW frame (no mutation).
    """
    missing = [c for c in shared_features if c not in df.columns]
    if missing:
        raise KeyError(f"CICIoT2023 frame is missing shared features: {missing}")
    return df.loc[:, shared_features].copy()


def save_shared_features(
    shared_features: list[str],
    ciciomt_header: list[str],
    ciciot_header: list[str],
    dropped_ciciomt_only: list[str],
    ciciot_only: list[str],
    out_path: Path,
) -> None:
    """Persist the exact shared-feature list + provenance to JSON."""
    payload = {
        "n_shared_features": len(shared_features),
        "shared_features": shared_features,
        "ciciomt2024_only_dropped": dropped_ciciomt_only,
        "ciciot2023_only_ignored": ciciot_only,
        "n_ciciomt2024_features": len([c for c in ciciomt_header]),
        "n_ciciot2023_features": len([c for c in ciciot_header]) - 1,
        "note": (
            "Shared-schema retrain. The CICIoMT2024-only features were "
            "unavailable in CICIoT2023 and were dropped, not reconstructed. "
            "'Duration' is absent in CICIoT2023 in addition to the six named "
            "in the brief — reported honestly, dropped like the others."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
