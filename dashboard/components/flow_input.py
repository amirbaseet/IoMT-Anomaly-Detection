"""44-feature input parsers + range validators for Page 2 (Single Flow Analyzer).

Three input modes:
- `parse_paste_row(text)`         — comma-separated row of 44 numeric values
- `parse_csv_upload(file)`        — CSV with header row; extras ignored, all
                                    44 expected columns must be present
- `load_sample_flow(name)`        — pre-selected row from X_test.npy

Validation feedback pattern (matches spec §Page 2 step 4):
- GREEN  — all 44 numeric, all in [p0.1, p99.9] of training distribution
- YELLOW — 1-3 features outside [p0.1, p99.9] (still scored, flagged OOD)
- RED    — wrong count, non-numeric, or required columns missing
            (Score button stays disabled)

The canonical feature order is sourced from `results/shap/config.json`'s
`feature_names` list — exactly the same order the upstream models expect.
Loading once (cached) and treating it as immutable for the rest of the page.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SHAP_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "results/shap/config.json"
FEATURE_RANGES_PATH: Final[Path] = Path(__file__).resolve().parent / "feature_ranges.json"
X_TEST_PATH: Final[Path] = PROJECT_ROOT / "preprocessed/full_features/X_test.npy"

N_FEATURES: Final[int] = 44
OOD_YELLOW_MAX: Final[int] = 3  # >3 features OOD escalates to a warning, not a block

# Pre-selected demo flows (resolved from y_test.csv first-occurrence search;
# row indices documented for reproducibility — see scripts/precompute_dashboard_artifacts.py).
SAMPLE_FLOWS: Final[dict[str, int]] = {
    "ARP_Spoofing (test row 0)": 0,
    "Benign (test row 1744)": 1744,
    "Recon_Ping_Sweep (test row 106007)": 106007,
    "MQTT_Malformed_Data (test row 101319)": 101319,
}


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a paste / upload / sample-flow validation pass."""
    ok: bool                              # safe to score (RED → False)
    severity: str                         # 'green' | 'yellow' | 'red'
    message: str                          # user-facing summary
    values: np.ndarray | None             # shape (44,) float32 if ok else None
    ood_features: tuple[str, ...] = ()    # names of OOD features (for yellow)


@st.cache_data(ttl=None, show_spinner=False)
def feature_names() -> list[str]:
    cfg = json.loads(SHAP_CONFIG_PATH.read_text())
    names = cfg["feature_names"]
    if len(names) != N_FEATURES:
        raise RuntimeError(f"expected {N_FEATURES} features, got {len(names)}")
    return list(names)


@st.cache_data(ttl=None, show_spinner=False)
def feature_ranges() -> dict[str, dict[str, float]]:
    return json.loads(FEATURE_RANGES_PATH.read_text())


def _check_ood(values: np.ndarray) -> tuple[str, ...]:
    """Return tuple of feature names that fall outside [p0.1, p99.9].

    p0.1 == p1.0 (clamped/saturated bounds in CICIoMT2024 — see plan note R10);
    we only flag values strictly outside the published [p0.1, p99.9] envelope.
    """
    ranges = feature_ranges()
    names = feature_names()
    ood: list[str] = []
    for i, name in enumerate(names):
        rng = ranges[name]
        v = float(values[i])
        if v < rng["p0.1"] or v > rng["p99.9"]:
            ood.append(name)
    return tuple(ood)


def _make_result(values: np.ndarray) -> ValidationResult:
    """Apply OOD check and assemble a ValidationResult.

    All numeric+count gating must pass before this is called.
    """
    ood = _check_ood(values)
    if not ood:
        return ValidationResult(
            ok=True,
            severity="green",
            message=f"All {N_FEATURES} features parsed; values within training distribution.",
            values=values.astype(np.float32),
        )
    if len(ood) <= OOD_YELLOW_MAX:
        return ValidationResult(
            ok=True,
            severity="yellow",
            message=(
                f"{len(ood)} feature{'s' if len(ood) != 1 else ''} outside "
                f"training distribution [p0.1, p99.9]: {', '.join(ood)}. "
                f"Still scored, but treat the prediction as out-of-distribution."
            ),
            values=values.astype(np.float32),
            ood_features=ood,
        )
    return ValidationResult(
        ok=True,
        severity="yellow",
        message=(
            f"{len(ood)} features outside training distribution "
            f"(showing first 5: {', '.join(ood[:5])}). "
            f"This is far from any training sample — score with extreme caution."
        ),
        values=values.astype(np.float32),
        ood_features=ood,
    )


def parse_paste_row(text: str) -> ValidationResult:
    """Parse a comma-separated row of 44 numeric values."""
    raw = (text or "").strip()
    if not raw:
        return ValidationResult(False, "red", "Paste a comma-separated row of 44 values.", None)
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    if len(parts) != N_FEATURES:
        return ValidationResult(
            False, "red",
            f"Expected {N_FEATURES} comma-separated values, got {len(parts)}.",
            None,
        )
    try:
        values = np.asarray([float(p) for p in parts], dtype=np.float64)
    except ValueError as exc:
        return ValidationResult(False, "red", f"Non-numeric value: {exc}.", None)
    if not np.all(np.isfinite(values)):
        return ValidationResult(False, "red", "Input contains NaN or inf.", None)
    return _make_result(values)


def parse_csv_upload(uploaded) -> ValidationResult:
    """Parse a Streamlit-uploaded CSV file (header row required).

    Extra columns are ignored; all 44 expected names must be present.
    Multi-row CSVs use the first row only — this is a single-flow page.
    """
    try:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    except Exception as exc:
        return ValidationResult(False, "red", f"Could not parse CSV: {exc}.", None)
    if df.empty:
        return ValidationResult(False, "red", "Uploaded CSV is empty.", None)

    expected = feature_names()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        return ValidationResult(
            False, "red",
            f"Missing required columns: {preview}{more}.",
            None,
        )

    row = df.iloc[0]
    try:
        values = np.asarray([float(row[c]) for c in expected], dtype=np.float64)
    except (ValueError, TypeError) as exc:
        return ValidationResult(False, "red", f"Non-numeric value in row 0: {exc}.", None)
    if not np.all(np.isfinite(values)):
        return ValidationResult(False, "red", "Row 0 contains NaN or inf.", None)
    return _make_result(values)


@st.cache_data(ttl=None, show_spinner=False)
def _load_test_row(idx: int) -> np.ndarray:
    """Lazily load one row of X_test.npy (mmap, single-row slice)."""
    return np.asarray(np.load(X_TEST_PATH, mmap_mode="r")[idx]).astype(np.float64)


def load_sample_flow(label: str) -> ValidationResult:
    """Load a pre-selected demo flow by display label."""
    if label not in SAMPLE_FLOWS:
        return ValidationResult(False, "red", f"Unknown sample: {label!r}.", None)
    values = _load_test_row(SAMPLE_FLOWS[label])
    return _make_result(values)


def values_to_paste_string(values: np.ndarray, *, decimals: int = 6) -> str:
    """Format a 44-feature row as the comma-separated string Page 2 displays."""
    if values.shape != (N_FEATURES,):
        raise ValueError(f"expected shape ({N_FEATURES},), got {values.shape}")
    return ", ".join(f"{float(v):.{decimals}g}" for v in values)
