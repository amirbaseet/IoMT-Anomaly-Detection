"""
Shared utilities for the deliverable-production scripts.

All scripts run from the project root. Paths are resolved relative to a single
PROJECT_ROOT constant. Heavy artefacts are loaded lazily through `load_*`
helpers so a script that doesn't need a 16 MB SHAP tensor doesn't pay for it.
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Root + canonical subdirectory paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
PREPROCESSED = PROJECT_ROOT / "preprocessed"
EDA_OUTPUT = PROJECT_ROOT / "eda_output"
DELIVERABLES = PROJECT_ROOT / "deliverables"
FIGURES = DELIVERABLES / "figures"

# Phase → on-disk path mapping (see artifact_manifest.md §"Directory mapping")
PHASE_PATHS = {
    "phase4": RESULTS / "supervised",
    "phase5": RESULTS / "unsupervised",
    "phase5_unscaled": RESULTS / "unsupervised_unscaled",
    "phase6": RESULTS / "fusion",
    "phase6B": RESULTS / "zero_day_loo",
    "phase6C": RESULTS / "enhanced_fusion",
    "phase7": RESULTS / "shap",
}

# Reproducibility tripwire — asserted in 04_fusion_phase6.py
TRIPWIRE_STRICT_AVG_P95 = 0.8035264623662012
TRIPWIRE_STRICT_AVG_P93 = 0.8589586873140701  # §15D anchor (continuous sweep)
TRIPWIRE_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Helpers — printing + loading
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    """Print a clean section banner."""
    bar = "=" * 78
    print(bar)
    print(f"  {title}")
    print(bar)


def emit(phase: str, metric: str, value: object, source: str) -> None:
    """Parseable stdout: '[PhaseX] <metric> = <value>  (source: <path>)'."""
    if isinstance(value, float):
        v = f"{value:.10f}".rstrip("0").rstrip(".")
        if "." not in v:
            v = f"{v}.0"
    else:
        v = str(value)
    print(f"[{phase}] {metric} = {v}  (source: {source})")


def emit_pct(phase: str, metric: str, value: float, source: str) -> None:
    """Emit a percentage with two decimals."""
    print(f"[{phase}] {metric} = {value * 100:.2f}%  (source: {source})")


def load_json(rel_path: str | Path) -> dict:
    """Load a JSON artefact relative to PROJECT_ROOT."""
    p = PROJECT_ROOT / rel_path
    with p.open() as fp:
        return json.load(fp)


def load_csv(rel_path: str | Path):
    """Load a CSV artefact relative to PROJECT_ROOT via pandas."""
    import pandas as pd
    p = PROJECT_ROOT / rel_path
    return pd.read_csv(p)


def load_npy(rel_path: str | Path):
    """Load a .npy artefact relative to PROJECT_ROOT via numpy."""
    import numpy as np
    p = PROJECT_ROOT / rel_path
    return np.load(p)


def check_artifact_exists(rel_path: str | Path) -> bool:
    """Return True if the artefact exists under PROJECT_ROOT."""
    return (PROJECT_ROOT / rel_path).exists()
