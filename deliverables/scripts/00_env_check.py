"""
Phase 0 — Environment check (no phase content; sanity before running 01–06).

WHAT WE FOUND
  Project requires Python 3.13 with TensorFlow 2.21 wheels (downgraded from
  3.14 in Phase 5 — see Project_Journey Phase 5 problems table). All seven
  required libraries (numpy, pandas, sklearn, xgboost, tensorflow, shap,
  scipy) must be importable. All required result subdirectories must exist
  on disk (this is a structural precondition, not a metric to reproduce).

WHY WE CHOSE THIS APPROACH
  Alternatives considered:
    - Skip env check and rely on import errors in downstream scripts
    - Run pip-check (network-bound, slow, not deterministic)
  Decision criterion: fail-fast with one clear error message instead of
    seven downstream-script tracebacks pointing at the same root cause.
  Tradeoff accepted: this script duplicates a small amount of the manifest's
    "what must exist" list to stay self-contained.
  Evidence: deliverables/artifact_manifest.md, requirements.txt.

Run from project root:
  venv/bin/python -m deliverables.scripts.00_env_check
"""
from __future__ import annotations

import importlib
import sys

from deliverables.scripts._common import (
    PROJECT_ROOT, PHASE_PATHS, banner, emit, check_artifact_exists,
)

REQUIRED_PACKAGES = [
    "numpy", "pandas", "sklearn", "xgboost", "tensorflow", "shap", "scipy",
]

REQUIRED_ARTEFACT_FILES = [
    "results/supervised/metrics/E7_multiclass.json",
    "results/unsupervised/thresholds.json",
    "results/fusion/metrics/h1_h2_verdicts.json",
    "results/zero_day_loo/metrics/h2_loo_verdict.json",
    "results/enhanced_fusion/metrics/h2_enhanced_verdict.json",
    "results/enhanced_fusion/metrics/ablation_table.csv",
    "results/shap/shap_values/shap_values.npy",
    "preprocessed/full_features/y_test.csv",
]


def main() -> int:
    banner("Phase 0 — Environment check")
    print(f"Python: {sys.version.split()[0]}")
    emit("Phase 0", "python_version", sys.version.split()[0], "sys.version")

    fail = 0
    for pkg in REQUIRED_PACKAGES:
        try:
            m = importlib.import_module(pkg)
            v = getattr(m, "__version__", "unknown")
            print(f"  {pkg}: {v}")
            emit("Phase 0", f"{pkg}_version", v, "importlib")
        except ImportError as e:
            print(f"  {pkg}: NOT INSTALLED — {e}")
            fail += 1

    print()
    print("Required artefact directories:")
    for label, path in PHASE_PATHS.items():
        ok = path.exists()
        print(f"  {label:<18} {path.relative_to(PROJECT_ROOT)}: {'OK' if ok else 'MISSING'}")
        if not ok:
            fail += 1

    print()
    print("Spot-check required files:")
    for rel in REQUIRED_ARTEFACT_FILES:
        ok = check_artifact_exists(rel)
        print(f"  {'OK' if ok else 'MISSING'}: {rel}")
        if not ok:
            fail += 1

    if fail:
        print()
        print(f"!!! {fail} environment problems detected — fix before running 01–06 !!!")
        return 1

    print()
    print("OK — environment is ready for the deliverable run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
