# CLAUDE.md — deliverables/scripts (frozen production pipeline)

The numbered, fail-fast deliverable pipeline orchestrated by `run_all.py`. Pure re-derivation of
published thesis numbers from on-disk artifacts — treat as FROZEN; changing a stage can break a
bit-exact tripwire. Read root `../../CLAUDE.md` first.

## Run
```
venv/bin/python -m deliverables.scripts.run_all        # from project root (or run_all.py directly)
```
Runs the numbered stages `00 → 07` as subprocesses, fail-fast, `cwd=PROJECT_ROOT`.

## Invariants (do not regress)
- **Bit-exact tripwires** in `_common.py`: `TRIPWIRE_STRICT_AVG_P95 = 0.8035264623662012` and
  `TRIPWIRE_STRICT_AVG_P93 = 0.8589586873140701` (both ± `TRIPWIRE_TOLERANCE = 1e-9`).
  `04_fusion_phase6.py` asserts them. A drift = a real regression; investigate, never loosen the
  tolerance. (root INV-02)
- **Path-portable** — `PROJECT_ROOT = Path(__file__).resolve().parents[2]` (`_common.py:16`).
  These scripts do NOT hardcode absolute paths (unlike `thesis/crossdataset/src/config.py`);
  keep it that way.
- **Parseable output** — metrics print via `_common.emit()`, one metric per line. Keep the format.
- **`07_tau_sweep.py`** (untracked WIP at time of writing) is a standalone Tier-1 verifier
  (NOT in `run_all`'s stage list) — pure
  post-processing of the frozen E7 softmax (`results/supervised/predictions/E7_test_proba.npy`)
  → `results/tausweep/`. No training, no argparse.
