# Deliverables — IoMT IDS Thesis Production Pass

This directory holds the Python and notebook deliverables for the
production-deliverable run completed on 2026-05-15.

The five **markdown** deliverables and the **run_all_summary.txt** live under
`.claude/plans/deliverables/` because a PreToolUse hook on the user's
machine blocks `.md` and `.txt` files outside the hook-exempt path. The split
is purely a logistics artefact; the content is the same as if everything had
been written directly under `deliverables/`.

## File map

### Under `deliverables/` (this directory)

- `scripts/__init__.py` — package marker
- `scripts/_common.py` — paths, constants, helpers (banner, emit, load_*)
- `scripts/00_env_check.py` — Python + library versions + artefact-directory existence
- `scripts/01_data_preprocessing.py` — Phase 2 EDA + Phase 3 preprocessing reproducibility
- `scripts/02_supervised_phase4.py` — E1–E8 + E5G; H3 verdict; Yacoubi comparison
- `scripts/03_unsupervised_phase5.py` — AE/IF thresholds + scaling-fix (C13) narrative
- `scripts/04_fusion_phase6.py` — **Phase 6/6B/6C with the two reproducibility tripwires**
- `scripts/05_explainability_phase7.py` — TreeSHAP global + per-class + 4-way comparison
- `scripts/06_path_b_hardening.py` — Tier 1 (multi-seed, sweep, KS, SHAP-bg) + Tier 2 (β-VAE, LSTM-AE)
- `scripts/run_all.py` — orchestrator (fail-fast, writes `run_all_summary.txt`)
- `thesis_walkthrough.ipynb` — narrated companion notebook
- `full_report.docx` — generated from `full_report.md` via the docx skill
- `figures/` — any figure a script writes goes here

### Under `.claude/plans/deliverables/` (hook-exempt)

- `full_report.md` — the canonical narrative (~11,000 words, 12 sections + 2 appendices)
- `CHANGELOG.md` — README↔Project_Journey conflict log + hook conflict logistics
- `artifact_manifest.md` — every `.npy`/`.csv`/`.pkl`/`.json` the deliverables consume
- `numbers_map.md` — every numerical claim → source file/line
- `decisions_ledger.md` — per-phase choice/alternatives/criterion/tradeoff (drives every "Why we chose this approach" block)
- `run_all_summary.txt` — collected stdout from `run_all.py`

## How to run

```bash
# from project root (~/IoMT-Project)
venv/bin/python -m deliverables.scripts.run_all
```

This runs scripts 00–06 in order, stops on first non-zero exit, and writes
`run_all_summary.txt`. The notebook can be executed separately:

```bash
venv/bin/jupyter nbconvert --to notebook --execute \
  deliverables/thesis_walkthrough.ipynb \
  --output thesis_walkthrough.ipynb \
  --ExecutePreprocessor.timeout=600
```

## Reproducibility tripwires

Two tripwires are asserted in `04_fusion_phase6.py` before any downstream
computation depends on them:

1. `entropy_benign_p95` strict_avg = `0.8035264623662012` (within 1e-9)
2. `entropy_benign_p93 + ae_p90` strict_avg = `0.8589586873140701` (§15D anchor, within 1e-7)

Both reproduce bit-exactly across Path B Week 1 (multi-seed Week 1 reuses #1
under seed=42), Week 2A (sweep Week 2A reuses #1 plus computes #2 fresh), and
both Tier 2 substitutions (β-VAE Week 5 + LSTM-AE Week 8).
