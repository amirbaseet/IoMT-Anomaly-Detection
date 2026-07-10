# FINDINGS — rediscovery-killers

Verified facts that cost a session to learn. Cite `file:line`. Add a row whenever a fact
surprises you and would surprise the next cold session.

- **No test/lint/CI.** No pytest, ruff, black, pre-commit, or `.github/`. The only gate is
  `run_all`'s bit-exact tripwires + `scripts/verify_*.py`. Don't hunt for a test runner.
- **`notebooks/` holds `.py` modules, not notebooks.** Each is `python notebooks/<name>.py` with a
  `__main__` guard. The real `.ipynb` are `IoMT_Anomaly_Detection.ipynb` (root) and
  `deliverables/thesis_walkthrough.ipynb`.
- **Only 2 files have argparse:** `notebooks/multi_seed_loo.py`, `thesis/crossdataset/src/run.py`.
  Everything else is module-level constants + `__main__`.
- **LIME is not installed and not imported** — SHAP/TreeSHAP only. (`requirements.txt` has no
  lime; no `import lime` anywhere.)
- **THE FOUR TRAPS are code comments, not a doc** — inline `# TRAP 1-4` in
  `thesis/crossdataset/src/*.py` (`models.py`, `ae_worker.py`, `labels.py`, `ciciot_sampler.py`,
  `features.py`, `metrics.py`, `config.py`, `run.py`). No `CLAUDE_CODE_BRIEF` contains them.
- **Large parts of the tree are untracked WIP** — e.g. root `CLAUDE_CODE_BRIEF_*.md`,
  `Thesis_Generalization_Analysis_*.md`, and `results/tausweep/` are untracked at time of writing.
  A fresh clone will NOT have them; check `git status` before assuming a path exists.
- **Stale pre-`/code/`-move path (`/Users/amoorabaseet/IoMT-Project`) survives in 17 tracked
  `.log`/`.ipynb`/`.json` artifacts** — harmless records of past runs; the compat symlink
  `~/IoMT-Project` covers them. The one live source hit was fixed: `thesis/crossdataset/src/config.py`
  now derives repo-internal paths from `__file__`.
- **Two committed venvs** (`venv/`, `venv_old/`) inflate greps and the stale-path count — exclude
  them when searching.
- **`.claude/` is gitignored** (`.gitignore:40`) — anything under it (agent config, hookify local
  rules) is machine-local and will NOT ship in a clone.
- **`.claude/plans/` is an artifact dump, not plans** — 46 files, only `lstm_ae_plan.md` is a real
  forward plan; `.claude/plans/deliverables/` is a stale near-dup of `deliverables/`.
- **Canonical number registry:** `deliverables/numbers_map.md` (every claim -> on-disk source).
  Conflict tie-break log: `deliverables/CHANGELOG.md` (README wins headline).
- **Tripwire values:** `entropy_benign_p95` strict_avg = `0.8035264623662012`; §15D p93 anchor =
  `0.8589586873140701`; tolerance `1e-9` (`deliverables/scripts/_common.py:35-37`).
- **Scalers:** `preprocessed/scaler_full.pkl`, `preprocessed/scaler_reduced.pkl` (fit train-only).
- **Two seed sets, don't conflate:** main pipeline `[1, 7, 42, 100, 1729]`; crossdataset `(42, 43, 44)`.
- **Installed xgboost 3.2.0 is outside the manifest pin `<3.0`** — a known drift; numbers were
  produced on the installed version, so don't "fix" numbers to match the pin.
