# CLAUDE.md — IoMT Anomaly Detection (M.Sc. thesis)

Four-layer anomaly detection on CICIoMT2024: XGBoost (supervised) + autoencoder/VAE +
Isolation Forest, fused by an entropy gate; per-class TreeSHAP explainability. Python 3.13,
no GPU. Explainability is **SHAP only — LIME is NOT implemented** (DN-01).

Stack (verified installed): Python 3.13.13 · xgboost 3.2.0 · tensorflow 2.21 / keras 3.14 ·
scikit-learn 1.8 · shap 0.51 · pandas 2.3 · numpy 2.2. `requirements.txt` pins
`xgboost>=2.0,<3.0` but 3.2.0 is installed — a known drift; do not "correct" numbers to it.

This root is THIN. In a subsystem? Read its CLAUDE.md:
- `deliverables/scripts/CLAUDE.md` — frozen production pipeline (`run_all` + tripwires)
- `thesis/crossdataset/CLAUDE.md` — cross-dataset generalization (the four TRAPS live here)
- `notebooks/CLAUDE.md` — the `.py` training/eval modules (NOT ipynb)
- `dashboard/CLAUDE.md` — Streamlit app
Reference (link, don't inline): `README.md` (§-indexed compendium) · `deliverables/numbers_map.md`
(number→source) · `decisions/` (why-we-chose ledgers — **local, gitignored; absent in a clone**) ·
`docs/` (per-phase). Agent cycle:
`docs/agents-roster.md`. Rediscovery-killers: `docs/FINDINGS.md`.

## 1. DO NOT (hard prohibitions)
- **DN-01** — Do NOT claim a capability the code lacks. LIME is not in this repo (SHAP/TreeSHAP
  only); audit the code before any method/contribution claim. [burned: 7d86720]
- **DN-02** — Do NOT feed the tree-model RobustScaler ColumnTransformer to the AE/IF/VAE.
  Distance/reconstruction models need a dedicated StandardScaler fit on **benign-train**; the
  wrong scaler once caused a 510× AE-loss error. [92e59d4]
- **DN-03** — Do NOT fit any scaler / SMOTE / threshold on val, test, or the pre-split pool.
  Train-only (benign-train for the AE). [TRAP 1]
- **DN-04** — Do NOT edit a reported metric, citation, or figure without tracing it to its
  canonical source (`deliverables/numbers_map.md` / source paper / results artifact).
  numbers_map wins. [283a838, 7d86720]
- **DN-05** — Do NOT present a single-seed result as robust; aggregate over the seed set (INV-04).
  [3f61e59]
- **DN-06** — Do NOT `git add -A` / `--all` / blanket-stage. The tree carries untracked live
  thesis work, large derived artifacts, and two committed venvs — stage by explicit path only.
  Enforced on this machine by a local hookify guard (`.claude/hookify.*.local.md`, gitignored —
  not inherited by a clone).

## 2. CRITICAL INVARIANTS
- **INV-01** — Feature & label order is canonical: source it from the canonical `config.json`
  feature list (`feature_names` / `feature_names_reduced`), match by name then reorder — never
  by position; compare two models in the identical label space. [TRAP 3; faa4d14; README §5]
- **INV-02** — Fusion is tripwired: seed=42 must reproduce `entropy_benign_p95` strict_avg =
  `0.8035264623662012 ± 1e-9` (`_common.py:35`; §15D p93 anchor `0.8589586873140701` too).
  A drift is a real regression — investigate, don't loosen `TRIPWIRE_TOLERANCE`.
- **INV-03** — CICIoT2023 CSV ingestion strips trailing CRLF on the Label column (TRAP 2,
  `labels.py:41`). Silent label-match failure otherwise.
- **INV-04** — Seed sets: main pipeline `[1, 7, 42, 100, 1729]`; crossdataset `(42, 43, 44)`.
  `random_state=42` for reproducibility, the full set for stability.
- **INV-05** — Dedup caveat: headline numbers sit ~0.5–1.4 pp below published values (deduped
  test set). Every literature-comparison cell must name its caveat in the same row.

## 3. WORKFLOW
- **Before code:** read the subsystem CLAUDE.md + relevant `docs/phase*.md`. (`decisions/` holds
  local-only rationale ledgers — gitignored, absent in a clone; consult if present.) Don't
  re-derive a settled decision.
- **Writing code:** no test suite (below) — the safety net is the tripwire harness + verifiers;
  preserve them.
- **After code — gates (run verbatim):**
  ```
  venv/bin/python -m deliverables.scripts.run_all        # fail-fast, bit-exact tripwires
  venv/bin/python scripts/verify_report_numbers.py
  venv/bin/python scripts/verify_duplicate_counts.py
  ```
- **After any doc/status/number:** run `/fact-check` before committing (write→verify).
- **New feature:** `/plan-feature` → house-format plan in `docs/plans/`.
- **Fixed a rule-rooted bug?** add/upgrade a DN-/INV- rule here.

## 4. CURRENT STATUS
### Systems built
| Layer | What | Where |
|---|---|---|
| L1 supervised | XGBoost/RF, E7 macro-F1 = 0.9076 | notebooks/supervised_training.py, results/supervised/ |
| L2 unsupervised | AE + Isolation Forest, AE test AUC = 0.9892 | notebooks/lstm_ae_train.py, results/unsupervised_unscaled/ |
| Fusion | entropy-gate 5-case | notebooks/fusion_engine.py, results/fusion/ |
| Explainability | per-class TreeSHAP | notebooks/shap_analysis.py, results/shap/ |
| Generalization | CICIoMT2024→CICIoT2023 | thesis/crossdataset/, results/tausweep/ |
| Dashboard | Streamlit (local) | dashboard/ |

(Numbers above trace to `deliverables/numbers_map.md` rows 16 & 21.)

### Tests
No pytest / lint / CI in this repo. The gate is `run_all`'s bit-exact tripwires +
`scripts/verify_*.py` number audits. Say "tripwires reproduce", never "tests pass".

### Recently closed (newest first)
- **Literature-figure restore (283a838)** — re-verified COCIA/Recon_VulScan/AIAI vs numbers_map.
- **LIME cleanup + sigma fix (7d86720)** — removed LIME capability claims; sigma 0.022→0.023 ×7.
- **Path B Tier-2 LSTM-AE (c3e3f34)** — Layer-2 substitution + 4-issue calibration audit.

### Workflow next (work this queue, don't re-derive)
1. Fold `results/tausweep/tau_sweep_summary.md` into the thesis generalization section.
2. Fix the one live stale path — `thesis/crossdataset/src/config.py:17-21` hardcodes the
   pre-move `/Users/amoorabaseet/IoMT-Project`.
3. Decide xgboost pin: bump manifest to allow 3.2.0, or pin the env down.
4. Cleanup: `venv/` and `venv_old/` are committed to the tree — decide whether to untrack.

❓ Pending decisions: keep vs archive `Project_Journey_Complete.md`; whether the venvs stay tracked.
