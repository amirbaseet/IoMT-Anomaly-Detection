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

## 4. Skill config
<!-- machine-read by skills (/brief, /senior-review, /red-team, /fact-check). Precedence: explicit args > this section > skill defaults. -->
- oracles: deliverables/numbers_map.md (every reported number — DN-04), results/ artifacts, tripwire anchors in deliverables/scripts/_common.py:35; README.md §-index for prose claims
- stack: Python 3.13.13 · xgboost 3.2.0 (manifest pins <3.0 — known drift, do not "correct") · tensorflow 2.21 · scikit-learn 1.8 · shap 0.51 · seeds [1,7,42,100,1729] / crossdataset (42,43,44)
- default-flags: senior-review=--paper, red-team=evaluator,ops
- languages: docs=English, figures=_en+_tr pairs, stakeholder=Turkish

## 5. CURRENT STATUS
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
- **#31–#34 integration + standalone gap report v1.0 (lit-review ed3c43d→9c7ab57)** — 3-BLUE-agent propose pass (37 blocks, indexed in `INTEGRATION-PROPOSALS-FINDINGS.md`) → **G2 decision: Strong-on-conjunction, harmonized in both docs** → 49 blocks applied across the gap doc (G2/G3 rewritten, six-column §3.3 ladder, new §5.7 PSM-HT head-to-head, §0 fact-5 frozen-vs-expanded frame, rosters G8=5/G10=4, G13 parity row) and v6.6 (Table 2.1 rows 30–33, **Table 2.2 R10 removed as a duplicate of row 32/Mahbub**, membership 33=31+2, §2.2.5 three-implement-and-evaluate + Jodayree subsection, §2.4 rewordings). Cross-doc verify: G2 harmony + zero numbering contamination confirmed; 6 fixes applied (XAI count is **22/33** — machine-counted; dangling G13; ViT row; ordinal; numbering-map guard added to the sweep file). Then **`Research_Gap_Report_v1.0.md` drafted per the frozen brief** (exec summary / landscape / G1–G13 table / zero-day frontier / positioning / 33-row corpus appendix, ~4,000 words); senior review (--paper): **GO-WITH-FIXES, factual core fully verified, overclaim audit passed**; 3 CRITICALs (Appendix-B quarantine arithmetic, provenance-line 31+2, missing Saeed in the roster) + 12 recommendations all applied. Done-check satisfied: review ran, no CRITICAL survives, committed. **Report awaits advisor length/format confirmation before sending (brief's one open unknown).** Known debt: v6.6 §2.6 reference entries for rows 30–33 + related-work renumbering (disclosed in its version note).
- **Corpus expansion #31–#34 + 2026-07 sweep (lit-review 7160a8b→8f0e71d)** — brief frozen (`.claude/plans/2026-07-29-litgap-report-brief.md`, local-only: `.claude/` gitignored) for the advisor-requested standalone English gap report; 102-agent deep-research sweep (19 confirmed/6 refuted claims → `SWEEP-2026-07-FINDINGS.md`; two refutations later **overturned by primary-source fetches** — lesson recorded: 0-3 votes on redirect-gated sources measure fetchability, not truth). Four entrants summarized and adversarially verified: **#31 Jodayree PSM-HT** (HIGH-risk zero-day rival: 18-type LOO all-succeeding, energy+EVT, session-disjoint splits, no dedup, ~167:1 window-flattened imbalance), **#32 Palaniappan MI+DL-BDA** (Wilcoxon+Cohen's-d → G10's exceptions become four; Recall@FPR≤1%; 4th resolvable code link), **#33 Mahbub CWD-DA** (71-instance Ping Sweep class — sharpest G9 exhibit; Tables 5≡7 identity; note: commit 8003333 briefly pushed unfixed files under a fixing message — corrected honestly in fca4684), **#34 Al-Hasani FSLM+KDN** (abstract-tier like #09; its abstract's "1.5M/42/15/real-hospital" dataset description conflicts with canon on every element). ALL_SUMMARIES rebuilt ×4 (33 files, #01–#34, gap at #09; full-text tier = 32 — only #34 is abstract-tier among the files; #09 has no file). ⚠ Gap doc + v6.6 were left STALE vs the expanded corpus at this arc's close — **resolved by the next arc (integration applied 2026-07-29, see entry above)**.
- **Chapter-2 prose v6.6 (lit-review 1cd14db→ea57fc1)** — drift-checked `Literature_Review_Chapter2_v6.5.md` against the corrected gap doc (41 findings, 10 CRITICAL — spec in `thesis/lit_review/V6.5-DRIFT-FINDINGS.md`), then rewrote as **v6.6**: §2.4.2(a) rebuilt on the DR-6 float64-train-split↔float32 resolution (the disproved "pre-redistributed dataset / 500×" theory removed); G2/G5/G7 premises corrected to survive published precedent; dedup roster 6→9; leakage frame 2→4 axes; Hafid operational head-to-head added; gap table extended to G13 (new G8–G12 + former profiling gap renumbered G13); 8 `[Phase-log:]` tags citing 4 phase docs. Adversarial verify (report in-session, not persisted): **39/41 spec findings fixed, 10/10 §2.4.2(a) reconciliation numbers exact vs `dr6_out/`, 0 absence-credit regressions** → follow-up fixes applied, incl. 3 pre-existing v6.5 reference errors (row/ref 18 authorship is **Akkal** et al., ICSPIS 2024 — load-bearing for the DR-6 provenance argument; #14 is ICISSP 2025, not SECRYPT; #27 IncFL is an arXiv preprint, not a paged conference paper). Editorial decisions closed: corpus-membership rule declared in §2.2 (29 = 28 full-text + #9 abstract-only; absence claims over the 28) and six-works head-to-head framing (Lipsa routed to §2.4.3). One flagged item inside v6.6: the ~86% DDoS-ICMP within-class duplication figure is marked "[re-anchor to numbers_map before submission]" — it currently traces only to `Project_Journey_Complete.md`.
- **Lit-review senior review + DR-6 resolution (320b5d7 · lit-review 55dfc58 imported-with-fixes-applied + b41e026/33efead DR-6 · pcap aaf8917)** — 3-agent adversarial review of `thesis/lit_review/Chapter2_Synthesis_and_Gap_Analysis.md` (verdict NO-GO→GO-WITH-FIXES: G2/G7 false as worded vs Alfageer #21/Akar #08; 27→29-paper frame; dedup roster 5/27→9/29) + **43 of 46** BLUE proposals applied (3 superseded duplicates skipped; the rewrite predates the repo's first commit, so no reviewable diff — `FINDINGS.md`/`PROPOSALS-FINDINGS.md` are the provenance): new gaps G8–G12, DR-9–DR-12, Büken/Hafid head-to-heads, core internal numbers carry resolvable `[Phase-log:]` citations (cite-once-inherit convention, §0 fact 4; one cites `pathB_hardening.md`). **DR-6 closed by recomputation** (`iomt-pcap-experiments/dr6_*.py`): 5,119 = float64-exact duplicates of the released train split (reproduced exactly; Akkal's subset truly has 672 — his 5,119 is pipeline-inherited); thesis 36.95%/44.72% reproduce at **float32** to 4 decimals (2,645,751/721,914). Wording "bit-identical at float32" applied in `docs/phase2_eda.md`; cross-check: `scripts/verify_duplicate_counts.py`'s features-only pooled count (3,368,126) reproduces DR-6's float32 full-merge figure. lit_review now versioned in **private** repo `iomt-lit-review` (canonical; parent repo gitignores the dir; Downloads copy secondary).
- **Literature-figure restore (283a838)** — re-verified COCIA/Recon_VulScan/AIAI vs numbers_map.
- **LIME cleanup + sigma fix (7d86720)** — removed LIME capability claims; sigma 0.022→0.023 ×7.
- **Path B Tier-2 LSTM-AE (c3e3f34)** — Layer-2 substitution + 4-issue calibration audit.

### Workflow next (work this queue, don't re-derive)
0. ~~Gap-doc + v6.6 integration of #31–#34 + report draft~~ **DONE 2026-07-29** (see top entry). Remaining from that arc: confirm report length/format with the advisor before sending (`/stakeholder-message` candidate); v6.6 §2.6 reference entries for rows 30–33 + related-work renumbering.
1. Fold `results/tausweep/tau_sweep_summary.md` into the thesis generalization section.
2. Decide xgboost pin: bump manifest to allow 3.2.0, or pin the env down.
3. Cleanup: `venv/` and `venv_old/` are committed to the tree — decide whether to untrack.
4. README duplicate-wording sweep (surfaced by DR-6, README has uncommitted edits — do when committing it): "exact duplicates" needs the float32 qualifier at **L547, L844, L2522**; L547's "**not reported in any prior paper**" is the falsified rarity overclaim the review killed (9/29 touch dedup) — reword per the gap doc's §3.1 defensible statement; L2065's "shared pre-redistributed input dataset" inference is **superseded** — DR-6 shows 5,119 is the correct float64 count of the released train split (Riyadi/Kharoubi scopes = train dir; Akkal inherited). Optionally add the precision note to `numbers_map.md` rows 36–37 (DN-04: it's the oracle — same-value, qualifier only).

❓ Pending decisions: keep vs archive `Project_Journey_Complete.md` (note: v6.6's ~86% flood-duplication figure currently traces only to this file — re-anchor to `numbers_map.md` before archiving); whether the venvs stay tracked. (`chore/bootstrap-claude` in sync with origin as of 2026-07-29.)
