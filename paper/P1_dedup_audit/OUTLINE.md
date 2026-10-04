# P1 — Extended outline + figure/table plan            status: SELF-CHECKED · decisions C/D/F/H resolved · 2026-07-30

**Governing spec:** `.claude/plans/2026-07-30-p1-dedup-paper-brief.md` (FROZEN 2026-07-30). Deviations are
recorded in §9 as *amendment candidates*, never applied silently.
**Venue:** Internet of Things (Elsevier) · full-length research article · target 8–12k words · English · `_en` figures only.
**Oracle rule (DN-04):** every number below carries its source. `NM` = `deliverables/numbers_map.md`;
`dr6` = `iomt-pcap-experiments/dr6_out/dr6_float32_check.json`; `panel` = `dr6_out/dr6_panel.json`;
`dr6b` = `dr6_out/dr6b_perclass_f32.json`; `GR` = `thesis/lit_review/Research_Gap_Report_v1.0.md`;
`v6.6` = `thesis/lit_review/Literature_Review_Chapter2_v6.6.md`.
**NEW-ORACLE gate: CLEARED 2026-07-30.** The 12 previously-unanchored values (float32/float64 counts and
rates, pooled merge, both cross-split identity pairs, the intra-class identity, the per-class table pointer)
are now rows in `numbers_map.md` §2. No `[NO-ROW]` blocker remains.

---

## 0. The claim ladder (what the paper argues, in order)

1. CICIoMT2024's duplicate content is **precision-dependent**: 0.07% at float64, **36.95%** at float32 — the
   precision every model actually computes in. (dr6; NM §2 rows 36–37)
2. The literature's three "5,119 duplicates" reports are **not wrong and not comparable** — they are the
   float64-exact count of one split, reproduced to the row. (panel; GR §3)
3. The redundancy is **structured, not diffuse**: 99.5% of it sits in six TCP/IP flood sub-types, it is
   **entirely intra-class**, and it is **not volume-driven** (1.64M-row UDP flood has zero duplicates). (dr6b)
4. Two *distinct* leakage mechanisms must be separated: **within-split redundancy** (large) and
   **cross-split identity** (small). Conflating them is why the field's dedup discussion is confused. (dr6)
5. The consequences are **mostly negative results**, measured not inferred: a duplicated *test set* inflates
   metrics by a small separable amount (+0.35–0.40 pp accuracy, 5 seeds), duplicated *training data* has no
   separable effect, and the raw-vs-dedup comparison the field makes sits inside its own seed variance —
   alongside a published +26pp effect that does not reproduce, a resampling method that degrades macro-F1 in
   4/4 configs, and a 0.733→0.999 baseline chasm.
6. Therefore: a **reporting protocol** (precision, scope, stage, split, per-class) without which
   cross-paper accuracy comparison on this dataset is uninterpretable.

★── Paper's one-sentence claim: *duplicate leakage in CICIoMT2024 is a precision-and-scope artifact,
    it is 500× larger than the literature reports, it is concentrated in six flood classes, its
    measurable cost is test-side metric inflation rather than training-side memorization, and the
    field cannot compare results until it states the five parameters that determine it.* ──★

---

## 1. Section-by-section outline (with evidence mapping)

### Title / Abstract / Highlights / Keywords  (~400 words)
- Working title: *"How much of CICIoMT2024 is a copy? A per-split, precision-stated audit of duplicate
  leakage in the reference IoMT intrusion-detection benchmark"*.
- Abstract must state all five protocol parameters and both headline pairs (0.07%/36.95%).
- Highlights: Elsevier convention is 3–5 bullets, ≤85 characters each — **[GfA-UNVERIFIED]** (§10).
- **Evidence:** NM §2 rows 31–37 · dr6.

### 1. Introduction  (~1,000 words)
- CICIoMT2024 as the field's reference substrate: **33 studies in under two years**. (GR §1.1)
- The interpretability crisis: Dadkhah's untuned 19-class baseline **0.733** vs downstream **0.96–0.999**. (GR §1.2)
- Gap statement: **9 of 31** full-text-verified studies touch dedup in some form; **zero** report a per-split
  rate or a leakage-impact analysis. (GR headline 1, `:15` — **not** the G1 table row, which says 9/33; see §9-H)
- Contributions list (5): per-split precision-stated measurement · the 5,119 reconciliation ·
  per-class structure · mechanism decomposition · reporting protocol + public repro repo.
- **Absence-claim caveat (mandatory):** corpus snapshot **2026-07-29**, 31 full-text-verified;
  freshness re-sweep required before submission. (GR §5 provenance line, brief line 16)

### 2. Related work  (~1,200 words)
- 2.1 Dedup practice in the CICIoMT2024 corpus — the 9/31 roster, per-study, with what each *did not* report. → **T5**
- 2.2 The 5,119 cluster (Riyadi #16, Akkal #18, Kharoubi #25 — GR Appendix A numbering). (GR App. A rows 153/155/162)
- 2.3 Leakage in ML-security evaluation generally — duplicate/near-duplicate leakage precedents outside IoMT.
  **[EXTERNAL-CITES-NEEDED]** — no existing artifact covers this; needs a small literature pull (§9).
- 2.4 Substrate fragmentation as the confound that makes the audit necessary: 5–46 features, 16k–8.78M rows,
  ≥8 label spaces. (GR headline 2 / G9)
- **Discipline:** every absence claim is scoped to "the 31 full-text-verified studies" (GR §1.1 wording).

### 3. Data and method  (~1,200 words)
- 3.1 The released artifact: 72 CSVs, Wi-Fi+MQTT, **7,160,831** train / **1,614,182** test / **8,775,013** total,
  45 features, 19 classes. (NM §2 rows 31–33, 41, 44) → **T1**
- 3.2 Duplicate definition = exact match on the 45-column feature vector; **REP45 vs REP46** (with label). (panel header)
- 3.3 Precision as an experimental variable: float64-as-printed vs float32-as-computed; why float32 is the
  operational precision (GBDT/NN frameworks compute in float32). (GR §3)
- 3.4 Scope as an experimental variable: per-file / per-split / pooled-merge / published-subset. (panel scopes)
- 3.5 Hashing method: `pandas.util.hash_pandas_object(df, index=False)` on the 45-column frame, float32 cast
  via `df.astype(np.float32)`, duplicate count = `len(h) − len(np.unique(h))`. (`dr6_float32_check.py:35–46`)
  ⚠ **Method difference to disclose:** the audit scripts hash the CSVs **as read** — they do *not* replace
  ±inf with NaN, whereas the thesis EDA pipeline does so before `drop_duplicates`
  (`notebooks/ciciomt2024_eda.py`, quoted in `docs/phase2_eda.md:77–83`). The two agree to four decimals
  (36.95%/44.72%), which is itself the cross-validation — but the paper must state which convention each
  number came from rather than implying one procedure.
- 3.6 Reproduction package → the public repo (§8), scripts + JSON only, no CIC raw data.
- **Honest-stack note:** Python 3.13.13; xgboost 3.2.0 installed vs `requirements.txt` pinning `<3.0` —
  stated as known drift if pipeline versions are cited. (CLAUDE.md stack line)

### 4. Result 1 — the precision collapse  (~900 words)
| | float64 (as printed) | float32 (as computed) |
|---|---|---|
| train | **5,119** (0.0715%) | **2,645,751** (36.9475%) |
| test | **2,065** (0.1279%) | **721,914** (44.7232%) |
| pooled merge | 7,379 (0.0841%) | 3,368,126 (38.3831%) |
- All six cells: dr6, and all six are now anchored as `numbers_map.md` §2 rows (added 2026-07-30); rates
  36.95/44.72 additionally on the original NM rows 36–37.
- Cross-check that the two independent pipelines agree: pooled float32 **3,368,126** reproduces
  `scripts/verify_duplicate_counts.py`'s features-only pooled count. (CLAUDE.md DR-6 arc entry)
- Consistency identity to print in-text: 7,160,831 − 4,515,080 = **2,645,751** ✓ (NM rows 32/34 vs dr6).
- → **F1** (precision-collapse curve), **T2** (the panel).

### 5. Result 2 — the 5,119 reconciliation  (~900 words)
- 7,155,712 unique + **5,119** = **7,160,831** = the train directory's exact row count. (GR §3; panel REP45/train)
- REP46 (features+label) gives the **same** 5,119 → the collapse introduces **no label ambiguity**. (panel REP46/train)
- Akkal's subset: every DDoS-flavoured train-only scope yields **672** at float64, never 5,119 → his figure is
  pipeline-inherited, his own subset's true count is 672. (dr6 `akkal_train_only/*`; GR §3)
- **672 vs 674 is itself an exhibit, not a discrepancy** (checked 2026-07-30): `dr6`'s Akkal predicates are
  **train-only** (4.78M–7.04M rows → 672); `panel`'s identical-named scopes are **train+test**
  (5.85M–8.62M rows → 674). Same six subsets, +2 duplicates purely from widening scope — the paper's
  scope-sensitivity thesis at its smallest scale. Cite each with its scope label; docs' 672 stands.
  (`dr6_float32_check.py:80` `rs = [r for r in tr ...]` vs `dr6_reconcile_5119.py:114` `rows_for → files`)
- Verdict wording (fixed by GR §3): *"Nobody miscounted; the counts differ by precision and scope."*
- → **T4**.

### 6. Result 3 — where the redundancy lives  (~1,200 words)
- Per-class within-class float32 rates, 19 classes, train+test, plus each class's share of duplicate mass. → **T3**, **F2**
- Headlines (all dr6b; first three rows also NM rows 38–40):
  - **TCP_IP-DDoS-ICMP 86.32% train / 94.37% test** — 50.2% of all train duplicate mass alone.
  - **TCP_IP-\* floods = 99.5%** of train duplicate mass (2,632,808 / 2,645,751).
  - **Recon** sub-types: 6.5–15.6% by rate but only **~0.5%** of mass — "negligible non-flood" is false by rate,
    true by mass. State both. (CLAUDE.md DR-6b arc correction)
  - **Zero** at float32: Benign, all five MQTT classes, and **TCP_IP-DDoS-UDP** — 1,635,956 rows, 0 duplicates.
    → redundancy is **protocol-structural, not volume-driven**. This is the section's strongest single exhibit.
- Structural claim, now anchored in `numbers_map.md` §2: within-class duplicate sums equal the pooled
  per-split counts **exactly** (2,645,751 train / 721,914 test) → **zero cross-class duplicate collisions** at
  float32; corroborated at float64 by REP46 returning the same 5,119 as REP45.
- Rarest-class tie-in: Recon-Ping_Sweep 740 raw − 51 dup = **689** unique = NM's rarest-class count (NM row 46) —
  i.e. the corrected rarest-class identity is itself a dedup consequence.

### 7. Result 4 — two mechanisms, not one  (~900 words)
- **M1 within-split redundancy** (large): collapses effective sample size; a test metric computed over
  721,914 duplicated test rows is dominated by a small set of distinct vectors.
- **M2 cross-split identity** (small): at float32, **461** unique vectors occur in both splits, carried by
  **≈13,533** test rows (**0.84%** of the test split); at float64, **195** vectors / **≈357** rows (0.022%).
  (all four now anchored in `numbers_map.md` §2)
  - Derivation (no new compute): dups_pooled − (dups_train + dups_test) = 3,368,126 − 3,367,665 = **461**;
    row share from `dr6/f32/test_rows_with_vector_in_train_pct = 0.008384` × 1,614,182 = 13,533.3.
  - ⚠ Two precision caveats the manuscript must respect: (i) that JSON key is named `_pct` but stores a
    **fraction** — never cite 0.008384 as a percentage (→ §9-B); (ii) the fraction is stored rounded to 6 dp,
    so the row counts are **±1** (13,532–13,534; 356–357). Either print them as "≈" or have the repro repo's
    `verify.py` emit exact integers — preferred. The **461** and **195** vector counts are exact.
- Why this matters: the field's instinct is "leakage = train→test copying". Here the dominant mechanism is
  **M1**, which no amount of cross-split checking detects. → **F3** (four-axis schematic), **F4** (effective-N).
- Four leakage axes, two corrected corpus-wide; axes 3–4 open. (GR §3 four-axes block / G11)

### 8. Result 5 — measured consequences (the negative results)  (~1,300 words)
- 8.1 **What duplicate leakage actually costs — measured, not inferred (C1, 5 seeds).** The controlled 2×2
  ablation (identical E7 pipeline, raw vs deduplicated data, INV-04 seed set) replaces the cross-paper
  premium entirely. Three results, in order of strength:
  - **POSITIVE, separable: a duplicated test set inflates reported metrics.** macro-F1 **+0.00784 ± 0.00077**
    (raw-trained) and **+0.00798 ± 0.00025** (dedup-trained); accuracy **+0.35 pp ± 0.09** and
    **+0.40 pp ± 0.09**. Sign-consistent across all five seeds *and* both training arms — eight independent
    measurements inside +0.0076…+0.0085. This is the paper's measured leakage cost.
  - **NEGATIVE, not separable: duplicated *training* data has no measurable effect.** macro-F1
    **−0.00064 ± 0.02468** (raw test) / **−0.00078 ± 0.02438** (dedup test): a near-zero mean with a σ ~40×
    larger, and the sign flips between seeds (+0.032 at seed 42, −0.030 at seed 1). Gradient-boosted trees on
    millions of rows are indifferent to exact-duplicate rows — duplicates re-weight patterns already present.
    State this plainly; it is a genuine negative result, not a failed experiment.
  - **NEGATIVE, and the sharpest point in the paper: the comparison the literature makes is inside its own
    noise.** Raw-everywhere vs deduplicated-everywhere gives macro-F1 **+0.00863 ± 0.02468** — not separable.
    Every published raw-vs-deduplicated accuracy comparison on this dataset, *including this project's own
    former "0.53 pp memorization premium"*, is within single-configuration seed variance. The premium claim is
    **retired**, not softened.
  - Consequence for INV-05: the ~0.5–1.4 pp gap between this thesis's headline numbers and published ones is
    consistent with **test-side inflation alone** (+0.35–0.40 pp accuracy), with no training-side component.
  - **Also measured: deduplication moves the fitted preprocessing statistics.** 37 of 104 fitted scaler parameter
    values differ between arms (RobustScaler scale up to **5.67×**; MinMax unchanged). Leakage is not only about rows
    seen twice — it distorts every statistic fitted on the training set, which is why cross-space scoring is
    invalid and why "dedup or not" changes the feature space, not just the row count.
  - **Honesty note the manuscript must carry:** the published E7 macro-F1 (0.9076) is the **maximum** of the
    five seed draws, ≈1 σ above the 5-seed mean 0.8909 ± 0.0168. P1 reports mean ± σ; the thesis's own
    single-seed headline should be re-stated the same way (DN-05).
- 8.2 **Resampling.** SMOTETomek degrades macro-F1 in **4/4** configs: −0.0114 RF/reduced, −0.0171 RF/full,
  −0.0449 XGB/reduced, −0.0368 XGB/full (NM rows 90–93), on deduplicated data. → **F5**, **T6**
  ⚠ NM row 75 says "macro-F1 degrades in **0/4** configs" — contradicted by its own rows 90–93 (all negative)
  and by GR G7 ("all 4 configs"). Oracle-wording defect, second of its kind after σ. → §9 item D.
- 8.3 **Entropy criterion.** Yacoubi-AIAI attributes a 0.735→0.998 RF jump (~26pp) to the `entropy` split
  criterion; the controlled re-test gives **+0.47pp** (E5 0.8551 vs E5G 0.8504), within noise, and the
  production model is XGBoost, which has no such knob. (gap doc §2.4 / DR-7 · `docs/phase4_supervised.md`)
- 8.4 **The baseline chasm.** Dadkhah untuned 0.733 vs downstream 0.96–0.999: part better models, a
  *measurable* part leakage. (GR §1.2) — wording must stay "a measurable part", never "mostly".
- 8.5 One-line cross-reference only to the zero-day work (P2 boundary — brief line 28).

### 9. A reporting protocol for CICIoMT2024  (~1,000 words)
- Five mandatory parameters: **precision** · **scope** (which directories/files) · **stage** (dedup before or
  after split/resample) · **split** (official file-level vs re-split) · **granularity** (per-split + per-class).
- Plus: report a metric pair (raw, deduplicated) rather than choosing; report macro-F1 **and** MCC; state
  imbalance retained vs flattened. (GR G4/G9/G10 · §7 open problem 1)
- Near-duplicate honesty: window-averaged features mean any exact-match rate is a **lower bound**. (GR §3)
- → **T7** (the checklist, designed to be reused as a reviewer aid).

### 10. Threats to validity  (~600 words)
- Exact-match only → lower bound; no near-duplicate metric offered.
- Wi-Fi+MQTT scope; BLE not audited (a separate 27-feature schema). (GR §2 analysis)
- Cross-paper premium (§8.1) is not a controlled ablation unless §9-C is executed.
- Corpus absence claims are a 2026-07-29 snapshot in a weekly-publishing field.
- Single-analyst verification; hashing collision probability stated, not assumed.

### 11. Conclusion  (~400 words)
### Declarations: CRediT · declaration of interest · data availability (repo + CIC download link) · AI-disclosure — **[GfA-UNVERIFIED]**
### References — target 45–60; sourced from v6.6 §2.6 (refs 1–47, clean) + GR Appendix A.

---

## 2. Table plan

| # | Table | Data source | Status |
|---|---|---|---|
| T1 | Dataset as released (rows/split, classes, features, imbalance) | NM §2 rows 31–33, 41–46 | ready |
| T2 | Precision × scope duplicate panel (6 cells + cross-split) | dr6 | ready; **[NO-ROW]** for float64 cells |
| T3 | Per-class within-class duplicate rate + mass share, 19 classes × train/test | dr6b | ready (3 of 38 cells anchored: NM 38–40) |
| T4 | The 5,119 reconciliation: 3 papers × claimed scope vs measured | panel + dr6 `akkal_*` | ready (§9-A resolved: 672 = train-only, 674 = train+test) |
| T5 | Dedup practice roster — 9 of 31 studies, what each did / did not report | GR App. A + v6.6 Table 2.1 | ready |
| T6 | Negative-result exhibits (SMOTETomek ×4, entropy criterion, Dadkhah chasm) | NM 82–93 + phase4 docs + GR §1.2 | ready; **§9-D wording fix first** |
| T7 | The proposed five-parameter reporting checklist | this paper (new) | authored, not derived |

## 3. Figure plan (`_en` labels only, new figures — none of the 39 thesis figures fit)

| # | Figure | Type | Data source | Status |
|---|---|---|---|---|
| F1 | Duplicate rate vs numeric precision (float64 → float32 measured; f16 + decimal-rounding sweep) | line/step, train+test | dr6 for 2 points; **sweep = new script** | needs §9-E |
| F2 | Per-class duplicate rate (bars) with duplicate-mass Pareto overlay, 19 classes | dual-axis bar+line | dr6b | ready |
| F3 | Four leakage axes × corpus correction status (which axis, corrected by whom) | schematic | GR §3 + G11 | ready (authored diagram) |
| F4 | Effective sample size: nominal vs distinct test rows per class | bar | dr6b + NM §2 | ready (derived from T3) |
| F5 | SMOTETomek macro-F1 delta, 4 configs | bar, signed | NM rows 90–93 | ready |
| F6 *(optional)* | Cross-split identity: 461 shared vectors / 13,533 test rows in context | small multiple | dr6 arithmetic | needs **[NO-ROW]** rows |

## 4. Prerequisite oracle rows — **DONE 2026-07-30** (all 12 added to `numbers_map.md` §2)

1. Float64-exact duplicate counts + rates: 5,119 / 0.0715% train; 2,065 / 0.1279% test; 7,379 / 0.0841% pooled → source `dr6`.
2. Float32 pooled-merge: 3,368,126 / 38.3831% → source `dr6`.
3. Cross-split identity, float32: **461** shared unique vectors; **13,533** test rows; **0.84%** of test → derived from `dr6`.
4. Cross-split identity, float64: **195** shared vectors (panel); 357 test rows / 0.02% (derived).
5. Structural: within-class duplicate sums = pooled per-split counts exactly (zero cross-class collisions, float32) → `dr6b`.
6. Per-class table: the remaining 35 cells of T3 (or one row pointing at `dr6b` as the table's oracle).
7. Wording fix to existing row 75 (see §9-D) — same values, corrected sentence.

## 5. Word budget
Intro 1.0k · Related 1.2k · Method 1.2k · R1 0.9k · R2 0.9k · R3 1.2k · R4 0.9k · R5 1.3k · Protocol 1.0k ·
Threats 0.6k · Concl. 0.4k · Abstract+front 0.4k = **~11.0k** — inside the 8–12k target with ~1k slack.

## 6. Reproduction repo — `ciciomt2024-dedup-audit`
- Contents: `dr6_reconcile_5119.py`, `dr6_float32_check.py`, `dr6b_perclass_f32.py`, a new `precision_sweep.py`
  (if §9-E approved), a `verify.py` that re-asserts every manuscript number from the JSONs, the three result
  JSONs, README with CIC download instructions, LICENSE, `requirements.txt`.
- Excluded: CIC raw data (link only) · any lit-review content · anything from `iomt-pcap-experiments` beyond
  the four scripts. Hard-coded absolute paths in the current scripts must be parameterized before publishing.
- Gates before public: secret scan · CIC licence check on publishing derived aggregate counts (brief line 43)
  · `verify.py` reproduces every cited number.

## 7. Order of work
1. Resolve §9 items A–E with the user (A, C, D block drafting).
2. Add the §4 oracle rows to `numbers_map.md`; commit (explicit path, DN-06).
3. Draft §§3–7 (method + results) — the oracle-dense core — then fact-check that batch.
4. Draft §§1–2, 8–11; fact-check.
5. Build F1–F5 (`_en`), T1–T7.
6. Scaffold + secret-scan the repro repo; wire `verify.py`; then flip public after the licence check.
7. `/senior-review --paper` → apply CRITICALs → package for hoca.

## 8. Boundaries held (brief §"Out of scope")
No SHAP/Cohen's-d material (P3) · no zero-day results beyond one cross-reference (P2) · no survey
publication (P4) · no energy+EVT ablation (P2) · no edits to v6.6 / gap doc except citation fixes.

## 9. Amendment candidates / decisions needed
- **A · 672 vs 674 — RESOLVED 2026-07-30, no decision needed.** Scope difference, verified in the scripts:
  `dr6` computes the Akkal candidates train-only (672), `panel` computes them train+test (674). Both correct;
  the docs' 672 is the train-only figure. Promoted into §5 as an exhibit rather than a defect.
- **B · JSON key defect.** `f32/test_rows_with_vector_in_train_pct` stores a fraction (0.008384), not a
  percentage. Fix the key name in the repro-repo copy; never cite the raw key value as a percentage.
- **C · The controlled raw-vs-dedup arm — DECIDED 2026-07-30: option C1 approved.** §8.1's memorization
  premium was a *cross-paper* accuracy difference (our deduplicated 99.27% vs another team's raw 99.80/99.811%),
  confounded by model, tuning, resampling, feature engineering and metric averaging. Resolution: run ONE
  controlled arm — the identical E7 configuration (XGBoost / Full-44 / no resampling, `random_state=42`) trained
  on the raw non-deduplicated train split and evaluated on **both** the raw and the deduplicated test splits,
  which also separates training-side memorization from test-side redundancy. Recorded as **Amendment 1** in the
  frozen brief. §8.1 cites the controlled arm as primary evidence; the cross-paper figures survive only as
  corroboration with confounders named. If the Δ falls inside the known σ = 0.023 band, that is the finding.
- **D · Oracle-wording defect — FIXED 2026-07-30.** `numbers_map.md` row 75 now reads "macro-F1 **improves in
  0/4 configs — it degrades in all 4**", with the four deltas inlined and the canonical source quoted
  (`README.md:2398`: "SMOTETomek degrades macro-F1 across all 4 configurations"). T6 is unblocked.
- **E · Precision sweep for F1.** Only two precision points exist (f64, f32). A sweep (f16, and decimal
  rounding to 1–7 significant digits) makes F1 a real curve and strengthens claim 1 at low cost — one new
  script in the repro repo. In scope as a derivation, or an amendment? Recommend: in scope, repo-side.
- **F · Manuscript home — CONFIRMED 2026-07-30.** `paper/P1_dedup_audit/` in `IoMT-Project`, committed by
  explicit path (DN-06).
- **H · Corpus denominator conflict — FIXED 2026-07-30.** `Research_Gap_Report_v1.0.md:84` said "9/33 studies"
  while `:15` said "Nine of the 31 full-text-verified studies". The G1 row now reads "9 of the 31
  full-text-verified studies", matching the report's own §1.1 absence-claim scoping rule. Verified to be the
  only occurrence of the 33-denominator form across the lit-review corpus. P1 §2.1 and T5 cite **9/31**.

## 10. Verification status of this outline (2026-07-30)
**Checked:** all six precision-panel cells and their rates recomputed from `dr6` row counts (all six match to
the printed digits); the arithmetic identities (7,160,831 − 4,515,080 = 2,645,751 · 7,155,712 + 5,119 =
7,160,831 · pooled − per-split = 461 · f64 residual = 195, matching `panel`'s independent overlap probe ·
740 − 51 = 689); the per-class table recomputed from `dr6b` (within-class sums = 2,645,751 / 721,914 exactly;
flood mass 99.51%; DDoS-ICMP 50.16% of mass; Recon 0.49%); every cited `numbers_map.md` line number
(31–46, 75, 82–93, 96, 98 — all say what is claimed); E5/E5G rows exist (NM lines 85–86; 0.8551 − 0.8504 =
+0.47pp); the 672/674 scope explanation and both quoted script line numbers; figure inventory
(39 figures in each of two directories, **0** with an `_en` suffix — the three `grep _en` hits are
`entropy`/`enhanced` substrings, not suffixes).
**Four WRONGs found and fixed in this document:** the ±inf→NaN attribution to the audit script (§3.5);
the ±1-precision overstatement on 13,533/357 (§7); "tracked" for an untracked directory (§9-F); the
9/31-vs-9/33 denominator, surfaced as §9-H and since fixed at source.
**Caveat on this check:** the independent-Explore-agent pass specified by `/fact-check` failed twice with
API 529 (overloaded); the verification above was run in-session by the same author as the outline. It is a
self-check, not an independent one — re-run `/fact-check` on this file when subagents are available, and
treat a zero-finding result as suspicious.
