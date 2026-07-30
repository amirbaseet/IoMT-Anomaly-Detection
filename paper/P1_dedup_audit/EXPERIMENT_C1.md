# C1 — controlled raw-vs-deduplicated arm            status: SPEC (awaiting go-ahead) · 2026-07-30

Authorized by **Amendment 1** of `.claude/plans/2026-07-30-p1-dedup-paper-brief.md`. Supplies the primary
evidence for manuscript §8.1, replacing the cross-paper accuracy difference.

## Design — a 2×2 matrix, not a single arm
The approved amendment asks for a raw-trained model evaluated on both test splits. Completing the square costs
one extra training run and separates the two mechanisms cleanly:

| trained on ↓ / tested on → | **raw test** | **dedup test** |
|---|---|---|
| **raw train** | C1-a — what the literature reports | C1-b — memorization exposed: honest test, leaky training |
| **dedup train** | C1-c — leak-free training, inflated test | C1-d — the thesis's published E7 (macro-F1 0.9076 / MCC 0.9906 / acc 99.27%) |

- **C1-a − C1-d** is the matched replacement for the 99.80→99.27 cross-paper Δ.
- **C1-b vs C1-c** decomposes it: C1-b isolates *training-side* memorization, C1-c isolates *test-side*
  redundancy inflation. No published study separates these.
- C1-d already exists as a number but will be re-run inside the same script so all four cells share one
  code path, one library version and one seed.

## Fixed configuration (deviating from any of these invalidates the comparison)
- Model: E7 exactly — `XGBClassifier`, Full-44 feature set, **no** resampling, `random_state=42`,
  hyperparameters read from `notebooks/supervised_training.py:207–217` (`get_xgb`), no `class_weight` /
  `scale_pos_weight` (that absence is itself documented in README §12.4).
- Task: 19-class multiclass. Metrics: macro-F1, accuracy, MCC, macro-precision, per-class F1.
- Split: the dataset's **official file-level** train/test directories; validation = the pipeline's own
  80/20 `train_test_split(random_state=42)` (`preprocessing_pipeline.py:339–343`). No re-splitting, no merging.
- Seeds: single seed 42 — this is a like-for-like comparison against E7, not a stability claim. If
  |C1-a − C1-d| falls inside the known σ = 0.023 band, that is the reported finding (DN-05 spirit).
- **INV-01 gate:** assert the raw arm's `LabelEncoder` mapping is identical to `preprocessed/config.json`'s
  before any cross-evaluation. Different label space ⇒ abort, not "reorder and hope".
- **DN-02/DN-03 gate:** the raw arm fits its scaler on raw-train only. Nothing is fitted on val or test.
  No AE/IF/VAE involvement — this is a tree-only ablation.

## Implementation plan (no forking of the frozen pipeline)
1. `experiments/build_raw_cleaned.py` — reproduce `notebooks/ciciomt2024_eda.py`'s merge + clean over the 72
   raw CSVs (±inf→NaN, median fill) **with `drop_duplicates` omitted**, writing
   `eda_output_raw/{train,test}_cleaned.csv`. Expected row counts: **7,160,831 / 1,614,182** — asserted, not hoped.
2. `experiments/run_raw_preprocessing.py` — `import notebooks.preprocessing_pipeline as pp`, override
   `pp.TRAIN_INPUT` / `pp.TEST_INPUT` / `pp.OUTPUT_DIR` / `pp.EXPECTED_*_ROWS`, then call its own functions.
   Reusing the module (never copying it) is what guarantees the two arms are treated identically. Only the
   full-44 tree variant is needed — SMOTE variants, AE sets and zero-day sets are skipped.
3. `experiments/run_c1_matrix.py` — train both models, score all four cells, write
   `results/c1_dedup_ablation/c1_matrix.json` + a per-class CSV.
4. New numbers → `deliverables/numbers_map.md` before any manuscript prose (DN-04).

## Cost and risk
- **Compute:** Phase 4 ran 24 trainings in 60 min on the 3.6M-row deduplicated split
  (`numbers_map.md` §11) ⇒ two arms ≈ tens of minutes. Raw preprocessing is the long pole (Phase 3 took
  228 min for the *full* artifact set; this subset is a fraction of it).
- **⚠ Disk is the real constraint.** The volume is **90% full — 45 GiB free**. `preprocessed/` is 5.7 GB for
  4.5M train rows, so the raw equivalent is ≈9 GB, plus ≈1.8 GB of raw cleaned CSVs ≈ **11 GB**. It fits, but
  it is a third of the remaining headroom. Mitigations: write only the full-44 variant; keep features as
  float32 `.npy`; delete `eda_output_raw/` once the `.npy` artifacts exist; or point the output at an external
  volume. **This needs a go-ahead before anything is written.**
- **Not a risk:** no existing artifact is overwritten. Every output path is new (`eda_output_raw/`,
  `preprocessed_raw/`, `results/c1_dedup_ablation/`); the frozen pipeline files are imported, never edited.

---

## RESULTS (2026-07-30) — status: COMPLETE

5 seeds [1, 7, 42, 100, 1729], 10 trainings, 122.9 min. Seed 42 reproduced `run_c1_matrix.py` exactly
(drift = none). Oracle rows live in `deliverables/numbers_map.md` §2; artifacts in
`results/c1_dedup_ablation/`.

| cell | trained | tested | macro-F1 (mean ± σ) | accuracy (mean ± σ) |
|---|---|---|---|---|
| C1-a | raw | raw | 0.8995 ± 0.0106 | 0.995158 ± 0.001077 |
| C1-b | raw | dedup | 0.8917 ± 0.0105 | 0.991652 ± 0.001928 |
| C1-c | dedup | raw | 0.8989 ± 0.0168 | 0.994840 ± 0.001128 |
| C1-d | dedup | dedup | 0.8909 ± 0.0168 | 0.990819 ± 0.002054 |

| contrast | macro-F1 | separable? |
|---|---|---|
| test-set effect, raw-trained | +0.00784 ± 0.00077 | **YES** — sign-consistent over 5 seeds |
| test-set effect, dedup-trained | +0.00798 ± 0.00025 | **YES** |
| training-set effect, on raw test | −0.00064 ± 0.02468 | no — sign flips between seeds |
| training-set effect, on dedup test | −0.00078 ± 0.02438 | no |
| naive raw-vs-dedup pairing | +0.00863 ± 0.02468 | no |

**Verdict.** The amendment's purpose is served, and the answer is not the one the brief anticipated. Duplicate
leakage costs metrics on the **test** side (+0.35–0.40 pp accuracy, +0.78–0.80 pp macro-F1, tight σ) and has
**no separable training-side effect**. The "memorization premium" framing — including this project's own
0.53 pp figure — is retired: the raw-vs-dedup pairing is inside single-configuration seed variance.

Side result: 37 of 44 fitted scaler parameters differ between the arms (RobustScaler scale up to 5.67×),
so deduplication changes the feature space itself, not merely the row count.

Design defects found and fixed during execution (all in the experiment code, not the pipeline): cross-space
scoring, inverted contrast labels, and an invalid σ borrowed from the fusion recall metric. See commit
`7379ad0`.
