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
