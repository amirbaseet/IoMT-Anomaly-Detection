# Cross-Dataset Generalization — Results Summary

**ShieldMind** · Train: CICIoMT2024 · Test: CICIoT2023 · Status: **Measured**

- Shared-feature schema: **38 features** (intersection; 7 CICIoMT2024-only features dropped, not reconstructed).
- Multi-seed: **3 seeds** (42, 43, 44); cross-dataset numbers are mean ± σ.
- Canonical reproducible sample (seed 42) saved to `sampled_test_set.parquet`.

> Caveat (per design doc §7): cross-dataset used a 38-feature shared-schema retrain; the 7 CICIoMT2024-only features (Duration, Srate, Drate, Magnitue, Radius, Covariance, Weight) were unavailable in CICIoT2023. A drop below the in-dataset baseline is the expected, honest generalization result — not a defect.

## Design A — Binary attack/benign generalization

### Supervised XGBoost (Layer 1)

| Metric | In-dataset (shared-feat, CICIoMT2024 test) | Cross-dataset (CICIoT2023) | Frozen thesis (README, 44-feat) |
|---|---|---|---|
| F1 (attack) | 0.9992 | 0.9636 ± 0.0001 | 0.9880 (best binary, macro) |
| accuracy | 0.9984 | 0.9298 ± 0.0002 | — |
| precision | 0.9992 | 0.9307 ± 0.0002 | — |
| recall | 0.9992 | 0.9990 ± 0.0000 | — |
| roc_auc | 0.9999 | 0.6904 ± 0.0003 | — |

### Unsupervised Flow AE (Layer 2) — threshold set on CICIoMT2024 benign-val

| Threshold | Metric | In-dataset | Cross-dataset (CICIoT2023) | Frozen thesis (README) |
|---|---|---|---|---|
| p90 | F1 (attack) | 0.9915 | 0.9638 ± 0.0001 | 0.9853 |
| p90 | recall | 0.9877 | 0.9995 ± 0.0000 | — |
| p99 | F1 (attack) | 0.8906 | 0.9633 ± 0.0001 | 0.9853 |
| p99 | recall | 0.8029 | 0.9984 ± 0.0000 | — |
| — | ROC-AUC | 0.9917 | 0.2586 ± 0.0010 | 0.9892 |

## Design B — 5 shared families (Benign, Spoofing, Recon, DoS, DDoS)

Out-of-scope CICIoT2023 classes (all MIRAI-*, SQLINJECTION, XSS, BACKDOOR_MALWARE, BROWSERHIJACKING, COMMANDINJECTION, UPLOADING_ATTACK, DICTIONARYBRUTEFORCE) were dropped from the B test set. MQTT was dropped from the CICIoMT2024 training set (no CICIoT2023 counterpart).

| Metric | In-dataset (shared-feat) | Cross-dataset (CICIoT2023) | Frozen thesis (README 6-class incl. MQTT) |
|---|---|---|---|
| macro-F1 (5 families) | 0.9037 | 0.1965 ± 0.0002 | 0.9363 |
| accuracy | 0.9978 | 0.2627 ± 0.0006 | — |

### Per-family F1 (cross-dataset, mean ± σ)

| Family | In-dataset F1 | Cross-dataset F1 | Cross-dataset recall | support (seed 42) |
|---|---|---|---|---|
| Benign | 0.9703 | 0.1990 ± 0.0015 | 0.2076 ± 0.0020 | 40209 |
| Spoofing | 0.5788 | 0.0372 ± 0.0003 | 0.0847 ± 0.0006 | 39772 |
| Recon | 0.9707 | 0.3421 ± 0.0016 | 0.5274 ± 0.0021 | 82546 |
| DoS | 0.9992 | 0.0131 ± 0.0008 | 0.0067 ± 0.0004 | 79553 |
| DDoS | 0.9997 | 0.3913 ± 0.0013 | 0.2957 ± 0.0010 | 240251 |

Confusion matrices: `confusion_A.csv` (binary, XGBoost), `confusion_B.csv` (5 families) — both for the canonical seed.

_Interpretation (honest generalization vs pipeline artifact) is left to the planning chat, per the brief._

<!-- ADDENDUM:START -->

## Addendum 1 — Design A base-rate diagnostics (Measured)

The ~0.96 attack F1 is a **base-rate artifact**: CICIoT2023 is ~93% attack, so flagging nearly everything as attack scores high F1/recall. **ROC-AUC is the headline transfer metric**, and benign recall / balanced accuracy expose the collapsed benign side.

| Detector | Benign recall | Attack recall | Balanced acc | ROC-AUC |
|---|---|---|---|---|
| XGBoost (cross) | 0.0005 ± 0.0000 | 0.9990 ± 0.0000 | 0.4998 ± 0.0000 | 0.6904 ± 0.0003 |
| XGBoost (in-dataset) | 0.9650 | 0.9992 | 0.9821 | 0.9999 |
| Flow AE p90 (cross) | 0.0000 ± 0.0000 | 0.9995 ± 0.0000 | 0.4998 ± 0.0000 | 0.2586 ± 0.0010 |
| Flow AE p99 (cross) | 0.0007 ± 0.0001 | 0.9984 ± 0.0000 | 0.4996 ± 0.0001 | 0.2586 ± 0.0010 |

**Finding — AE inversion.** The Flow AE cross-dataset ROC-AUC is **0.2586 ± 0.0010**, i.e. **below 0.5**: under the CICIoMT2024-benign-trained autoencoder, CICIoT2023 *attacks* reconstruct with **lower** error than CICIoT2023 *benign*. The reconstruction-error signal is inverted across testbeds — flooding traffic sits closer to the CICIoMT2024 benign manifold than CICIoT2023's own benign does. Verified not a scaling artifact: the standardizer and imputer are fit on CICIoMT2024 only and applied transform-only to CICIoT2023 (no `.fit`/`.fit_transform` ever touches the test set).

## Addendum 2 — Design B robustness: DoS+DDoS merged into 'Flooding' (4 families, Measured)

The DoS vs DDoS distinction is effectively per-flow-impossible (it is a property of the campaign, not the individual flow), so collapsing them into one **Flooding** family is the fairer transfer measure. Computed post-hoc by merging DoS+DDoS in both truth and prediction of the existing 5-family model — no penalty for DoS↔DDoS confusion.

| Metric | In-dataset (shared-feat) | Cross-dataset (CICIoT2023) |
|---|---|---|
| macro-F1 (4 families) | 0.8800 | 0.2578 ± 0.0001 |
| accuracy | 0.9982 | 0.3233 ± 0.0004 |

Per-family F1 (4-family):

| Family | In-dataset F1 | Cross-dataset F1 |
|---|---|---|
| Benign | 0.9703 | 0.1990 ± 0.0015 |
| Spoofing | 0.5788 | 0.0372 ± 0.0003 |
| Recon | 0.9707 | 0.3421 ± 0.0016 |
| Flooding | 1.0000 | 0.4530 ± 0.0008 |

For reference, the 5-family macro-F1 was 0.1965 ± 0.0002; merging DoS+DDoS moves it to 0.2578 ± 0.0001 — the delta quantifies how much of the Design-B degradation was DoS↔DDoS confusion versus genuine non-transfer.

<!-- ADDENDUM:END -->

## Addendum 3 — Inverse direction: train CICIoT2023 → test CICIoMT2024, FULL data (Measured 2026-09-28)

Source: `inverse_full_results.json` (written by `src/inverse_full_run.py`). This section sits outside the ADDENDUM markers because `src/addendum.py` rewrites the text between them. Note: a full re-run of `src/run.py` (via `report.write_summary`) overwrites this whole file, including this section.

**Result: the inverse direction collapses to chance on the full data** — ROC-AUC **0.4797**, balanced accuracy **0.5002**. The model flags almost every CICIoMT2024 row as attack (benign recall 0.0003, attack recall 1.0000). The forward direction is at chance on balanced accuracy (0.4998) but keeps some ranking signal (ROC-AUC 0.6904).

### Protocol

- **Train:** the 62 CICIoT2023 merged CSVs other than `Merged63.csv` (62 of the 63 in `MERGED_CSV.zip`), streamed from the zip → **44,591,073 rows** (1,041,192 benign = 2.33%). 9 empty-label rows dropped. The row counts reconcile with `artifacts/label_counts.json`: 44,591,073 + 428,161 holdout = 45,019,243 − 9.
- **Test:** all rows of CICIoMT2024 `data/test` (21 files) → **1,614,182 rows** (37,607 benign), selected by name to the same 38 shared features (`ciciomt_data.load_split` raises if one is missing).
- **Model:** `config.XGB_PARAMS_BASE` (the forward run's frozen params) via `xgb.train` + `QuantileDMatrix`, unweighted, threshold 0.5. NaN cells are filled with medians fit on the training data only (2,309 train cells, 22 holdout cells, 0 test cells). **One model seed (42), so no σ is reported.** Sampling-seed σ does not exist when all the data is used. With subsample/colsample 0.8, the model-seed σ is not zero, but it is not measured here.
- **Known-positive control:** `Merged63.csv` (428,161 rows) was held out and scored as a within-CICIoT2023 **script sanity cell** → ROC-AUC **0.9981**, bal-acc 0.9301. This is a sanity check. It does not replace the B2 control's 0.974.

**What the control does and does not show.** The holdout shows the CICIoT2023 zip reader and the model are sound. The CICIoMT2024 test loader (`ciciomt_data.load_split`) is a different code path; it is the one that scored 0.9999 in-dataset in the forward run. An AUC this close to 0.5 is chance under either label polarity. Together this points to a dataset shift (testbed and feature extraction) rather than a script bug. It does **not** rule out that features with the same name are computed differently in the two datasets.

### 2×2 generalization table (binary, XGBoost; ROC-AUC / balanced accuracy)

| train ↓ · test → | CICIoMT2024 | CICIoT2023 |
|---|---|---|
| **CICIoMT2024** | 0.9999 / 0.9821 (in-dataset, shared-feat) | 0.6904 ± 0.0003 / 0.4998 ± 0.0000 (forward, 3 seeds) |
| **CICIoT2023** | **0.4797 / 0.5002** (inverse, full data, 1 seed) | 0.9981 / 0.9301 (full-run holdout `Merged63`) · 0.974 / 0.7962 (B2 control, `scripts/b2_control.py` in the audit repo; 578k sample, random 75/25 split, 100 trees depth 6) |

### Preliminary → full

| Run | Train data | XGB params | MT test rows | ROC-AUC | bal-acc |
|---|---|---|---|---|---|
| Preliminary (`src/inverse_run.py`, run 2026-09-22, logged 2026-09-23) | 578,659-row class-capped seed-42 sample (`sampled_test_set.parquet`) | 100 trees, depth 6 | first 200k rows/file, then a 20k/file sample (288,597 rows) | 0.5155 | 0.5001 |
| **Full (`src/inverse_full_run.py`)** | 44,591,073 rows (62 of 63 files) | `XGB_PARAMS_BASE` (200 trees, depth 8, …) | all 1,614,182 | **0.4797** | **0.5002** |

At least six things changed at once, so the 0.036 AUC shift cannot be pinned on any one of them:
1. training data and its class mix
2. XGB params
3. test coverage
4. the holdout file
5. NaN handling (XGBoost's native missing-value routing vs train-median imputation)
6. the training API (`XGBClassifier` vs `xgb.train` + `QuantileDMatrix`)

Both runs sit at chance, so the conclusion does not depend on the shift.

### Reading

- **Each model separates its own dataset well, and transfer fails in both directions.** The CICIoMT2024 model scores 0.9999 in-dataset. The full CICIoT2023 model scores 0.9981 on the held-out `Merged63`, and the preliminary-protocol B2 control scores 0.974. Across datasets, the forward direction keeps weak ranking (AUC 0.6904) and the inverse has none (AUC 0.4797). Both are at chance on balanced accuracy.
- **The bal-acc near 0.5 is partly the base rate.** Both training sets are ~97–98% attack (CICIoT2023 train 97.67% by row count; CICIoMT2024 `data/train` 97.31% by `wc -l` minus headers), so at threshold 0.5 almost everything is called attack. ROC-AUC is the threshold-independent measure. In the inverse direction it is also at chance.
- **Doménech et al. (2025) — partial replication, with an open divergence on ROC-AUC.** They ran the same direction (train CICIoT2023 → test CICIoMT2024) on 44 common features.
  - 5-class: F1 falls from 0.86–0.94 (baseline) to 0.27–0.41 (their Table 5, different attacks) or 0.37–0.55 (Table 4, same attacks).
  - 2-class, same attacks (Table 4): XGB F1 falls from 0.9762 to 0.5523, but **ROC-AUC stays 0.9825**. 71.22% of benign traffic is flagged as attack.
  - We reproduce the threshold-level collapse (our benign recall 0.0003). We do **not** reproduce their retained ranking (our AUC 0.4797 vs their 0.9825).
  - Features (our 38 shared vs their 44 common), preprocessing, attack scope and split all differ. The cause of the AUC gap is **not yet explained**. This is a replication/robustness result, not a novel one.

## Addendum 4 — Feature-definition mismatch control (Measured 2026-09-29)

Sources: `feature_scale_check.csv` (per-feature medians and single-feature AUC per dataset), `inverse_full_drop_flipped.json`, `inverse_full_drop_scale.json`. Both JSONs were written by `src/inverse_full_run.py --drop …`. The protocol is otherwise identical to Addendum 3: the same 44,591,073 train rows, the same 1,614,182 test rows, `XGB_PARAMS_BASE`, 1 model seed.

**Result: removing 7 scale-mismatched features lifts the inverse ROC-AUC from 0.4797 to 0.7067.** Removing the 3 most strongly direction-flipped features changes almost nothing (0.4898). This is evidence that part of the inverse collapse comes from same-named columns that sit on different scales in the two datasets.

### The mismatch (`feature_scale_check.csv`)

Basis:
- **CICIoT2023:** `sampled_test_set.parquet` — 578,659 rows drawn from all 63 merged CSVs, seed-42 class-capped (40k benign, ≤20k per attack class).
- **CICIoMT2024:** all 1,614,182 `data/test` rows.

The within-CICIoT2023 single-feature AUCs therefore reflect the class-capped mix, not the 97.7%-attack training distribution.

| Feature | CICIoT2023 benign median | CICIoMT2024 benign median |
|---|---|---|
| `IAT` | 0.006511 | 1.695e+08 |
| `Header_Length` | 29.6 | 1.921e+05 |
| `Variance` | 1.616e+05 | 0.9 |
| `rst_count` | 0 | 1372 |

- **Scale gaps:** 6 features have nonzero benign medians ≥10× apart: `IAT`, `Header_Length`, `Variance`, `Rate`, `Std`, `Max`. The next-largest gap is `psh_flag_number` at 5×.
- **Zero on one side:** 4 more have a benign median of zero in one dataset only: `rst_count` (0 vs 1372), `ack_count` (9 vs 0), `HTTPS` (0.8 vs 0), `UDP` (0.1 vs 0).
- **Direction flips:** 14 of 38 features have single-feature AUCs (benign vs attack, within each dataset) on opposite sides of 0.5. The other 11 are weak (each within 0.03 of 0.5 in at least one dataset). The 3 strongest flips are `Tot sum` (0.6555 vs 0.3063), `rst_count` (0.5860 vs 0.05758) and `Protocol Type` (0.5675 vs 0.4406).

### Control runs (train CICIoT2023 → test CICIoMT2024)

| Run | Features | Inverse ROC-AUC | Inverse bal-acc | Benign recall | Holdout `Merged63` ROC-AUC |
|---|---|---|---|---|---|
| All shared (Addendum 3) | 38 | 0.4797 | 0.5002 | 0.0003 | 0.9981 |
| Drop the 3 strongest flips: `Tot sum`, `rst_count`, `Protocol Type` | 35 | 0.4898 | 0.5004 | 0.0008 | 0.9981 |
| **Drop 7 scale-mismatched:** the 6 with ≥10× gaps + `rst_count` | 31 | **0.7067** | 0.5099 | 0.0200 | 0.9968 |

### Reading

- **Scale-mismatched features matter more than the flipped ones tested.** Dropping the 7 scale-mismatched features restores some ranking signal (+0.227 AUC). Dropping the 3 most strongly flipped features does not (+0.010). This does not separate scale from direction, because:
  - the two sets overlap (`rst_count` is in both);
  - they differ in size (3 vs 7), and no size-matched random-drop control was run;
  - 11 weaker flips and 3 zero-on-one-side features (`ack_count`, `HTTPS`, `UDP`) stay in both runs.
- **CICIoT2023 separation holds without these features.** The holdout ROC-AUC stays at 0.9981 and 0.9968 (both ≥0.996). With the 7 dropped, holdout bal-acc falls slightly, from 0.9301 to 0.9201.
- **Ranking partly recovers, threshold decisions do not.** AUC 0.7067 is still far below the 0.9968 within-dataset holdout. At threshold 0.5 the model still calls almost everything attack: bal-acc 0.5099, benign recall 0.0200. Calibration itself was not measured.
- **The inverse is now comparable to the forward direction.** 0.7067 is close to the forward MT→IoT2023 AUC of 0.6904, but the two differ in setup: the forward used 38 features and 3 sampling seeds; the inverse uses 31 features and 1 seed. The forward run was not re-run without these features.
- **Doménech gap.** Dropping the mismatched features closes about 0.23 of the 0.50 AUC gap to Doménech's 0.9825. The rest is unexplained. Whether their data has the same mismatch was not tested.

### Caveats

- **Post-hoc selection.** Both drop sets were chosen with CICIoMT2024 test labels: benign-conditional medians for the 7, within-test single-feature AUCs for the 3. These runs diagnose *why* transfer fails. They are not a deployable model result.
- **Imputation confound.** The drop-7 run also removes every NaN-bearing column (train NaN cells 2,309 → 0). The drop-3 run does not.
- **One model seed.** Seed variance was not measured.
- **Not isolated.** No per-feature ablation was run, so which of the 7 matter most is unknown.
- **`Variance` is a different quantity in the two releases (follow-up, measured 2026-10-02).** Source: `variance_definition_check.json` (written by `src/variance_definition_check.py`). In the CICIoT2023 sample, `Variance` equals `Std`² on all 578,639 finite rows (of 578,659; rtol 1e-3). In CICIoMT2024 `data/test`, `Variance` lies in [0, 1] on all 1,614,182 rows and equals `Std`² on 81.2% of them. The markdown extract of Doménech et al.'s Table A.9 pairs `Variance` with "Ratio of the variance" (layout mangled in the extract; no PDF on disk). So for this one feature the mismatch is a definition difference, not a testbed difference.
- **For the other six features the cause of the scale difference is unproven.** The medians suggest different definitions: the CICIoT2023 `IAT` benign median, 0.0065, fits seconds between packets; CICIoMT2024's 1.69e8 does not, and its unit is unknown. The raw CICIoT2023 pcaps and the packet-level CSVs with timestamps (`archive.zip`) on the SSD could test the `IAT` definition from the CICIoT2023 side.
