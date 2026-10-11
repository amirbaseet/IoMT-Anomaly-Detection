# same_extractor — controlled subset experiment (NFStream flows, CICIoMT2024 vs CICIoT2023)

**What this is.** A reproducible comparison of CICIoMT2024 and CICIoT2023 flows produced by the **same
extractor** (`iomt-pcap-experiments/stage2a_extract_features.py` → `extract_one`, NFStream), restricted to the
attack families that have pcaps on both sides. It is a **controlled subset experiment**, not "the
cross-dataset result": it does not replace `../results_summary.md` and nothing here edits it,
`deliverables/numbers_map.md`, or any existing file. Every number below is `post-hoc`.

**Status of this folder (2026-10-10):** steps 01–05 (arms 1–4, label-rule pre-check) and the 04b follow-up run (step 04 after the owner's ruling: group by `src_mac` only,
because each dataset has one independent benign capture). **Arms 1–4 were run after the owner's decisions of 2026-10-10. Arm 5, the background audit and the two sensitivities were declared in "Changed after seeing arms 1–4" and run on 2026-10-11 (results in the sections after it); later additions (float32 overlap re-measure, 4-family macro, ICMP shares) are labelled "added after" where they appear. The sections "Detection stage — PLAN" and "NOT YET RUN" below are the original text, kept as written, with status notes added.**

## Layout

```
same_extractor/
  README.md                        this file
  proposed_numbers_map_entries.md  proposals only, generated from results/*.json, status post-hoc
  src/_common.py                   paths, pair table, provenance, 59-column helper (imports NON_FEATURE_COLS)
  src/01_version_check.py          venv: iomt-pcap-experiments/venv (only one with nfstream 6.6.0)
  src/02_column_audit.py           venv: IoMT-Project/venv
  src/03_family_counts.py          venv: IoMT-Project/venv
  src/04_origin_control.py         venv: IoMT-Project/venv; seeds 42, 43, 44
  src/04b_followup.py              venv: IoMT-Project/venv; permutation null + seed-mechanism check + re-run of one reference
  src/05_detection_arms.py         arms 1–4 + uncapped sensitivity; venv: IoMT-Project/venv; --scratch <dir> holds the row cache (not in the repo)
  src/05b_csv_label_check.py       arm-5 pre-check on the official MERGED_CSV.zip; src/05c_packets_per_row.py derives packets/row
  src/render_proposed_entries.py   renders the proposal file; reads results only
  src/06_like_for_like.py, 07_background_audit.py, 08_sensitivities.py   arm 5, audit, sensitivities (declared in "Changed after seeing arms 1–4" below)
  src/06b_overlap_checks.py        added after review: overlap positive control + within-dataset baseline overlap (arm 5)
  src/09_float32_overlap_arms1to4.py  added after review: arms 1-4 overlap re-measured on float32 values
  src/render_readme_arm5.py        appended the arm-5 results section once
  src/10_macro4.py, 11_icmp_shares.py   added after the background audit: 4-family macro (no model), ICMP indicator shares; src/render_readme_icmp.py appended their section once
  results/                         JSON/CSV, each < 5 MB; no flow data
```

Run each step as `<venv>/bin/python -B -I src/NN_name.py` (`-B`: no `__pycache__` in either repo; `-I`: no
implicit `sys.path` entries). Step 01 needs `--scratch <dir>` for the extracted pcaps (outside the repo).
Result files are never overwritten: a second run refuses, it does not delete.

## Rules this follows

- **Model inputs = 59 columns** = all columns except `NON_FEATURE_COLS` (imported from `stage2c_dedup.py`,
  19 names) and the label. The 59 names are printed in `results/02_column_audit.json` and
  `results/03_family_counts.json`. `src_mac` / `src_capture` are used only for grouping and audits.
- **Provenance in every result file** (`meta` block): script path + sha256 (file bytes), helper sha256,
  python executable / prefix / versions, seeds, UTC run time (`libs.nfstream` is null outside step 01: the IoMT-Project venv has no nfstream; `01_version_check.json` holds the version, other steps carry it in `inputs_note`), input paths + sizes + sha256 (an empty list in `05_detection_arms.json` and `06b_overlap_checks.json`, whose inputs are scratch caches named in `inputs_note`),
  thesis repo commit + branch, and the full list of dirty tree entries (not just a boolean).
- **nfstream version of the inputs:** CICIoT2023 tables = 6.6.0 (`manifest.jsonl`, same stage2a sha256 as today).
  CICIoMT2024 `features_identity` = **not recorded**; step 01 tests whether today's extractor reproduces it
  on two pcaps, it does not recover the old version.
- **Labels** come from the capture filename in both pipelines (see `03_family_counts.py` docstring), so benign
  background inside an attack pcap carries the attack label in both tables.
- **Dedup** = `stage2c_dedup.py`'s method (59-column key, global within a dataset, keep first, files in sorted
  name order). The order dependence of cross-class duplicates is stated in the result; the within-label count is
  the order-free sensitivity. A control run must reproduce the stored `dedup_report.json` before counts are used.

## NOT YET RUN (original text of 2026-10-10; status 2026-10-11: both items have since been run — the like-for-like arm as arm 5, the NFStream-space within-dataset baselines as arms 1 and 2)

- **Like-for-like arm:** the CSV-based transfer (`../src/run.py`, the published-CSV features) re-run on the
  *same family subset*, so a change in AUC is not confused with a change of classes.
- **Within-dataset baseline in NFStream space:** the same models trained and tested inside one dataset on
  NFStream features.

## Limits

- **10 of 33 CICIoT2023 attack classes have no pcap** (owner's figure; checked 2026-10-11 outside this folder: 33 named attack labels in `thesis/crossdataset/artifacts/label_counts.json` (untracked, local), 23 of them in pcap filenames under `/Volumes/PortableSSD/CICIOT23/pcaps` and `pcaps_full`).
  Only the classes in `src/_common.py::PAIRS` are on the CICIoT2023 side.
- **`DDoS-ICMP_Flood` is a partial set** of the CICIoT2023 captures (the feature folder holds a subset of the
  numbered files; `results/03_family_counts.json` lists exactly which).
- **38 of 72 CICIoMT2024 pcaps are not on disk** (their stored features are kept). Re-derived in
  `results/01_version_check.json` → `coverage_of_stored_captures` from the two readable zips; a third zip is
  unreadable (`BadZipFile`) and a fourth holds CSVs and two PDFs only (49 `.csv`, 2 `.pdf` by `zipfile.namelist()`, checked 2026-10-11).
- **The stored CICIoMT2024 run's nfstream version was not recorded** (see above).
- CICIoMT2024 has both DoS and DDoS classes per protocol; CICIoT2023 pcaps cover DoS-SYN/TCP/UDP but
  **DDoS-ICMP**. Pair rows in step 03 keep the CICIoMT2024 classes as separate sub-rows; a pair row is not a
  claim that DoS equals DDoS.
- Benign "contiguity" uses a 60 s gap threshold chosen by the script author; raw gaps are in the result so it
  can be re-judged.

## Step 04 results (origin control, group = `src_mac`) and 04b follow-up

Basis for every figure: deduplicated benign flows, 59 model inputs, `StratifiedGroupKFold(5)` on `src_mac`, untuned XGBoost (`max_depth=6`, no subsampling), seeds 42/43/44 for the folds; `results/04_origin_control.json`, `results/04b_followup.json`.

| task | rows (classes) | `src_mac` groups | AUC | balanced accuracy at 0.5 |
|---|---|---|---|---|
| **Origin:** CICIoMT2024 benign (0) vs CICIoT2023 benign (1) | 119,381 (10,540 / 108,841) | 90 | 0.813 ± 0.019 (sd over 3 seeds) | 0.643 ± 0.007 |
| Time reference: CICIoMT2024 `Benign_train` vs `Benign_test` | 10,540 (8,640 / 1,900) | 23 | 0.654, ONE fold partition (see below) | 0.586, one partition |
| Time reference: CICIoT2023 first vs second half of the benign capture | 108,841 (70,579 / 38,262) | 67 | 0.628 ± 0.004 | 0.605 ± 0.004 |

**These are not comparable.** In the origin task the label is constant within a `src_mac` (a MAC belongs to one dataset; no MAC string occurs in both), so the test is "unseen device → which dataset".
In the two time references the label varies within a MAC (the same device appears in both halves), so the test is "same devices, earlier vs later". The 0.654 and 0.628 are kept as measured; they are not a baseline for 0.813 and no difference between them is claimed.

**Group-level permutation null (replaces reference (c), which was asked for in the step-04 report and had not been run).** The dataset label was shuffled across the 90 `src_mac` groups (each MAC keeps one label; 23 groups with label 0, 67 with label 1, as observed), rows inherit their group's label, everything else identical, 200 permutations.
Statistic = mean AUC over the 5 folds with fold seed 42 (200 × 5 fits; 3 seeds would triple the cost). Null AUC: mean 0.4997, sd 0.0932, 2.5th percentile 0.337, 97.5th percentile 0.722, maximum 0.748 (200 valid, none skipped).
The observed value with the same fold seed is 0.833 (3-seed mean 0.813): above all 200 null values, one-sided p = (1+0)/(1+200) = 0.005, the smallest value 200 permutations can give. This shows the origin signal is not what 90 randomly labelled device groups produce. It does not say what the signal is (device mix, traffic timing, testbed).
The null is wide (sd 0.09) because each fold holds only a few groups.

**Why the seeds changed the folds in two tasks and not in the CICIoMT2024 reference.** `StratifiedGroupKFold(shuffle=True)` permutes the groups, then re-sorts them by `np.argsort(-np.std(class_counts_per_group), kind="mergesort")` and assigns them greedily.
The shuffled order survives only among groups with exactly equal std; after that sort the assignment is deterministic. Measured in `results/04b_followup.json` → `seed_mechanism_diagnostics`:
origin: 90 groups, 85 distinct std values (10 groups in ties) → 3 different partitions over the 3 seeds; CICIoT2023 reference: 67 groups, 49 distinct values (26 in ties) → 3 different partitions;
CICIoMT2024 reference: 23 groups, 23 distinct values (0 ties) → ONE partition for all seeds, so seeds 42/43/44 gave identical folds and the "sd 0.0" was not a stability estimate. Shuffling was on, but ineffective there. The XGBoost seed does not change the model either (no subsampling, `hist`).
**Fix and re-run of that task:** `GroupKFold(5, shuffle=True, random_state=seed)` (no stratification, no std sort): 3 different partitions, every test fold has both classes, AUC 0.646 ± 0.047 (sd over seeds), balanced accuracy 0.559 ± 0.025; per seed AUC 0.616 / 0.700 / 0.623.
That re-run supersedes the 0.654 ± 0.000 as the CICIoMT2024 reference spread; 0.654 stays in `04_origin_control.json` unchanged. The CICIoT2023 reference and the origin task already varied by seed and are not re-run.

## Detection stage — PLAN (written 2026-10-10 before any detection run; arms 1–5 have since been run, see the result sections)

(Delivered inside this README because a global hook blocks new `.md` files except README/CLAUDE/AGENTS/CONTRIBUTING/FINDINGS; the owner chose this over a separate file, 2026-10-10.)
Written before any detection result exists. Everything here is fixed now; a change after the first run is a deviation and must be listed as one in the result files. Status of every future number: `post-hoc`.
Basis for counts: deduplicated rows, `results/03_family_counts.json` (59-column key, global within a dataset, stage2c method; control reproduced `dedup_report.json`).

### 1. Task, inputs, model
- **Task:** binary benign (0) vs attack (1), one model per arm and seed, trained on all families together. **Rows:** deduplicated only. **Inputs:** the 59 columns (names printed in each result file); `src_mac`, `src_capture` and timestamps only for splitting and audits.
- **Model:** `XGBClassifier(**config.XGB_PARAMS_BASE, objective="binary:logistic", eval_metric="logloss", random_state=seed)` exactly as `thesis/crossdataset/src/models.py::_xgb` builds it; imported from `thesis/crossdataset/src/config.py`, never retyped, never tuned. At the time of writing the file holds `n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=0.1, tree_method=hist, n_jobs=-1`. **No class weights** (the thesis models use none). Step 04's model (`max_depth=6`, no subsampling) is a different, untuned choice and is not reused.
- NaN/inf are asserted absent; if present, the thesis median imputer is fit on the training side only.
- **Training-set size (owner decision 2026-10-10, arms 1–5 alike):** per class, training rows are capped at 20,000 per family and 40,000 for benign (the values of `config.ATTACK_CAP` / `config.BENIGN_CAP`, read from the file), drawn without replacement with the seed; smaller classes kept whole. Test rows are not capped. **Reason:** flood dominance — floods are 99.93% of IoMT and 97.7% of IoT pair-family attack rows (deduplicated, 4,350,483 of 4,353,616 and 6,086,896 of 6,229,583), so an uncapped fit is mostly SYN/TCP. **It is NOT the thesis's training rule:** in `run.py` the thesis caps sample the CICIoT2023 TEST set (`sampled_test_set.parquet`) in the forward run, and `inverse_full_run.py` trained on all rows; `inverse_run.py` trains on the capped sample. **Sensitivity (declared before any run):** arms 3 and 4 once more with UNCAPPED training, seed 42 only, reported beside the capped result.

### 2. Families
Floods merged at protocol level (DoS and DDoS together); pair table from `src/_common.py::PAIRS`. Deduplicated flows, files in brackets:

| family | CICIoMT2024 | CICIoT2023 | small-support (< 1,000 dedup flows) |
|---|---|---|---|
| SYN flood | 2,228,567 (10) | 2,288,310 (8) | no |
| TCP flood | 2,007,522 (10) | 2,232,777 (11) | no |
| UDP flood | 111,373 (15) | 1,524,156 (17) | no |
| ICMP flood | 3,021 (15) | 41,653 (10, partial set) | no |
| port scan | **653** (2) | 17,678 (1) | **CICIoMT2024** |
| OS scan | 1,215 (2) | 23,803 (1) | no |
| ping sweep | **107** (2) | **716** (1) | **both** |
| vuln scan | **281** (2) | 73,690 (1) | **CICIoMT2024** |
| ARP spoofing | **877** (2) | 26,800 (2) | **CICIoMT2024** |
| Benign | 10,540 (2) | 108,841 (4) | — |

- **Small-support rule:** a family below 1,000 deduplicated flows in the dataset it is TESTED on is printed with the flag `SMALL_SUPPORT` and gets no per-family claim (it still counts inside pooled and macro numbers). Flagged: arm 3 (tests on IoT) ping sweep; arm 4 (tests on IoMT) port scan, ping sweep, vuln scan, ARP; arm 1 the same four; arm 2 ping sweep. The test-side count after splitting is also printed, and the flag also applies below 300.
- **Excluded:** the five CICIoMT2024 MQTT classes and every CICIoT2023 class outside the table. The ICMP pair is CICIoMT2024 {DoS_ICMP, DDoS_ICMP} vs CICIoT2023 DDoS-ICMP_Flood only; a pair is not a claim that DoS equals DDoS.

### 3. Arms and splits
All splits are made AFTER dedup, so no exact 59-column float64 vector sits on both sides inside one dataset (arms 1–2). That is true by construction and is not evidence of generalisation. **Added 2026-10-11:** XGBoost reads float32, and on float32 values 1.16% of arm 1 test rows (seed 42) and 0.51–1.92% of arm 2 test rows have a twin on the train side (`results/09_float32_overlap_arms1to4.json`). Benign is one capture per dataset (step 03), so no benign split is capture-disjoint.
- **Arm 1, within CICIoMT2024.** Attacks: the dataset's own file-level train/test split (`*_train` / `*_test` captures), capture-disjoint. Benign: `Benign_train` vs `Benign_test`, two contiguous pieces of one capture (gap 0.0 s, step 03), so a TIME split; flows whose first-seen is within 60 s of the cut are dropped from both sides (the 60 s is the step-03 contiguity constant, chosen by the author). Dedup keeps a shared vector in the first file in sorted-name order; which of `_test` / `_train` wins varies by class (`Benign_test` sorts before `Benign_train`, but `…SYN1_train` before `…SYN_test` because `1` < `_`): stated as a limit, no sensitivity run. Seeds change only the training draw and XGBoost's subsampling.
- **Arm 2, within CICIoT2023.** Families with ≥ 2 captures (SYN 8, TCP 11, UDP 17, ICMP 10, ARP 2): whole files go to test, about 30% of files and at least one per family, chosen by seed; training and test never share a capture. Families with ONE capture (port scan, OS scan, ping sweep, vuln scan) and benign (4 files = one capture): time split on `bidirectional_first_seen_ms`, first 70% train, last 30% test, 60 s embargo at the cut. Capture-disjoint is impossible there and `src_mac` grouping is degenerate (one attacker), so those four families share attacker and testbed across the cut and are expected to look easier. No `src_mac`-grouped benign sensitivity is planned.
- **Arm 3, IoMT → IoT.** Train: CICIoMT2024 pair-family + benign (capped as above). Test: all CICIoT2023 deduplicated rows in the same families + benign, uncapped. **Arm 4, IoT → IoMT:** the reverse.
- **Overlap report (arms 3–4):** share of test vectors whose exact 59-column float64 key also occurs in the training table, computed against the FULL deduplicated training-side table and against the capped sample actually used; overall and per family. Overlapping rows stay in the test set; a second AUC on the non-overlapping rows is also reported. Preprocessing is fit on the training side only.
- **Arm 5, like-for-like on the released CSVs.** The thesis CSV pipeline (`thesis/crossdataset/src/run.py` for IoMT → IoT, `inverse_run.py` for IoT → IoMT) re-run with the SAME family subset (MQTT and out-of-scope classes dropped, floods merged), the same frozen params, seeds 42/43/44, both directions, the same metrics. Its features are the CSV shared set, not the 59 NFStream columns; that is the thing compared, so nothing else may differ. To settle in code before running: (a) whether the thesis sample is deduplicated and on which columns; the same rule must apply to both datasets; (b) the SAME training cap rule as arms 3–4 (20,000 per family, 40,000 benign, drawn with the seed); only the feature set may differ; (c) arms 3–4 vs 5 are compared on the families present in both.
- **Arm 5 is NOT blocked (corrected before any run):** `config.CICIOT2023_MERGED_DIR` does not exist, but the CSVs are in `/Volumes/PortableSSD/CICIOT23/csv/MERGED_CSV.zip` (1,702,032,816 bytes); `read_member` from `inverse_full_run.py` is reused to stream them. Arm 5 does not run until the label-rule pre-check has been reported and the owner has answered.

**Pre-check before arm 5 — how the official CICIoT2023 CSVs label benign background inside attack captures: run by `src/05b_csv_label_check.py`, result in `results/05_csv_label_check.json` (see the report; arm 5 waits for the owner).** Before it ran: not verified. The CSVs were thought absent; `artifacts/label_counts.json` holds only per-label row totals (BENIGN 1,051,373 rows), which are not diagnostic because they come from a different extractor (301,193 raw NFStream benign flows); none of the `thesis/crossdataset/*.md` files I grepped for "background" states the rule. Procedure once the CSVs return: (1) quote the dataset authors' description of labelling with its source; (2) for the ping-sweep capture (1,686 NFStream flows vs 2,161 `RECON-PINGSWEEP` CSV rows) and the ARP capture, list NFStream flows whose protocol that attack cannot produce (DNS, mDNS, DHCP, NTP) and check whether matching CSV rows carry the attack label or `BENIGN`; (3) write the finding with the rows used to `results/05_csv_label_check.json`. If background is `BENIGN` in the CSVs but attack in our filename-labelled NFStream table, the tables define "attack" differently and arm 5 is not like-for-like without saying so. Until the owner answers, arm 5 carries the flag `LABEL_RULE_UNVERIFIED`.

### 4. Metrics
Per arm, per seed, on the test side: **ROC-AUC**, **balanced accuracy at threshold 0.5**, **benign recall at 0.5** (benign test count printed). **Two macros, both equal-weight means over families of the AUC of that family's attack rows vs ALL benign test rows.** (i) **Common-family macro (headline when arms are compared with each other, owner decision 2026-10-10):** over the families that are not small-support in either dataset: SYN, TCP, UDP, ICMP, OS scan. (ii) **Per-arm macro (secondary):** over the non-small-support families of that arm's test side, which differ between arms. Pooled numbers are dominated by SYN/TCP floods; they are also reported, labelled "pooled, flood-dominated". Mean and sd over seeds 42/43/44; per-family and per-seed rows in the result files. The step-04b check is repeated: the script asserts that the 3 seeds produce different draws and fails if they do not. Threshold 0.5 is fixed, never tuned on target data.

### 5. What each outcome would mean, and would NOT show
Context: step 04 shows benign flows separate by dataset origin (AUC 0.813 vs null 0.50 ± 0.09), so a dataset signature exists and any cross-dataset drop is confounded with it.

| outcome | would mean | would NOT show |
|---|---|---|
| Arms 1, 2 high (AUC ≳ 0.95, benign recall high) | the 59 features separate these families inside one dataset | generalisation; arm 2's four single-capture families and both benign time splits share attacker/testbed across the cut |
| Arms 1/2 high, arms 3/4 much lower | features learned in one dataset do not carry over on this subset | that the extractor or features are at fault; testbed, devices, attack tools and capture setup differ at once, and benign differs too (step 04) |
| Arms 3/4 high | the 59-feature signal of these families transfers between these two testbeds | that CICIoMT2024's published results transfer; only 9 of 33 CICIoT2023 attack classes (plus benign) are used, MQTT is dropped, ICMP is a partial set |
| Arm 3 high, arm 4 low (or reverse) | transfer is asymmetric on this subset | a cause; train size, family mix and overlap differ by direction |
| High overlap share in arms 3/4 | many test vectors were literally seen in training (floods repeat), so AUC is partly memorised | how much of the AUC is due to overlap, unless the non-overlap AUC is read beside it |
| Benign recall low, AUC high | scores rank well but 0.5 is the wrong operating point on the target side | that the model fails; the threshold is fixed |
| Arm 5 ≈ arms 3/4 | the CSV-vs-NFStream feature change does not drive the earlier result | that CSVs are equivalent; the label-rule check must be clean first |
| Arm 5 ≠ arms 3/4 | the feature set or label rule matters on the SAME families | which of the two; they differ together |
| Family flagged `SMALL_SUPPORT` | a number is printed | anything about that family |

This stage does not test whether CICIoMT2024's reported detection numbers are inflated.

### 6. Outputs and budget
Outputs: `src/05_detection_arms.py` (arms 1–4 and the uncapped sensitivity), `src/05b_csv_label_check.py` (label pre-check), `src/06_like_for_like.py` (arm 5; it ran 2026-10-11 after the owner answered), `results/05_detection_arms.json`, `results/05_csv_label_check.json`, `results/06_like_for_like.json`, each with the `meta` provenance block, < 5 MB, no flow data, run with `python -B -I`, temp data in the scratchpad only.
**Budget vs need (estimates except row counts and RAM):** rows — IoMT 4.35M attack (4,353,616) + 10.5k benign, IoT 6.23M attack + 108.8k benign (pair families, deduplicated), training ≤ ~0.25M rows per fit; memory — 59 × float64 = 472 B/row → IoT ≈ 3.0 GB, IoMT ≈ 2.2 GB, ≈ 5.2 GB both, machine has 25,769,803,776 bytes (`sysctl hw.memsize`); disk — results < 5 MB, optional scratch cache ≈ 3 GB, 68 GiB free (`df -h`, 2026-10-10); wall-clock — load + dedup ≈ 20 min (step 03 took about that), 12 + 6 fits at 1–3 min each plus scoring up to 6.3M rows → about 1–1.5 h total.

### 7. Changed before any run (2026-10-10, owner decisions after the first plan)
1. **Training cap confirmed** (20,000 per family, 40,000 benign, drawn with the seed) for arms 1–5 alike; the stated reason is corrected to flood dominance, and the claim that the thesis uses the same rule is removed (section 1).
2. **Sensitivity added:** arms 3 and 4 with uncapped training, seed 42 only, reported beside the capped result.
3. **Arm 5 cap rule:** the same cap rule as arms 3–4; "the thesis caps as the thesis has them" is removed.
4. **Arm 5 unblocked:** CSVs located in `MERGED_CSV.zip` on the portable SSD; the label-rule pre-check runs first and arm 5 then waits for the owner.
5. **Headline changed:** common-family macro (SYN, TCP, UDP, ICMP, OS scan) is the headline for comparing arms; the per-arm macro is secondary.
6. **Report fields:** per arm both macros, pooled AUC, balanced accuracy, benign recall, overlap share and non-overlap AUC.
7. **Plan text corrected by me:** which of `_test` / `_train` wins dedup varies by class (arm 1 limit); results script split into `05_detection_arms.py` and `05b_csv_label_check.py`.

### 8. Open items for the owner
- ❓ Arm 5 waits for the label-rule pre-check report. (Status 2026-10-11: answered; arm 5 ran, flag `LABEL_RULE_INFERRED_SAME`.)

## Step 05 results — arms 1–4 (run 2026-10-10; arm 5 NOT run)

Basis: deduplicated test flows, 59 NFStream inputs, thesis XGB params (`config.XGB_PARAMS_BASE`, objective binary:logistic), training capped at 20,000 per family / 40,000 benign, test uncapped, seeds 42/43/44 (mean ± sd over 3 seeds); `results/05_detection_arms.json`. All `post-hoc`. Dedup control passed for both datasets (kept counts per family equal `03_family_counts.json`).

| arm | test flows (benign) | **common macro AUC** (SYN, TCP, UDP, ICMP, OS scan) | per-arm macro AUC | pooled AUC (flood-dominated) | balanced acc @0.5 (pooled) | benign recall @0.5 | overlap share (full train table) | non-overlap pooled AUC |
|---|---|---|---|---|---|---|---|---|
| 1 within CICIoMT2024 | 937,672 (1,847) | **0.992 ± 0.000** | 0.992 ± 0.000 | 0.9998 ± 0.0000 | 0.986 ± 0.000 | 0.973 ± 0.000 | n/a (0 on float64 keys by construction; 1.16% / 0.5–1.9% on float32 keys, arm 1 / arm 2, see "Added after review") | n/a |
| 2 within CICIoT2023 | 1,655,960 (32,625) — **seed 42 only; seed 43: 1,949,028, seed 44: 1,734,820** (benign 32,625 in all) | **0.970 ± 0.001** | 0.928 ± 0.003 | 0.9920 ± 0.0016 | 0.835 ± 0.012 | 0.679 ± 0.027 | n/a (0 on float64 keys by construction; 1.16% / 0.5–1.9% on float32 keys, arm 1 / arm 2, see "Added after review") | n/a |
| 3 IoMT → IoT | 6,338,424 (108,841) | **0.826 ± 0.003** | 0.741 ± 0.001 | 0.9243 ± 0.0048 | 0.731 ± 0.007 | 0.506 ± 0.015 | 0.0030 ± 0.0000 | 0.924 ± 0.005 |
| 4 IoT → IoMT | 4,364,156 (10,540) | **0.919 ± 0.008** | 0.919 ± 0.008 | 0.9984 ± 0.0002 | 0.641 ± 0.009 | 0.281 ± 0.017 | 0.0043 ± 0.0000 | 0.998 ± 0.000 |

Per-arm macro families: arm 1 and arm 4 = the five common families (the others are small-support there); arm 2 adds port scan, vuln scan, ARP; arm 3 adds port scan, vuln scan, ARP. Non-overlap common macro for arms 3/4: 0.824 ± 0.003 and 0.912 ± 0.009.

**Sensitivity — uncapped training, seed 42 only** (same test rows; capped seed-42 value in brackets):

| arm | train rows | common macro | per-arm macro | pooled AUC | balanced acc | benign recall |
|---|---|---|---|---|---|---|
| 3 IoMT → IoT | 4,364,156 | 0.7595 (0.8279) | 0.7012 (0.7399) | 0.8487 (0.9291) | 0.6214 (0.7380) | 0.3160 (0.5163) |
| 4 IoT → IoMT | 6,338,424 | 0.8906 (0.9281) | 0.8906 (0.9281) | 0.9985 (0.9986) | 0.6375 (0.6311) | 0.2751 (0.2624) |

**Flags and limits of these numbers (read before using them):**
- **Small support:** per-family AUCs for port scan, ping sweep, vuln scan and ARP are flagged `SMALL_SUPPORT` in arms 1 and 4 (and ping sweep in arms 2, 3); the files hold them, this README makes no per-family claim.
- **Operating point:** in arms 3 and 4 the ranking is good (pooled AUC ≥ 0.92) but benign recall at the fixed 0.5 threshold is low; nothing here shows the model is unusable or usable at another threshold.
- **Common vs pooled:** the pooled AUC is dominated by SYN/TCP floods (IoMT 99.93%, IoT 97.7% of attack rows); compare arms on the common macro.
- **Arm 2 caveat:** port scan, OS scan, ping sweep, vuln scan and benign are single captures split by time, so attacker and testbed are shared across the cut.
- **Arm 1 caveat:** benign is Benign_train vs Benign_test, one capture cut in time (embargo 60 s dropped 67 rows at seed 42).
- **Overlap method:** a 128-bit key from two pandas row hashes over the 59 float64 values, not a byte compare (the model reads float32; the overlap on float32 values is re-measured under "Added after review").
- **Arm 1 spread:** the split is fixed, so its sd covers only the training draw and XGBoost subsampling; the script asserts that the 3 seeds give different draws (it passed).

## Pre-check for arm 5 — official CICIoT2023 CSV label rule (run 2026-10-10)

`results/05_csv_label_check.json`, `results/05c_packets_per_row.json`. 63 members of `MERGED_CSV.zip` read, 45,019,243 rows; all five checked attack labels and BENIGN are present. The official rows carry no timestamp, address, port or file name, so a row cannot be traced to a pcap: **the evidence is indirect.** `read_member` was not reused as is (it returns only a binary label); the script streams the same members with the same label normalisation.

| class | official rows | our NFStream packets (class files) | packets per official row | official rows mostly ICMP / TCP / UDP (fraction ≥ 0.5) | our packets by protocol ICMP / TCP / UDP |
|---|---|---|---|---|---|
| RECON-PINGSWEEP | 2,161 | 19,919 | 9.22 | 0.000 / 0.746 / 0.139 | 0.020 / 0.743 / 0.233 |
| MITM-ARPSPOOFING | 294,469 | 3,025,366 | 10.27 | 0.000 / 0.732 / 0.263 | 0.010 / 0.720 / 0.268 |
| RECON-OSSCAN | 93,970 | 962,137 | 10.24 | 0.000 / 0.907 / 0.103 | 0.010 / 0.830 / 0.158 |
| RECON-PORTSCAN | 78,730 | 806,215 | 10.24 | 0.000 / 0.891 / 0.115 | 0.009 / 0.826 / 0.163 |
| VULNERABILITYSCAN | 357,583 | 3,594,944 | 10.05 | 0.000 / 0.764 / 0.248 | 0.015 / 0.678 / 0.303 |
| BENIGN | 1,051,373 | 10,766,983 | 10.24 | 0.000 / 0.907 / 0.105 | 0.009 / 0.830 / 0.159 |

**What it shows.** (1) Packets per official row is about 10 for every class, benign included (9.2 for ping sweep): the official rows look like fixed-size windows of roughly 10 packets that cover each whole capture, not just the attack packets. (The window size itself is not verified here.) (2) The official RECON-PINGSWEEP rows are 0.0% mostly-ICMP and 74.6% mostly-TCP, close to our ping-sweep pcap where ICMP is 2.0% of packets and TCP 74.3%. If background had been removed or relabelled benign, those rows would be ICMP-dominated. Together this says background traffic inside attack captures **keeps the attack label in the official CSVs**, the same rule as our filename labelling.
**What it does not show.** It is inference from composition, not an authors' statement and not a row-level join; the two tables' attack classes are the same only in that sense. The CSV class sets can still differ from our pcap subsets (DDoS-ICMP_Flood is partial on our side). `LABEL_RULE_UNVERIFIED` is therefore changed to `LABEL_RULE_INFERRED_SAME` pending the owner's answer; arm 5 has not been run.

## Changed after seeing arms 1–4 (declared 2026-10-11, BEFORE arm 5, the audit and the sensitivities were run)

Everything here was added after the arms 1–4 results were known. The pre-declared headline (common-family macro AUC) stays. Nothing below changes a number in arms 1–4. Status of every number produced from these settings: `post-hoc`.

**Correction to the arms 1–4 table above.** The arm 2 test count 1,655,960 is **seed 42 only**: the held-out capture files differ per seed, so the test set differs (seed 43: 1,949,028; seed 44: 1,734,820; benign 32,625 in all three; `results/05_detection_arms.json -> per_seed.arm2_within_iot.<seed>.n_test`). This means the arm 2 sd also contains the effect of which captures were held out, not only the training draw. Arms 1, 3 and 4 have the same test set in all seeds (937,672 / 6,338,424 / 4,364,156).

### A. Arm 5 — like-for-like on the official CSV features (`src/06_like_for_like.py`)
Settings, fixed now:
1. **Classes.** CICIoT2023 CSV labels = exactly the classes in `_common.PAIRS`: `DoS-SYN_Flood`, `DoS-TCP_Flood`, `DoS-UDP_Flood`, `DDoS-ICMP_Flood`, `Recon-PortScan`, `Recon-OSScan`, `Recon-PingSweep`, `VulnerabilityScan`, `MITM-ArpSpoofing`, `BENIGN` (matched after `labels.normalize_ciciot2023`). **No** `DDoS-SYN/TCP/UDP` flood labels. CICIoMT2024: DoS + DDoS per flood (the `PAIRS` file classes, via the same filename rule as step 05), **no MQTT**.
2. **Rows.** CICIoMT2024: `data/train` + `data/test` per-class CSVs (class from the filename). CICIoT2023: the 63 members of `MERGED_CSV.zip`.
3. **Dedup** within each dataset on the model-input columns, **before** the cap, global over all classes, files in sorted-name order, first occurrence kept (the step-03 rule). Raw → dedup counts are printed per family and written to the result file. **My choices where the brief is silent:** the key is the 38 shared columns (the primary set) for both feature sets, so the two sets see identical rows; the key is a 128-bit hash of the float64 values (two pandas row hashes), not a byte compare, because 45 M rows do not fit a `DataFrame.duplicated` pass.
4. **Feature sets.** (primary) the thesis shared set, 38 columns, read from `thesis/crossdataset/shared_features.json`. (secondary) the set left after the 7 features dropped in `results_summary.md` Addendum 4 (`IAT`, `Header_Length`, `Variance`, `rst_count`, `Rate`, `Std`, `Max`; 31 left). The thesis code holds that list **only as a command-line argument** (`inverse_full_run.py --drop`); the machine-readable copy is `dropped_features` in `thesis/crossdataset/inverse_full_drop_scale.json`, and the script reads it from there instead of retyping it. Both drop sets were chosen with CICIoMT2024 test labels (Addendum 4 caveat), so the secondary set is not independent of the CICIoMT2024 side.
5. **Model and metrics.** Same as arms 3–4: `models._xgb("binary:logistic","logloss")` from `config.XGB_PARAMS_BASE`, `random_state=seed`, cap 20,000 per family / 40,000 benign, seeds 42/43/44, test rows uncapped, both macros, pooled AUC, balanced accuracy, benign recall at 0.5, overlap share and non-overlap AUC (overlap measured on the model's own input columns, per feature set). Small-support rule as before (under 1,000 deduplicated flows in the tested dataset or under 300 on the test side). `inf` → NaN; NaN is filled with the median of the capped training sample (thesis TRAP 1); the NaN count is written to the result.
6. **Cross-dataset directions:** (5a) train all deduplicated CICIoMT2024 rows (official train + test files) → test all CICIoT2023 rows; (5b) the reverse; the same "whole table" rule as arms 3–4.
7. **Within-dataset CSV baselines.** CICIoMT2024: official `train` files → official `test` files (after the global dedup, so a test row that duplicates a train row is removed from whichever side sorts later; same limit as arm 1). CICIoT2023: **about 30% of the merged files (round(0.3 × 63) = 19 members, drawn by seed) held out as test, the rest train. This is a random split of merged files, NOT capture-disjoint**: each merged file mixes rows from several captures, so train and test share captures and the baseline is optimistic.
8. **Label rule.** The flag stays `LABEL_RULE_INFERRED_SAME`. No stronger check is possible: the official rows carry no identifier.

### B. Background audit (`src/07_background_audit.py`, new, no model)
On the experiment's own basis: the step-03 NFStream flows, 59-column key, global dedup survivors, per family and dataset.
- **Floods (SYN, TCP, UDP, ICMP):** share of surviving rows whose `protocol` matches the attack (ICMP = 1; SYN and TCP = 6; UDP = 17). Counts per protocol value are also written.
- **Scans and ARP (port scan, OS scan, ping sweep, vuln scan, ARP):** for each capture file, the **dominant raw-flow `src_mac`** = the `src_mac` with the most flows in the file before dedup. Then the share of surviving rows of that file that involve that MAC (as `src_mac`, as `dst_mac`, and either). The script prints the MAC, and whether it also appears in a benign capture (as `src_mac`, as `dst_mac`). Per-family numbers pool the files. Control: the survivor counts per family equal `03_family_counts.json`.
- What it can show: how much of an attack-labelled family is traffic of the attack protocol or of the attacker host. What it cannot show: that a non-matching row is background (it may be a legitimate part of the attack, e.g. DNS lookups by the attacker).

### C. Two sensitivities (`src/08_sensitivities.py`), declared before running
a. **Arms 1–4, 3 seeds: protocol-matched attack test rows.** For the four flood families, attack **test** rows are limited to protocol-matching rows (same protocol rule as B). Benign test rows and the training data are unchanged. The models are re-fitted with the same seeds; the unrestricted metrics of the re-fit must reproduce `05_detection_arms.json` (a control, reported). Reported per arm: per-family AUC for the flood families (restricted, with the original beside it), the **flood macro** (mean of the 4 flood AUCs), and the common macro with restricted flood rows. Scan and ARP families are not restricted (no protocol rule for them).
b. **Arms 2 and 4, seed 42: flood-only training.** Training = the four flood families + benign only (the same capped draw as the arm, with scan and ARP rows removed after the draw, so the flood and benign rows are the same ones). The test set is unchanged. Reported: benign recall at 0.5 beside the current value, plus pooled AUC, balanced accuracy and both macros.
- What (a) would show: if the flood AUC changes little, the protocol-mismatched rows are not what drives it. If it moves a lot, those rows matter. What (b) would show: whether the low benign recall in arm 4 is tied to scan and ARP rows in training. It would not show why.

### D. Cost
Estimates, not measurements: arm 5 ≈ 15–25 min wall-clock (reading 45 M CSV rows, 24 + 6 fits), audit ≈ 3 min, sensitivities ≈ 10 min. Memory: the CICIoT2023 pair-family rows only (estimate before the run: about 16 M rows × 38 × float32 ≈ 2.5 GB; measured afterwards: 16,449,300 raw pair-family rows, 7,450,563 kept after dedup, and only the kept rows are cached) plus a 16-byte hash per row for all 45 M rows ≈ 0.7 GB; machine 24 GiB.

## Results after the declaration: arm 5, background audit, two sensitivities

Run 2026-10-11 (UTC dates in the result files), after the declaration above. All `post-hoc`. Scripts: `src/06_like_for_like.py`, `src/07_background_audit.py`, `src/08_sensitivities.py`; results `results/06_like_for_like.json`, `07_background_audit.json`, `08_sensitivities.json`.

### Arm 5 — official CSV features, same family subset

Basis: deduplicated CSV rows, the 10 `PAIRS` classes only, training capped at 20,000 per family / 40,000 benign, test uncapped, mean ± sd over seeds 42/43/44, thesis XGB params. **Headline = common macro** (SYN, TCP, UDP, ICMP, OS scan).

| feature set | task | test flows (benign), seed 42 | **common macro AUC** | per-arm macro | pooled AUC | balanced acc @0.5 | benign recall @0.5 | overlap share | non-overlap pooled AUC |
|---|---|---|---|---|---|---|---|---|---|
| shared 38 (primary) | CICIoMT2024 official train → test | 1,548,396 (37,607) | **0.999 ± 0.000** | 0.994 ± 0.001 | 1.000 ± 0.000 | 0.970 ± 0.003 | 0.940 ± 0.006 | 0.0085 (06b) | n/a |
| shared 38 (primary) | CICIoT2023 19 of 63 files held out (random, not capture-disjoint) | 2,118,847 (302,805) | **0.966 ± 0.000** | 0.938 ± 0.001 | 0.989 ± 0.000 | 0.864 ± 0.001 | 0.742 ± 0.002 | 0.034 / 0.033 / 0.032 (06b, seeds 42/43/44) | n/a |
| shared 38 (primary) | 5a CICIoMT2024 → CICIoT2023 | 7,450,563 (1,044,774) | **0.878 ± 0.018** | 0.761 ± 0.029 | 0.884 ± 0.021 | 0.530 ± 0.035 | 0.066 ± 0.077 | 0.0000 ± 0.0000 | 0.884 ± 0.021 |
| shared 38 (primary) | 5b CICIoT2023 → CICIoMT2024 | 8,440,968 (230,339) | **0.545 ± 0.036** | 0.563 ± 0.049 | 0.516 ± 0.030 | 0.502 ± 0.002 | 0.003 ± 0.004 | 0.0000 ± 0.0000 | 0.516 ± 0.030 |
| shared 31 (drop 7, secondary) | CICIoMT2024 official train → test | 1,548,396 (37,607) | **0.993 ± 0.000** | 0.951 ± 0.001 | 0.999 ± 0.000 | 0.936 ± 0.001 | 0.873 ± 0.001 | 0.891 (06b) | n/a |
| shared 31 (drop 7, secondary) | CICIoT2023 19 of 63 files held out (random, not capture-disjoint) | 2,118,847 (302,805) | **0.962 ± 0.000** | 0.919 ± 0.001 | 0.984 ± 0.000 | 0.836 ± 0.002 | 0.689 ± 0.005 | 0.774 / 0.779 / 0.784 (06b, seeds 42/43/44) | n/a |
| shared 31 (drop 7, secondary) | 5a CICIoMT2024 → CICIoT2023 | 7,450,563 (1,044,774) | **0.845 ± 0.015** | 0.721 ± 0.004 | 0.868 ± 0.015 | 0.609 ± 0.011 | 0.275 ± 0.014 | 0.0000 ± 0.0000 | 0.868 ± 0.015 |
| shared 31 (drop 7, secondary) | 5b CICIoT2023 → CICIoMT2024 | 8,440,968 (230,339) | **0.722 ± 0.050** | 0.723 ± 0.035 | 0.667 ± 0.048 | 0.528 ± 0.007 | 0.056 ± 0.014 | 0.0000 ± 0.0000 | 0.667 ± 0.048 |

The `base_iot` test set differs per seed (test rows 2,118,847 / 2,150,445 / 2,200,079 for seeds 42 / 43 / 44); the other three tasks have a fixed test set.

**Raw → deduplicated rows per family** (dedup key = the 38 shared columns, global within the dataset, first kept; the CICIoT2023 raw counts also equal the official label counts in `thesis/crossdataset/artifacts/label_counts.json`: **yes, all 10**).

| family | CICIoMT2024 raw → dedup | CICIoT2023 raw → dedup |
|---|---|---|
| SYN | 1,514,857 → 1,514,841 | 1,942,176 → 1,020,163 |
| TCP | 1,449,543 → 1,449,542 | 2,558,256 → 1,019,694 |
| UDP | 2,702,529 → 2,702,529 | 3,177,323 → 1,768,428 |
| ICMP | 2,401,899 → 2,401,242 | 6,893,259 → 1,809,166 |
| port_scan | 106,603 → 100,965 | 78,730 → 73,104 |
| os_scan | 20,666 → 19,653 | 93,970 → 89,201 |
| ping_sweep | 926 → 926 | 2,161 → 2,158 |
| vuln_scan | 3,207 → 3,140 | 357,583 → 355,846 |
| arp | 17,791 → 17,791 | 294,469 → 268,029 |
| Benign | 230,339 → 230,339 | 1,051,373 → 1,044,774 |

Controls: for CICIoMT2024 the hash dedup and `DataFrame.duplicated` mark the identical rows (7,392 duplicates, masks identical: True). Non-finite cells left after dedup: CICIoMT2024 0, CICIoT2023 446 (filled with the training-sample median). Rows in all classes after dedup, CICIoT2023: 20,203,779 of 45,019,243.

**Same direction, NFStream vs CSV features** (common macro, mean ± sd over 3 seeds; the test sets differ, so this is a comparison of settings, not of one model on one test set):

| direction | NFStream 59 features (arm 3/4) | CSV shared 38 (arm 5) | CSV shared 31 (arm 5, secondary) |
|---|---|---|---|
| CICIoMT2024 → CICIoT2023 | 0.826 ± 0.003 | 0.878 ± 0.018 | 0.845 ± 0.015 |
| CICIoT2023 → CICIoMT2024 | 0.919 ± 0.008 | 0.545 ± 0.036 | 0.722 ± 0.050 |

### Background audit (no model; NFStream flows, 59-column dedup basis, pair families)

Control: survivor counts per family equal `03_family_counts.json` in both datasets (passed: True).

**Floods — share of surviving rows whose `protocol` matches the attack** (ICMP = 1, SYN/TCP = 6, UDP = 17):

| family | CICIoMT2024 matching / survivors | CICIoT2023 matching / survivors |
|---|---|---|
| SYN (protocol 6) | 99.91% of 2,228,567 | 98.54% of 2,288,310 |
| TCP (protocol 6) | 99.94% of 2,007,522 | 98.15% of 2,232,777 |
| UDP (protocol 17) | 97.89% of 111,373 | 95.23% of 1,524,156 |
| ICMP (protocol 1) | 3.18% of 3,021 | 4.56% of 41,653 |

**Scans and ARP — per capture file: dominant raw-flow `src_mac` and share of the file's surviving rows that involve it** (either = as `src_mac` or `dst_mac`):

| dataset | family | capture file | dominant raw `src_mac` | its share of RAW flows as src | survivors | survivors involving it (either) | in a benign capture as src / dst |
|---|---|---|---|---|---|---|---|
| CICIoMT2024 | port_scan | Recon-Port_Scan_test.pcap.csv | `00:27:10:cb:17:60` | 95.9% | 102 | 27.5% | True / True |
| CICIoMT2024 | port_scan | Recon-Port_Scan_train.pcap.csv | `00:27:10:cb:17:60` | 42.9% | 551 | 7.4% | True / True |
| CICIoMT2024 | os_scan | Recon-OS_Scan_test.pcap.csv | `00:27:10:cb:17:60` | 86.5% | 438 | 47.0% | True / True |
| CICIoMT2024 | os_scan | Recon-OS_Scan_train.pcap.csv | `f4:b1:9c:c9:79:ff` | 26.3% | 777 | 15.1% | True / True |
| CICIoMT2024 | ping_sweep | Recon-Ping_Sweep_test.pcap.csv | `00:0c:29:b9:07:f6` | 45.5% | 43 | 44.2% | True / True |
| CICIoMT2024 | ping_sweep | Recon-Ping_Sweep_train.pcap.csv | `00:0c:29:b9:07:f6` | 43.5% | 64 | 46.9% | True / True |
| CICIoMT2024 | vuln_scan | Recon-VulScan_test.pcap.csv | `fc:67:1f:e1:9e:58` | 46.9% | 160 | 23.1% | True / True |
| CICIoMT2024 | vuln_scan | Recon-VulScan_train.pcap.csv | `fc:67:1f:e1:9e:58` | 16.1% | 121 | 30.6% | True / True |
| CICIoMT2024 | arp | ARP_Spoofing_test.pcap.csv | `8c:ae:4c:dd:1e:39` | 41.0% | 130 | 53.1% | True / True |
| CICIoMT2024 | arp | ARP_Spoofing_train.pcap.csv | `8c:ae:4c:dd:1e:39` | 39.4% | 747 | 52.2% | True / True |
| CICIoT2023 | port_scan | Recon-PortScan.pcap.csv | `dc:a6:32:dc:27:d5` | 63.2% | 17,678 | 7.2% | True / True |
| CICIoT2023 | os_scan | Recon-OSScan.pcap.csv | `dc:a6:32:dc:27:d5` | 60.9% | 23,803 | 11.3% | True / True |
| CICIoT2023 | ping_sweep | Recon-PingSweep.pcap.csv | `3c:18:a0:41:c3:a0` | 46.8% | 716 | 69.3% | True / True |
| CICIoT2023 | vuln_scan | VulnerabilityScan.pcap.csv | `ac:17:02:05:34:27` | 29.7% | 73,690 | 44.9% | True / True |
| CICIoT2023 | arp | MITM-ArpSpoofing.pcap.csv | `3c:18:a0:41:c3:a0` | 29.5% | 23,755 | 83.0% | True / True |
| CICIoT2023 | arp | MITM-ArpSpoofing1.pcap.csv | `9c:8e:cd:1d:ab:9f` | 22.3% | 3,045 | 3.2% | True / True |

### Sensitivity (a): arms 1–4, flood attack TEST rows limited to protocol-matching rows (3 seeds)

Control: the unrestricted re-fit reproduces `05_detection_arms.json` (max abs difference over pooled AUC, both macros, balanced accuracy, benign recall, all 12 fits: **0**).

| arm | flood macro (4 floods), all rows → matching rows | common macro, all → matching | ICMP AUC, all → matching | ICMP test rows, seed 42: all → matching |
|---|---|---|---|---|
| 1 within CICIoMT2024 | 0.993 ± 0.000 → 0.994 ± 0.004 | 0.992 ± 0.000 → 0.993 ± 0.003 | 0.974 ± 0.000 → 0.977 ± 0.015 (SMALL_SUPPORT on ICMP) | 683 → 24 |
| 2 within CICIoT2023 | 0.982 ± 0.001 → 0.992 ± 0.001 | 0.970 ± 0.001 → 0.978 ± 0.001 | 0.939 ± 0.002 → 0.973 ± 0.000 | 12,862 → 593 |
| 3 CICIoMT2024 → CICIoT2023 | 0.867 ± 0.005 → 0.867 ± 0.005 | 0.826 ± 0.003 → 0.826 ± 0.003 | 0.707 ± 0.003 → 0.691 ± 0.006 | 41,653 → 1,898 |
| 4 CICIoT2023 → CICIoMT2024 | 0.931 ± 0.008 → 0.988 ± 0.001 | 0.919 ± 0.008 → 0.965 ± 0.003 | 0.739 ± 0.029 → 0.962 ± 0.003 (SMALL_SUPPORT on ICMP) | 3,021 → 96 |

Per-family flood AUC (all rows → matching rows), mean over 3 seeds:

| arm | SYN | TCP | UDP | ICMP |
|---|---|---|---|---|
| 1 within CICIoMT2024 | 1.000 → 1.000 | 1.000 → 1.000 | 0.999 → 1.000 | 0.974 → 0.977 |
| 2 within CICIoT2023 | 0.996 → 0.998 | 0.999 → 0.999 | 0.993 → 0.997 | 0.939 → 0.973 |
| 3 CICIoMT2024 → CICIoT2023 | 0.973 → 0.977 | 0.989 → 0.993 | 0.796 → 0.807 | 0.707 → 0.691 |
| 4 CICIoT2023 → CICIoMT2024 | 0.999 → 1.000 | 0.999 → 0.999 | 0.990 → 0.994 | 0.739 → 0.962 |

### Sensitivity (b): arms 2 and 4, seed 42, training on the four flood families + benign only

| arm | training rows (flood-only / current) | benign recall @0.5, current → flood-only | balanced acc | pooled AUC | common macro | per-arm macro |
|---|---|---|---|---|---|---|
| 2 within CICIoT2023 | 120,000 / 172,285 | **0.7099 → 0.9511** | 0.8488 → 0.9608 | 0.9903 → 0.9887 | 0.9694 → 0.9658 | 0.9308 → 0.8979 |
| 4 CICIoT2023 → CICIoMT2024 | 120,000 / 198,394 | **0.2624 → 0.4602** | 0.6311 → 0.7297 | 0.9986 → 0.9981 | 0.9281 → 0.9030 | 0.9281 → 0.9030 |

### Reading (written by hand from the tables above; every figure is in a table above)
1. **The order of 5a and arm 3 depends on the ICMP family, and that family is a different row population in the two tables.** 5-family headline: 5a (CSV 38) 0.878 vs arm 3 0.826, so 5a is higher by 0.052 (5a sd 0.018). 4-family macro without ICMP: 5a 0.849 vs arm 3 0.855, a difference of 0.006, smaller than the 5a seed sd (0.022), so 5a is no longer higher. The other direction does not depend on ICMP: 5b (CSV 38) 0.545 (5-family) and 0.556 (4-family) vs arm 4 0.919 and 0.964; at 38 features 5b is at chance level (pooled AUC 0.516, benign recall 0.003). What the ICMP family holds: in the released CSVs 99.67–99.95% of the ICMP-flood rows have the ICMP indicator ≥ 0.5 (all rows), while in the NFStream tables only 2.76% (CICIoMT2024) and 3.86% (CICIoT2023) of the ICMP-family flows have protocol 1 before dedup (3.18% / 4.56% after). Any CSV-vs-NFStream comparison that includes ICMP therefore compares different row populations (whether the non-ICMP flows are background or part of the attack is not measured, item 6); the tables do not separate extractor, feature scale, flow definition and test set.
2. **Dropping the 7 features** moves 5b from 0.545 to 0.722 and 5a from 0.878 to 0.845. The drop set was chosen with CICIoMT2024 test labels (Addendum 4), so the 5b gain is not an independent result.
3. **Cross-dataset overlap is 0.0000 at four decimals in both directions:** exactly 0 at 38 features, 2.7e-7 and 2.4e-7 at 31 features (a handful of rows); no other match under the 128-bit hash key over float32-widened values. The tool passes a positive control (1,000 training rows used as test rows are all found, all four feature-set/dataset pairs; `results/06b_overlap_checks.json`). Within-dataset the `n/a (dedup)` entries were wrong and are replaced by measured shares: dedup was float64 over 38 columns, so rows equal in float32 (0.85–3.4% at 38 columns) or equal after dropping 7 columns (77–89% at 31 columns) remain across the train/test split. The shared-31 baselines are therefore close to memorisation checks, not generalisation tests.
4. **5a is unstable across seeds:** benign recall at the fixed 0.5 threshold is 0.066 ± 0.077 (seed values from 0.0006 to 0.15, see the log or the JSON). Treat the 0.5 operating point as unreliable here; the AUC columns are the usable part.
5. **Baselines.** Both within-dataset CSV baselines rank well (common macro 0.999 and 0.966). The CICIoT2023 one is optimistic by construction (random split of merged files, train and test share captures).
6. **ICMP flood family (audit).** On the NFStream basis only 3.18% (CICIoMT2024) and 4.56% (CICIoT2023) of the surviving ICMP-flood rows have protocol 1. SYN, TCP and UDP floods are 95.23–99.94% protocol-matching. The pre-dedup share is now measured (`results/11_icmp_shares.json`): 2.76% and 3.86%, so dedup does not cause it (dedup raises the share slightly). Not measured: whether the non-ICMP rows are background or part of the attack.
7. **Scans and ARP (audit).** The dominant raw `src_mac` carries only 16.1–95.9% of its capture's raw flows, and the survivors involving it range from 3.2% to 83.0%. In every one of the 16 captures the dominant MAC also appears in a benign capture, as source and as destination. So the share is not a background rate: it says how much of the surviving traffic touches one frequent host, nothing more.
8. **Sensitivity (a).** Arm 3 does not move (flood macro 0.867 → 0.867). Arm 4 rises from 0.9315 to 0.9885, almost entirely through ICMP (0.739 → 0.962), and its common macro from 0.919 to 0.965. The matching ICMP test rows are few: 96 of 3,021 at seed 42 (flagged SMALL_SUPPORT; arm 1: 24 of 683). Arm 2 rises from 0.982 to 0.992 (ICMP 0.939 → 0.973, 593 rows). So arm 4's flood and common macros rise when the non-ICMP rows of the ICMP family are removed; arm 3's ICMP AUC falls (0.707 → 0.691); the restricted ICMP AUC rests on very few rows.
9. **Sensitivity (b).** Without scan and ARP rows in training, benign recall rises in both arms (arm 2: 0.7099 → 0.9511; arm 4: 0.2624 → 0.4602) while the common macro falls (0.9694 → 0.9658; 0.9281 → 0.9030) and the port scan, OS scan, vuln scan and ARP AUCs fall (ping sweep is flat in arm 2 and falls in arm 4; `results/08_sensitivities.json -> b_flood_only_training_seed42`). Part of the low benign recall is therefore tied to those training rows; in arm 4 it stays below 0.5 without them, so the rest is not explained by this test. One seed only.
10. **Not shown anywhere here:** that the CICIoMT2024 and CICIoT2023 attack classes are the same attack; that the CSV and NFStream flows describe the same traffic; any statement about deployment.

### Added after review (not pre-declared): overlap re-measured on float32 values
`src/06b_overlap_checks.py`, `src/09_float32_overlap_arms1to4.py`. Cause: step 05 deduplicated float64 values, XGBoost reads float32, so rows that differ only below float32 precision count as different in the dedup and as the same input in the model. Key = 128-bit hash of float32 values widened to float64 (not a byte compare). Re-fit control: max abs difference to `05_detection_arms.json` = 0 (`results/09_float32_overlap_arms1to4.json`).

| arm | test rows with a train-side twin, float64 key (step 05) → float32 key | non-overlap common macro, float64 key → float32 key (seeds 42 / 43 / 44) |
|---|---|---|
| 3 CICIoMT2024 → CICIoT2023 | 0.30% → 0.96% | 0.827 / 0.826 / 0.821 → 0.825 / 0.824 / 0.819 |
| 4 CICIoT2023 → CICIoMT2024 | 0.43% → 1.27% | 0.922 / 0.905 / 0.908 → 0.919 / 0.901 / 0.905 |

Within-dataset, float32 key, test rows with a train-side twin: arm 1 (seed 42) 1.16%; arm 2 1.72% / 0.51% / 1.92% (seeds 42 / 43 / 44). Reading: the overlap is about three times larger than step 05 reported, still ≤ 1.3% cross-dataset, and removing it moves the common macro by 0.002–0.004. It does not change any arm 3/4 conclusion.

## ICMP decision: 4-family macro and ICMP indicator shares (added after the background audit)

Owner ruling: the pre-declared 5-family headline (SYN, TCP, UDP, ICMP, OS scan) stays. The 4-family macro (SYN, TCP, UDP, OS scan; ICMP excluded) is added beside it. No model was fitted: `src/10_macro4.py` reads `05_detection_arms.json` and `06_like_for_like.json`; `src/11_icmp_shares.py` reads the released CSVs and the NFStream tables. Mean ± sd (ddof=1) over seeds 42/43/44. No family was dropped for small support in any entry.

| task | 5-family headline | **4-family macro** (added) |
|---|---|---|
| arm 1 within CICIoMT2024 (NFStream) | 0.992 ± 0.000 | **0.997 ± 0.000** |
| arm 2 within CICIoT2023 (NFStream) | 0.970 ± 0.001 | **0.978 ± 0.001** |
| arm 3 CICIoMT2024 → CICIoT2023 (NFStream) | 0.826 ± 0.003 | **0.855 ± 0.004** |
| arm 4 CICIoT2023 → CICIoMT2024 (NFStream) | 0.919 ± 0.008 | **0.964 ± 0.003** |
| CSV 38: CICIoMT2024 official train → test | 0.999 ± 0.000 | **0.999 ± 0.000** |
| CSV 38: CICIoT2023 held-out files (random split) | 0.966 ± 0.000 | **0.958 ± 0.000** |
| CSV 38: 5a CICIoMT2024 → CICIoT2023 | 0.878 ± 0.018 | **0.849 ± 0.022** |
| CSV 38: 5b CICIoT2023 → CICIoMT2024 | 0.545 ± 0.036 | **0.556 ± 0.040** |
| CSV 31: CICIoMT2024 official train → test | 0.993 ± 0.000 | **0.991 ± 0.000** |
| CSV 31: CICIoT2023 held-out files (random split) | 0.962 ± 0.000 | **0.952 ± 0.000** |
| CSV 31: 5a CICIoMT2024 → CICIoT2023 | 0.845 ± 0.015 | **0.807 ± 0.019** |
| CSV 31: 5b CICIoT2023 → CICIoMT2024 | 0.722 ± 0.050 | **0.723 ± 0.038** |

**Per-family AUC vs all benign, mean over 3 seeds** (SYN / TCP / UDP / OS scan):

| direction | NFStream (arm 3 / arm 4) | CSV 38 (5a / 5b) | CSV 31 (5a / 5b) |
|---|---|---|---|
| CICIoMT2024 → CICIoT2023 | 0.973 / 0.989 / 0.796 / 0.662 | 0.981 / 0.950 / 0.810 / 0.655 | 0.943 / 0.908 / 0.816 / 0.563 |
| CICIoT2023 → CICIoMT2024 | 0.999 / 0.999 / 0.990 / 0.870 | 0.533 / 0.518 / 0.508 / 0.664 | 0.959 / 0.600 / 0.487 / 0.846 |

**(a) Released CSVs, all rows (raw, not deduplicated): share of rows with the `ICMP` indicator ≥ 0.5**

| dataset | class | rows | share ICMP ≥ 0.5 |
|---|---|---|---|
| CICIoMT2024 | Benign | 230,339 | 0.0017% |
| CICIoMT2024 | DDoS_ICMP | 1,887,175 | 99.9534% |
| CICIoMT2024 | DoS_ICMP | 514,724 | 99.6699% |
| CICIoT2023 | DDoS-ICMP_Flood | 6,893,259 | 99.8851% |
| CICIoT2023 | Benign | 1,051,373 | 0.0026% |

**(b) NFStream flow tables, BEFORE dedup: share of flows with `protocol` = 1** (after-dedup value from `07_background_audit.json` beside it):

| dataset | family | flows before dedup | protocol 1 before dedup | protocol 1 after dedup |
|---|---|---|---|---|
| CICIoMT2024 | ICMP | 3,519 | 2.76% | 3.18% of 3,021 |
| CICIoMT2024 | Benign | 18,929 | 0.26% | not computed |
| CICIoT2023 | ICMP | 68,613 | 3.86% | 4.56% of 41,653 |
| CICIoT2023 | Benign | 301,193 | 1.37% | not computed |

### Fact-check of the Reading section and the float32 table (2026-10-11, against the result JSONs)
Every figure in Reading items 1–10 and in the float32 table was recomputed from `results/05–09*.json`. Found and fixed: (1) item 3 said overlap equals zero "for both feature sets"; it is exactly 0 only at 38 features (2.7e-7 / 2.4e-7 at 31); (2) item 6 listed the pre-dedup share as unmeasured; step 11 measured it; (3) item 8 rounded 0.9315 → 0.9885 to 0.931 → 0.988 and attributed the arm 4 rise to the ICMP rows with a cause-and-effect verb; (4) item 1 is rewritten per the ICMP decision. The hand-typed float32 table matched the JSONs in all cells. The text above that Reading section was not re-checked in this pass.

### Fact-check of the text above the Reading section (2026-10-11, against the result JSONs, the scripts and the disk)
Checked: the family table (all 10 rows, counts and file numbers), flood shares (99.93%, 97.7%) and their sums, step 04 and 04b (table, permutation null, seed-mechanism counts, re-run), the arms 1–4 table, per-arm macro families, small-support flags, uncapped sensitivity, embargo count, XGB params, caps, pre-check table (official rows, packets per row, protocol shares), zip sizes, memory figure, arm 5/audit/sensitivity declarations against `06`–`08`, code comments for the splits. Fixed in place: status line (said arm 5 was still to run); "NOT YET RUN" and "PLAN ONLY" headings (both stale); layout (wrong "declared above", three scripts missing); provenance bullet (nfstream version and input list are not in every file); "no vector on both sides … by construction" and the `n/a` overlap cells (true for float64 only); "4.36M attack rows" (4,353,616 = 4.35M); fourth zip "CSVs only" (two PDFs too); open item "arm 5 waits" (answered). Not re-checkable here: budget estimates (20 min, 1–1.5 h, 68 GiB free at the time; 60 GiB free now), the owner's "10 of 33 classes have no pcap", the thesis sentence about `inverse_full_run.py` training on all rows (not read in this pass).

### Reviewer edits (2026-10-11, from the review session; wording only, no new script output)
Line 65: the "10 of 33" figure is now marked as checked, with its basis. Limits table: "10 of 33 classes are used" corrected to 9 of 33 attack classes plus benign. Reading item 1: "different traffic" replaced by "different row population", because per packet the NFStream ICMP family is also ICMP-dominated and the composition of the non-ICMP flows is not measured (item 6).
