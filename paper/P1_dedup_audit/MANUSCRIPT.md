# How much of CICIoMT2024 is a copy? A per-split, precision-stated audit of duplicate leakage in the reference IoMT intrusion-detection benchmark

> **DRAFT — sections 3–7 only** · target venue: *Internet of Things* (Elsevier) · drafted 2026-07-30
> Governing spec: `.claude/plans/2026-07-30-p1-dedup-paper-brief.md` (FROZEN) + `OUTLINE.md`.
> Sections 1–2 and 8–11 are stubs. Every number carries an oracle reference in an HTML comment; those
> comments are stripped at submission and must survive `/fact-check` before that happens.

## 1. Introduction

*[TO DRAFT — see OUTLINE.md §1. Needs the corpus freshness re-sweep before absence claims are final.]*

## 2. Related work

*[TO DRAFT — see OUTLINE.md §2. Needs the external leakage-precedent citation pull (§2.3).]*

---

## 3. Data and method

### 3.1 The released artifact

CICIoMT2024's Wi-Fi and MQTT subset ships as **72 CSV files** — 51 covering the training split and 21 the
test split — with the split expressed at the file level by directory. Each row is a flow-level feature vector
of **45 numeric attributes**; the class label is carried by the filename rather than by a column, and
volumetric attack families are distributed across numbered files (`TCP_IP-DDoS-ICMP1` … `TCP_IP-DDoS-ICMP8`)
that together constitute one class.
<!-- oracle: numbers_map.md §2 rows 31-33 (row counts), row 41 (45 features), row 44 (19 classes);
     file counts from the data/{train,test} directory listings -->

| Property | Train | Test | Total |
|---|---|---|---|
| Files | 51 | 21 | 72 |
| Rows as released | 7,160,831 | 1,614,182 | 8,775,013 |
| Features | 45 | 45 | 45 |
| Classes | 19 | 19 | 19 |

The 19 classes comprise 18 attack types plus `Benign`, and the distribution is severely imbalanced: after the
deduplication described below, the ratio between the largest and smallest class is **2,374:1**.
<!-- oracle: numbers_map.md §2 row 45 -->

Throughout this paper we quantify duplication over the released artifact as distributed. We deliberately do
**not** merge the two directories before measuring, because the file-level split is the dataset's own
evaluation protocol and merging it destroys the very boundary whose integrity is in question. Pooled-merge
figures are reported separately and labelled as such.

### 3.2 What counts as a duplicate

A duplicate is an exact match of the **45-column feature vector**. Two representations are measured:

- **REP45** — the 45 released feature columns only.
- **REP46** — REP45 plus the class label recovered from the filename, i.e. the post-merge table that every
  study which concatenates the CSVs actually works with.

The distinction matters for a specific reason. If duplicate vectors were shared *between* classes,
deduplication would face a label-ambiguity problem: which label survives? Reporting both representations
settles that question empirically rather than by assumption.
<!-- oracle: dr6_reconcile_5119.py:9-11 (REP45/REP46 definitions) -->

### 3.3 Precision as an experimental variable

The released CSVs print values at float64 precision. Every mainstream implementation applied to this dataset —
gradient-boosted decision trees, and neural frameworks by default — computes in **float32**. Two rows that
differ only beyond the seventh significant digit are printed as distinct in the CSV and are *literally the
same training example* to the model that consumes them.

Duplication is therefore not a single quantity but a function of the precision at which it is evaluated, and
a duplicate count reported without its precision is unfalsifiable. We measure at both:

- **float64-as-printed** — bit-identical to the 16th digit; what a naive `drop_duplicates` on the CSV as read
  reports.
- **float32-as-computed** — the operational precision, obtained by casting the 45 feature columns to
  `float32` before comparison.

### 3.4 Scope as an experimental variable

The second unstated parameter is scope: *which rows were counted*. We evaluate per-file, per-split (train,
test), pooled-merge (train + test as one table), and the published subsets that individual studies report
having used. As §5 shows, scope alone accounts for a discrepancy the literature has left unexplained.

### 3.5 Procedure

Each CSV is read with the 45 feature columns typed as `float32`, or left at `float64`, according to the
representation under test. Rows are hashed with `pandas.util.hash_pandas_object(df, index=False)` — a 64-bit
row hash over the ordered column tuple — and the duplicate count for a scope is `len(h) − len(unique(h))`,
the number of rows in excess of the distinct vectors present. Cross-split identity is measured as the
intersection of the two splits' distinct-vector sets.
<!-- oracle: dr6_float32_check.py:35-46 -->

Two honest caveats about the procedure:

**Hash collisions.** Over 8,775,013 rows a 64-bit row hash has an expected false-collision count of
2.1 × 10⁻⁶ colliding pairs under the birthday bound — negligible relative to the counts reported here, but
not identically zero. The
float64 counts were additionally confirmed against the row arithmetic of the released directories (§5), which
does not depend on hashing at all.

**±inf handling.** The audit scripts hash the CSVs as read. The reference pipeline against which we
cross-check (§3.6) replaces ±inf with NaN before its own deduplication step. The two procedures agree on the
headline rates to four decimal places, which is the cross-validation; where a number originates from one
convention rather than the other, we say so.
<!-- oracle: docs/phase2_eda.md:77-83 (pipeline's inf handling) vs dr6_float32_check.py (as-read) -->

### 3.6 Independent cross-checks

Every headline count in §4 was produced by at least two independent code paths: the audit scripts released
with this paper, and a separate end-to-end preprocessing pipeline written earlier without reference to them.
The float32 train count agrees exactly — 2,645,751 rows removed, recoverable as 7,160,831 − 4,515,080 — and a
third implementation, built for the ablation in §8, reproduced both split counts to the row.
<!-- oracle: numbers_map.md §2 rows 34-37; eda_output_raw/build_report.json (third path) -->

### 3.7 Reproduction package

All scripts, all result JSONs, and a verifier that re-asserts every number in this paper from those JSONs are
released as `ciciomt2024-dedup-audit` *[URL on acceptance]*. The package contains **no CICIoMT2024 data**; the
dataset must be obtained from the Canadian Institute for Cybersecurity directly. Environment: Python 3.13.13,
pandas 2.3, NumPy 2.2, scikit-learn 1.8, XGBoost 3.2.0.
<!-- oracle: CLAUDE.md stack line. NOTE: requirements.txt pins xgboost<3.0 while 3.2.0 is installed —
     known drift, to be stated honestly wherever pipeline versions are cited (brief line 23). -->

---

## 4. Result 1 — duplication is precision-dependent, by a factor of 500

Measured over the released artifact at the two precisions of §3.3:

| Precision | Meaning | Train duplicates | Test duplicates | Pooled merge |
|---|---|---|---|---|
| float64 (as printed) | identical to the 16th digit | **5,119** (0.0715%) | **2,065** (0.1279%) | 7,379 (0.0841%) |
| **float32 (as computed)** | identical at ~7 significant digits | **2,645,751 (36.9475%)** | **721,914 (44.7232%)** | 3,368,126 (38.3831%) |
<!-- oracle: numbers_map.md §2 duplicate rows; all six cells from dr6_out/dr6_float32_check.json -->

The same rows, the same comparison, the same dataset: **0.07% or 36.95%**, depending entirely on a parameter
no study in this literature states. The float32 figure is the operationally relevant one, because it describes
the data as the model actually receives it.

Three properties of this result deserve emphasis.

**It is a lower bound, not an estimate.** Exact matching cannot see near-duplicates, and CICIoMT2024's feature
construction manufactures them: attributes such as `Rate`, `Srate`, `IAT`, `AVG`, `Std` and `Variance` are
window-averaged statistics, so consecutive windows over a sustained volumetric flood differ in their
low-order digits while describing the same behaviour. Any exact-match rate at any precision is a floor on the
true redundancy.

**The collapse is not gradual.** Between the two precisions the count rises by a factor of **517**
(2,645,751 / 5,119) — consistent with a duplicate mass whose rows agree to roughly seven significant digits
and diverge thereafter, which is exactly the signature of window-averaged features over near-stationary
traffic.

**Deduplication removes more than a third of the benchmark.** Dropping float32 duplicates takes the training
split from 7,160,831 to **4,515,080** rows and the test split from 1,614,182 to **892,268**. Two thirds of the
nominal size of the field's reference dataset survives; the remainder is copies.
<!-- oracle: numbers_map.md §2 rows 34-35 -->

---

## 5. Result 2 — the literature's "5,119 duplicates" is correct, and not comparable

Three published studies using CICIoMT2024 report removing exactly **5,119** duplicate rows. Set against §4's
float32 measurement this looks like a 500-fold contradiction in the literature. It is not a contradiction, and
the resolution is this paper's thesis in miniature.

**The reconciliation.** At float64 the released training directory contains 7,155,712 distinct feature
vectors. Adding the 5,119 duplicate rows recovers **7,155,712 + 5,119 = 7,160,831** — the exact row count of
the training directory. The published figure is the float64-exact duplicate count of the *training split*,
reproduced here to the row.
<!-- oracle: dr6_panel.json REP45/train; numbers_map.md §2 float64 rows -->

**Scope, at two scales.** Two of the three studies describe operating on the training directory, and for that
scope 5,119 is right. The third reports a subset of 4,971,919 rows, which matches no scope of the released
data we could construct; every DDoS-flavoured train-only subset we tested yields **672** float64 duplicates,
never 5,119, so that study's figure is inherited from a shared pipeline rather than measured on its own
subset. Those same six subsets evaluated over train **and** test yield **674** — two additional duplicates
from widening the scope alone. Scope-sensitivity is visible even at the scale of single-digit counts.
<!-- oracle: dr6_float32_check.json akkal_train_only/* (672, train-only) vs
     dr6_panel.json REP45/akkal_* (674, train+test) — scope difference verified in both scripts -->

**No study states the precision at which it counted, and none reports a per-split rate.** Of the 31
full-text-verified studies in our corpus, nine touch deduplication in some form, and zero
report train and test rates separately or analyse the consequence for their reported metrics.
<!-- NOTE: an earlier draft also claimed 'zero state a precision'. That specific absence is NOT covered by
     the verified corpus notes and must be checked paper-by-paper before it may be asserted. -->
<!-- oracle: Research_Gap_Report_v1.0.md:15 (9 of the 31); corpus snapshot 2026-07-29 —
     RE-SWEEP REQUIRED BEFORE SUBMISSION -->

**Nobody miscounted.** The counts differ by precision and by scope, and the precision-dependence itself —
5,119 becoming 2,645,751 on identical rows — is the finding.

**A byproduct: deduplication introduces no label ambiguity.** REP46 (features + label) returns the *same*
5,119 duplicates as REP45 (features only) at float64, and at float32 the sum of within-class duplicate counts
across all 19 classes equals the pooled per-split count exactly (2,645,751 train; 721,914 test). No duplicate
vector is shared between two classes at either precision. Deduplication on this dataset is therefore
unambiguous — a practical point for anyone implementing it, and not an obvious one a priori.
<!-- oracle: dr6_panel.json REP45/train == REP46/train == 5119; within-class sums over
     dr6b_perclass_f32.json == pooled f32 counts (numbers_map.md intra-class row) -->

---

## 6. Result 3 — the redundancy is structured, and not where volume would predict

Duplication is not spread across CICIoMT2024. Measured within each class at float32:

| Released class | Train rows | Train dup | Train rate | Test rows | Test dup | Test rate | Share of train dup mass |
|---|---|---|---|---|---|---|---|
| TCP_IP-DDoS-ICMP | 1,537,476 | 1,327,218 | 86.32% | 349,699 | 330,026 | 94.37% | 50.16% |
| TCP_IP-DDoS-TCP | 804,465 | 556,198 | 69.14% | 182,598 | 173,863 | 95.22% | 21.02% |
| TCP_IP-DoS-ICMP | 416,292 | 270,979 | 65.09% | 98,432 | 89,981 | 91.41% | 10.24% |
| TCP_IP-DDoS-SYN | 801,962 | 224,313 | 27.97% | 172,397 | 83,476 | 48.42% | 8.48% |
| TCP_IP-DoS-TCP | 380,384 | 159,203 | 41.85% | 82,096 | 39,513 | 48.13% | 6.02% |
| TCP_IP-DoS-SYN | 441,903 | 94,868 | 21.47% | 98,595 | 1,053 | 1.07% | 3.59% |
| Recon-Port_Scan | 83,981 | 10,096 | 12.02% | 22,622 | 3,031 | 13.40% | 0.38% |
| Recon-OS_Scan | 16,832 | 2,618 | 15.55% | 3,834 | 893 | 23.29% | 0.10% |
| Recon-VulScan | 2,173 | 141 | 6.49% | 1,034 | 61 | 5.90% | 0.01% |
| Recon-Ping_Sweep | 740 | 51 | 6.89% | 186 | 17 | 9.14% | 0.00% |
| ARP_Spoofing | 16,047 | 37 | 0.23% | 1,744 | 0 | 0.00% | 0.00% |
| TCP_IP-DoS-UDP | 566,950 | 29 | 0.01% | 137,553 | 0 | 0.00% | 0.00% |
| Benign | 192,732 | 0 | 0.00% | 37,607 | 0 | 0.00% | 0.00% |
| MQTT-DDoS-Connect_Flood | 173,036 | 0 | 0.00% | 41,916 | 0 | 0.00% | 0.00% |
| MQTT-DDoS-Publish_Flood | 27,623 | 0 | 0.00% | 8,416 | 0 | 0.00% | 0.00% |
| MQTT-DoS-Connect_Flood | 12,773 | 0 | 0.00% | 3,131 | 0 | 0.00% | 0.00% |
| MQTT-DoS-Publish_Flood | 44,376 | 0 | 0.00% | 8,505 | 0 | 0.00% | 0.00% |
| MQTT-Malformed_Data | 5,130 | 0 | 0.00% | 1,747 | 0 | 0.00% | 0.00% |
| TCP_IP-DDoS-UDP | 1,635,956 | 0 | 0.00% | 362,070 | 0 | 0.00% | 0.00% |
| **Total** | **7,160,831** | **2,645,751** | **36.95%** | **1,614,182** | **721,914** | **44.72%** | **100%** |
<!-- oracle: dr6_out/dr6b_perclass_f32.json — table generated directly from the artifact;
     headline cells also numbers_map.md §2 rows 38-40 -->

Four observations, in descending order of consequence.

**Six classes carry 99.5% of it.** The `TCP_IP-*` flood families account for **2,632,808 of 2,645,751**
duplicate training rows — **99.51%**. `TCP_IP-DDoS-ICMP` alone carries **50.16%**, at a within-class rate of
86.32% in train and 94.37% in test. On this dataset, "the duplicate problem" is the volumetric TCP/IP flood
problem.

**Redundancy is protocol-structural, not volume-driven.** The largest class in the training split,
`TCP_IP-DDoS-UDP` with **1,635,956 rows, contains zero duplicates at float32**, while `TCP_IP-DDoS-ICMP` at a
comparable 1,537,476 rows is 86.32% duplicate. Volume does not predict redundancy; the protocol and its
feature signature do. Any account of this dataset's duplication that reasons from class size is wrong.

**Zero is real for a third of the label space.** `Benign` and all five MQTT classes contain no float32
duplicates whatsoever, in either split. Deduplication does not touch the benign class — which matters, because
the benign class governs false-alarm behaviour at deployment prevalence.

**Rate and mass tell different stories for the `Recon` family.** The four `Recon` sub-types show non-trivial
within-class rates (6.49%–15.55% in train) but contribute only **0.49%** of the duplicate mass. A summary
claiming duplication is negligible outside the floods is true by mass and false by rate; both belong in the
record. One consequence is easy to miss: the rarest class of the deduplicated dataset, `Recon-Ping_Sweep`,
arrives at its 689 unique training rows only after 51 of its 740 released rows are removed as copies.
Class-rarity claims on this dataset are themselves deduplication-dependent.
<!-- oracle: dr6b_perclass_f32.json; 740-51=689 cross-checks numbers_map.md §2 row 46 (rarest class) -->

---

## 7. Result 4 — three leakage mechanisms, and the field looks for the smallest one

"Leakage" in this literature is discussed, where it is discussed at all, as train-to-test contamination. That
framing does not describe what is present here. Three mechanisms must be separated; the first two differ in
magnitude by more than an order of magnitude, and the third is not a row-level phenomenon at all.

**M1 — within-split redundancy (large).** 36.95% of training rows and 44.72% of test rows are copies of
another row *in the same split*. On the test side this collapses the effective sample size: a metric computed
over 1,614,182 test rows is in fact computed over 892,268 distinct vectors, with the surplus concentrated in
the six flood classes of §6, which are correspondingly over-weighted in every prevalence-weighted statistic.
No cross-split check detects M1, because nothing crosses the split.

**M2 — cross-split identity (small).** At float32, **461** distinct feature vectors occur in both the training
and the test split, carried by approximately **13,533 test rows — 0.84%** of the test split. At float64 the
same figures are **195** vectors and approximately **357** rows (0.02%).
<!-- oracle: numbers_map.md §2 cross-split rows; 461 = f32/full − (f32/train + f32/test);
     row counts derived from stored 6-dp fractions, hence ±1 -->

The asymmetry is the point. The mechanism the field would look for, if it looked, is **M2** — and M2 is
genuinely small. The mechanism that dominates is **M1**, which is invisible to the check M2 would motivate. A
study that verified "no test row appears in training" would pass on this dataset and still report metrics over
a test set that is 44.72% copies.

**M3 — statistic contamination.** A third mechanism appeared while we were controlling for the first two, and
we have not seen it named in this literature. Preprocessing statistics are fitted on the training split, so
the duplicate mass distorts them. Refitting the identical preprocessing pipeline on the deduplicated training
data rather than the raw training data moves **37 of the 104 fitted parameter values** across its three scaler
groups: RobustScaler centres differ for 7 of 21 features and scales for 16 of 21, with a maximum relative difference of **5.67×**;
StandardScaler means and scales differ for all 7 of its features; the MinMax group, whose features are binary
indicators, is unchanged in all 16.
<!-- oracle: results/c1_dedup_ablation/c1_matrix.json:scaler_shift_raw_vs_dedup -->

Deduplication is therefore not a row-count operation — it changes the feature space in which the model is
trained. That has a methodological consequence beyond this dataset: a deduplicated and a non-deduplicated arm
of the same pipeline are **not** mutually scoreable. A model from one arm evaluated against the other arm's
scaled test matrix receives systematically mis-scaled inputs. We made exactly this error while building the
ablation of §8, and it produced a plausible-looking result — an apparent collapse to 63% accuracy — that was
an artifact of the scaling mismatch and nothing else. We report it because the failure mode is easy to reach
and hard to notice: it produces numbers that are wrong in the direction that flatters the hypothesis.

**Four leakage axes, two corrected corpus-wide.** Beyond M1–M3, the corpus exhibits two further axes that this
work does not correct: destruction of the official file-level split by merge-and-re-split, and resampling
applied before splitting. Across the four axes, the corpus corrects duplicate rows exactly (this work) and
device-level overlap approximately (one study, using predicted device labels); the split-boundary and
resample-order axes remain open.
<!-- oracle: Research_Gap_Report_v1.0.md §3 four-axes block, G11 -->

---

## 8. Result 5 — measured consequences

*[TO DRAFT — the C1 ablation is complete and oracle-anchored (numbers_map.md §2, C1 block); OUTLINE.md §8.1
carries the agreed framing: test-side inflation is separable, training-side is not, and the naive
raw-vs-dedup pairing sits inside seed variance. §§8.2–8.5 (SMOTETomek, entropy criterion, Dadkhah chasm)
unstarted.]*

## 9. A reporting protocol for CICIoMT2024

*[TO DRAFT — OUTLINE.md §9: five mandatory parameters + the T7 checklist.]*

## 10. Threats to validity

*[TO DRAFT — OUTLINE.md §10. Must carry: exact-match lower bound, Wi-Fi+MQTT scope (no BLE), the
hash-collision bound, single-dataset scope, corpus snapshot date.]*

## 11. Conclusion

*[TO DRAFT]*
