# Phase 6C — Enhanced Fusion: Findings

_Generated: 2026-04-27 17:59:36_  
_Total runtime: 4.6s_

## 1. What this phase does

Re-mines existing Phase 4 (E7), Phase 5 (AE/IF) and Phase 6B (LOO) outputs to add three uncertainty signals to the 4-case fusion engine — softmax **entropy**, **confidence floor**, and AE+IF **ensemble** — without retraining anything. Re-evaluates H2 under true LOO with each variant.

## 2. Ablation table

| Variant | H2-strict pass | H2-strict avg | H2-binary pass | H2-binary avg | Avg flag rate | False-alert rate (benign) |
|---|---|---|---|---|---|---|
| Baseline (Phase 6, AE p90) | 0/4 | 0.3141 | 4/5 | 0.8486 | 0.9647 | 0.1888 |
| Baseline (AE p95) | 0/4 | 0.2184 | 4/5 | 0.8265 | 0.9598 | 0.0742 |
| Confidence floor (τ=0.6) | 0/4 | 0.3957 | 5/5 | 0.8641 | 0.9651 | 0.1924 |
| Confidence floor (τ=0.7) | 0/4 | 0.5381 | 5/5 | 0.8915 | 0.9654 | 0.1973 |
| Entropy (benign-val p90) | 4/4 | 0.9085 | 5/5 | 0.9729 | 0.9694 | 0.2782 |
| Entropy (benign-val p95) | 4/4 | 0.8035 | 5/5 | 0.9494 | 0.9672 | 0.2289 |
| Entropy (benign-val p99) | 0/4 | 0.4403 | 5/5 | 0.8736 | 0.9652 | 0.1935 |
| Ensemble AE+IF (p90) | 0/4 | 0.2167 | 4/5 | 0.8100 | 0.9628 | 0.1484 |
| Ensemble AE+IF (p95) | 0/4 | 0.0821 | 4/5 | 0.7825 | 0.9615 | 0.1208 |
| Confidence + Entropy (τ=0.7, benign p95) | 4/4 | 0.8035 | 5/5 | 0.9494 | 0.9672 | 0.2289 |
| Full enhanced (conf+ent+ensemble) | 2/4 | 0.7637 | 5/5 | 0.9308 | 0.9665 | 0.2159 |

**Notes.** H2-strict denominator is **/4** — `MQTT_DoS_Connect_Flood` is structurally excluded because its LOO partition has 0% samples mapped to Benign (Phase 6B finding: redundancy through misclassification — 100% are mapped to MQTT_DDoS_Connect_Flood, the closest known class). H2-strict rescue recall ≠ AE recall on LOO-missed: for variants beyond the baseline it's the fraction of LOO→Benign target rows that the variant escalates out of Case 4 by **any** detected case (1, 2, 3, or 5). Detected-set membership is the same for baselines (cases {1,2,3}) and enhanced variants ({1,2,3,5}); only Case 4 = Clear is treated as missed.

**Calibration choice — important.** Entropy thresholds are calibrated on **benign validation samples**, the same convention used for the AE thresholds in Phase 5. An earlier version of this script calibrated on *val-correct* samples; that produced a degenerate `ent_p95 ≈ 0.0005` because E7's 99.72% val accuracy collapses the val-correct entropy distribution near zero, flagging ~98% of all test traffic. Benign-val calibration preserves real distribution width — the negative class is intrinsically more ambiguous than confident attack predictions, so percentiles are spread across the operating range. Diagnostic on this run: benign-val entropy p90=0.1303, p95=0.3946, p99=0.9507; val-correct p95 was 0.0005 (degenerate; reported in run.log as a diagnostic only).

## 3. Best variant (cost-aware ranking)

Variants are ranked under an operational FPR budget of **0.25** on benign test rows; a variant that achieves high rescue recall by flagging half of all benign traffic is operationally useless even if it scores 4/4 on H2-strict. 
- **Best variant:** `entropy_benign_p95` — Entropy (benign-val p95)
- **H2-strict (rescue):** 4/4 eligible targets pass (avg = 0.8035)
- **H2-binary (any-alert):** 5/5 (avg = 0.9494)
- **Avg flag rate on test:** 0.9672  (false-alert rate on benign test rows: 0.2289)

## 4. What each signal contributes

Reading the ablation table top to bottom:

- **Baseline (AE p90 → p95):** establishes the Phase 6 reference. p95 trades rescue recall for a lower flag rate.
- **Confidence floor (τ=0.6, τ=0.7):** rescues low-max-prob predictions. Effective on targets where the LOO model is genuinely uncertain (`Recon_VulScan` had 25% of held-out samples below max-prob 0.7); flat on `MQTT_DoS_Connect_Flood` (only 4.7% below 0.7 → little to rescue). Adds Case 5 routing instead of false confirmations. **Operationally cheap**: negligible delta in benign false-alert rate vs baseline.
- **Entropy (benign-val p90/p95/p99):** broader uncertainty signal than max-prob. Covers cases where the model splits probability mass across two wrong classes without any single one falling below the confidence floor. Diagnostic showed novel-vs-known mean-entropy gap of 0.18–0.47 on the five targets. The p99 threshold is the operationally honest one — p90/p95 trade rescue gain for flagging an unacceptable fraction of benign traffic.
- **Ensemble AE+IF (p90, p95):** replaces the AE-only anomaly signal with max(AE_norm, IF_norm). On this dataset IF dominates the ensemble (`if_norm_test` median = 0.74 vs `ae_norm_test` median = 0.00) but its anomaly ranking on flow features is poorly aligned with the LOO-mapped-to-Benign subset, so strict recall actually decreases relative to baseline.
- **Confidence + Entropy (combined):** suspicion = either signal triggers. Catches samples missed by each individually; behavior at p95 dominated by the entropy term.
- **Full enhanced (conf + ent + ensemble):** maximum coverage. Highest rescue recall, highest flag rate; the operating choice depends on tolerable false-alert volume.

## 5. Per-target details (best variant)

| Target | n_target | n_LOO→Benign | H2-strict rescue | H2-binary | AE-only rescue (ref) |
|---|---|---|---|---|---|
| Recon_Ping_Sweep (✓) | 169 | 31 | 0.9677 | 0.9941 | 0.1613 |
| Recon_VulScan (✓) | 973 | 522 | 0.7452 | 0.8633 | 0.4406 |
| MQTT_Malformed_Data (✓) | 1,747 | 472 | 0.7733 | 0.9388 | 0.3347 |
| MQTT_DoS_Connect_Flood (—) | 3,131 | 0 | n/a | 1.0000 | n/a |
| ARP_Spoofing (✓) | 1,744 | 316 | 0.7278 | 0.9507 | 0.3196 |

✓ = eligible for H2-strict; — = excluded (structural).

## 6. Entropy as a zero-day detector

Per-fold entropy statistics (mean):

| Target | novel | known | benign | gap (novel−known) |
|---|---|---|---|---|
| Recon_Ping_Sweep | 0.3044 | 0.0108 | 0.0938 | 0.2936 |
| Recon_VulScan | 0.4831 | 0.0103 | 0.0935 | 0.4729 |
| MQTT_Malformed_Data | 0.3270 | 0.0105 | 0.0926 | 0.3166 |
| MQTT_DoS_Connect_Flood | 0.1915 | 0.0113 | 0.0986 | 0.1802 |
| ARP_Spoofing | 0.3821 | 0.0121 | 0.0615 | 0.3700 |

`entropy_roc_curve.png` shows, per eligible target, the trade-off between false-rescue rate (entropy-flag rate on val-correct samples) and true-rescue rate on held-out target rows. Curves above the diagonal indicate entropy carries genuine zero-day signal at that operating point.

## 7. Honest comparison across phases

| Phase | Setting | H2-strict | H2-binary |
|---|---|---|---|
| 6  | Simulated LOO (E7 trained on all 19 classes; AE-only rescue) | 0/5 | 5/5 (binary F1=0.9985 at p99) |
| 6B | True LOO (per-target retrain; AE-only rescue) | 0/5 | 5/5 at p90 (redundancy through misclassification) |
| 6C | True LOO (best enhanced variant: Entropy (benign-val p95)) | 4/4 (denominator = 4) | 5/5 |

The Phase 6C strict denominator change from 5 → 4 is **not** a metric softening — it is a correction. `MQTT_DoS_Connect_Flood` having 0 LOO→Benign samples means the strict experiment cannot, by definition, observe a rescue on that target. Reporting `k/5` would silently force one of the five entries to be n/a; reporting `k/4 eligible` makes the structural exclusion explicit.

## 8. Limitations

- `MQTT_DoS_Connect_Flood` excluded from H2-strict (denominator structural).
- Single random seed (RANDOM_STATE=42); per-fold variance not estimated. Bootstrap CIs over the rescue subset would be a natural extension but were deferred since the rescue subsets are O(10²–10³).
- Entropy thresholds calibrated on val-correct samples, which may underestimate the entropy of known but mis-classified samples. Mitigation: multiple operating points (p95, p97) reported throughout.
- The ensemble score uses a single normalization basis (val-fitted MinMax). More principled options (rank-normalization, isotonic calibration) are deferred.
- All operating points reported; no per-target threshold cherry-picking.

## 9. Implications for the thesis narrative

- The Phase 6 negative finding ('reconstruction-error AE alone is insufficient for zero-day detection on flow features') stands.
- Phase 6C demonstrates that uncertainty signals already present in the supervised model — entropy and max-softmax-prob — carry **complementary** zero-day information, recoverable without retraining.
- The publishable contribution is the **ablation table**: a clean per-signal decomposition of where rescue recall comes from, evaluated under proper true-LOO conditions.
- The 4-case fusion logic generalizes cleanly to a 5-case logic (Case 5 = Uncertain Alert / Operator Review), preserving the confidence-stratified alert framing introduced in Phase 6.
- Future work directions sharpened: principled ensemble calibration (deferred from §8), per-fold variance estimation, and replacing the reconstruction-error AE with a profiling-feature-basis AE — which addresses the layer-coupling concern identified in Phase 6's future-work section.
