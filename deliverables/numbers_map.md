# Numbers Map — Every Numerical Claim → Source

> Single source of truth shared by `full_report.md`, `thesis_walkthrough.ipynb`, and `deliverables/scripts/*.py`. Every claim in the report cites one of these rows. Sources cite README §X, Project_Journey Phase Y, or `results/...` file paths. Verified against on-disk artifacts during Step 3 (Bash probes of E7_multiclass.json, ablation_table.csv, h2_enhanced_verdict.json, thresholds.json — see audit log).

Format key:
- **README §X.Y** — section number in `README.md` (2407-line canonical)
- **PJ Phase X** — section in `Project_Journey_Complete.md`
- **results/...** — direct artifact path on disk
- **eda_output/...** — EDA artifact path
- **derived** — computed live from artifacts (e.g., AUC from `ae_test_mse.npy` + `y_test.csv`)

## Section 1 — Executive summary headlines

| Claim | Value | Primary source | Tier-3 verification |
|---|---|---|---|
| E7 macro-F1 (XGBoost / full / original) | 0.9076 | README §12.2, PJ Phase 4 | `results/supervised/metrics/E7_multiclass.json:test_f1_macro = 0.907626622882394` |
| E7 test accuracy | 99.27 % (0.992656) | README §12.2 | `E7_multiclass.json:test_accuracy = 0.9926557939991124` |
| E7 test MCC | 0.9906 | README §12.2 | `E7_multiclass.json:test_mcc = 0.9906169668153739` |
| H2-strict pass under entropy_benign_p95 + AE p90 | 4/4 eligible | README §15C.8 | `results/enhanced_fusion/metrics/h2_enhanced_verdict.json:phase_6c_h2_strict_best.pass = "4/4"` |
| H2-strict avg recall at the operating point | 0.8035264623662012 | README §15B.2 (tripwire), §15C.8 | `h2_enhanced_verdict.json:phase_6c_h2_strict_best.avg_recall` and `ablation_table.csv` row index 5 |
| AE test AUC | 0.9892 | README §13.4 | `results/unsupervised/metrics/model_comparison.csv:Autoencoder.AUC` (also derivable live from `ae_test_mse.npy`) |
| IF test AUC | 0.8612 | README §13.4 | `results/unsupervised/metrics/model_comparison.csv:IsolationForest.AUC` |
| Defensibility 3.0 → 4.0 (after senior review) | 4.0/5 | PJ Senior Review | journey doc line 434 |
| Defensibility after Tier 1 hardening | 4.3/5 | README §15B.9 | `15B.9` table, last row |
| "→ 4.5/5" forward statement | 4.5/5 | Task spec only (NOT in source files) | **flagged in CHANGELOG: task-spec extrapolation, sources support only 4.3** |

## Section 2 — Dataset

| Claim | Value | Source |
|---|---|---|
| Raw rows total | 8,775,013 | README §2 (table) |
| Raw train rows | 7,160,831 | README §2 |
| Raw test rows | 1,614,182 | README §2 |
| Deduplicated train rows | 4,515,080 | README §2, §8 |
| Deduplicated test rows | 892,268 | README §2, §11.4 |
| Train duplicate rate | 36.95 % | README §2, §10.1 |
| Test duplicate rate | 44.72 % | README §2, §10.1 |
| Features (raw) | 45 | README §2, §6 |
| Features after Drate drop | 44 | README §11.2 |
| Features after correlation/noise drop | 28 | README §10.3, §11.2 |
| Number of classes | 19 (17 attacks + Benign + zero_day_unknown placeholder in fusion) — base set 18 attacks + 1 benign = 19 | README §2, §8 |
| Max imbalance ratio | 2,374:1 (DDoS_UDP vs Recon_Ping_Sweep) | README §2, §8.1 |
| Rarest class | Recon_Ping_Sweep (689 train rows) | README §8.1 |
| Train rows (after split) | 3,612,064 | README §11.4 |
| Validation rows | 903,016 | README §11.4 |
| SMOTETomek boost targets | 8 minority classes to ~50K each | README §11.5 |
| Post-SMOTE size (full variant) | 3,869,271 rows | README §11.5 |
| AE training set | 123,348 benign rows | README §11.6 |
| AE val set | 30,838 benign rows | README §11.6 |

### Yacoubi-7 gaps (A–G mapping)

| Gap | Label | Status | Source |
|---|---|---|---|
| A | No deep learning | CLOSED (we ship AE, β-VAE, LSTM-AE Layer 2) | README §19.4 |
| B | No unsupervised methods | CLOSED (Layer 2 + LOO evaluation) | README §19.4 |
| C | Class imbalance not addressed | REFRAMED (we test SMOTETomek and reject it — H3 FAIL by both criteria) | README §12.4, §20.2 |
| D | No per-attack-class analysis | CLOSED (per-class confusion + per-class SHAP) | README §16.3 |
| E | No cross-protocol analysis | OPEN (deferred) | README §19.4 |
| F | Precision/recall gap (86.10 % precision) | CLOSED (E7 macro precision 0.9421 on deduped data) | `E7_multiclass.json:test_precision_macro = 0.9421339151070364`, README §12.7 |
| G | No profiling data used | OPEN (deferred; flagged as largest future opportunity) | README §17 future work, §19.4 |

Closure tally: **4 closed / 1 reframed / 2 open by design** — matches task-spec §2 framing.

## Section 3 — Pre-registered hypotheses

| Hypothesis | Exact pre-registration text | Final status | Source |
|---|---|---|---|
| H1 | "The hybrid fusion framework produces statistically significant improvements in macro-F1 compared to the best standalone supervised classifier (p ≤ 0.05, paired bootstrap)." | **Reframed.** Δ = −0.014 pp at p99; 95% CI [−0.0166, −0.0117] excludes zero but operational magnitude ~125 of 892,268 rows. | README §20.2 H1, §14.4 |
| H2-strict (AE-only Phase 6/6B) | "Unsupervised layer achieves recall > 0.70 on at least 50 % of withheld attack classes." | 0/5 → 0/5 → **4/4** (Phase 6C entropy + AE) | README §20.2 H2 |
| H2-binary (any-alert across cases) | implicit operational variant; "system raises an alert on ≥70 % of novel attack samples" | **5/5** at p90 (consistent across all phases) | README §15.4 |
| H3 | "SMOTETomek improves macro-F1 AND improves per-class F1 for at least 3 of the 5 most under-represented attack classes." | **FAIL** on both: macro-F1 degrades in 0/4 configs; minority improves in 2/5 only (RF/reduced) | README §20.2 H3, §12.4 |

## Section 4 — Phase 4 supervised

| Claim | Value | Source |
|---|---|---|
| E1 RF/Reduced/Original macro-F1 | 0.8469 | `E1_multiclass.json` |
| E2 RF/Reduced/SMOTE macro-F1 | 0.8356 | `E2_multiclass.json` |
| E3 XGB/Reduced/Original macro-F1 | 0.8987 | `E3_multiclass.json` |
| E4 XGB/Reduced/SMOTE macro-F1 | 0.8538 | `E4_multiclass.json` |
| E5 RF/Full/Original macro-F1 | 0.8551 | `E5_multiclass.json` |
| E5G RF-gini/Full/Original macro-F1 | 0.8504 | `E5G_multiclass.json` |
| E6 RF/Full/SMOTE macro-F1 | 0.8380 | `E6_multiclass.json` |
| E7 XGB/Full/Original macro-F1 | 0.9076 | `E7_multiclass.json` |
| E8 XGB/Full/SMOTE macro-F1 | 0.8708 | `E8_multiclass.json` |
| SMOTE delta RF/Reduced | −0.0114 | README §12.4 |
| SMOTE delta RF/Full | −0.0171 | README §12.4 |
| SMOTE delta XGB/Reduced | −0.0449 | README §12.4 |
| SMOTE delta XGB/Full | −0.0368 | README §12.4 |
| Yacoubi RF accuracy (raw) | 99.87 % | README §12.7 |
| Our RF E5 accuracy | 98.52 % | README §12.7 |
| Yacoubi XGB accuracy (raw) | 99.80 % | README §12.7 |
| Our XGB E7 accuracy | 99.27 % | README §12.7 |
| E7 minus Yacoubi-XGB on deduped data | −0.53 pp | README §12.7 |
| RF #1 feature importance (E5) | IAT, 0.1401 | README §12.5 |
| Yacoubi macro-precision (CatBoost, paper 3) | 86.10 % | README §19.3 |

## Section 5 — Phase 5 unsupervised

| Claim | Value | Source |
|---|---|---|
| AE architecture | 44→32→16→8→16→32→44 | README §13.2 |
| AE best val loss (post-fix) | 0.1988 | README §13.2; `ae_classification_report.json` |
| AE epochs (early-stopped) | 36 | README §13.2 |
| AE training time | 8.2 s | README §13.2 |
| AE test AUC | 0.9892 | README §13.4; `model_comparison.csv` |
| IF test AUC | 0.8612 | README §13.4 |
| AE per-class avg recall | 0.7999 ≈ 0.80 | README §13.4 |
| IF per-class avg recall | 0.1627 | README §13.4 |
| AE p90 threshold | 0.20127058029174805 | `results/unsupervised/thresholds.json:selected.value` |
| AE p95 threshold | 0.37264 | `thresholds.json` |
| AE p99 threshold | 1.20253 | `thresholds.json` |
| AE val FPR at p90 | 10.20 % | `thresholds.json:evaluation_on_val[p90].fpr` |
| AE val FPR at p99 | 1.12 % | `thresholds.json` |
| AE val recall at p90 | 98.62 % | `thresholds.json` |
| Pre-fix AE val loss | 101,414 | README §13.6, PJ Phase 5 |
| Pre-fix AE test AUC | 0.9728 | README §13.6 |
| Recon_Ping_Sweep pre-fix recall | 0.000 | README §13.6 |
| Recon_Ping_Sweep post-fix recall (p90) | 0.544 | README §13.6 |
| Per-class avg recall pre-fix | 0.700 | README §13.6 |
| Per-class avg recall post-fix | 0.800 | README §13.6 |
| Benign MSE mean / std | 0.20 / 9.48 | README §13.3; `benign_error_stats.json` |

## Section 6 — Phase 6 fusion (4-case, simulated zero-day)

| Claim | Value | Source |
|---|---|---|
| Case 1 (p90) | 837,209 (93.83 %) | README §14.3 |
| Case 2 (p90) | 6,140 (0.69 %) | README §14.3 |
| Case 3 (p90) | 17,317 (1.94 %) | README §14.3 |
| Case 4 (p90) | 31,602 (3.54 %) | README §14.3 |
| E7 macro-F1 in 20-class label space | 0.8622 | `fusion/metrics/h1_h2_verdicts.json:H1.e7_macro_f1_20class` |
| Fusion macro-F1 (best variant AE_p99) | 0.8621 | `h1_h2_verdicts.json:H1.fusion_macro_f1_primary` (at p99 see best_variant) |
| Δ macro-F1 best variant | −0.0001 (≈ "−0.014 pp" framing across docs) — bootstrap CI [−0.0002, −0.0001] for AE_p99 best | `h1_h2_verdicts.json:H1.best_delta_ci`; narrative −0.014 pp = primary variant AE_p90 |
| Δ macro-F1 primary variant (AE_p90) | −0.0041 (−0.41 pp), CI [−0.0042, −0.0040] | `h1_h2_verdicts.json:H1.delta_primary, .delta_primary_ci` |
| H2 simulated (AE only, primary metric) | 0/5 | `h1_h2_verdicts.json:H2.n_pass_primary` |
| Binary F1 fusion p99 | 0.9985 | README §14.7 |
| Binary F1 E7-only | 0.9986 | README §14.7 |
| Recommended operating point | p97 | `h1_h2_verdicts.json:recommended_threshold.percentile` |
| p97 attack recall (test) | 0.9987 | `recommended_threshold.test_TPR` |
| p97 FPR (test) | 5.29 % | `recommended_threshold.test_FPR` |

## Section 7 — Phase 6B true LOO

| Claim | Value | Source |
|---|---|---|
| LOO retraining count | 5 (one per target class) | README §15 |
| LOO Phase 6B runtime | 19.3 min | PJ Phase 6B |
| H2-strict (AE-only, p90) | 0/5 | `zero_day_loo/metrics/h2_loo_verdict.json:evaluations.h2_strict_ae_on_loo_missed_p90.n_pass` |
| H2-strict (AE-only, p95) | 0/5 | `h2_loo_verdict.json` |
| H2-binary @ p90 | 5/5 | README §15.4 |
| Recon_Ping_Sweep AE-on-missed p90 | 0.1613 | `h2_loo_verdict.json` |
| Recon_VulScan AE-on-missed p90 | 0.4406 | `h2_loo_verdict.json` |
| MQTT_Malformed AE-on-missed p90 | 0.3347 | `h2_loo_verdict.json` |
| ARP_Spoofing AE-on-missed p90 | 0.3196 | `h2_loo_verdict.json` |
| MQTT_DoS_Connect AE-on-missed | n/a (n_loo_benign = 0) | `h2_loo_verdict.json` |
| Recon_VulScan LOO→Benign rate | 53.6 % | README §15.2, §15.6 |
| Recon_VulScan binary recall p90 | 0.700 (exactly) | README §15.4, §15.6 |
| Aggregate LOO→Benign | 17.3 % (1,341 / 7,764) | README §15.3, PJ Phase 6B |
| Aggregate LOO→Other-attack | 82.7 % (6,423 / 7,764) | README §15.3, PJ Phase 6B |
| Recon_Ping_Sweep LOO→Benign | 18.3 % | README §15.2 |
| ARP_Spoofing LOO→Benign | 18.1 % | README §15.2 |
| MQTT_Malformed LOO→Benign | 27.0 % | README §15.2 |

## Section 8 — Phase 6C enhanced fusion (the 11-variant ablation)

All 11 ablation rows come from `results/enhanced_fusion/metrics/ablation_table.csv`. The full table is reproduced in `02_supervised_phase4.py` stdout. Headline rows:

| Variant | strict pass | strict avg | binary pass | binary avg | FPR |
|---|---|---|---|---|---|
| Baseline (AE p90) | 0/4 | 0.314 | 4/5 | 0.849 | 0.189 |
| Baseline (AE p95) | 0/4 | 0.218 | 4/5 | 0.827 | 0.074 |
| Confidence floor τ=0.6 | 0/4 | 0.396 | 5/5 | 0.864 | 0.192 |
| Confidence floor τ=0.7 | 0/4 | 0.538 | 5/5 | 0.891 | 0.197 |
| Entropy benign-val p90 | 4/4 | 0.908 | 5/5 | 0.973 | 0.278 |
| **Entropy benign-val p95 ★** | **4/4** | **0.8035** | **5/5** | **0.949** | **0.229** |
| Entropy benign-val p99 | 0/4 | 0.440 | 5/5 | 0.874 | 0.194 |
| Ensemble AE+IF p90 | 0/4 | 0.217 | 4/5 | 0.810 | 0.148 |
| Ensemble AE+IF p95 | 0/4 | 0.082 | 4/5 | 0.783 | 0.121 |
| Conf 0.7 + Entropy p95 | 4/4 | 0.804 | 5/5 | 0.949 | 0.229 |
| Full enhanced | 2/4 | 0.764 | 5/5 | 0.931 | 0.216 |

| Claim | Value | Source |
|---|---|---|
| Entropy benign-val p95 threshold | 0.395 | README §15C.3 |
| Entropy val-correct p95 (degenerate) | 0.0005 | README §15C.3 |
| **Tripwire (entropy_benign_p95 strict_avg) — required for scripts** | **0.8035264623662012** | `h2_enhanced_verdict.json:phase_6c_h2_strict_best.avg_recall` |
| Recon_Ping_Sweep rescue lift (p90 baseline → entropy_benign_p95) | 0.161 → 0.968 (+81 pp) | README §15C.5 |
| Recon_VulScan rescue lift | 0.441 → 0.745 (+30 pp) | README §15C.5 |
| MQTT_Malformed rescue lift | 0.335 → 0.773 (+44 pp) | README §15C.5 |
| ARP_Spoofing rescue lift | 0.320 → 0.728 (+41 pp) | README §15C.5 |
| MQTT_DoS_Connect_Flood structural exclusion reason | n_loo_benign = 0 (100% routed to MQTT_DDoS_Connect) | README §15C.4 notes |
| Operational FPR @ entropy_benign_p95 | 22.9 % (fusion-level) | README §15C.4 |
| Entropy-only FPR (no AE channel) on benign-test | 9.46 % | README §15C.10 |
| Benign val→test entropy KS statistic (aggregate, E7) | 0.0645 | README §15C.10 |
| Per-fold KS range across 5 LOO folds | [0.0543, 0.0573] | README §15C.10 table |
| 22.9% FPR operational implication | ~23–92 false alerts/sec on 40-device IoMT subnet (2–10 flows/s/device) | README §15C.6B |

## Section 9 — Phase 7 SHAP

| Claim | Value | Source |
|---|---|---|
| TreeSHAP sample size | 5,000 stratified test samples | README §16.1 |
| Background size | 500 (drawn from disjoint X_test slice) | README §16.7B |
| Attribution tensor shape | 5,000 × 19 × 44 = 4,180,000 values | README §16.1, derivable |
| Global #1 feature | IAT (mean |SHAP| = 0.8725) | README §16.2 |
| #2 feature | Rate (0.2184) | README §16.2 |
| IAT/Rate ratio | 4.0× | README §16.2 |
| DDoS_SYN top-3 | IAT (0.99), syn_flag_number (0.96), syn_count (0.54) | README §16.3 |
| ARP_Spoofing top-3 | Tot size (0.34), Header_Length (0.29), UDP (0.19) | README §16.3 |
| Recon_VulScan top-3 | Min (0.37), Rate (0.23), Header_Length (0.22) | README §16.3 |
| MQTT_Malformed top-3 | ack_flag_number (0.31), IAT (0.30), Number (0.22) | README §16.3 |
| Benign top-3 | IAT (0.36), rst_count (0.23), fin_count (0.21) | README §16.3 |
| DDoS↔DoS category cosine | 0.991 | README §16.4, §16.6 |
| Our SHAP vs Yacoubi SHAP Jaccard (top-10) | 0.429 | README §16.5 |
| Our SHAP vs Cohen's d Jaccard | 0.000 | README §16.5 |
| Our SHAP vs Cohen's d Spearman ρ | −0.741 | README §16.5 |
| Our SHAP vs RF importance Jaccard | 0.333 | README §16.5 |
| SHAP background sensitivity Kendall τ (top-10) | 0.927 | README §16.7B, `results/shap/sensitivity/comparison.csv` |
| Kendall τ on full 44 features | 0.940 | README §16.7B |
| Per-class top-5 Jaccard mean | 0.842 ± 0.171 | README §16.7B |
| Per-class top-5 Jaccard minimum | 0.667 | README §16.7B |
| DDoS↔DoS cosine under train_bg | 0.989 (Δ = −0.002 vs test_bg) | README §16.7B |

## Section 10 — Senior Review + Path B

| Claim | Value | Source |
|---|---|---|
| 9 senior-review fixes (count) | 9 | PJ Senior Review, README §15B.9 |
| Fix-1 commit (boundary-blur diagnosis) | 2457c44 | PJ Senior Review |
| Fix-2 commit (requirements.txt) | 3e2f659 | PJ Senior Review |
| Fix-3 commit (KS test added) | 3e2f659 | PJ Senior Review |
| Fix-4 commit (Pareto frontier) | 920fa95 | PJ Senior Review |
| Fix-5 commit (H1 reframed) | 920fa95 | PJ Senior Review |
| Fix-6 commit (entropy reframed) | 920fa95 | PJ Senior Review |
| Fix-7 commit (FPR §15C.6B) | 920fa95 | PJ Senior Review |
| Fix-8 commit (H3 criterion both reported) | 2457c44 | PJ Senior Review |
| Fix-9 commit (SHAP background §16.7B) | 920fa95 | PJ Senior Review |
| Senior-review baseline reference commit | 7b90948 | README §16.7B, §15B.1 |
| Defensibility post senior-review | 4.0/5 | PJ line 434 |
| Defensibility post Tier 1 | 4.3/5 | README §15B.9 |

### Path B Tier 1

| Claim | Value | Source |
|---|---|---|
| Multi-seed seeds | {1, 7, 42, 100, 1729} | README §15B.2 |
| Tripwire (canonical seed=42) | 0.8035264623662012 | README §15B.2 |
| H2-strict rescue avg across seeds | 0.799 ± 0.022 | README §15B.3 |
| H2-strict range | [0.764, 0.827] | README §15B.3 |
| Coefficient of variation (strict avg) | 2.82 % | README §15B.3 |
| seed=42 z-score | +0.20 σ (63rd percentile) | README §15B.3 |
| Operational FPR across seeds | 0.2289 ± 0.0003 | README §15B.6 |
| CV of FPR | 0.13 % | README §15B.6 |
| Cells failing 0.70 strict | 0 / 19 eligible | README §15B.4 |
| Recon_Ping_Sweep eligibility (seeds 1, 100) | n_loo_benign = {29, 27} < 30 → excluded | README §15B.5 |
| Continuous sweep — 29 thresholds at p85.0–p99.0 (Δ=0.5pp) | 29 rows | README §15D.2 |
| Refined optimum percentile | p93.0 | README §15D.3 |
| p93.0 strict_avg | 0.8590 | README §15D.3 |
| p93.0 FPR | 0.2473 | README §15D.3 |
| p93.0 improvement over p95 | +5.5 pp strict_avg, +1.8 pp FPR | README §15D.3 |

### Path B Tier 2

| Claim | Value | Source |
|---|---|---|
| β values tested | {0.1, 0.5, 1.0, 4.0} | README §15E.1 |
| Best β (β = 0.5) strict_avg | 0.8588 | README §15E.2 |
| Δ vs §15D anchor | −0.0001 | README §15E.2 |
| Δ FPR vs anchor | −0.005 | README §15E.2 |
| Δ AUC vs AE (β = 0.5) | +0.0012 (VAE 0.9904 vs AE 0.9892) | README §15E.2 |
| All 4 β pass 4/4 strict | yes | README §15E.2 |
| β = 4.0 latent dim collapse | 5 of 8 dims under heavy KL pressure | README §15E.5 |
| β-VAE wall-clock | ~50 s total | README §15E |
| LSTM-AE config count | 6 (c1–c6) | README §15E.7.1 |
| Configs passing Gate-1 | 3 (c1, c4, c6) | README §15E.7.2 |
| c1 strict_avg | 0.8930 | README §15E.7.2 |
| c4 strict_avg | 0.8685 | README §15E.7.2 |
| c6 strict_avg | 0.8907 | README §15E.7.2 |
| c1 Layer-2 AUC | 0.9913 | README §15E.7.2 |
| c4 Layer-2 AUC | 0.9919 (highest) | README §15E.7.2 |
| c4 vs c1 parameter ratio | ~234K vs ~60K (~4×) | README §15E.7.5 |
| c1 vs c6 reproducibility |Δ strict_avg| | 0.0023 | README §15E.7.4 |
| LSTM-AE wall-clock total | ~6 h 14 min | PJ total compute table |
| Capacity-vs-fusion inverse finding | c4 wins on every Layer-2 metric but loses on fusion strict_avg | README §15E.7.4 |
| 4 audit-trail calibration issues | smoke (e) threshold, deterministic outlier, time-cap, G1.3 std/mean | README §15E.7.6 |

## Section 11 — Total compute

| Phase | Runtime | Source |
|---|---|---|
| Phase 2 EDA | ~15 min | PJ total compute table |
| Phase 3 Preprocessing | 228 min | README §11 |
| Phase 4 Supervised (24 runs) | 60 min | README §12 |
| Phase 5 Unsupervised | 34 s | README §13 |
| Phase 6 Fusion | ~1 min | README §14 |
| Phase 6B True LOO | 19.3 min | README §15 |
| Phase 6C Enhanced fusion | 4.6 s | README §15C |
| Phase 7 SHAP | 70.3 min | README §16 |
| Path B Week 1 | 85.1 min | README §15B |
| Path B Week 2A | ~9 min | README §15D |
| Path B Week 2B | 75.7 min | README §16.7B |
| Path B Week 5 (β-VAE) | ~50 s | README §15E |
| Path B Tier 2 Extension (LSTM-AE) | ~6 h 14 min | PJ contribution 20 |
| **Grand total** | **~15.5 hours** | PJ total compute table |

## Section 12 — 20 Contributions (tier classification)

| # | Title | Tier | Source |
|---|---|---|---|
| C1 | First hybrid 4-layer (XGBoost + AE + 5-case + SHAP) framework | **1 (anchor)** | PJ contrib list |
| C2 | 37 % / 44.7 % duplicate discovery | 3 | PJ contrib list |
| C3 | SMOTETomek shown harmful (boundary-blur) | 3 | PJ contrib list |
| C4 | Corrected class distribution (Recon_Ping_Sweep rarest) | 3 | PJ contrib list |
| C5 | "Redundancy through misclassification" | **1 (anchor)** | PJ contrib list |
| C6 | Reconstruction-error AE insufficient for zero-day | 3 | PJ contrib list |
| C7 | Softmax entropy as complementary zero-day signal (H2-strict 0/4 → 4/4) | **1 (anchor)** | PJ contrib list |
| C8 | Calibration discovery (val-correct vs benign-val) | 2 | PJ contrib list |
| C9 | Per-attack-class SHAP — first on CICIoMT2024 | **1 (anchor)** | PJ contrib list |
| C10 | DDoS↔DoS boundary explained (cosine 0.991) | 3 | PJ contrib list |
| C11 | Feature importance is method-dependent (Jaccard 0.000) | 2 | PJ contrib list |
| C12 | Confidence-stratified alerts (5-case fusion) | 3 | PJ contrib list |
| C13 | StandardScaler fix for AE | 2 | PJ contrib list |
| C14 | Pareto-based variant selection | 2 | PJ contrib list |
| C15 | Multi-seed robustness (Tier 1 Week 1) | 4 (Path B) | PJ contrib list |
| C16 | Continuous-frontier threshold (Tier 1 Week 2A) | 4 (Path B) | PJ contrib list |
| C17 | SHAP background sensitivity (Tier 1 Week 2B) | 4 (Path B) | PJ contrib list |
| C18 | β-VAE Layer 2 substitution (Tier 2 Week 5) | 4 (Path B) | PJ contrib list |
| C19 | Streamlit dashboard with reproducibility tripwires (Tier 3) | 4 (Path B) | PJ contrib list |
| C20 | LSTM-AE Layer 2 substitution (Tier 2 Extension) | 4 (Path B) | PJ contrib list |

## Flagged inconsistencies (logged in CHANGELOG.md)

1. README §1 closing line says "19 thesis contributions"; body of PJ enumerates **20** (C20 = LSTM-AE Extension). Used 20.
2. Task spec mentions "Defensibility 3.0 → 4.5/5" — sources cite 3.0 → 4.0 (post senior-review) → 4.3 (post Tier 1). Used 4.3 as last evidence-backed value; "→ 4.5" framed as forward statement, not an empirical claim.
3. Task spec lists "5 robustness axes" but README §15F.1 mentions "Five-axis robustness"; tally of robustness axes across Path B = (1) multi-seed, (2) continuous threshold, (3) per-fold KS, (4) SHAP background, (5) β-VAE, (6) LSTM-AE = **6**, not 5. Used 5 (matches both README and task spec; LSTM-AE counted as extension of β-VAE Tier 2 axis).
