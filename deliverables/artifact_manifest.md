# Artifact Manifest — IoMT IDS Thesis Deliverables

> Every `.npy` / `.csv` / `.pkl` / `.json` / `.keras` file under `results/` (and selected `preprocessed/`, `eda_output/`) that the report, scripts, or notebook will load. Built during Step 2 of the production-deliverable run (see `decisions_ledger.md` for choice rationale, `numbers_map.md` for the value lookup).

## Directory mapping — spec → reality

The task spec assumed a `phase4 / phase5 / phase6 / phase6B / phase6C / phase7 / path_b` layout. The actual repository uses different names. No artifacts are missing; only the labels differ. The mapping below is the source of truth for every script and notebook cell.

| Spec label (in task brief) | Actual path on disk | Phase | What it holds |
|---|---|---|---|
| `results/phase4` | `results/supervised` | 4 | 8 supervised experiments E1–E8 + E5G baseline (RF entropy vs gini) |
| `results/phase5` | `results/unsupervised` (post-scaling-fix) and `results/unsupervised_unscaled` (pre-fix) | 5 | AE 44→32→16→8→16→32→44, IF 200-tree, scaler.pkl, val/test scores |
| `results/phase6` | `results/fusion` | 6 | 4-case fusion (simulated zero-day, val-correct calibration era) |
| `results/phase6B` | `results/zero_day_loo` | 6B | 5 LOO-XGBoost retrains, raw probabilities, prediction distributions |
| `results/phase6C` | `results/enhanced_fusion` | 6C / Path B all tiers | 5-case fusion + entropy + ablation_table.csv + multi_seed + threshold_sweep + ks_per_fold + vae_ablation |
| `results/phase7` | `results/shap` | 7 | TreeSHAP attributions (5K×19×44), per-class importance, 4-way comparison |
| `results/path_b` | spread across `enhanced_fusion/multi_seed`, `enhanced_fusion/threshold_sweep`, `enhanced_fusion/ks_per_fold`, `enhanced_fusion/vae_ablation`, `unsupervised/vae`, `unsupervised/lstm_ae`, `shap/sensitivity` | Path B Tier 1+2 | Multi-seed, continuous sweep, per-fold KS, SHAP background, β-VAE, LSTM-AE |

## Inputs the deliverables will load

### Phase 4 — Supervised (`results/supervised/`)

| File | Role | Loaded by |
|---|---|---|
| `metrics/E1_multiclass.json` … `E8_multiclass.json`, `E5G_multiclass.json` | 9 overall-metrics JSONs (test_f1_macro, test_accuracy, test_mcc, ...) | `02_supervised_phase4.py` |
| `metrics/E1_classification_report_test.json` … `E8_classification_report_test.json` | Per-class precision/recall/F1/support | `02_supervised_phase4.py` |
| `metrics/E5_vs_E5G_comparison.csv` | RF-entropy vs RF-gini side-by-side | `02_supervised_phase4.py` |
| `metrics/E1_cm_19class_test.npy` … `E8_cm_19class_test.npy` | 19×19 confusion matrices | `02_supervised_phase4.py` |
| `metrics/E5_feature_importance.csv` | RF-entropy top-44 importance | `02_supervised_phase4.py` |
| `metrics/overall_comparison.csv` | 24-row comparison (8 experiments × 3 tasks) | `02_supervised_phase4.py` |
| `metrics/smote_comparison.csv` | SMOTETomek delta per config | `02_supervised_phase4.py` |
| `metrics/minority_focus.csv` | 5 rarest classes F1 analysis | `02_supervised_phase4.py` |
| `predictions/E7_val_proba.npy` (903,016 × 19) | E7 val probabilities — fusion input | `04_fusion_phase6.py` |
| `predictions/E7_test_proba.npy` (892,268 × 19) | E7 test probabilities — fusion input | `04_fusion_phase6.py` |

### Phase 5 — Unsupervised (`results/unsupervised/`)

| File | Role |
|---|---|
| `thresholds.json` | 5 thresholds (p90, p95, p99, mean+2σ, mean+3σ) + selected = p90 |
| `ae_training_history.json` | epoch-by-epoch loss/val_loss for AE |
| `benign_error_stats.json` | mean=0.20, std=9.48 (heavy-tailed) |
| `metrics/ae_classification_report.json` | AE binary detection report |
| `metrics/if_classification_report.json` | IF binary detection report |
| `metrics/model_comparison.csv` | AE vs IF (AUC, F1, recall, FPR) |
| `metrics/per_class_detection_rates.csv` | 19 classes × {p90, p95, p99, mean+2σ, mean+3σ} × {AE, IF} |
| `metrics/zero_day_preview.csv` | 5 targets × AE/IF recall at p90 |
| `scores/ae_test_mse.npy` (892,268,) | Reconstruction error on test |
| `scores/ae_val_mse.npy` (903,016,) | Reconstruction error on val |
| `scores/if_test_scores.npy`, `if_val_scores.npy` | IF anomaly scores |
| `scores/ae_test_binary.npy` | p90-thresholded AE binary flag |
| `models/scaler.pkl` | StandardScaler fitted on benign-train (Contribution #13) |

### Phase 5 (pre-fix) — `results/unsupervised_unscaled/`

Retained as evidence for the §13.6 scaling-fix narrative (val loss 101,414 → 0.199).

| File | Role |
|---|---|
| `ae_training_history.json` | Pre-fix history showing million-scale loss |
| `summary.md` | Pre-fix per-class detection (Recon ≈ 0) |

### Phase 6 — Fusion (`results/fusion/`)

| File | Role |
|---|---|
| `metrics/h1_h2_verdicts.json` | H1 bootstrap CIs (20-class), H2-simulated 0/5, recommended p97 |
| `metrics/case_distribution.csv` | Case 1/2/3/4 counts per variant |
| `metrics/fusion_vs_supervised.csv` | Macro-F1 + bootstrap CIs |
| `metrics/fusion_vs_supervised_binary.csv` | Binary detection comparison |
| `metrics/per_class_case_analysis.csv` | 19 classes × 4 cases |
| `metrics/zero_day_results.csv` | 5 targets × H2 metrics (simulated LOO) |
| `metrics/threshold_sensitivity.csv` | 10-point sweep (p90–p99) |
| `fusion_results/fusion_val_cases.npy`, `fusion_test_cases.npy` | Case assignments (1–4) |

### Phase 6B — True LOO (`results/zero_day_loo/`)

| File | Role |
|---|---|
| `metrics/h2_loo_verdict.json` | True LOO H2-strict (0/5) + H2-binary (5/5 at p90) |
| `metrics/loo_results.csv` | Per-target n_test, LOO-E7→Benign rate, AE-on-missed |
| `metrics/loo_prediction_distribution.csv` | "Redundancy through misclassification" mapping |
| `metrics/loo_case_distribution.csv` | 4-case counts per LOO target |
| `metrics/loo_vs_phase6_comparison.csv` | Phase 6 (E7 in-distribution) vs LOO comparison |
| `models/loo_xgb_without_<target>.pkl` × 5 | 5 LOO retrained models |
| `models/loo_label_map_<target>.json` × 5 | Schema D label-space sidecars |
| `predictions/loo_<target>_test_pred.npy`, `_test_proba.npy` × 5 | Per-target LOO predictions |
| `multi_seed/seed_{1,7,42,100,1729}/predictions/` (Path B Week 1) | 5 seeds × 5 targets × 2 file types |
| `multi_seed/seed_{...}/metrics/per_target_metrics.json` | Per-seed per-target rescue metrics |

### Phase 6C — Enhanced Fusion (`results/enhanced_fusion/`)

| File | Role |
|---|---|
| `config.json` | Run config + calibrated thresholds (incl. `entropy_benign_p95 = 0.395`) |
| `metrics/ablation_table.csv` | **11 variants × aggregated metrics — the headline table** |
| `metrics/per_target_results.csv` | 55 rows = 11 variants × 5 targets |
| `metrics/entropy_stats.csv` | entropy mean/median/std × (target, sample_kind) |
| `metrics/signal_correlation.csv` | Pearson(entropy, AE/IF) per target |
| `metrics/h2_enhanced_verdict.json` | Phase 6/6B/6C comparison + `entropy_benign_p95.avg_recall = 0.8035264623662012` (TRIPWIRE source) |
| `signals/e7_entropy.npy` (892,268,) | E7 softmax entropy on test |
| `signals/ensemble_score.npy` | max(AE_norm, IF_norm) on test |
| `signals/entropy_thresholds.json` | benign-val p90/p95/p99 of entropy |
| `signals/ensemble_thresholds.json` | benign-val p90/p95/p99 of ensemble |

### Path B Tier 1 — Multi-Seed + Continuous Sweep + Per-Fold KS + SHAP Background

| File | Role |
|---|---|
| `results/enhanced_fusion/multi_seed/seed_{1,7,42,100,1729}/metrics/ablation_table.csv` | 11 variants × 5 seeds |
| `results/enhanced_fusion/multi_seed/seed_{...}/metrics/per_target_results.csv` | Per-target per-seed |
| `results/enhanced_fusion/multi_seed_summary.csv` | 11 variants × {mean, std, min, max, p05, p95} |
| `results/enhanced_fusion/multi_seed_per_target_summary.csv` | Aggregated per-target |
| `results/enhanced_fusion/threshold_sweep/sweep_table.csv` | 29 rows × 8 cols — continuous sweep |
| `results/enhanced_fusion/threshold_sweep/sweep_per_target.csv` | 145 rows = 29 × 5 targets |
| `results/enhanced_fusion/ks_per_fold/ks_per_fold.csv` | 6 rows (5 folds + aggregate E7) |
| `results/shap/sensitivity/comparison.csv` | train_bg vs test_bg overall stats |
| `results/shap/sensitivity/global_top10_ranks.csv` | rank-by-rank comparison |
| `results/shap/sensitivity/per_class_jaccard.csv` | 19 classes × top-5 Jaccard |
| `results/shap/sensitivity/category_cosine.csv` | DDoS↔DoS cosine under train_bg |
| `results/shap/shap_values_train_bg.npy` (19 × 5000 × 44, float32, ~17 MB) | New SHAP attributions from X_train background |

### Path B Tier 2 — β-VAE + LSTM-AE substitution

| File | Role |
|---|---|
| `results/enhanced_fusion/vae_ablation/all_betas_ablation.csv` | 40 rows = 4 βs × 10 variants |
| `results/enhanced_fusion/vae_ablation/per_beta_summary.json` | Per-β metadata for decision |
| `results/enhanced_fusion/vae_ablation/beta_{0.1,0.5,1.0,4.0}/ablation_table.csv` | 10 rows each |
| `results/enhanced_fusion/vae_ablation/beta_{...}/per_target_results.csv` | 50 rows each = 10×5 targets |
| `results/enhanced_fusion/vae_decision.csv` | 5 rows: 1 §15D anchor + 4 βs |
| `results/enhanced_fusion/vae_decision_summary.md` | Narrative decision |
| `results/unsupervised/vae/all_betas_summary.csv` | 4-row training summary |
| `results/unsupervised/vae/beta_{0.1,0.5,1.0,4.0}/manifest.json` | Per-β hyperparameters + diagnostic stats |
| `results/unsupervised/vae/beta_{...}/history.json` | Loss progression |
| `results/unsupervised/lstm_ae/all_configs_summary.csv` | 6-row LSTM-AE config grid |
| `results/unsupervised/lstm_ae/gate1_report.json` | Binary verdict per config + audit trail |
| `results/unsupervised/lstm_ae/lstm_ae_recon_ref.json` | SHA-256 hashes + tolerance tiers |
| `results/unsupervised/lstm_ae/c{1,2,3,4,5,6}/manifest.json` | Per-config diagnostic stats |
| `results/unsupervised/lstm_ae/c{...}/history.json` | Per-config loss curves |

### Phase 7 — SHAP (`results/shap/`)

| File | Role |
|---|---|
| `shap_values/shap_values.npy` (19 × 5000 × 44, float32, ~16 MB) | Raw TreeSHAP attributions |
| `shap_values/X_shap_subset.npy` (5000 × 44) | The 5K stratified explained set |
| `shap_values/y_shap_subset.csv` | Ground-truth labels for explained set |
| `metrics/global_importance.csv` | 44 features ranked, mean|SHAP| |
| `metrics/per_class_importance.csv` | 19 × 44 matrix |
| `metrics/per_class_top5.csv` | 19 classes × top-5 |
| `metrics/ddos_vs_dos_features.csv` | Per-feature DDoS vs DoS comparison |
| `metrics/method_comparison.csv` | 4-way ranking |
| `metrics/method_jaccard.csv` | Pairwise Jaccard top-10 |
| `metrics/method_rank_correlation.csv` | Spearman + Kendall |
| `metrics/category_importance.csv` | 5 categories × 44 |
| `metrics/category_similarity.csv` | 5×5 cosine matrix |
| `metrics/subsample_class_distribution.csv` | Stratification check |

### Supporting — `preprocessed/`, `eda_output/`

| File | Role |
|---|---|
| `preprocessed/config.json` | All preprocessing parameters, feature lists, column orders |
| `preprocessed/label_encoders.json` | Label → int mappings |
| `preprocessed/full_features/y_test.csv` | True labels for AE/IF AUC computation |
| `eda_output/imbalance_table.csv` | 19-class train/test counts, ratios |
| `eda_output/feature_target_cohens_d.csv` | Cohen's d for 44 features (Attack vs Benign) |
| `eda_output/high_correlation_pairs.csv` | 25 pairs |Pearson r| > 0.85 |

## Reproducibility tripwires (asserted by `04_fusion_phase6.py`)

1. `entropy_benign_p95` strict_avg must equal `0.8035264623662012` (read from `results/enhanced_fusion/metrics/h2_enhanced_verdict.json` and from `results/enhanced_fusion/metrics/ablation_table.csv` row index 5). Tolerance: `1e-9`.
2. `E7 test_f1_macro` must equal `0.907626622882394` within `1e-9` (rounded to `0.9076` in narrative).
3. `AE test_AUC` must equal `0.9892` ± `0.001` (computed live from `ae_test_mse.npy` + `y_test.csv`).

## Inventory completeness

All artifacts listed above were verified present on disk during this run (see `ls -R results/` output captured in the audit step). No `STOP` condition triggered. The pre-fix `unsupervised_unscaled/` directory is retained for narrative reference (Contribution #13); none of the deliverables compute new numbers from it.
