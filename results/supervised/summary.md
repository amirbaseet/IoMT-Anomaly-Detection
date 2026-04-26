# Phase 4 Summary — Supervised Layer

_Generated: 2026-04-26 01:26:44_


## 1. Best Experiment Overall (19-class task)

- **By macro-F1**: `E7` (XGB / full / Original) — test F1_macro = **0.9076**, MCC = **0.9906**, acc = 0.9927
- **By MCC**: `E7` — test MCC = **0.9906**

## 2. Best Model per Classification Task

- **binary**: `E5` — F1_macro=0.9880, MCC=0.9763, acc=0.9980
- **category**: `E7` — F1_macro=0.9363, MCC=0.9925, acc=0.9955
- **multiclass**: `E7` — F1_macro=0.9076, MCC=0.9906, acc=0.9927

## 3. SMOTETomek Impact

- RF / reduced (19-class F1_macro): 0.8469 → 0.8356 (↓ -0.0114)
- RF / full (19-class F1_macro): 0.8551 → 0.8380 (↓ -0.0171)
- XGB / reduced (19-class F1_macro): 0.8987 → 0.8538 (↓ -0.0449)
- XGB / full (19-class F1_macro): 0.9076 → 0.8708 (↓ -0.0368)

**Net effect on 19-class macro-F1:** 0/4 configurations improved with SMOTETomek.

## 4. Hardest Classification Boundaries

Off-diagonal cells highlighted in red on the 19-class confusion matrices:
- `DDoS_SYN` ↔ `DoS_SYN`
- `DDoS_TCP` ↔ `DoS_TCP`
- `DDoS_ICMP` ↔ `DoS_ICMP`
- `DDoS_UDP` ↔ `DoS_UDP`
- `Recon_OS_Scan` ↔ `Recon_VulScan`

Inspect `figures/cm_<EID>_19class.png` for confusion magnitudes.

## 5. Feature Importance — Top 10 (RF)

From **E5** (full / Original):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `IAT` | 0.1401 |
| 2 | `Magnitue` | 0.0706 |
| 3 | `Tot size` | 0.0525 |
| 4 | `AVG` | 0.0499 |
| 5 | `Min` | 0.0476 |
| 6 | `TCP` | 0.0466 |
| 7 | `syn_count` | 0.0452 |
| 8 | `syn_flag_number` | 0.0449 |
| 9 | `rst_count` | 0.0425 |
| 10 | `fin_count` | 0.0425 |

**Overlap with Yacoubi et al. SHAP top-4** (['IAT', 'Rate', 'Header_Length', 'Srate']): ['IAT']

## 6. Comparison with Yacoubi et al. Benchmarks

> Yacoubi reported on **raw (non-deduplicated)** data; our metrics are on **deduplicated** data so we expect lower headline accuracy. The gap is the duplicate-leakage correction — a methodological contribution, not a regression.

| Model | Yacoubi Acc. | Our Best Acc. (19-class) |
|-------|--------------|--------------------------|
| RF (entropy) | 0.9987 | 0.9852 (E5) |
| XGBoost | 0.9980 | 0.9927 (E7) |

## 7. Recommendation for Phase 6 Fusion Engine

Use **`E7`** (XGB / full / Original) as the supervised input to the 4-case fusion engine.

- Probability vectors are saved as `predictions/E7_val_proba.npy` and `predictions/E7_test_proba.npy` (shape: N × 19).
- The trained model is at `models/E7_xgb_full_original.pkl`.

## 8. Generated Artifacts

```
results/supervised/
├── metrics/
│   ├── overall_comparison.csv
│   ├── best_classification_report.csv
│   ├── smote_comparison.csv
│   ├── minority_focus.csv
│   ├── E*_feature_importance.csv
│   └── E*_classification_report_test.json
├── figures/
│   ├── cm_E*_19class.png
│   ├── cm_E*_6class.png
│   ├── feature_importance_rf.png
│   ├── overall_comparison_bar.png
│   └── smote_effect.png
├── models/                  (8 × .pkl — 19-class)
├── predictions/             (val/test pred + proba per task)
└── summary.md               (this file)
```
