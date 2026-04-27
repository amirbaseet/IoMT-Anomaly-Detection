# Phase 6 — Fusion Engine & Zero-Day Simulation Summary

_Generated 2026-04-26 21:44:34 (v3 — H1 label-space bug fixed)_

## 1. Configuration

- Primary variant: **AE_p90**
- AE threshold p90: `0.2058`
- AE threshold p95: `0.3845`
- AE threshold p99: `1.3152`
- Benign class id: `1` | Zero-day pseudo-class id: `19`
- Test samples: `892,268` | Val samples: `903,016`
- Bootstrap iterations: 200 (seed=42)
- H1 evaluated in 20-label space (includes `zero_day_unknown`; E7 scored in same space for apples-to-apples comparison)
- H2 primary metric: AE recall on samples E7 misclassified as benign
- H2 sample-size guard: n_called_benign >= 30

## 2. Case Distribution (test set)

| variant   |   threshold |   case1_n |   case1_pct |   case2_n |   case2_pct |   case3_n |   case3_pct |   case4_n |   case4_pct |
|:----------|------------:|----------:|------------:|----------:|------------:|----------:|------------:|----------:|------------:|
| AE_p90    |        0.21 |    837209 |       93.83 |      6140 |        0.69 |     17317 |        1.94 |     31602 |        3.54 |
| AE_p95    |        0.38 |    834022 |       93.47 |      2236 |        0.25 |     20504 |        2.30 |     35506 |        3.98 |
| AE_p99    |        1.32 |    747621 |       83.79 |       325 |        0.04 |    106905 |       11.98 |     37417 |        4.19 |
| IF        |      nan    |    496369 |       55.63 |      4118 |        0.46 |    358157 |       40.14 |     33624 |        3.77 |

## 3. Fusion vs E7 — 20-class macro-F1 with bootstrap CI

E7 baseline macro-F1 (20-class): **0.8622** [0.8586, 0.8655] | MCC: 0.9906 | acc: 0.9927

| variant   |   macro_f1 |   macro_f1_ci_lo |   macro_f1_ci_hi |    mcc |   accuracy |   delta_f1_vs_E7 |   delta_ci_lo |   delta_ci_hi | h1_significant   |
|:----------|-----------:|-----------------:|-----------------:|-------:|-----------:|-----------------:|--------------:|--------------:|:-----------------|
| AE_p90    |     0.8582 |           0.8546 |           0.8615 | 0.9824 |     0.9862 |          -0.0041 |       -0.0042 |       -0.0040 | False            |
| AE_p95    |     0.8610 |           0.8574 |           0.8643 | 0.9878 |     0.9904 |          -0.0012 |       -0.0013 |       -0.0012 | False            |
| AE_p99    |     0.8621 |           0.8584 |           0.8654 | 0.9902 |     0.9924 |          -0.0001 |       -0.0002 |       -0.0001 | False            |
| IF        |     0.8594 |           0.8557 |           0.8626 | 0.9848 |     0.9881 |          -0.0029 |       -0.0030 |       -0.0028 | False            |

## 4. Binary Detection (Cases 1+2+3 vs Case 4)

| variant   |   accuracy |   precision |   recall |     f1 |    mcc |
|:----------|-----------:|------------:|---------:|-------:|-------:|
| E7_only   |     0.9973 |      0.9987 |   0.9985 | 0.9986 | 0.9665 |
| AE_p90    |     0.9912 |      0.9919 |   0.9989 | 0.9954 | 0.8855 |
| AE_p95    |     0.9954 |      0.9964 |   0.9988 | 0.9976 | 0.9413 |
| AE_p99    |     0.9971 |      0.9984 |   0.9986 | 0.9985 | 0.9636 |
| IF        |     0.9928 |      0.9939 |   0.9986 | 0.9963 | 0.9079 |

## 5. Simulated Zero-Day under E7-Blindness

> **Methodological note.** This is *not* leave-one-attack-out in the strict sense — E7 is trained on all 19 classes, including the 5 targets. The simulation measures: when E7 misclassifies a target attack as benign, does the AE catch it? True LOO would require retraining E7 five times (deferred to future work).

| target                 |   n_test |   e7_recall |   e7_called_benign_n |   e7_called_benign_pct | h2_sample_sufficient   |   ae_recall_on_missed_p90 |   ae_recall_on_missed_p95 |   ae_recall_on_missed_p99 |   ae_recall_p90 |   ae_recall_p95 |   ae_recall_p99 |   binary_detected_recall_p90 |   confirmed_or_zeroday_p90 |
|:-----------------------|---------:|------------:|---------------------:|-----------------------:|:-----------------------|--------------------------:|--------------------------:|--------------------------:|----------------:|----------------:|----------------:|-----------------------------:|---------------------------:|
| Recon_Ping_Sweep       |      169 |       0.710 |                   25 |                 14.793 | False                  |                     0.200 |                     0.080 |                     0.000 |           0.544 |           0.509 |           0.225 |                        0.882 |                      0.544 |
| Recon_VulScan          |      973 |       0.332 |                  535 |                 54.985 | True                   |                     0.357 |                     0.264 |                     0.049 |           0.627 |           0.568 |           0.425 |                        0.646 |                      0.627 |
| MQTT_Malformed_Data    |     1747 |       0.828 |                  159 |                  9.101 | True                   |                     0.182 |                     0.138 |                     0.006 |           0.555 |           0.461 |           0.226 |                        0.926 |                      0.555 |
| MQTT_DoS_Connect_Flood |     3131 |       0.999 |                    0 |                  0.000 | False                  |                   nan     |                   nan     |                   nan     |           1.000 |           1.000 |           0.983 |                        1.000 |                      1.000 |
| ARP_Spoofing           |     1744 |       0.710 |                  252 |                 14.450 | True                   |                     0.222 |                     0.151 |                     0.032 |           0.548 |           0.266 |           0.022 |                        0.888 |                      0.548 |

## 6. Hypothesis Verdicts

### H1 — Fusion improves macro-F1 (paired bootstrap)

- E7 baseline (20-class): 0.8622 [0.8586, 0.8655]
- Fusion (AE_p90): 0.8582 [0.8546, 0.8615]
- Δ = -0.0041 95% CI [-0.0042, -0.0040]
- Best variant (AE_p99): Δ CI [-0.0002, -0.0001]
- **Verdict (primary): FAIL — Δ CI excludes 0 (negative; fusion hurts macro-F1)**
- **Verdict (best variant): FAIL — Δ CI excludes 0 (negative; fusion hurts macro-F1)**

> 20-class macro-F1 penalises every false `zero_day_unknown` alarm equally. Binary detection (§4) is more representative of operational value.

### H2 — AE catches what E7 misses on ≥50% of zero-day targets

**Primary metric: AE recall on samples E7 misclassified as benign.**

- Targets passing (best threshold, AE-on-missed ≥ 0.7): **0/5**
  - ⚠ Recon_Ping_Sweep: insufficient samples (n_called_benign < 30); excluded
  - ✗ Recon_VulScan: best AE-on-missed = 0.357
  - ✗ MQTT_Malformed_Data: best AE-on-missed = 0.182
  - ⚠ MQTT_DoS_Connect_Flood: insufficient samples (n_called_benign < 30); excluded
  - ✗ ARP_Spoofing: best AE-on-missed = 0.222
- **Verdict: FAIL ✗**

**Auxiliary (raw AE per-class recall, Phase-5 framing):** 1/5 pass.

## 7. Threshold Sensitivity (val for selection, test for reporting)

|   percentile |   threshold |   val_attack_recall |   val_benign_fpr |   val_binary_f1 |   test_attack_recall |   test_benign_fpr |   test_binary_f1 |
|-------------:|------------:|--------------------:|-----------------:|----------------:|---------------------:|------------------:|-----------------:|
|      50.0000 |      0.0152 |              0.9997 |           0.5001 |          0.9888 |               0.9995 |            0.5355 |           0.9881 |
|      60.0000 |      0.0231 |              0.9996 |           0.4003 |          0.9910 |               0.9994 |            0.4402 |           0.9901 |
|      70.0000 |      0.0496 |              0.9996 |           0.3007 |          0.9931 |               0.9992 |            0.3530 |           0.9919 |
|      80.0000 |      0.1018 |              0.9995 |           0.2012 |          0.9953 |               0.9991 |            0.2818 |           0.9934 |
|      85.0000 |      0.1421 |              0.9994 |           0.1521 |          0.9963 |               0.9990 |            0.2500 |           0.9940 |
|      90.0000 |      0.2058 |              0.9993 |           0.1027 |          0.9974 |               0.9989 |            0.1844 |           0.9954 |
|      92.0000 |      0.2576 |              0.9993 |           0.0831 |          0.9978 |               0.9989 |            0.1451 |           0.9963 |
|      95.0000 |      0.3845 |              0.9993 |           0.0534 |          0.9984 |               0.9988 |            0.0830 |           0.9976 |
|      97.0000 |      0.5615 |              0.9992 |           0.0335 |          0.9989 |               0.9987 |            0.0529 |           0.9982 |
|      99.0000 |      1.3152 |              0.9992 |           0.0139 |          0.9993 |               0.9986 |            0.0373 |           0.9985 |

## 8. Recommended Operating Threshold

Selected on val (FPR < 0.05): **p97** (threshold = 0.5615)

- Val:  TPR = 0.9992, FPR = 0.0335
- Test: TPR = 0.9987, FPR = 0.0529, binary F1 = 0.9982

## 9. Per-class case rates (primary variant)

| class                   |   Case1_pct |   Case2_pct |   Case3_pct |   Case4_pct |   n_test |
|:------------------------|------------:|------------:|------------:|------------:|---------:|
| ARP_Spoofing            |       51.61 |        3.21 |       33.94 |       11.24 |     1744 |
| Benign                  |        2.75 |       15.40 |        0.28 |       81.56 |    37607 |
| DDoS_ICMP               |       99.59 |        0.00 |        0.41 |        0.01 |    19673 |
| DDoS_SYN                |       99.72 |        0.00 |        0.28 |        0.00 |    88921 |
| DDoS_TCP                |       98.80 |        0.00 |        1.19 |        0.01 |     8735 |
| DDoS_UDP                |       99.98 |        0.00 |        0.02 |        0.00 |   362070 |
| DoS_ICMP                |       97.56 |        0.00 |        2.43 |        0.01 |     8451 |
| DoS_SYN                 |      100.00 |        0.00 |        0.00 |        0.00 |    97542 |
| DoS_TCP                 |       99.99 |        0.00 |        0.01 |        0.00 |    42583 |
| DoS_UDP                 |       99.99 |        0.00 |        0.01 |        0.00 |   137553 |
| MQTT_DDoS_Connect_Flood |      100.00 |        0.00 |        0.00 |        0.00 |    41916 |
| MQTT_DDoS_Publish_Flood |       26.11 |        0.00 |       73.89 |        0.00 |     8416 |
| MQTT_DoS_Connect_Flood  |       99.97 |        0.00 |        0.03 |        0.00 |     3131 |
| MQTT_DoS_Publish_Flood  |        6.21 |        0.00 |       93.79 |        0.00 |     8505 |
| MQTT_Malformed_Data     |       53.86 |        1.66 |       37.03 |        7.44 |     1747 |
| Recon_OS_Scan           |       84.80 |        1.67 |        7.99 |        5.54 |     2941 |
| Recon_Ping_Sweep        |       51.48 |        2.96 |       33.73 |       11.83 |      169 |
| Recon_Port_Scan         |       95.83 |        0.08 |        3.73 |        0.36 |    19591 |
| Recon_VulScan           |       43.06 |       19.63 |        1.95 |       35.35 |      973 |

## 10. Files generated

- `fusion_results/fusion_{val,test}_cases.npy` — case arrays
- `fusion_results/fusion_{val,test}_labels.csv` — decoded (redundant w/ npy + dict, kept for inspection)
- `metrics/case_distribution.csv`
- `metrics/fusion_vs_supervised.csv` (macro-F1 + bootstrap CIs)
- `metrics/fusion_vs_supervised_binary.csv`
- `metrics/per_class_case_analysis.csv`
- `metrics/zero_day_results.csv`
- `metrics/threshold_sensitivity.csv` (val + test)
- `metrics/h1_h2_verdicts.json`
- `figures/*.png` (5 plots)
- `config.json`
