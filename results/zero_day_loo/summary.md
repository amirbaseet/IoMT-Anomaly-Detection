# Phase 6B — True Leave-One-Attack-Out Zero-Day Results

_Run started:  2026-04-26T23:30:21_  
_Run finished: 2026-04-26T23:49:37_  
_Total runtime: 19.3 min_

## 1. Setup

- Trained **5** XGBoost models, each excluding one target class entirely from training.
- Hyperparameters identical to E7; only the training data changes.
- The AE and IF were trained on benign-only data and remain **unchanged** — their scores carry over from Phase 6 unmodified.
- AE thresholds: `{'p90': 0.20127058029174805, 'p95': 0.37264156341552734, 'p99': 1.2025282382965088}`
- H2 criterion: AE recall ≥ 0.7 on ≥ 3 of 5 held-out target classes.

## 2. Per-target results

| Target | n test | LOO-E7→Benign | LOO-E7→Other | AE recall (p95) | AE on LOO-missed (p95) | Binary recall (p95) |
|---|---:|---:|---:|---:|---:|---:|
| Recon_Ping_Sweep | 169 | 18.3% | 81.7% | 0.509 | 0.065 | 0.828 |
| Recon_VulScan | 973 | 53.6% | 46.4% | 0.573 | 0.345 | 0.649 |
| MQTT_Malformed_Data | 1,747 | 27.0% | 73.0% | 0.470 | 0.258 | 0.800 |
| MQTT_DoS_Connect_Flood | 3,131 | 0.0% | 100.0% | 1.000 | — | 1.000 |
| ARP_Spoofing | 1,744 | 18.1% | 81.9% | 0.286 | 0.206 | 0.856 |

## 3. H2 re-evaluation under true LOO

**Primary verdict (strict, AE on LOO-missed @ p95): FAIL** (0/5 targets ≥ 0.7).

### Strict: AE recall on LOO-missed @ p95

**FAIL** — 0/5 targets ≥ 0.7

- ✗ `Recon_Ping_Sweep` → 0.065
- ✗ `Recon_VulScan` → 0.345
- ✗ `MQTT_Malformed_Data` → 0.258
- ✗ `MQTT_DoS_Connect_Flood` → —
- ✗ `ARP_Spoofing` → 0.206

### Strict: AE recall on LOO-missed @ p90

**FAIL** — 0/5 targets ≥ 0.7

- ✗ `Recon_Ping_Sweep` → 0.161
- ✗ `Recon_VulScan` → 0.441
- ✗ `MQTT_Malformed_Data` → 0.335
- ✗ `MQTT_DoS_Connect_Flood` → —
- ✗ `ARP_Spoofing` → 0.320

### Relaxed: AE recall on all target samples @ p95

**FAIL** — 1/5 targets ≥ 0.7

- ✗ `Recon_Ping_Sweep` → 0.509
- ✗ `Recon_VulScan` → 0.573
- ✗ `MQTT_Malformed_Data` → 0.470
- ✓ `MQTT_DoS_Connect_Flood` → 1.000
- ✗ `ARP_Spoofing` → 0.286

### Relaxed: AE recall on all target samples @ p90

**FAIL** — 1/5 targets ≥ 0.7

- ✗ `Recon_Ping_Sweep` → 0.544
- ✗ `Recon_VulScan` → 0.630
- ✗ `MQTT_Malformed_Data` → 0.558
- ✓ `MQTT_DoS_Connect_Flood` → 1.000
- ✗ `ARP_Spoofing` → 0.553

### Binary: any-alert recall (Cases 1+2+3) @ p95

**PASS** — 4/5 targets ≥ 0.7

- ✓ `Recon_Ping_Sweep` → 0.828
- ✗ `Recon_VulScan` → 0.649
- ✓ `MQTT_Malformed_Data` → 0.800
- ✓ `MQTT_DoS_Connect_Flood` → 1.000
- ✓ `ARP_Spoofing` → 0.856

## 4. Phase 6 (simulated) vs Phase 6B (true LOO)

| Target | P6 E7 | LOO E7 | P6 AE-missed (p95) | LOO AE-missed (p95) | P6 Binary (p95) | LOO Binary (p95) |
|---|---:|---:|---:|---:|---:|---:|
| Recon_Ping_Sweep | 0.710 | 0.000 | 0.080 | 0.065 | — | 0.828 |
| Recon_VulScan | 0.332 | 0.000 | 0.264 | 0.345 | — | 0.649 |
| MQTT_Malformed_Data | 0.828 | 0.000 | 0.138 | 0.258 | — | 0.800 |
| MQTT_DoS_Connect_Flood | 0.999 | 0.000 | — | — | — | 1.000 |
| ARP_Spoofing | 0.710 | 0.000 | 0.151 | 0.206 | — | 0.856 |

## 5. What does the blind LOO-E7 think held-out attacks are?

- **Recon_Ping_Sweep** → Recon_OS_Scan (44.4%), ARP_Spoofing (37.3%), Benign (18.3%)
- **Recon_VulScan** → Benign (53.6%), Recon_Port_Scan (20.7%), Recon_OS_Scan (16.3%), Recon_Ping_Sweep (8.1%), ARP_Spoofing (0.8%)
- **MQTT_Malformed_Data** → ARP_Spoofing (54.4%), Benign (27.0%), MQTT_DDoS_Connect_Flood (13.7%), Recon_OS_Scan (3.9%), Recon_Port_Scan (1.0%)
- **MQTT_DoS_Connect_Flood** → MQTT_DDoS_Connect_Flood (87.0%), MQTT_DDoS_Publish_Flood (12.7%), Recon_OS_Scan (0.2%), MQTT_Malformed_Data (0.1%)
- **ARP_Spoofing** → Recon_Port_Scan (46.6%), Recon_VulScan (26.6%), Benign (18.1%), MQTT_Malformed_Data (8.6%), Recon_Ping_Sweep (0.1%)

## 6. Discussion

Phase 6 reported H2 as FAIL (0/5) but was based on a simulated LOO — the supervised model E7 was trained on all 19 classes including each target, so the only samples it 'missed' were edge cases near a decision boundary, where the AE has the least leverage. Under true LOO, the supervised model has zero exposure to the held-out class.

With true LOO, the AE flagged ≥ 70% of LOO-E7-missed samples on **0/5** targets at p95 (and **0/5** at p90). The relaxed criterion (AE recall on *all* target samples, not just those E7 calls benign) passes on **1/5** at p95. The fused binary detector — the practical IDS metric — raised an alert (Cases 1+2+3) on **4/5** targets.

Why the gap between strict and binary criteria: when the LOO-E7 misclassifies a held-out attack as a different known attack (Case 1 or Case 3), the IDS still triggers a response — the operator sees an alert, even if the assigned class is wrong. Pure 'zero-day warnings' (Case 2: E7 says benign, AE flags anomalous) are only the subset where the supervised model entirely missed the sample. That subset is precisely what the H2 strict criterion stresses, and where the AE has to carry the full burden alone.

The LOO-E7 prediction distribution (Section 5) tells us how 'detectable as some attack' the held-out class is to a model that has never seen it: classes that get mapped to neighbouring attacks (e.g. Recon family → other Recon variants) keep binary recall high even when zero-day warnings are rare. Classes that get mapped to Benign place the entire detection load on the AE.

## 7. Implications for IoMT deployment

- The fused binary alert (any of Cases 1, 2, 3) is the metric an IoMT operator actually consumes. A misclassified-but-flagged attack still triggers triage; only Case 4 silently passes through.
- For classes the LOO-E7 confidently mislabels as a sibling attack, the system degrades gracefully — coverage stays high without the AE doing heavy lifting.
- For classes the LOO-E7 routes to Benign, the AE is the sole line of defence. Operators should treat the AE recall on the LOO-missed subset as the conservative lower bound on true zero-day coverage.

## 8. Limitations

- Single random seed for the LOO XGBoost models; per-fold variance not estimated.
- The 5 targets cover Recon, MQTT, and ARP families but exclude DDoS/DoS, where many sibling labels remain in training — those would test a harder LOO scenario.
- AE thresholds are fixed from the original benign validation; they are not re-tuned per fold, which is the conservative choice but may understate AE recall.
- The LOO-E7 still sees 18 of 19 attack classes during training, so its 'novel' decision is biased toward the closest known class. A field deployment in a new hospital would face many simultaneous unknowns.

## 9. Future work

- Repeat LOO with multiple seeds; report mean ± std on every recall figure.
- Sweep the AE threshold and plot precision–recall curves on each held-out class to characterise the operating point trade-off.
- Add a calibrated low-confidence floor on the LOO-E7 softmax (e.g. flag samples where max-prob < τ) to convert Case 3 into Case 2-style warnings — this would directly raise zero-day recall without retraining either model.
- Test a stricter LOO that holds out an entire attack family (all Recon, all MQTT) rather than a single class, to estimate cross-family generalisation.

## 10. Output index

- `metrics/loo_results.csv` — per-target metrics
- `metrics/loo_vs_phase6_comparison.csv` — side-by-side with Phase 6
- `metrics/loo_prediction_distribution.csv` — what LOO-E7 thinks held-out classes are
- `metrics/loo_case_distribution.csv` — fusion case breakdown per threshold
- `metrics/h2_loo_verdict.json` — H2 evaluation under all criteria
- `figures/loo_zero_day_results.png`
- `figures/loo_vs_phase6_comparison.png`
- `figures/loo_prediction_distribution.png`
- `figures/loo_case_distribution.png`
- `models/loo_xgb_without_*.pkl` — saved retrained models
- `predictions/loo_*_test_pred.npy`, `loo_*_test_proba.npy`
- `config.json` — full run configuration
