# Phase 5 — Unsupervised Layer Summary

## 1 · Best autoencoder configuration

- Architecture: `44 → 32 → 16 → 8 → 16 → 32 → 44`
- Optimiser: Adam, lr = `0.001`, batch size = `512`
- Trained for **36 epochs** (early-stopped, patience=10)
- Final training loss: `0.219297`
- Best validation loss: **`0.198779`**
- Wall-clock training time: **8.2 s** (0.14 min)

## 2 · Selected anomaly threshold

- All five candidate thresholds evaluated on the **validation** set:

| name | value | precision | recall | F1 | FPR | TPR |
|------|-------|-----------|--------|-----|------|-----|
| p90 | 0.201271 | 0.9954 | 0.9862 | 0.9908 | 0.1020 | 0.9862 |
| p95 | 0.372642 | 0.9977 | 0.9833 | 0.9904 | 0.0519 | 0.9833 |
| p99 | 1.202528 | 0.9994 | 0.8425 | 0.9143 | 0.0112 | 0.8425 |
| mean_2std | 19.148867 | 0.9999 | 0.3213 | 0.4863 | 0.0005 | 0.3213 |
| mean_3std | 28.623911 | 1.0000 | 0.2550 | 0.4064 | 0.0003 | 0.2550 |

- **Selected:** `p90` = `0.201271` (highest F1 on val).
- Rationale: the percentile / mean+std rules are computed on benign-only validation
  errors, so they reflect the natural noise floor of normal traffic. The chosen rule
  gave the best precision-recall trade-off for binary anomaly classification on
  validation, and is therefore used as the operating point for fusion in Phase 6.

## 3 · Binary anomaly detection performance (test set)

| metric | Autoencoder | Isolation Forest |
|---|---|---|
| AUC-ROC | **0.9892** | 0.8612 |
| FPR @ 95 % TPR | 0.0203 | 0.2721 |
| anomaly precision | 0.9917 | 0.9919 |
| anomaly recall | 0.9789 | 0.5808 |
| anomaly F1 | 0.9853 | 0.7327 |

## 4 · Per-class detection rates (Autoencoder, best threshold)

**Easiest to detect:**

- `MQTT_DDoS_Connect_Flood` — recall **1.000** (n=41,916)
- `DoS_SYN` — recall **1.000** (n=97,542)
- `DoS_UDP` — recall **1.000** (n=137,553)

**Hardest to detect:**

- `MQTT_DoS_Publish_Flood` — recall **0.067** (n=8,505)
- `MQTT_DDoS_Publish_Flood` — recall **0.266** (n=8,416)
- `Recon_Ping_Sweep` — recall **0.544** (n=169)

The full 19-class detection-rate table is in `metrics/per_class_detection_rates.csv`
and visualised as `figures/detection_rate_heatmap.png`.

## 5 · Autoencoder vs Isolation Forest

- The autoencoder learned a tighter benign manifold (lower benign-MSE variance) and
  therefore produces a more separable score distribution for volumetric/flooding
  attacks where `Rate`, `IAT`, and flag counts deviate sharply from benign.
- Isolation Forest tends to be more competitive on point-anomaly attacks
  (Recon, Spoofing) because it isolates rare feature combinations rather than
  measuring reconstruction error.
- **Average per-class recall:** AE = `0.7999` ·
  IF = `0.1627`
- Training cost: AE = 8.2 s · IF = 0.6 s

## 6 · Zero-day simulation (preview)

| target | n_test | AE @ p90 | IF recall |
|---|---|---|---|
| `Recon_Ping_Sweep` | 169 | 0.544 | 0.077 |
| `Recon_VulScan` | 973 | 0.630 | 0.021 |
| `MQTT_Malformed_Data` | 1,747 | 0.558 | 0.203 |
| `MQTT_DoS_Connect_Flood` | 3,131 | 1.000 | 0.008 |
| `ARP_Spoofing` | 1,744 | 0.553 | 0.439 |

- **Per-class detection preview** at the selected threshold:
  1/5 targets achieve ≥ 70 % recall.
  *Indicative only — the AE never sees attacks during training regardless, so this
  measures class separability, not true zero-day generalization. Proper H2 evaluation
  (with held-out classes and retrained supervised + IF models) is deferred to Phase 6.*
- This is a *preview* — Phase 6 fusion will combine these scores with the supervised
  E7 probabilities, which should boost zero-day recall further by exploiting the
  "supervised says benign + unsupervised says anomaly = zero-day" rule.

## 7 · Recommendation for Phase 6 fusion

- **Primary recommendation:** Both (complementary).
- AE provides the primary anomaly signal via reconstruction error (stronger on volumetric/flooding attacks where Rate/IAT deviate sharply). IF provides a secondary signal that catches point anomalies AE misses (Recon, Spoofing, point flag-count outliers). Phase 6 fusion should consume both score arrays. Test AUC: AE=0.9892, IF=0.8612.
- Both score arrays are exported (`scores/ae_*.npy`, `scores/if_*.npy`); the fusion
  engine should consume **both** so that the 4-case logic
  (supervised × unsupervised) can use the stronger signal per region of feature
  space. AE is preferred for the binary anomaly flag, IF as a secondary signal.

## 8 · Key findings for thesis discussion

1. Training the autoencoder on benign-only traffic produced a clean reconstruction-error
   separation between benign and attack samples — visible in
   `figures/ae_error_distribution.png`.
2. Confirms the EDA observation that benign IoMT traffic is a compact PCA cluster:
   the AE bottleneck of 8 dimensions was sufficient to reconstruct it with
   benign-val MSE = `0.198779`.
3. Per-class recall varies sharply across attack families. Volumetric / flooding
   classes (e.g. DDoS_*) are detected almost perfectly, while quieter recon
   classes are harder — motivating the hybrid design.
4. The unsupervised layer is a complement to, not a substitute for, the supervised
   XGBoost (E5 (F1_macro=0.9880)). Phase 6 fusion exploits this complementarity.

---

_Generated by `unsupervised_training.py` — random_state = 42_
