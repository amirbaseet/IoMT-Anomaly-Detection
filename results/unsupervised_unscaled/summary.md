# Phase 5 — Unsupervised Layer Summary

## 1 · Best autoencoder configuration

- Architecture: `44 → 32 → 16 → 8 → 16 → 32 → 44`
- Optimiser: Adam, lr = `0.001`, batch size = `512`
- Trained for **98 epochs** (early-stopped, patience=10)
- Final training loss: `50368.121094`
- Best validation loss: **`101414.609375`**
- Wall-clock training time: **20.4 s** (0.34 min)

## 2 · Selected anomaly threshold

- All five candidate thresholds evaluated on the **validation** set:

| name | value | precision | recall | F1 | FPR | TPR |
|------|-------|-----------|--------|-----|------|-----|
| p90 | 100121.484375 | 0.9953 | 0.9758 | 0.9854 | 0.1027 | 0.9758 |
| p95 | 199849.843750 | 0.9976 | 0.9741 | 0.9857 | 0.0529 | 0.9741 |
| p99 | 2575986.750000 | 0.6513 | 0.0009 | 0.0017 | 0.0104 | 0.0009 |
| mean_2std | 1202433.375000 | 0.7282 | 0.0029 | 0.0057 | 0.0239 | 0.0029 |
| mean_3std | 1752942.750000 | 0.6995 | 0.0018 | 0.0036 | 0.0173 | 0.0018 |

- **Selected:** `p95` = `199849.843750` (highest F1 on val).
- Rationale: the percentile / mean+std rules are computed on benign-only validation
  errors, so they reflect the natural noise floor of normal traffic. The chosen rule
  gave the best precision-recall trade-off for binary anomaly classification on
  validation, and is therefore used as the operating point for fusion in Phase 6.

## 3 · Binary anomaly detection performance (test set)

| metric | Autoencoder | Isolation Forest |
|---|---|---|
| AUC-ROC | **0.9728** | 0.8616 |
| FPR @ 95 % TPR | 0.0172 | 0.2715 |
| anomaly precision | 0.9986 | 0.9919 |
| anomaly recall | 0.9684 | 0.5789 |
| anomaly F1 | 0.9833 | 0.7311 |

## 4 · Per-class detection rates (Autoencoder, best threshold)

**Easiest to detect:**

- `MQTT_DDoS_Connect_Flood` — recall **1.000** (n=41,916)
- `DoS_SYN` — recall **1.000** (n=97,542)
- `DoS_UDP` — recall **1.000** (n=137,553)

**Hardest to detect:**

- `Recon_Ping_Sweep` — recall **0.000** (n=169)
- `Recon_Port_Scan` — recall **0.009** (n=19,591)
- `Recon_OS_Scan` — recall **0.014** (n=2,941)

The full 19-class detection-rate table is in `metrics/per_class_detection_rates.csv`
and visualised as `figures/detection_rate_heatmap.png`.

## 5 · Autoencoder vs Isolation Forest

- The autoencoder learned a tighter benign manifold (lower benign-MSE variance) and
  therefore produces a more separable score distribution for volumetric/flooding
  attacks where `Rate`, `IAT`, and flag counts deviate sharply from benign.
- Isolation Forest tends to be more competitive on point-anomaly attacks
  (Recon, Spoofing) because it isolates rare feature combinations rather than
  measuring reconstruction error.
- **Average per-class recall:** AE = `0.6996` ·
  IF = `0.1647`
- Training cost: AE = 20.4 s · IF = 0.6 s

## 6 · Zero-day simulation (preview)

| target | n_test | AE @ p95 | IF recall |
|---|---|---|---|
| `Recon_Ping_Sweep` | 169 | 0.000 | 0.077 |
| `Recon_VulScan` | 973 | 0.023 | 0.020 |
| `MQTT_Malformed_Data` | 1,747 | 0.212 | 0.235 |
| `MQTT_DoS_Connect_Flood` | 3,131 | 0.999 | 0.008 |
| `ARP_Spoofing` | 1,744 | 0.392 | 0.429 |

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
- AE provides the primary anomaly signal via reconstruction error (stronger on volumetric/flooding attacks where Rate/IAT deviate sharply). IF provides a secondary signal that catches point anomalies AE misses (Recon, Spoofing, point flag-count outliers). Phase 6 fusion should consume both score arrays. Test AUC: AE=0.9728, IF=0.8616.
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
   benign-val MSE = `101414.601562`.
3. Per-class recall varies sharply across attack families. Volumetric / flooding
   classes (e.g. DDoS_*) are detected almost perfectly, while quieter recon
   classes are harder — motivating the hybrid design.
4. The unsupervised layer is a complement to, not a substitute for, the supervised
   XGBoost (E5 (F1_macro=0.9880)). Phase 6 fusion exploits this complementarity.

---

_Generated by `unsupervised_training.py` — random_state = 42_
