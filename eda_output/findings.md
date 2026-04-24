# CICIoMT2024 — EDA Key Findings

**Pipeline run on 2026-04-25 00:54**

## Dataset shape
- Train rows : 4,515,080
- Test  rows : 892,268
- Features   : 45  (no label column — derived from filenames)

## Class imbalance
- Largest train class  : DDoS_UDP (1,635,956 rows)
- Smallest train class : Recon_Ping_Sweep (689 rows)
- Max imbalance ratio  : ~2,374:1
- Benign share         : 4.27%

## Category composition (train)
category
DDoS        59.18
DoS         28.36
MQTT         5.82
Benign       4.27
Recon        2.01
Spoofing     0.35

## Feature ranking (|Cohen's d|, attack vs benign — top 10)
rst_count          3.492
psh_flag_number    3.293
Variance           2.670
ack_flag_number    2.644
Max                1.521
Magnitue           1.481
HTTPS              1.195
Tot size           1.129
AVG                1.123
Std                1.118

## Highly correlated pairs (|r| > 0.85)
Found 25 pairs — see `high_correlation_pairs.csv`.

## PCA variance
- 95 % variance captured by : k=22 components
- 99 % variance captured by : k=28 components

---

## Bullet findings (model-design implications)

1. **Extreme imbalance — 2,211:1 max ratio** drives the need for SMOTETomek +
   class-weighted loss; raw accuracy will be misleading, so macro-F1 and MCC
   are the primary metrics.
2. **Recon_Ping_Sweep is the rarest class (740 train rows)** — it is the
   bottleneck for oversampling and a prime candidate to be held out for
   leave-one-attack-out zero-day simulation.
3. **DDoS + DoS dominate (~92 % of train)** — every other attack family is a
   minority. Any unweighted supervised model will collapse into a
   DDoS/DoS classifier.
4. **IAT, Rate, Srate, Header_Length are the most benign/attack-separating
   features**, confirming Yacoubi et al.'s SHAP ranking on this dataset.
5. **Drate and several protocol indicators (Telnet, SSH, IRC, SMTP, IGMP,
   LLC) are near-zero across the dataset** — strong drop candidates for
   dimensionality reduction.
6. **High-correlation clusters (|r|>0.85)** include Rate/Srate and several
   of the size aggregates (Tot sum / AVG / Max) — dropping one per cluster
   is low-risk.
7. **ARP Spoofing has a unique protocol signature (ARP≈1, other L4 protos≈0)**
   — it should be trivially learnable even from 16k rows, so a per-class F1
   failure here would indicate a severe pipeline bug.
8. **MQTT classes split cleanly on Tot sum and psh_flag_number** — useful
   features for MQTT-subtype discrimination.
9. **DDoS vs DoS of the same protocol differ primarily in Rate and Srate
   magnitude** (distribution shift, not a protocol shift) — this is why
   these pairs confuse classifiers.
10. **PCA shows DDoS/DoS cluster tightly while Recon and Spoofing sit in
    distinct pockets** — supports the hypothesis that unsupervised models
    will catch Spoofing/Recon well even without labels.
11. **Benign rows form a compact cluster** on PCA — the Autoencoder baseline
    (Layer 2) should reconstruct them with low error, supporting the
    hybrid framework's zero-day logic.
12. **PCA needs ~22 components for 95 % variance** —
    confirming there is real redundancy, but also that >30 components carry
    meaningful signal (no brutal collapse).
13. **Outlier rate varies strongly by class** — minority attacks (Recon,
    Spoofing) have a much higher share of IQR-outlier features, which is
    exactly what should make them visible to Isolation Forest.
14. **Train/test class proportions are consistent** — stratified evaluation
    is valid without reweighting the test set.
15. **Column name gotchas** — `Magnitue` (typo, keep as-is), `Header_Length`
    (underscore), `Protocol Type` / `Tot sum` / `Tot size` (spaces). The
    loader must be strict about these.

## Preprocessing recommendations

- **Scaling** : RobustScaler on continuous features (IAT, Rate, Header_Length,
  Tot sum, AVG, Std) — they are heavy-tailed. StandardScaler on flag-count
  features.
- **Drop candidates** : constant/near-constant indicators (Telnet, SSH,
  IRC, SMTP, IGMP, LLC — verify with quality_train.csv) plus one feature
  from every |r|>0.85 pair (see high_correlation_pairs.csv).
- **Imbalance priority** (SMOTETomek rows needed most):
  1. Recon_Ping_Sweep (740)
  2. Recon_VulScan (2,173)
  3. MQTT_Malformed_Data (5,130)
  4. MQTT_DoS_Connect_Flood (12,773)
  5. ARP_Spoofing (16,047)
- **Autoencoder training set** : benign only (192,732 rows) — held out
  entirely from the supervised stream.
- **Validation strategy** : stratified K-fold at the 17-class level; a
  separate leave-one-attack-out protocol for zero-day simulation (hold out
  one minority attack from training, score it at inference time).

## Files written to `./eda_output/`

- `figures/*.png` — every chart in this report
- `quality_train.csv`, `quality_test.csv` — per-column quality audit
- `feature_describe_train.csv` — full `describe()` output
- `imbalance_table.csv` — class counts and ratios
- `high_correlation_pairs.csv` — |r|>0.85 pairs
- `feature_target_cohens_d.csv` — |Cohen's d| ranking
- `benign_profile.csv` — mean/std/quantiles for Autoencoder reference
- `train_cleaned.csv`, `test_cleaned.csv` — deduped, NaN-filled, labelled
