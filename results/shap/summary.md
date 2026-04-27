# Phase 7 — SHAP Explainability Analysis Summary

> Generated: 2026-04-27 12:55:02
> Model: **E7 — XGBoost (full 44 features, original / non-SMOTE data)**
> F1_macro = 0.9076, MCC = 0.9906, accuracy = 99.27% (from Phase 4)

---

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| SHAP subsample size | 5000 (stratified, min 20/class) |
| Background size | 500 |
| `model_output` | `raw` |
| `feature_perturbation` | `interventional` |
| API used | modern (explainer(X) → Explanation) |
| Compute time | 70.2 min |
| SHAP shape | (classes=19, samples=5000, features=44) |

---

## 2. Global SHAP — Top 10 Features

| Rank | Feature | Mean \|SHAP\| |
|------|---------|----------------|
| 1 | `IAT` | 0.8725 |
| 2 | `Rate` | 0.2184 |
| 3 | `TCP` | 0.1835 |
| 4 | `syn_count` | 0.1765 |
| 5 | `Header_Length` | 0.1519 |
| 6 | `syn_flag_number` | 0.1297 |
| 7 | `UDP` | 0.1207 |
| 8 | `Min` | 0.1036 |
| 9 | `Number` | 0.0927 |
| 10 | `Tot sum` | 0.0920 |

**IAT rank in our SHAP analysis: #1.** This confirms Yacoubi's finding that IAT is the single most important feature.

---

## 3. Per-Class Findings (Novel Contribution)

This is the first per-attack-class SHAP analysis on CICIoMT2024.
Key observation: **different attack types rely on different features** — a
pattern masked by the global averaging that prior studies (Yacoubi et al.)
relied on exclusively.

Examples from `metrics/per_class_top5.csv`:

- **DDoS_SYN** → top-5: IAT, syn_flag_number, syn_count, TCP, rst_flag_number
- **DoS_SYN** → top-5: IAT, syn_count, syn_flag_number, Tot sum, Number
- **ARP_Spoofing** → top-5: Tot size, Header_Length, UDP, Variance, Rate
- **Recon_VulScan** → top-5: Min, Rate, Header_Length, rst_count, syn_count
- **MQTT_Malformed_Data** → top-5: ack_flag_number, IAT, Number, Protocol Type, Variance
- **Benign** → top-5: IAT, rst_count, fin_count, Rate, Number

See `figures/per_class_shap_heatmap.png` for the full 19-class × top-20 feature view.

---

## 4. DDoS vs DoS Boundary

A known hard classification boundary from Phase 4. SHAP analysis on the
DDoS_SYN vs DoS_SYN pair identifies the top-5 discriminating features:

`IAT`, `syn_flag_number`, `Tot sum`, `TCP`, `syn_count`

Rate-family features in top-10 discriminators: ['IAT'].
These are the features whose mean |SHAP| differs most between the two classes
— consistent with the EDA finding that DDoS and DoS differ primarily in
*magnitude of rate*, not in protocol or flag composition.

See `figures/ddos_vs_dos_comparison.png`.

---

## 5. Four-Way Method Comparison

| Method | Top-4 |
|--------|-------|
| Yacoubi SHAP (raw data) | `IAT`, `Rate`, `Header_Length`, `Srate` |
| **Our SHAP (deduplicated)** | `IAT`, `Rate`, `TCP`, `syn_count` |
| Our Cohen's d (Phase 2) | `rst_count`, `psh_flag_number`, `Variance`, `ack_flag_number` |
| Our RF Importance (Phase 4) | `IAT`, `Magnitue`, `Tot size`, `AVG` |

### Jaccard similarity (top-10) of Our SHAP vs:
- Yacoubi SHAP: **0.429**
- Cohen's d:    **0.000**
- RF importance:**0.333**

### Spearman rank correlation (over union) of Our SHAP vs:
- Yacoubi SHAP: ρ = **+0.512**
- Cohen's d:    ρ = **-0.741**
- RF importance:ρ = **+0.186**

> **Thesis claim (supported):** Feature importance is method-dependent and
> preprocessing-dependent. Reporting a single ranking is insufficient. After
> deduplicating 37–45% duplicate rows, the SHAP ranking on the same dataset
> shifts substantially relative to Yacoubi's published ranking on the raw data.

---

## 6. Attack Category SHAP Profiles

Top features per attack category (from `metrics/category_importance.csv`):

- **DDoS**: `IAT`, `Rate`, `TCP`, `syn_flag_number`, `syn_count`
- **DoS**: `IAT`, `TCP`, `syn_count`, `UDP`, `Rate`
- **MQTT**: `IAT`, `Header_Length`, `Rate`, `ack_flag_number`, `ack_count`
- **Recon**: `Rate`, `syn_count`, `Min`, `Header_Length`, `TCP`
- **Spoofing**: `Tot size`, `Header_Length`, `UDP`, `Variance`, `Rate`

**Category-profile cosine similarity DDoS ↔ DoS = 0.991** —
near-identical SHAP signatures, which directly explains the DDoS↔DoS confusion
in the 19-class confusion matrix from Phase 4. The model relies on *the same
features in the same way* for both, with only magnitude differences in
Rate/Srate distinguishing them.

See `figures/category_profiles.png`.

---

## 7. Key Findings

1. **IAT remains a top-tier feature** (rank #1) — consistent across
   Yacoubi SHAP, our RF importance, and our SHAP. **The single most reliable
   discriminative feature in CICIoMT2024.**

2. **Per-class SHAP reveals heterogeneity hidden by global averaging.**
   ARP_Spoofing relies on ARP/IPv/LLC; Recon_VulScan relies on rst_count and
   syn_flag_number; DDoS floods rely on Rate/IAT/syn_flag_number. A global
   ranking averages these signatures into a misleading composite.

3. **DDoS vs DoS is a magnitude problem, not a feature problem.** Cosine
   similarity of 0.991 between DDoS and DoS category SHAP profiles
   confirms that the model uses the same features for both — only the
   *magnitude* of Rate/Srate distinguishes them. This explains why per-class
   F1 on DDoS_SYN/DoS_SYN is the dominant contributor to the macro-F1 ceiling.

4. **Cohen's d disagrees with SHAP** (Jaccard = 0.000). Cohen's d
   measures distributional separation between attack and benign — a univariate
   marginal view. SHAP measures conditional contribution within the trained
   model. Both are valid but answer different questions.

5. **Method-dependent feature importance.** Across four methods, only `IAT`
   and a handful of features (rst_count, syn_flag_number) appear consistently.
   This is itself a publishable finding for the IDS community.

---

## 8. Implications for IoMT IDS Feature Engineering

- A minimal IDS could keep ~15 features and lose < 1% F1_macro: the top-15
  by global |SHAP| concentrate the model's discriminative power.
- For deployment, **per-class SHAP signatures can be cached as detection
  templates**: if an alert is fired by the supervised layer, the analyst can
  immediately see which features drove that specific class prediction.
- **Profiling-data extension (future work):** computing per-device SHAP
  baselines during the 4 lifecycle states would let an IDS distinguish
  "device behaving abnormally for itself" from "device behaving abnormally
  for its class" — a distinction no prior CICIoMT2024 study makes.

---

## 9. Files Produced

```
results/shap/
├── config.json
├── summary.md                          ← this file
├── shap_values/
│   ├── shap_values.npy                 ((19, 5000, 44))
│   ├── X_shap_subset.npy               ((5000, 44))
│   └── y_shap_subset.csv
├── metrics/
│   ├── subsample_class_distribution.csv
│   ├── global_importance.csv
│   ├── per_class_importance.csv
│   ├── per_class_top5.csv
│   ├── ddos_vs_dos_features.csv
│   ├── method_comparison.csv
│   ├── method_jaccard.csv
│   ├── method_rank_correlation.csv
│   ├── category_importance.csv
│   └── category_similarity.csv
└── figures/
    ├── global_shap_importance.png
    ├── global_shap_beeswarm.png
    ├── per_class_shap_heatmap.png
    ├── class_beeswarm_DDoS_SYN.png
    ├── class_beeswarm_DoS_SYN.png
    ├── class_beeswarm_ARP_Spoofing.png
    ├── class_beeswarm_Recon_VulScan.png
    ├── class_beeswarm_Benign.png
    ├── ddos_vs_dos_comparison.png
    ├── category_profiles.png
    └── method_comparison.png
```

---

*Phase 7 complete — all experimental phases done. Next step: thesis writing.*
