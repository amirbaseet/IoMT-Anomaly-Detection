# Cross-Dataset Generalization — Results Summary

**ShieldMind** · Train: CICIoMT2024 · Test: CICIoT2023 · Status: **Measured**

- Shared-feature schema: **38 features** (intersection; 7 CICIoMT2024-only features dropped, not reconstructed).
- Multi-seed: **3 seeds** (42, 43, 44); cross-dataset numbers are mean ± σ.
- Canonical reproducible sample (seed 42) saved to `sampled_test_set.parquet`.

> Caveat (per design doc §7): cross-dataset used a 38-feature shared-schema retrain; the 7 CICIoMT2024-only features (Duration, Srate, Drate, Magnitue, Radius, Covariance, Weight) were unavailable in CICIoT2023. A drop below the in-dataset baseline is the expected, honest generalization result — not a defect.

## Design A — Binary attack/benign generalization

### Supervised XGBoost (Layer 1)

| Metric | In-dataset (shared-feat, CICIoMT2024 test) | Cross-dataset (CICIoT2023) | Frozen thesis (README, 44-feat) |
|---|---|---|---|
| F1 (attack) | 0.9992 | 0.9636 ± 0.0001 | 0.9880 (best binary, macro) |
| accuracy | 0.9984 | 0.9298 ± 0.0002 | — |
| precision | 0.9992 | 0.9307 ± 0.0002 | — |
| recall | 0.9992 | 0.9990 ± 0.0000 | — |
| roc_auc | 0.9999 | 0.6904 ± 0.0003 | — |

### Unsupervised Flow AE (Layer 2) — threshold set on CICIoMT2024 benign-val

| Threshold | Metric | In-dataset | Cross-dataset (CICIoT2023) | Frozen thesis (README) |
|---|---|---|---|---|
| p90 | F1 (attack) | 0.9915 | 0.9638 ± 0.0001 | 0.9853 |
| p90 | recall | 0.9877 | 0.9995 ± 0.0000 | — |
| p99 | F1 (attack) | 0.8906 | 0.9633 ± 0.0001 | 0.9853 |
| p99 | recall | 0.8029 | 0.9984 ± 0.0000 | — |
| — | ROC-AUC | 0.9917 | 0.2586 ± 0.0010 | 0.9892 |

## Design B — 5 shared families (Benign, Spoofing, Recon, DoS, DDoS)

Out-of-scope CICIoT2023 classes (all MIRAI-*, SQLINJECTION, XSS, BACKDOOR_MALWARE, BROWSERHIJACKING, COMMANDINJECTION, UPLOADING_ATTACK, DICTIONARYBRUTEFORCE) were dropped from the B test set. MQTT was dropped from the CICIoMT2024 training set (no CICIoT2023 counterpart).

| Metric | In-dataset (shared-feat) | Cross-dataset (CICIoT2023) | Frozen thesis (README 6-class incl. MQTT) |
|---|---|---|---|
| macro-F1 (5 families) | 0.9037 | 0.1965 ± 0.0002 | 0.9363 |
| accuracy | 0.9978 | 0.2627 ± 0.0006 | — |

### Per-family F1 (cross-dataset, mean ± σ)

| Family | In-dataset F1 | Cross-dataset F1 | Cross-dataset recall | support (seed 42) |
|---|---|---|---|---|
| Benign | 0.9703 | 0.1990 ± 0.0015 | 0.2076 ± 0.0020 | 40209 |
| Spoofing | 0.5788 | 0.0372 ± 0.0003 | 0.0847 ± 0.0006 | 39772 |
| Recon | 0.9707 | 0.3421 ± 0.0016 | 0.5274 ± 0.0021 | 82546 |
| DoS | 0.9992 | 0.0131 ± 0.0008 | 0.0067 ± 0.0004 | 79553 |
| DDoS | 0.9997 | 0.3913 ± 0.0013 | 0.2957 ± 0.0010 | 240251 |

Confusion matrices: `confusion_A.csv` (binary, XGBoost), `confusion_B.csv` (5 families) — both for the canonical seed.

_Interpretation (honest generalization vs pipeline artifact) is left to the planning chat, per the brief._

<!-- ADDENDUM:START -->

## Addendum 1 — Design A base-rate diagnostics (Measured)

The ~0.96 attack F1 is a **base-rate artifact**: CICIoT2023 is ~93% attack, so flagging nearly everything as attack scores high F1/recall. **ROC-AUC is the headline transfer metric**, and benign recall / balanced accuracy expose the collapsed benign side.

| Detector | Benign recall | Attack recall | Balanced acc | ROC-AUC |
|---|---|---|---|---|
| XGBoost (cross) | 0.0005 ± 0.0000 | 0.9990 ± 0.0000 | 0.4998 ± 0.0000 | 0.6904 ± 0.0003 |
| XGBoost (in-dataset) | 0.9650 | 0.9992 | 0.9821 | 0.9999 |
| Flow AE p90 (cross) | 0.0000 ± 0.0000 | 0.9995 ± 0.0000 | 0.4998 ± 0.0000 | 0.2586 ± 0.0010 |
| Flow AE p99 (cross) | 0.0007 ± 0.0001 | 0.9984 ± 0.0000 | 0.4996 ± 0.0001 | 0.2586 ± 0.0010 |

**Finding — AE inversion.** The Flow AE cross-dataset ROC-AUC is **0.2586 ± 0.0010**, i.e. **below 0.5**: under the CICIoMT2024-benign-trained autoencoder, CICIoT2023 *attacks* reconstruct with **lower** error than CICIoT2023 *benign*. The reconstruction-error signal is inverted across testbeds — flooding traffic sits closer to the CICIoMT2024 benign manifold than CICIoT2023's own benign does. Verified not a scaling artifact: the standardizer and imputer are fit on CICIoMT2024 only and applied transform-only to CICIoT2023 (no `.fit`/`.fit_transform` ever touches the test set).

## Addendum 2 — Design B robustness: DoS+DDoS merged into 'Flooding' (4 families, Measured)

The DoS vs DDoS distinction is effectively per-flow-impossible (it is a property of the campaign, not the individual flow), so collapsing them into one **Flooding** family is the fairer transfer measure. Computed post-hoc by merging DoS+DDoS in both truth and prediction of the existing 5-family model — no penalty for DoS↔DDoS confusion.

| Metric | In-dataset (shared-feat) | Cross-dataset (CICIoT2023) |
|---|---|---|
| macro-F1 (4 families) | 0.8800 | 0.2578 ± 0.0001 |
| accuracy | 0.9982 | 0.3233 ± 0.0004 |

Per-family F1 (4-family):

| Family | In-dataset F1 | Cross-dataset F1 |
|---|---|---|
| Benign | 0.9703 | 0.1990 ± 0.0015 |
| Spoofing | 0.5788 | 0.0372 ± 0.0003 |
| Recon | 0.9707 | 0.3421 ± 0.0016 |
| Flooding | 1.0000 | 0.4530 ± 0.0008 |

For reference, the 5-family macro-F1 was 0.1965 ± 0.0002; merging DoS+DDoS moves it to 0.2578 ± 0.0001 — the delta quantifies how much of the Design-B degradation was DoS↔DDoS confusion versus genuine non-transfer.

<!-- ADDENDUM:END -->
