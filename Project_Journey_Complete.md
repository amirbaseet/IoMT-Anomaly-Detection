# IoMT Anomaly Detection — Complete Project Journey

> **Student:** Amro — M.Sc. Artificial Intelligence and Machine Learning in Cybersecurity, Sakarya University
> **Thesis:** "A Hybrid Supervised-Unsupervised Framework for Anomaly Detection and Zero-Day Attack Identification in IoMT Networks Using the CICIoMT2024 Dataset"
> **Duration:** April 2026 | **Machine:** MacBook Air M4, 24GB RAM, Python 3.13
> **GitHub:** github.com/amirbaseet/IoMT-Anomaly-Detection
> **Status:** ALL EXPERIMENTAL PHASES COMPLETE — Thesis writing phase

---

## The Big Picture

We built a 4-layer hybrid intrusion detection system for medical IoT (IoMT) networks:

1. **Layer 1 (Supervised):** XGBoost classifies 19 known attack types → F1_macro = 0.9076
2. **Layer 2 (Unsupervised):** Autoencoder detects anomalies by learning "normal" traffic → AUC = 0.9892
3. **Layer 3 (Fusion):** 4-case decision logic combining both layers → binary F1 = 0.9985
4. **Layer 4 (XAI):** Per-class SHAP explains why the model flags each attack type differently

We tested 3 hypotheses — all "failed" in their strict form, but produced genuine research contributions more valuable than simple "pass" results:

| Hypothesis | Claim | Result | What we learned |
|-----------|-------|--------|----------------|
| **H1** | Fusion improves macro-F1 over XGBoost | **FAIL** (Δ = −0.0001) | Value is in case stratification, not aggregate metrics |
| **H2 strict** | AE catches ≥70% of zero-day | **FAIL** (0/5 targets) | AE and E7 share feature space → overlapping blind spots |
| **H2 binary** | Hybrid system detects novel attacks | **PASS** (5/5 at p90) | Novel mechanism: "redundancy through misclassification" |
| **H3** | SMOTETomek improves minority F1 | **FAIL** (all 4 configs worse) | class_weight='balanced' + SMOTE = compounding correction |

---

## Phase 1 — Literature Review & Problem Definition

### What we did
- Analyzed the CICIoMT2024 dataset paper (Dadkhah et al., 2024) in depth — 40 IoMT devices, 18 attack types, 45 features, WiFi/MQTT/BLE protocols
- Critically reviewed the Yacoubi et al. trilogy (3 papers, 2025-2026) — the primary reference achieving 99.87% accuracy
- Surveyed 12 total studies using CICIoMT2024 and 10 related IoMT IDS works
- Identified 7 research gaps no prior study addressed simultaneously
- Formalized 3 research questions, 3 hypotheses, and 5 research objectives
- Evaluated 3 research directions — chose Option A: Hybrid Framework

### Key findings
- No study uses profiling data — the most unique feature of CICIoMT2024 remains unexploited
- No study combines supervised + unsupervised on this dataset
- No study does per-attack-class SHAP — Yacoubi only did global SHAP
- Yacoubi's 99.87% accuracy masks 86.10% macro-precision → systematic minority-class failures

### Problems encountered
- **None** — desk research phase

### Output
- Literature_Review_Chapter2.md, Thesis_Proposal_IoMT_Research_Directions.md, yacoubi_critical_review.md, CICIoMT2024_Deep_Analysis.md

---

## Phase 2 — Exploratory Data Analysis (EDA)

### What we did
- Downloaded CICIoMT2024 dataset (72 CSV files), loaded 8.78 million raw rows
- Checked for duplicates, missing values, correlations, PCA, Cohen's d effect sizes
- Generated 15+ publication-quality figures

### Key findings

**Discovery #1: 37% of training data is exact duplicates (FIRST REPORT EVER)**
- Training: 7.16M → 4.52M rows (36.95% duplicates); Test: 1.61M → 892K (44.72%)
- No prior paper reported this — Yacoubi's 99.87% partly inflated by data leakage

**Discovery #2: Real class distribution differs from literature**
- Rarest class: Recon_Ping_Sweep (689 rows), NOT ARP_Spoofing
- Maximum imbalance: 2,374:1 (DDoS_UDP vs Recon_Ping_Sweep)

**Discovery #3: Cohen's d contradicts Yacoubi's SHAP — ZERO overlap in top-4**

**Discovery #4: Benign traffic forms compact PCA cluster** — promising for Autoencoder

### Problems encountered

| Problem | Fix |
|---------|-----|
| Dataset download requires session cookies | wget with cookies + DownThemAll fallback |
| Literature reports wrong class counts | Verified from raw CSVs, documented corrections |

### Output
- `eda_output/` — findings.md, 15+ figures, train_cleaned.csv (4.52M), test_cleaned.csv (892K)

---

## Phase 3 — Preprocessing & Feature Engineering

### What we did
- Two feature variants: Full (44) and Reduced (28)
- 3-group ColumnTransformer scaling: RobustScaler + StandardScaler + MinMaxScaler
- Stratified 80/20 split: 3,612,064 train / 903,016 val / 892,268 test
- SMOTETomek on 8 minority classes (boosted to ~50K each)
- Benign-only AE data: 123,348 train + 30,838 val
- 5 leave-one-attack-out zero-day datasets

### Problems encountered

| Problem | Severity | Fix |
|---------|----------|-----|
| Runtime 228 minutes | Low | Ran overnight with caffeinate |
| RobustScaler leaves features with std>1000 | **Critical** | Not detected here — broke AE in Phase 5. Fixed then with StandardScaler |

### Output
- `preprocessed/` — 5.7 GB of .npy arrays, config.json, label_encoders.json, scalers

---

## Phase 4 — Supervised Model Training (Layer 1)

### What we did
- 8 experiments × 3 tasks = 24 training runs (RF vs XGBoost × Full vs Reduced × Original vs SMOTE)
- Evaluated with macro-F1, MCC, accuracy, per-class F1

### Key results

**Winner: E7 (XGBoost / full / original) — F1_macro=0.9076, acc=99.27%**

**H3 REJECTED: SMOTETomek degraded ALL 4 configs** (−0.01 to −0.04 F1)

**Full features (44) beat reduced (28) consistently** — correlation-based dropping too aggressive

**IAT confirmed #1 feature** (RF importance = 0.14)

### Problems encountered

| Problem | Severity | Fix |
|---------|----------|-----|
| RF max_depth=None → estimated 8-15 hrs | High | Capped at 30 → runtime 60 min total |
| verbose=1 spams thousands of lines | Low | Set verbose=0 |
| joblib compress=3 adds ~1 hour | Medium | Changed to compress=0 |
| First run interrupted (Ctrl+C) | Low | Resume logic + caffeinate on restart |
| class_weight='balanced' + SMOTE double-corrects | Medium | Documented as thesis finding (H3 FAIL) |

### Output
- `results/supervised/` — 8 models, 24 metrics, comparison tables, confusion matrices. Runtime: 60 min

---

## Phase 5 — Unsupervised Model Training (Layer 2)

### What we did
- Autoencoder (44→32→16→8→16→32→44) trained on 123K benign-only rows
- Isolation Forest (200 trees) trained on same benign data
- 5 threshold candidates evaluated (p90, p95, p99, mean+2σ, mean+3σ)
- Per-class detection rates for all 19 classes

### Key results (after scaling fix)
- AE test AUC: **0.9892** vs IF: 0.8612
- AE per-class avg recall: **0.80** vs IF: 0.16
- High detection (>95%): All DDoS/DoS floods
- Medium (50-87%): Recon, ARP_Spoofing, MQTT_Malformed
- Low (<30%): MQTT Publish floods

### Problems encountered

| Problem | Severity | Fix |
|---------|----------|-----|
| **🔴 AE loss in the MILLIONS (101,414 at best)** | **CRITICAL** | **StandardScaler fitted on benign-train** |
| Recon detection 0-2% (invisible to AE) | Critical | Same fix — scaling equalized feature contributions |
| Python 3.14 incompatible with TF 2.21 | Medium | Downgraded to Python 3.13 |
| Fat-tailed benign MSE (std=9.48 vs mean=0.20) | Medium | Percentile thresholds instead of mean+kσ |
| p90 has 18.6% FPR on benign | Medium | Noted p99 for fusion; all thresholds saved |
| Per-class recall accidentally includes Benign FPR | Medium | Filtered Benign before averaging |

**The scaling fix was the biggest technical problem in the entire project:**

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| AE val loss | 101,414 | **0.199** |
| AE test AUC | 0.9728 | **0.9892** |
| Recon_Ping_Sweep recall | 0.000 | **0.544** |
| Recon_OS_Scan recall | 0.014 | **0.865** |
| Per-class avg recall | 0.700 | **0.800** |

**Root cause:** Phase 3's RobustScaler preserves heavy tails (good for trees, bad for AE). Features like Covariance (std=5005) dominated the MSE loss. XGBoost is scale-invariant so Phase 4 was unaffected.

### Output
- `results/unsupervised/` — AE model, IF model, scaler.pkl, 8 score arrays, 7 figures. Runtime: 34 sec

---

## Phase 6 — Fusion Engine (Layer 3)

### What we did
- Implemented 4-case fusion: Confirmed Alert / Zero-Day Warning / Low-Confidence / Clear
- Applied at 3 thresholds (p90, p95, p99) + IF baseline
- H1 evaluation with paired bootstrap (200 iterations, 95% CI)
- Simulated zero-day on 5 targets
- 10-point threshold sensitivity sweep

### Key results
- Case distribution (p90): 93.83% Case 1, 0.69% Case 2, 1.94% Case 3, 3.54% Case 4
- **H1: FAIL** — Δ = −0.0001 at p99. Zero_day_unknown class penalizes macro-F1
- **H2 (simulated): FAIL** — 0/5 targets. But this wasn't true LOO → led to Phase 6B
- Binary F1 = 0.9985 at p99 — system works for detection
- Recommended operating point: p97 (99.87% recall, 5.3% FPR)

### Problems encountered

| Problem | Severity | Fix |
|---------|----------|-----|
| H1 label space bug (E7 in 19-class vs fusion in 20-class) | High | v3: both in 20-class space |
| Case 2 precision ~6% (94% are false alarms on benign) | Medium | Fundamental limitation; documented |
| Simulated LOO isn't real zero-day test | High | Created Phase 6B |

### Output
- `results/fusion/` — case arrays, metrics, h1_h2_verdicts.json, 5 figures. Runtime: ~1 min

---

## Phase 6B — True Leave-One-Attack-Out Zero-Day

### What we did
- Retrained XGBoost 5 times, each time EXCLUDING one target class entirely
- AE/IF NOT retrained (benign-only — unaffected by removing attack classes)
- Measured: when the blind LOO-E7 misses a novel attack, does the AE catch it?

### Why this was needed
Phase 6 was like testing a doctor on a disease they already studied. Phase 6B removes the disease from medical school entirely.

### Key results

**H2 Strict: FAIL (0/5)** — AE catches only 6-44% of samples LOO-E7 calls "Benign"

**H2 Binary: PASS (5/5 at p90)** — System alerts on ≥70% of novel attacks

**🔑 Key discovery: "Redundancy through misclassification"**
LOO-E7 doesn't call novel attacks "benign" — it maps them to the closest known class:
- Recon_Ping_Sweep → 82% mapped to Recon_OS_Scan/ARP_Spoofing (18% benign)
- MQTT_DoS_Connect → 100% mapped to MQTT_DDoS_Connect (0% benign!)
- ARP_Spoofing → 82% mapped to Recon_Port_Scan/Recon_VulScan (18% benign)

The IDS fires a wrong-class alert, but still an alert. Detection via feature-space proximity, not AE novelty detection.

**Stress test:** Recon_VulScan — 53.6% routed to Benign (worst case), binary recall = 0.700 exactly at p90.

### Problems encountered

| Problem | Fix |
|---------|-----|
| Runtime estimate 5 hours | Actually 19 minutes (each XGBoost ~4 min, not 60) |
| Label space changes per fold (18 classes instead of 19) | Per-fold encoder + inverse mapping |

### Output
- `results/zero_day_loo/` — 5 LOO models, predictions, metrics, 4 figures. Runtime: 19.3 min

---

## Phase 7 — SHAP Explainability (Layer 4)

### What we did
- TreeSHAP on E7 (XGBoost) with 5,000 stratified test samples, 500 background
- Computed 19 classes × 5,000 samples × 44 features = 4.18M SHAP values
- Global SHAP importance (top-20 + beeswarm)
- **Per-class SHAP analysis — 19 separate importance profiles (NOVEL — first on CICIoMT2024)**
- DDoS vs DoS boundary analysis
- Four-way feature importance comparison (Our SHAP vs Yacoubi SHAP vs Cohen's d vs RF Importance)
- Attack category profiles with cosine similarity matrix
- 11 publication-quality figures

### Key results

**IAT confirmed #1 (mean |SHAP| = 0.87, 4× stronger than #2)**

**Per-class SHAP reveals hidden heterogeneity (NOVEL):**
| Attack | Top feature | What drives detection |
|--------|------------|----------------------|
| DDoS_UDP | IAT (5.45!) | Extreme timing deviation |
| DDoS_SYN | syn_flag_number (0.96) | SYN flood signature |
| ARP_Spoofing | Tot size (0.34) | Packet structure anomaly |
| Recon_VulScan | Min (0.37) | Scan pattern (small probes) |
| MQTT_Malformed | ack_flag_number (0.31) | Flag anomaly |
| Benign | IAT (0.36) | Normal connection lifecycle |

Global averaging masks these distinct per-class signatures entirely.

**DDoS vs DoS: cosine similarity = 0.991** — near-identical SHAP profiles. Only IAT magnitude differs. Directly explains the confusion boundary.

**Four-way comparison — feature importance is method-dependent:**
- Our SHAP vs Yacoubi SHAP: Jaccard = **0.429** (moderate — shifted by deduplication)
- Our SHAP vs Cohen's d: Jaccard = **0.000** (ZERO overlap!)
- Our SHAP vs Cohen's d: Spearman ρ = **−0.741** (NEGATIVE correlation!)

**Thesis finding:** Statistical separation ≠ model reliance. Reporting one ranking is insufficient.

### Problems encountered

| Problem | Fix |
|---------|-----|
| SHAP computation took 70 min (expected 15-30) | Ran with caffeinate — completed fine |
| SHAP API version differences (old list vs new Explanation) | Script detects and handles both |

### Output
- `results/shap/` — shap_values.npy (19×5000×44), 10 metrics CSVs, 11 figures. Runtime: 70.3 min

---

## Complete List of Thesis Contributions

1. **First hybrid supervised-unsupervised framework on CICIoMT2024** — no prior work combines XGBoost + Autoencoder with structured fusion logic
2. **37% duplication discovery** — first report ever; explains inflated metrics in all prior literature
3. **SMOTETomek + class_weight='balanced' shown harmful** — contradicts common IoMT IDS assumption
4. **Corrected class distribution** — Recon_Ping_Sweep rarest (not ARP_Spoofing), real ratio 2,374:1
5. **"Redundancy through misclassification"** — novel zero-day mechanism: supervised model maps unknown attacks to similar known classes
6. **AE reconstruction-error insufficient for zero-day on flow features** — genuine negative finding redirecting future research
7. **Per-attack-class SHAP analysis** — first on CICIoMT2024; reveals heterogeneous feature importance masked by global averaging
8. **DDoS↔DoS boundary explained** — cosine similarity 0.991 proves identical feature reliance
9. **Feature importance is method-dependent** — Cohen's d vs SHAP: Jaccard=0.000, Spearman ρ=−0.741
10. **Confidence-stratified alerts (4-case fusion)** — operational value beyond binary IDS
11. **StandardScaler fix for AE on ColumnTransformer output** — practical ML pipeline lesson

---

## Total Compute Time

| Phase | Task | Runtime |
|-------|------|---------|
| Phase 2 | EDA | ~15 min |
| Phase 3 | Preprocessing | 228 min |
| Phase 4 | Supervised (24 runs) | 60 min |
| Phase 5 | Unsupervised (AE + IF) | 34 sec |
| Phase 6 | Fusion engine | ~1 min |
| Phase 6B | True LOO (5 XGBoost) | 19.3 min |
| Phase 7 | SHAP analysis | 70.3 min |
| **Total** | | **~6.5 hours** |

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 (downgraded from 3.14 for TF 2.21) |
| Supervised | XGBoost 2.x, scikit-learn (Random Forest, Isolation Forest) |
| Deep Learning | TensorFlow 2.21 (Keras) |
| XAI | SHAP 0.45+ (TreeExplainer) |
| Imbalance | imbalanced-learn (SMOTETomek) |
| Scaling | scikit-learn (StandardScaler, RobustScaler, MinMaxScaler) |
| Data | NumPy, Pandas, Matplotlib, Seaborn |
| Hardware | MacBook Air M4, 24GB RAM (CPU only) |
| Dataset | CICIoMT2024 (Canadian Institute for Cybersecurity) |

---

## File Structure

```
~/IoMT-Project/
├── notebooks/
│   ├── eda.py                        # Phase 2
│   ├── preprocessing_pipeline.py     # Phase 3
│   ├── supervised_training.py        # Phase 4
│   ├── unsupervised_training.py      # Phase 5
│   ├── fusion_engine.py              # Phase 6
│   ├── loo_zero_day.py               # Phase 6B
│   └── shap_analysis.py             # Phase 7
├── preprocessed/                     # 5.7 GB
├── results/
│   ├── supervised/                   # 8 models, 24 metrics
│   ├── unsupervised/                 # AE + IF + scaler
│   ├── fusion/                       # 4-case fusion
│   ├── zero_day_loo/                 # 5 LOO models
│   └── shap/                         # SHAP values + 11 figures
├── eda_output/                       # 15+ EDA figures
└── README.md                         # 1,757 lines, 24 sections
```

---

_Last updated: April 27, 2026 — ALL EXPERIMENTAL PHASES COMPLETE_
_Next step: Thesis writing_
