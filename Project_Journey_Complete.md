# IoMT Anomaly Detection — Complete Project Journey

> **Student:** Amro — M.Sc. Artificial Intelligence and Machine Learning in Cybersecurity, Sakarya University
> **Thesis:** "A Hybrid Supervised-Unsupervised Framework for Anomaly Detection and Zero-Day Attack Identification in IoMT Networks Using the CICIoMT2024 Dataset"
> **Duration:** April 2026 | **Machine:** MacBook Air M4, 24GB RAM, Python 3.13
> **GitHub:** github.com/amirbaseet/IoMT-Anomaly-Detection
> **Status:** ALL EXPERIMENTAL PHASES COMPLETE (1, 2, 3, 4, 5, 6, 6B, 6C, 7) + senior review fixes applied + Path B Tier 1 hardening complete (Weeks 1, 2A, 2B; senior review §1.2 / §1.4 / §1.5 closed) — Thesis writing phase

---

## The Big Picture

We built a 4-layer hybrid intrusion detection system for medical IoT (IoMT) networks:

1. **Layer 1 (Supervised):** XGBoost classifies 19 known attack types → F1_macro = 0.9076, accuracy = 99.27%
2. **Layer 2 (Unsupervised):** Autoencoder detects anomalies by learning "normal" traffic → AUC = 0.9892
3. **Layer 3 (Fusion):** 5-case decision logic combining both layers → binary F1 = 0.9985
4. **Layer 4 (XAI):** Per-class SHAP explains why the model flags each attack type differently

We tested 3 hypotheses, then improved the system in Phase 6C:

| Hypothesis | Claim | Phase 6 result | Phase 6C result | What we learned |
|---|---|---|---|---|
| **H1** | Fusion improves macro-F1 over XGBoost | Δ = −0.014 pp | unchanged | Operationally negligible (~125 of 892,268 rows); 95% CI [−0.0166, −0.0117] excludes zero but magnitude doesn't matter. Reframed as "no operationally meaningful difference," not "FAIL." |
| **H2 strict** | AE catches ≥70% of zero-day | 0/5 FAIL | **4/4 PASS** | Entropy + AE fusion lifts strict avg from 0.314 to 0.804. Entropy alone is insufficient (Recon_VulScan TPR = 0.473 < 0.50); AE channel rescues that target. |
| **H2 binary** | Hybrid system detects novel attacks | 5/5 PASS | 5/5 PASS | Novel mechanism: "redundancy through misclassification." 82.7% of novel attacks routed to similar known attacks, not "Benign." |
| **H3** | SMOTETomek improves minority F1 | 0/5 FAIL | unchanged | Mechanism is **boundary-blur on overlapping DDoS↔DoS and Recon↔Recon classes**, NOT class_weight interaction (XGBoost has no class_weight, yet degrades more than RF). |

**Key result:** H2-strict goes from 0/4 (Phase 6/6B) to 4/4 (Phase 6C) at +4 pp benign FPR cost. The supervised model already knew when it was confused — softmax entropy is the signal that exposes that knowledge.

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

**Winner: E7 (XGBoost / full / original) — F1_macro = 0.9076, acc = 99.27%**

**H3 REJECTED: SMOTETomek degraded ALL 4 configs** (−0.011 to −0.045 F1)

| Config | F1 (Original) | F1 (SMOTE) | Δ |
|---|---|---|---|
| RF/Reduced (E1→E2) | 0.8469 | 0.8356 | −0.011 |
| RF/Full (E5→E6) | 0.8551 | 0.8380 | −0.017 |
| XGB/Reduced (E3→E4) | 0.8987 | 0.8538 | −0.045 |
| XGB/Full (E7→E8) | 0.9076 | 0.8708 | −0.037 |

**The honest mechanism (revised after senior review):** SMOTETomek consistently degrades macro-F1 by 0.011–0.045 across both classifiers and both feature sets. The mechanism is **boundary-blur on already-overlapping class boundaries** (specifically DDoS↔DoS and Recon_OS_Scan↔Recon_VulScan, visible in confusion matrices and confirmed by SHAP cosine similarity = 0.991 for DDoS↔DoS in Phase 7). SMOTE generates synthetic minority points by interpolating between existing samples; when minority classes are *already adjacent* to structurally similar classes in feature space, the synthetic points fall on or across the decision boundary rather than reinforcing the minority cluster.

**The class_weight='balanced' interaction is NOT the mechanism.** XGBoost arms (E3, E4, E7, E8) use no class_weight and no scale_pos_weight, yet degrade *more* (−0.037, −0.045) than RF arms with class_weight='balanced' (−0.011, −0.017). If the mechanism were "compounding correction," XGBoost arms should be relatively unharmed. The opposite is observed.

**Full features (44) beat reduced (28) consistently** — correlation-based dropping was too aggressive

**IAT confirmed #1 feature** (RF importance = 0.14)

### Problems encountered

| Problem | Severity | Fix |
|---------|----------|-----|
| RF max_depth=None → estimated 8-15 hrs | High | Capped at 30 → runtime 60 min total |
| verbose=1 spams thousands of lines | Low | Set verbose=0 |
| joblib compress=3 adds ~1 hour | Medium | Changed to compress=0 |
| First run interrupted (Ctrl+C) | Low | Resume logic + caffeinate on restart |
| H3 mechanism initially misdiagnosed as "double correction" | Medium (post-hoc) | Senior review caught it; rewrote diagnosis as boundary-blur, supported by experimental matrix and SHAP |

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
- **H1 reframed:** Δ macro-F1 = −0.014 pp, 95% CI [−0.0166, −0.0117] (excludes zero, but magnitude is operationally negligible — ~125 of 892,268 rows)
- **H2 (simulated): FAIL** — 0/5 targets. But this wasn't true LOO → led to Phase 6B
- Binary F1 = 0.9985 at p99 — system works for detection
- Recommended operating point: p97 (99.87% recall, 5.3% FPR)

### Problems encountered

| Problem | Severity | Fix |
|---------|----------|-----|
| H1 label space bug (E7 in 19-class vs fusion in 20-class) | High | v3: both in 20-class space |
| Case 2 precision ~6% (94% are false alarms on benign) | Medium | Fundamental limitation; documented |
| Simulated LOO isn't real zero-day test | High | Created Phase 6B |
| H1 originally framed as "FAIL" / catastrophic | Medium (post-hoc) | Senior review: reframed to "no operationally meaningful difference"; same numbers, honest framing |

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

**Aggregate:** 1,341 of 7,764 LOO target samples → Benign (17.3%); 6,423 → other attacks (82.7%). Reproduced empirically from raw .npy files; aligns with the "redundancy through misclassification" claim.

**Stress test:** Recon_VulScan — 53.6% routed to Benign (worst case), binary recall = 0.700 exactly at p90.

### Problems encountered

| Problem | Fix |
|---------|-----|
| Runtime estimate 5 hours | Actually 19 minutes (each XGBoost ~4 min, not 60) |
| Label space changes per fold (18 classes instead of 19) | Per-fold encoder + inverse mapping (Schema D — sidecar JSONs) |

### Output
- `results/zero_day_loo/` — 5 LOO models, predictions, metrics, 4 figures. Runtime: 19.3 min

---

## Phase 6C — Enhanced Fusion (Entropy + Confidence + Ensemble)

### What we did

Phase 6B established that AE-only zero-day detection fails (0/5 strict). Phase 6C re-mines the existing model outputs to extract uncertainty signals XGBoost already produces — without retraining anything.

**Three new signals extracted from saved arrays:**

1. **Softmax entropy** — Shannon entropy of XGBoost's probability vector. High entropy = model is confused = potential novel attack.
2. **Confidence floor** — `max(softmax)`. Below threshold → route to AE for zero-day check.
3. **Ensemble unsupervised score** — `max(normalized_AE_MSE, normalized_IF_score)`.

**Generalized 4-case → 5-case fusion** (added Case 5 = Uncertain Alert / Operator Review).

**Built complete ablation table:** 11 fusion variants × 5 LOO targets = 55 evaluations.

### Key results

**Diagnostic that validated the approach:**

| Target (held-out) | Entropy mean (novel) | Entropy mean (known) | Gap |
|---|---|---|---|
| Recon_VulScan | 0.483 | 0.090 | **0.393** |
| ARP_Spoofing | 0.382 | 0.000 | **0.382** |
| MQTT_Malformed | 0.327 | 0.135 | 0.192 |
| Recon_Ping_Sweep | 0.304 | 0.118 | 0.187 |
| MQTT_DoS_Connect | 0.192 | 0.116 | 0.076 |

XGBoost knows it doesn't know — its softmax distribution is more spread out for novel classes. Confirmed across all 5 targets.

**Ablation table (post-recalibration):**

| Variant | H2-strict pass | H2-strict avg | H2-binary pass | H2-binary avg | Avg flag rate | Benign FPR |
|---|---|---|---|---|---|---|
| Baseline (AE p90) | 0/4 | 0.314 | 4/5 | 0.849 | 0.965 | 0.189 |
| Baseline (AE p95) | 0/4 | 0.218 | 4/5 | 0.827 | 0.960 | 0.074 |
| Confidence floor τ=0.6 | 0/4 | 0.396 | 5/5 | 0.864 | 0.965 | 0.192 |
| Confidence floor τ=0.7 | 0/4 | 0.538 | 5/5 | 0.891 | 0.965 | 0.197 |
| Entropy (benign-val p90) | 4/4 | 0.908 | 5/5 | 0.973 | 0.969 | 0.278 |
| **Entropy (benign-val p95) ★** | **4/4** | **0.804** | **5/5** | **0.949** | **0.967** | **0.229** |
| Entropy (benign-val p99) | 0/4 | 0.440 | 5/5 | 0.874 | 0.965 | 0.194 |
| Ensemble AE+IF (p90) | 0/4 | 0.217 | 4/5 | 0.810 | 0.963 | 0.148 |
| Ensemble AE+IF (p95) | 0/4 | 0.082 | 4/5 | 0.783 | 0.962 | 0.121 |
| Confidence + Entropy (τ=0.7, p95) | 4/4 | 0.804 | 5/5 | 0.949 | 0.967 | 0.229 |
| Full enhanced (conf+ent+ensemble) | 2/4 | 0.764 | 5/5 | 0.931 | 0.967 | 0.216 |

**Best variant via Pareto analysis:** `entropy_benign_p95` — 4/4 strict, 5/5 binary, FPR = 22.9% (the natural "elbow" of the Pareto frontier between rescue gain and FPR cost).

**Per-target rescue lift (best variant vs baseline):**

| Target | Baseline AE p90 | Entropy p95 | Δ |
|---|---|---|---|
| Recon_Ping_Sweep | 0.161 | 0.968 | **+81 pp** |
| Recon_VulScan | 0.441 | 0.745 | **+30 pp** |
| MQTT_Malformed_Data | 0.335 | 0.773 | **+44 pp** |
| ARP_Spoofing | 0.320 | 0.728 | **+41 pp** |

All four eligible targets cross the 0.70 strict-pass threshold for the first time across all phases.

**H2 final verdict:**

| Phase | Setting | H2-strict | H2-binary |
|---|---|---|---|
| 6 | Simulated LOO, AE-only | 0/5 | 5/5 (binary F1=0.9985) |
| 6B | True LOO, AE-only | 0/5 | 5/5 at p90 (redundancy via misclassification) |
| **6C** | True LOO, entropy_benign_p95 + AE p90 | **4/4** | **5/5** |

(Denominator /4 — MQTT_DoS_Connect_Flood structurally excluded, 0 LOO→Benign samples.)

### Problems encountered

| Problem | Severity | Fix |
|---------|----------|-----|
| **First run: entropy threshold 0.0005 flagged 98% of test (CRITICAL bug)** | **CRITICAL** | Re-calibrated entropy on benign-val (not val-correct) — same convention as AE; threshold became 0.395, FPR became realistic |
| AE+IF ensemble HURTS strict recall (intuition was wrong) | Medium | Documented as honest negative finding; IF dominates AE in normalization, anomaly ranking misaligned with LOO-missed subset. Final system uses AE only |
| Single FPR budget (0.25) was a post-hoc cutoff | Medium (post-hoc) | Senior review: replaced with Pareto frontier analysis. The chosen point is now defensible as the "elbow" of the frontier, not an arbitrary cut |
| Benign val→test entropy distribution shift not measured | Medium (post-hoc) | Senior review: added KS test (KS = 0.0645, modest shift, documented in §15C.10) |

### Calibration discovery (methodological contribution)

The **first run of Phase 6C** calibrated entropy on val-CORRECT samples (samples E7 classified correctly). This produced `ent_p95 = 0.0005` — degenerate, because E7 is 99.72% accurate on val and its correct-prediction entropy distribution is collapsed near zero. The threshold flagged 98% of all test samples → "detection system" became a "flag everything" system.

**The fix:** Calibrate entropy on benign-VAL samples (the same convention used for the AE p90 threshold in Phase 5). Benign traffic is intrinsically more ambiguous than confident attack predictions, so percentiles spread across the operating range. Threshold became 0.395, FPR became defensible.

This is a **publishable methodological finding**: val-correct calibration of uncertainty thresholds is degenerate when the supervised model is highly accurate. Benign-val calibration is the correct convention for IDS uncertainty signals.

### Output
- `results/enhanced_fusion/` — signals/, metrics/ (ablation_table.csv, per_target_results.csv, entropy_stats.csv, h2_enhanced_verdict.json), figures/ (6 plots + Pareto frontier added in senior review fixes), summary.md
- Runtime: 4.6 sec

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

**DDoS vs DoS: cosine similarity = 0.991** — near-identical SHAP profiles. Only IAT magnitude differs. Directly explains the confusion boundary AND supports the H3 boundary-blur diagnosis.

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
| SHAP background drawn from X_test, not X_train (unconventional) | Senior review: defended in §16.7B (TreeSHAP `feature_perturbation='interventional'` is invariant to background source for i.i.d.-similar data; disjoint slicing prevents self-attribution) |

### Output
- `results/shap/` — shap_values.npy (19×5000×44), 10 metrics CSVs, 11 figures. Runtime: 70.3 min

---

## Senior Review — Stress Test & Remediation

After all 7 experimental phases were complete, an external senior review (10+ years intrusion detection experience, uncertainty-aware ML, academic publishing) audited the project. The review delivered:

### What the review verified as correct (no changes needed)
- ✅ Train/val/test splits — clean, no leakage (`preprocessing_pipeline.py:339-345`)
- ✅ Random state consistent (42) across all 8 scripts
- ✅ Scaler fit on train only, applied to test
- ✅ AE benign-only training — properly isolated from val/test
- ✅ LOO label-space mapping (Schema D sidecars) — verified correct, no off-by-one
- ✅ "Redundancy through misclassification" percentages — reproduce within 0.5 pp from raw .npy files
- ✅ H2-strict 4/4 — bootstrap-robust (1000 iters), 5%-lower TPR ≥ 0.687 across all 4 targets
- ✅ No actual code bugs producing wrong numbers

### What needed fixing (all 9 fixes applied)

| # | Fix | What changed | Status |
|---|---|---|---|
| 1 | H3 "double correction" diagnosis refuted by own data | Replaced with boundary-blur mechanism, supported by experimental matrix and SHAP cosine similarity | ✅ commit 2457c44 |
| 2 | requirements.txt missing 6 of 11 packages | Pinned all 11 packages with version bounds, added Python 3.13 + Apple Silicon notes | ✅ commit 3e2f659 |
| 3 | Benign val→test entropy shift never measured | Added KS test (KS = 0.0645, p ≈ 2.6e-69 from large n; modest shift); §15C.10 limitation now documents the 9.46% entropy-only FPR vs 5% nominal target | ✅ commit 3e2f659 |
| 4 | FPR = 0.25 budget was post-hoc cutoff | Replaced with Pareto frontier plot + analysis; entropy_benign_p95 defended as the "elbow" of the frontier, not an arbitrary cut | ✅ commit 920fa95 |
| 5 | H1 framed as "FAIL" sounds catastrophic | Reframed across 7 hits in README + journey doc as "no operationally meaningful difference" (Δ = −0.014 pp, CI [−0.0166, −0.0117], ~125 of 892,268 rows) | ✅ commit 920fa95 |
| 6 | Entropy contribution overclaimed as "carries actionable zero-day signal" | Reframed as "complementary to AE": entropy alone fails Recon_VulScan (TPR = 0.473 < 0.50); the AE channel rescues that target. Contribution is the *fusion* of entropy + AE | ✅ commit 920fa95 |
| 7 | 22.9% benign FPR not quantified operationally | New §15C.6B: ~23-92 false alerts/sec on a 40-device IoMT subnet; two architectural responses (hierarchical aggregation, confidence-stratified routing) make case-stratified fusion alerts tractable | ✅ commit 920fa95 |
| 8 | H3 hypothesis criterion (≥3/5 minority improve) didn't match reported metric (macro-F1) | Tightened to report both: 0/4 macro-F1 + 2/5 minority improvements (still FAIL by both criteria) | ✅ commit 2457c44 |
| 9 | SHAP background from X_test (unconventional) had no defense | New §16.7B: TreeSHAP invariance argument, self-attribution prevention rationale, train-drawn alternative listed as future work | ✅ commit 920fa95 |

**Defensibility score:** 3.0 → 4.0 / 5 (per senior reviewer's own scoring rubric).

**Critical:** None of these fixes changed any experimental number. The data, models, metrics, and verdicts are all unchanged. The fixes are framing, methodology defense, and one new figure (Pareto frontier).

---

## Path B — Tier 1 Hardening

After the senior review remediation closed the framing/methodology gaps, three deeper items remained: per-fold variance not estimated (§1.5 of the review), threshold-grid coarseness (§1.4), and SHAP background source unverified (§1.2). Path B Tier 1 hardening closes all three with empirical evidence — multi-seed validation, a continuous threshold sweep, and a SHAP background sensitivity check. None of these reruns retrain a single model from the original 7 phases; they all operate on saved arrays + targeted re-inference.

### Week 1 — Multi-seed LOO Validation

#### What we did
- Trained 25 LOO-XGBoost models = 5 seeds {1, 7, 42, 100, 1729} × 5 zero-day targets, with seed=42 hardlinked from the canonical baseline (so 20 *new* trainings; 5 reuses). Each fold drops one attack class, fits XGBoost on the remaining 18 classes, and predicts on val/test.
- Re-applied the Phase 6C `entropy_benign_p95` fusion variant per seed, recording H2-strict avg, H2-binary avg, strict-pass count, and benign-test FPR. Hard tripwire: `entropy_benign_p95` strict_avg under seed=42 must reproduce 0.8035264623662012 within 1e-9 — caught any silent drift in the multi-seed driver vs the canonical Phase 6C run.
- Driver: `notebooks/multi_seed_fusion.py` (orchestrates) + `notebooks/multi_seed_loo.py` (training fan-out).

#### Key results
- **H2-strict avg = 0.799 ± 0.022** across 5 seeds; seed=42 sits at the 63rd percentile of the multi-seed distribution.
- **0/19 eligible (seed × target) cells fail the 0.70 strict threshold.** The denominator is 19 (not 20) because one seed × one target combination drops below the n=30 minimum eligibility floor — a structural property of the LOO partition for the smallest target (Recon_Ping_Sweep, n_test = 169), not a failure.
- **Operational FPR is effectively constant**: 0.2289 ± 0.0003 across all 5 seeds (CV = 0.13%) — the AE p90 channel dominates the FPR and is seed-invariant.
- **seed=42 reproducibility tripwire**: actual = 0.8035264623662012, reference = 0.8035264623662012, **diff = 0.000e+00** (passed).
- The multi-seed result reframes Phase 6C's "4/4 PASS" from a single-seed point estimate to a **distribution-level claim**: H2-strict 4/4 holds across all 5 seeds, and the spread (σ = 0.022) is small relative to the 0.70 criterion margin (0.093).

#### Output
- README §15B (multi-seed validation narrative + eligibility table)
- `results/zero_day_loo/multi_seed/seed_{1, 7, 42, 100, 1729}/`
- `results/enhanced_fusion/multi_seed/seed_{...}/metrics/{ablation_table.csv, per_target_results.csv}`
- Wall-clock: **85.1 min** on M4 (dominated by the 20 new XGBoost trainings; fusion re-evaluation is sub-second per seed)

### Week 2A — Continuous Threshold Sweep + Per-Fold KS

#### What we did
- **Task 1 — Continuous entropy threshold sweep:** swept 29 thresholds at percentiles {85.0, 85.5, ..., 99.0} of the benign-val E7 entropy distribution (Δ = 0.5pp), reusing the Phase 6C 5-case `entropy_fusion` logic verbatim with a hard reproducibility tripwire that asserts `strict_avg(p=95.0) == 0.8035264623662012` within 1e-9 before sweeping.
- **Task 2 — Per-fold KS test for benign val→test entropy shift:** decomposed the §15C.10 aggregate KS = 0.0645 into per-LOO-fold KS values. For each of the 5 zero-day targets, loaded `loo_xgb_without_<target>.pkl`, ran `predict_proba(X_val)` inline (LOO val proba is not saved on disk), and ran a two-sample KS on benign-val vs benign-test entropy distributions. A 6th aggregate row uses E7 directly to reproduce the §15C.10 figure as a sanity check.
- Drivers: `notebooks/threshold_sweep.py`, `notebooks/ks_per_fold.py`. No retraining; predict_proba is inference-only with `del + gc.collect()` between folds.

#### Key results
- **Continuous sweep — strictly monotone Pareto frontier:** strict_avg and FPR both decrease monotonically as the threshold percentile rises; every continuous point is Pareto-optimal. The discrete grid {p90, p95, p97, p99} hid this structure.
- **Refined operational optimum at p = 93.0:** under the §15C OPERATIONAL_FPR_BUDGET = 0.25 constraint, strict_avg = 0.859, FPR = 0.247, 4/4 strict pass. **+5.5pp improvement** on H2-strict avg over the §15C.6 published recommendation of `entropy_benign_p95` (strict_avg = 0.804, FPR = 0.229), at the cost of +1.8pp higher FPR.
- **Plateau structure observed:** the 4/4 strict-pass count holds from p85 down to p95.0 (FPR = 0.229), then drops to 3/4 at p95.5 (FPR = 0.223) and 0/4 by p98. The published p95.0 sits exactly at the lip of the plateau; p93.0 sits comfortably in the middle, providing both higher recall and stability margin against threshold drift. p95.0 remains valid but is no longer optimal.
- **Per-fold KS uniformity:** 5 LOO folds give KS values in [0.0543, 0.0573] — total spread = 0.0031, all 5 within ±0.0017 of each other. The val→test entropy shift is **uniform across folds**; no individual fold drives the aggregate. The 6th aggregate row reproduces §15C.10's KS = 0.0645 exactly, confirming the per-fold/aggregate distinction is just the dimensionality difference between the 18-class LOO-XGBoost entropy distributions and the 19-class E7 entropy distribution. Strengthens (not weakens) the §15C.10 framing of "small-to-moderate shift, not a structural break."

#### Output
- README §15D (new continuous-threshold-sweep section, ~350 words) and §15C.10 (per-fold KS table appended to the existing limitation paragraph, with the 18-vs-19-class footnote)
- `results/enhanced_fusion/threshold_sweep/{sweep_table.csv (29 rows), sweep_per_target.csv (145 rows), pareto_continuous.png, strict_avg_vs_threshold.png}`
- `results/enhanced_fusion/ks_per_fold/{ks_per_fold.csv (6 rows), ks_per_fold.png}`
- Wall-clock: **~9 min total** on M4 (Task 1: 4.7 s including reproducibility guard + 29 fusion evaluations; Task 2: 19.2 s including 5 LOO `predict_proba(X_val)` calls + 1 aggregate row)

### Week 2B — SHAP Background Sensitivity (BULLETPROOF)

#### What we did
- Re-ran TreeSHAP on the **same** 5,000-sample explained set (`X_shap_subset.npy`) with the **same** sampling protocol — `np.random.default_rng(42 + 1).choice(..., size=500, replace=False)` matching `shap_analysis.py:280` — and the **same** explainer arguments (`feature_perturbation="interventional"`, `model_output="raw"`). The *only* changing variable is the background source pool: `X_train` (3.6M rows) instead of test-disjoint `X_test` slice. This is the apples-to-apples comparison the §16.7B invariance argument predicts.
- Compared the new attribution tensor against the Phase 7 baseline along three dimensions: global Kendall τ on full 44-feature ranks AND on the top-10 union; per-class top-5 Jaccard for all 19 classes; DDoS↔DoS category cosine similarity (§16.4 reported 0.991, target reproduction within ±0.01).
- Driver: `notebooks/shap_sensitivity.py`. No retraining; only TreeSHAP is recomputed.

#### Key results
- **Decision: BULLETPROOF.** Kendall τ on top-10 union = **0.927** (above the 0.9 cutoff); τ on full 44 features = **0.940** (informational).
- **Per-class top-5 Jaccard = 0.842 ± 0.171** across 19 classes; min = 0.667 (4/5 features match); **all 19 classes ≥ 0.6**. **9/19 classes have IDENTICAL top-5** (Jaccard = 1.0): every DDoS/DoS variant except DoS_ICMP, plus MQTT_DDoS_Connect_Flood, MQTT_DoS_Publish_Flood, Recon_VulScan.
- **8 of the 10 globally-ranked top features** (IAT, Rate, TCP, syn_count, Header_Length, syn_flag_number, UDP, Min) have **identical ranks** under both backgrounds; only Number / Tot sum / Protocol Type rotate at ranks 9–11.
- **DDoS↔DoS category cosine reproduces:** 0.991 (test_bg) vs 0.989 (train_bg) — |Δ| = 0.002, **within fp32 noise floor and 5× tighter than the ±0.01 target**. The §16.4 finding "DDoS and DoS share a near-identical feature signature" is not an artifact of the test-side background.
- The §16.7B invariance argument is now backed by empirical evidence on the actual model + dataset, not just by the theoretical interventional-SHAP property.

#### Output
- README §16.7B updated: "future work" closer replaced with the empirical-verification paragraph; invariance and self-attribution-prevention paragraphs preserved.
- `results/shap/sensitivity/{comparison.csv, global_top10_ranks.csv, per_class_jaccard.csv, category_cosine.csv, top10_rank_comparison.png, per_class_jaccard.png, run.log}` (the 16.7 MB `shap_values_train_bg.npy` is auto-gitignored per `results/**/*.npy` but saved locally for re-analysis)
- Wall-clock: **75.7 min** on M4, no GPU (matched Phase 7's 70-min ballpark on identical dimensions)

---

## All Hypothesis Results — Final

| Hypothesis | Original Claim | Result | Evidence | Thesis Value |
|-----------|---------------|--------|----------|-------------|
| **H1** | Fusion improves macro-F1 over E7 | **No operationally meaningful difference** | Δ = −0.014 pp, 95% CI [−0.0166, −0.0117] excludes zero but magnitude is ~125 of 892,268 rows | Value is in case stratification (Cases 1-5), not aggregate metrics. Binary F1 = 0.9985. |
| **H2 strict** | AE catches ≥70% of zero-day | Phase 6/6B: 0/5 FAIL → **Phase 6C: 4/4 PASS** | Entropy + AE fusion lifts strict avg from 0.314 to 0.804 at +4 pp benign FPR | Entropy is the complementary signal that solves the AE blind spot. Calibration discovery (val-correct vs benign-val) is itself publishable. |
| **H2 binary** | Hybrid system detects novel attacks | **5/5 PASS** (consistent across phases) | 70-100% per-target binary recall via "redundancy through misclassification" | 82.7% of novel attacks routed to similar known attacks, not "Benign" — novel detection mechanism for IoMT IDS literature. |
| **H3** | SMOTETomek improves minority F1 | **0/4 macro-F1 + 2/5 minority — FAIL** | All 4 configs degrade −0.011 to −0.045; mechanism is boundary-blur on overlapping classes (NOT class-weight interaction; XGBoost has no class_weight yet degrades more than RF) | Boundary-blur explanation supported by Phase 4 confusion matrices + Phase 7 SHAP cosine = 0.991 for DDoS↔DoS. Contradicts common IoMT IDS oversampling assumption. |

---

## Complete List of Thesis Contributions (17 total)

1. **First hybrid 4-layer (XGBoost + AE + 5-case fusion + SHAP) framework on CICIoMT2024** — no prior work combines all 4 layers; Yacoubi is supervised-only
2. **37% duplication discovery in CICIoMT2024 train (44.7% in test)** — first report ever; explains inflated metrics in all prior literature
3. **SMOTETomek shown harmful on this dataset** — rejected via boundary-blur mechanism (not class-weight interaction); contradicts common IoMT IDS oversampling assumption
4. **Corrected class distribution** — Recon_Ping_Sweep rarest (not ARP_Spoofing), real ratio 2,374:1 (not "100:1")
5. **"Redundancy through misclassification"** — novel zero-day detection mechanism (Phase 6B); 82.7% of novel attacks route to similar known attacks, not "Benign"
6. **AE reconstruction-error insufficient for zero-day on flow features** — genuine negative finding (Phase 6/6B), redirecting future research toward complementary uncertainty signals
7. **Softmax entropy as complementary zero-day signal under true LOO** — Phase 6C; first demonstration on CICIoMT2024 that entropy + AE fusion lifts H2-strict from 0/4 to 4/4
8. **Calibration discovery: val-correct vs benign-val** — methodological contribution; val-correct calibration is degenerate for highly-accurate models, benign-val is the correct convention for IDS uncertainty signals
9. **Per-attack-class SHAP analysis** — first on CICIoMT2024; reveals heterogeneous feature importance masked by global averaging
10. **DDoS↔DoS boundary explained** — cosine similarity 0.991 proves identical feature reliance; Only IAT magnitude separates them
11. **Feature importance is method-dependent** — Cohen's d vs SHAP: Jaccard = 0.000, Spearman ρ = −0.741; reporting one ranking is insufficient
12. **Confidence-stratified alerts (5-case fusion)** — operational value beyond binary IDS; routes Cases 1/2/3/5 to different SOC tiers
13. **StandardScaler fix for AE on ColumnTransformer output** — practical ML pipeline lesson: tree models are scale-invariant, AE/IF are not
14. **Pareto-based variant selection methodology** — replaces arbitrary FPR budget; defensible across operational ranges, allows committee/practitioner to pick their own operating point
15. **Multi-seed robustness validation under true LOO** (Path B Week 1) — H2-strict 4/4 holds across 5 seeds with σ = 0.022 well inside the 0.70 criterion margin; FPR is seed-invariant (CV = 0.13%). Identifies a **structural eligibility floor** for tiny LOO targets (Recon_Ping_Sweep, n_test = 169) where one seed × one target combination falls below the n=30 minimum — a property of the LOO partition, not a metric failure. Reframes the 4/4 claim from a single-seed point estimate to a distribution-level statement.
16. **Continuous-frontier threshold methodology** (Path B Week 2A) — replaces the discrete 4-point grid {p90, p95, p97, p99} with a 29-threshold continuous sweep at 0.5pp resolution. Reveals a **strict-pass plateau structure** that the discrete grid hid: p95.0 sits exactly at the lip (one half-percentile drops 4/4 → 3/4) while p93.0 sits in the middle of the plateau with both higher recall (+5.5pp) and stability margin against threshold drift. Refines (does not contradict) the §15C.6 published recommendation.
17. **Empirical SHAP background sensitivity verification** (Path B Week 2B) — converts the §16.7B invariance argument from theory to empirical evidence on the actual model + dataset. Same 5,000-sample explained set + same uniform-random sampling protocol as Phase 7; only the source pool changes (X_train vs test-disjoint X_test). Result: Kendall τ = 0.927 over the top-10 union (BULLETPROOF), 19/19 classes with per-class top-5 Jaccard ≥ 0.6, DDoS↔DoS cosine reproduction within fp32 noise floor (|Δ| = 0.002).

---

## Total Compute Time

| Phase | Task | Runtime |
|-------|------|---------|
| Phase 2 | EDA | ~15 min |
| Phase 3 | Preprocessing | 228 min |
| Phase 4 | Supervised (24 runs) | 60 min |
| Phase 5 | Unsupervised (AE + IF) | 34 sec |
| Phase 6 | Fusion engine | ~1 min |
| Phase 6B | True LOO (5 XGBoost retrains) | 19.3 min |
| Phase 6C | Enhanced fusion (no retraining) | 4.6 sec |
| Phase 7 | SHAP analysis | 70.3 min |
| **Phase 1–7 experimental work** | | **~6.5 hours** |
| Path B Week 1 | Multi-seed LOO (20 new XGBoost trains + fusion re-eval) | 85.1 min |
| Path B Week 2A | Continuous threshold sweep + per-fold KS (no retrain) | ~9 min |
| Path B Week 2B | SHAP background sensitivity (TreeSHAP recompute, no retrain) | 75.7 min |
| **Path B Tier 1 hardening** | | **~170 min (~2.8 hours)** |
| **Grand total compute** | | **~9.3 hours** |

Senior review remediation (post-hoc, no compute): ~6 hours of writing + Pareto plot generation.

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
| Statistics | scipy (KS test, bootstrap) |
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
│   ├── enhanced_fusion.py            # Phase 6C
│   ├── pareto_frontier.py            # Senior review fix #4
│   └── shap_analysis.py              # Phase 7
├── preprocessed/                     # 5.7 GB
├── results/
│   ├── supervised/                   # 8 models, 24 metrics
│   ├── unsupervised/                 # AE + IF + scaler
│   ├── fusion/                       # 4-case fusion (Phase 6)
│   ├── zero_day_loo/                 # 5 LOO models (Phase 6B)
│   ├── enhanced_fusion/              # 5-case + entropy (Phase 6C)
│   │   ├── signals/
│   │   ├── metrics/                  # ablation_table.csv (key output)
│   │   └── figures/                  # 7 plots including pareto_frontier.png
│   └── shap/                         # SHAP values + 11 figures
├── eda_output/                       # 15+ EDA figures
├── requirements.txt                  # 11 packages, version-bounded
├── README.md                         # Updated with Phase 6C + senior review fixes
└── .gitignore
```

---

## Final System Architecture (Phase 6C)

**4 layers, 5-case fusion logic:**

```
IoMT traffic (44 features)
        │
        ▼
Preprocessing + StandardScaler (benign-fitted)
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
Layer 1: XGBoost E7    [confidence, entropy]   Layer 2: Autoencoder
F1_macro = 0.9076      (extracted from softmax) AUC = 0.9892
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                               ▼
Layer 3: 5-case fusion engine (entropy_benign_p95 + AE p90)
H2-strict 4/4 PASS | H2-binary 5/5 PASS | benign FPR 22.9%
                               │
        ┌──────────┬──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
     Case 1     Case 2     Case 3     Case 5     Case 4
   Confirmed  Zero-Day   Low-conf  Uncertain    Clear
                          (NEW in Phase 6C)
                               │
                               ▼
              Layer 4: SHAP per-class explainability
              IAT #1 globally; per-class signatures vary widely
```

**Operational alert routing:**
- Case 1 → BLOCK (high confidence attack)
- Case 2 → QUARANTINE + investigate (potential novel attack)
- Case 3 → MONITOR (model says attack but no anomaly)
- Case 4 → ALLOW
- Case 5 → OPERATOR REVIEW (model confused by entropy/conf, AE clean)

---

## What's Next — Thesis Writing Phase

All experimental work is complete. The remaining work is exposition:

- [ ] Chapter 1 — Introduction
- [ ] Chapter 2 — Literature Review (already drafted as Literature_Review_Chapter2.md)
- [ ] Chapter 3 — Dataset & Preprocessing
- [ ] Chapter 4 — Methodology (4-layer framework + 5-case fusion logic)
- [ ] Chapter 5 — Results & Discussion (Phases 4-7 + 6B + 6C)
- [ ] Chapter 6 — Conclusion & Future Work
- [ ] Defense presentation (PowerPoint)

**Future work (deferred from this thesis):**
- ~~Multi-seed LOO validation (3 days compute) — would tighten H2-strict 4/4 confidence further~~ **Done in Path B Week 1** (5 seeds in 85.1 min, 0/19 eligible cells fail; H2-strict avg 0.799 ± 0.022).
- ~~Continuous threshold sweep between p90 and p95 — may yield slightly tighter operating point~~ **Done in Path B Week 2A** (29 thresholds at p85.0–p99.0; refined optimum at p93.0 with strict_avg 0.859 vs published p95's 0.804 under the same FPR budget).
- ~~Train-drawn SHAP background sensitivity check — verify top-10 rank stability~~ **Done in Path B Week 2B** (Kendall τ = 0.927 over top-10 union — BULLETPROOF; 19/19 per-class Jaccard ≥ 0.6; DDoS↔DoS cosine reproduces within fp32 noise floor).
- **Profiling-feature-basis AE** — still open. Addresses layer-coupling concern (AE and XGBoost share the same 44 features). Phase 6's future-work item that Phase 6C did not address.
- **VAE replacement for reconstruction-error AE** — still open. More principled OOD detection; would couple cleanly with the entropy fusion if the latent-space density gives a complementary signal to softmax entropy.

---

_Last updated: April 30, 2026 — Path B Tier 1 hardening complete (Weeks 1, 2A, 2B). All 7 experimental phases + senior review remediation + multi-seed LOO validation (§15B) + continuous threshold sweep (§15D, refined operating point at p93.0) + per-fold KS (§15C.10, uniform across folds) + SHAP background sensitivity (§16.7B, BULLETPROOF). Senior review §1.2 / §1.4 / §1.5 closed._
_Next step: Thesis writing_