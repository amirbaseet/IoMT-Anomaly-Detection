# Per-Phase Design Decisions vs. CICIoMT2024 Prior Work

> Maps this thesis's design decisions against prior work on CICIoMT2024, one table per phase.
> **Provenance:** competitor approaches/numbers derive from `Literature_Review_Chapter2_v6.4.md`,
> `Defense_Positioning_v2.3.md`, and `Chapter2_Synthesis_and_Gap_Analysis.md`; every competitor
> number is grep-verified against `~/Downloads/research/papers/*.md` before use. Our numbers derive
> from `deliverables/numbers_map.md`. The **Phase 6 section is synthesized from these truth files**
> (no canonical template was available), to the same 4-column spec and honesty gates as the rest.
> **Honesty rule:** every "we win" cell names its caveat (leakage / FPR / weak recall) in the same row.
> Rows with no prior art are kept and framed "no prior art does this," never omitted.

Phase order: **2 · 3 · 4 · 5 · 6 · 7**.

---

## Phase 2 — Data integrity (per-split deduplication + class-distribution analysis)

**Decision.** Count **exact-row duplicates per split** on the **raw 72-file distribution**, drop them, and report per-split duplicate rates + the post-dedup class distribution. Our numbers (`numbers_map.md`): **train 36.95 % / test 44.72 %** duplicates; train 7,160,831 → 4,515,080, test 1,614,182 → 892,268; rarest class **Recon_Ping_Sweep 689 rows**; max imbalance **2,374:1**.

| Competitor | Their approach → number | Ours | Edge or overlap (caveat in-row) |
|---|---|---|---|
| **Riyadi et al. (2025)** [#17] | XGBoost pipeline; removed **5,119 dups (~0.07 %)** on a pre-merged / redistributed set | per-split exact dedup → **36.95 % / 44.72 %** | ~500× larger removal — but on the **raw per-protocol 72-file distribution**, a *different* set than their pre-redistributed one. **Not a contradiction, and not a "first to find duplicates" claim** — the contribution is the **per-split analysis**, not discovery of duplicates. |
| **Kharoubi, Cherbal & Akkal (2025)** [#26] | "removed 5,119 duplicate records to prevent overfitting" | same | The **identical 5,119** recurs → shared pre-processed input stream; our per-split rate measures a **different quantity** on raw data. |
| **Akkal et al. (2024)** [#19] | "identified and removed **5,119** duplicate rows" (DDoS-focused) | same | Same figure a third time across three papers → common redistributed source; the per-split, all-class rate is **unmeasured in prior work**. |
| **Dadkhah et al. (2024, dataset)** [#01] | Releases the official train/test splits + class counts; no per-split dedup analysis | post-dedup rarest **Recon_Ping_Sweep 689**, imbalance **2,374:1** | Not a *correction* of Dadkhah — a **post-deduplication recomputation** on a different denominator; complementary to the official counts. |
| **No prior art does this** | — | per-split (train-vs-test **separate**) duplicate rates; the **44.72 % test > 36.95 % train** split-asymmetry | **No prior CICIoMT2024 paper reports per-split duplicate rates.** The asymmetry (test set more duplicate-heavy than train) is the leakage mechanism prior accuracy numbers ride on — and it is unique to this work. |

**Reading.** The single 5,119 figure shared by Riyadi/Kharoubi/Akkal is on a pre-redistributed merge; it is **not the same measurement** as our raw per-split rates, so the honest claim is "we did a per-split duplicate analysis no one else did," never "we were first to find duplicates."

---

## Phase 3 — Resampling decision (SMOTETomek tested and rejected; H3)

**Decision.** **Test SMOTETomek and reject it** — ship Original (no resampling). H3 result (`numbers_map.md`): macro-F1 degrades in **0/4 configs** (RF/Reduced −0.0114, RF/Full −0.0171, XGB/Reduced −0.0449, XGB/Full −0.0368); mechanism = boundary-blur on overlapping DDoS↔DoS / Recon classes.

| Competitor | Their approach → number | Ours | Edge or overlap (caveat in-row) |
|---|---|---|---|
| **Riyadi et al. (2025)** [#17] | XGBoost + **SMOTEENN** → claims resampling helps, **99.811 %** | We **test SMOTETomek and reject it**: macro-F1 degrades in **0/4 configs** (−0.011 to −0.045) | Opposite verdict — but **not like-for-like**: Riyadi reports **weighted accuracy** on a near-undeduplicated set (resampling can inflate weighted metrics), while our **macro-F1 on deduplicated data** exposes boundary-blur. We don't claim Riyadi is wrong on their data — we claim resampling **hurts the honest minority metric on clean data**. |
| **CICIoMT2024 papers applying SMOTE-family balancing by default** (common) | Resampling as an unquestioned preprocessing step | Empirical rejection **with a mechanism** (boundary-blur, cross-checked by the DDoS↔DoS SHAP cosine 0.991) | **No prior CICIoMT2024 paper reports an ablation that rejects resampling with a mechanism** — the field default is to apply it. |

---

## Phase 4 — Supervised classifier choice (XGBoost / RF / ensembles; full-vs-reduced; honest metrics)

**Decision.** Ship **E7 = XGBoost / Full 44 features / Original (no SMOTE)** on the **per-split-deduplicated 19-class** task, and report **macro-F1 + MCC** as primary — not accuracy on raw data. Our baseline: **macro-F1 0.9076, accuracy 99.27 %, MCC 0.9906, macro-precision 0.9421** (`numbers_map.md`; `E7_multiclass.json`).

| Competitor | Their approach → number | Ours | Edge or overlap (caveat in-row) |
|---|---|---|---|
| **Kharoubi, Cherbal & Akkal (2025)** [#26] | XGBoost / DT / RF / CNN / LSTM comparison → **99.83 % (19-class)**, 99.94 % binary | E7 99.27 % acc, **macro-F1 0.9076, MCC 0.9906**, deduped | They are **+0.56 pp on accuracy** — but on raw duplicate-heavy data (no per-split dedup; my split removes **37 %/45 %**), **accuracy-only (no macro-F1, no MCC), no zero-day, no XAI**. The ~0.5 pp gap is the **leakage signature**, not a quality deficit. |
| **Riyadi et al. (2025, JIOS)** [#17] | XGBoost + SMOTEENN + IG/PCA + Bayesian-opt → **99.811 % (19-class)**; removed only **5,119 dups (~0.07 %)** | E7 macro-F1 0.9076 + MCC 0.9906 on 37 %/45 %-deduped | **+0.54 pp accuracy**, but on a **near-undeduplicated set** (5,119 rows vs my ~2.6 M) and **weighted metrics only** (masks minority classes). Their "SMOTEENN helps" claim is **contradicted by our H3** (Phase 3, SMOTETomek degrades macro-F1 0/4). |
| **Yacoubi et al. (2025, AIAI)** [#03] | XGBoost + SHAP-driven k=15 feature selection → **99.80 %**, but **6-class (six categories)**, raw data; macro-F1 ~95.4 at that granularity | E7 on the **harder 19-class** deduped, macro-F1 0.9076 | **Coarser task** (6 categories, not 19 sub-types) on raw data — **not a like-for-like comparison**; the 99.80 % accuracy is at a granularity where minority sub-types are merged away. |
| **Yacoubi et al. (2026, Springer)** [#04] | RF / CatBoost / Stacking, 19-class raw → RF **99.39 % acc but macro-F1 87.56 %** (Stacking 86.91 %) | E7 **macro-F1 0.9076 (deduped) > their 0.8756 (raw)** | Higher macro-F1 on **cleaner data** — but **not like-for-like**: our 0.9076 is on **per-split-deduplicated** 19-class data, theirs (0.8756) on **undeduplicated**, so this is not a head-to-head win. *Caveat in-row:* their **accuracy (99.39 %) exceeds ours**; the honest read is the **~12 pp accuracy–F1 gap** exposing minority blindspot, plus our macro-F1 holding higher on harder (deduplicated) data. |
| **Doménech et al. (2025)** [#20] | RF + preprocessing optimization (windowing, balancing) → **99.85 % (0.9985)** | E7 deduped, macro-F1 0.9076 + MCC | **+0.58 pp accuracy**, but **no deduplication** and preprocessing-focused; the headline is accuracy on raw data, with no per-class / MCC honesty axis reported. |
| **No prior art does this** | — (none in the 28-paper corpus reports it) | E7 **MCC 0.9906** + **macro-F1 0.9076** on **per-split-deduplicated** 19-class data | **No CICIoMT2024 paper reports MCC, and none reports honest macro-F1 on per-split-deduplicated data.** This is the integrity-axis contribution, not an accuracy contest we could "win." |

**Reading.** Every raw-data leader (Kharoubi 99.83, Riyadi 99.811, Doménech 99.85, Yacoubi-AIAI 99.80) sits **above** E7 on accuracy and **below or absent** on the honest axis — accuracy on duplicate-heavy data is inflated, and the only paper that reports a 19-class macro-F1 (Yacoubi-Springer, 0.8756) lands **below** ours on cleaner data. The design decision is *not* "beat their accuracy"; it is "report the metric that survives deduplication." Feature-set note: where Yacoubi (SHAP k=15) and Riyadi (IG+PCA) **select** features, we keep **Full 44** because reduced drops the RF-importance #2/#3/#4 features and full beats reduced by 0.005–0.009 macro-F1.

*(All competitor numbers grep-verified: Kharoubi 99.83/99.94 [#26], Riyadi 99.811 + 5,119 dups [#17], Yacoubi-AIAI 99.80 [#03], Yacoubi-Springer RF 99.39/87.56 & Stacking 86.91 [#04], Doménech 0.9985 [#20]. Our numbers: `numbers_map.md` §1/§4.)*

---

## Phase 5 — Unsupervised layer (benign-only autoencoder + Isolation Forest)

**Decision.** Train a **benign-only AE (+ Isolation Forest)** as a *complementary* zero-day channel, evaluated **inside the fusion engine**, not as a standalone accuracy figure. Our anchor (`numbers_map.md`): AE test **AUC 0.9892**.

| Competitor | Their approach → number | Ours | Edge or overlap (caveat in-row) |
|---|---|---|---|
| **Abo-Haat & Zuhair (2026)** [#22] | KAN + VAE + meta-learning with **device-disjoint** evaluation → **86.5 % F1-macro** | benign-only AE (AUC 0.9892) + IF, evaluated in fusion | **Orthogonal leakage axes — NOT a head-to-head.** Their 86.5 % is lower because of **device-disjoint splitting** (a harder, device-level leakage control), **not a weak VAE**; our 0.9892 addresses **duplicate-row** leakage. Comparing the two numbers directly is a category error — neither axis subsumes the other. (Their attack-traffic device labels are themselves machine-predicted, so even their device-disjoint guarantee is approximate.) |
| **Chandekar et al. (2025)** [#05] | AE + Isolation Forest among many models, used for ordinary **anomaly flagging** (no zero-day / novelty mechanism) | benign-only AE + IF as a **complementary zero-day** channel feeding an entropy-gated fusion | Same components, different role: Chandekar flags anomalies with no novelty protocol; we **map the AE's per-class blind spots** and make them the architectural rationale for fusion. |
| **No prior art does this** | — | a benign-only reconstruction layer whose per-class blind spots are *measured* and then **closed by a supervised-entropy gate** | No CICIoMT2024 work pairs a benign-only AE with a supervised-entropy gate as **complementary** zero-day channels. |

*(Competitor: Abo-Haat 86.5 % + device-disjoint [#22] grep-verified; Chandekar AE+IF role qualitative [#05]. Ours: `numbers_map.md` §5.)*

---

## Phase 6 — Fusion engine + leave-one-attack-out  *(synthesized from truth files + `numbers_map.md`; no canonical template was available)*

**Decision.** A **4→5-case decision-fusion** engine over saved arrays (no retrain at fusion time), with a **softmax-entropy gate** calibrated on benign-validation, tested under **true per-attack leave-one-out**. Anchors (`numbers_map.md`): max-prob confidence floor (τ=0.6/0.7) → **0/4** strict rescues; **entropy_benign_p95 → 4/4** (strict_avg 0.8035); fusion aggregate **Δ −0.014 pp** (no macro-F1 gain); fusion-level benign **FPR 22.9 %**. Rows are **design decisions**.

| Decision axis | Prior art → approach | Ours | Edge or overlap (caveat in-row) |
|---|---|---|---|
| **Zero-day signal** | **Alfageer et al. (2026)** [#21] — AE gate + a **max-probability confidence threshold (τ = 0.65)** that rejects low-confidence predictions as "Unknown" (`Defense_Positioning_v2.3`) | AE gate + **softmax-entropy** gate. In *our own* ablation, a **max-probability confidence floor (τ=0.6, 0.7) rescues 0/4** eligible LOO targets while the **entropy gate rescues 4/4** (strict_avg 0.8035) | **Convergent idea / divergent mechanism.** Alfageer is genuine convergent prior art (AE gate + max-prob threshold τ = 0.65). The **lever is the 0/4-vs-4/4 ablation on our own data** — their signal (max-prob) fails strict rescue where entropy succeeds. *Caveat:* measured on our deduplicated per-attack-LOO setup; we do not re-run Alfageer's pipeline. |
| **Protocol granularity** | **Uddin et al. (2025)** [#18] — **category-level** leave-one-out (excludes a whole attack *family*); **Alfageer holds out 2 classes** (Recon-OS_Scan, MQTT-DDoS-Connect_Flood; `Defense_Positioning_v2.3`) | **True per-attack leave-one-out** — retrain the supervised model with each eligible *sub-type* withheld while its siblings remain in training | Strictest novelty protocol on the granularity axis. *Caveat:* per-attack LOO is **harder** than category LOO or Alfageer's 2-class hold-out (siblings remain → easier generalization for them), so our rescue numbers are **not comparable** to theirs — different difficulty, not a head-to-head. |
| **Fusion structure** | Two-stage AE→supervised pipelines (Alfageer) | A 4→5-case decision **truth-table** combinator over saved arrays | Overlap on "supervised + unsupervised combined"; the differentiator is the **explicit case stratification**, not a higher number. |
| **Operator routing** | Binary alert / single "Unknown" flag (Alfageer) | **5-tier routing**: Confirmed / Zero-Day-Warning / Low-Confidence / Clear / Uncertain-Review | No prior CICIoMT2024 fusion stratifies alerts into operator-actionable tiers; the value is **operational triage**, explicitly not metric lift. |
| **H1 — honest negative** | Fusion papers report **aggregate accuracy gains** | **Fusion does NOT beat the supervised baseline** on aggregate macro-F1: **Δ −0.014 pp** (CI excludes zero but ~125 / 892,268 rows) | We **decline the aggregate-lift claim**. Fusion's value is operational stratification + zero-day, **not** an aggregate-metric win — stated as an honest negative, not buried. |
| **Weak spots (stated up front)** | — | **22.9 % fusion-level benign FPR** at entropy_benign_p95; **Recon_VulScan** is the marginal eligible target — **entropy-gate rescue rate 0.745** (barely clears the 0.70 strict bar), vs **AE-only strict zero-day recall 0.441** (the two metrics are *not* the same and are labeled here) | The 4/4 pass is **bounded above by 4** (MQTT_DoS_Connect_Flood structurally excluded, `n_loo_benign = 0`), carries a **high operational FPR**, and Recon_VulScan sits **on the margin**. None of this is hidden. |
| **No prior art does this** | — | entropy-gated **5-case** fusion with a bit-exact reproducibility tripwire, evaluated under **true per-attack LOO** | No CICIoMT2024 work combines per-attack LOO + softmax-entropy gating + operator-tier routing. |

*(Competitor mechanisms verified: Alfageer AE-gate + confidence-thresholding [#21], with **τ = 0.65** and the **2 held-out classes** from `Defense_Positioning_v2.3.md`; Uddin category-level LOO [#18]. Ours: `numbers_map.md` §6–§8. **Synthesized**, not a canonical template.)*

---

## Phase 7 — Explainability (per-class TreeSHAP)

**Decision.** **Per-class TreeSHAP** on the **deduplicated 19-class XGBoost**, explicitly **contrasted against global SHAP** and **cross-checked against an independent Cohen's-d ranking**. Anchors (`numbers_map.md`): 4,180,000 attributions; DDoS↔DoS cosine **0.991**; SHAP vs Cohen's-d Jaccard **0.000** (ρ −0.741).

| Competitor | Their approach → number | Ours | Edge or overlap (caveat in-row) |
|---|---|---|---|
| **Lipsa, Dash & Ivković (2025)** [#06] | RF **99 %** + **per-class SHAP waterfalls** (per attack type) | per-class TreeSHAP on **deduplicated 19-class XGBoost**, vs global, cross-checked vs Cohen's-d | **Lipsa is genuine per-class-SHAP precedent — so we do NOT claim "first per-class SHAP."** Our claim is the **conjunction Lipsa lacks**: deduplicated + 19-class XGBoost + explicit global-vs-per-class contrast + independent Cohen's-d cross-check (Jaccard 0.000). |
| **Yacoubi et al. (2025, COCIA)** [#02] | RF + CatBoost with **global SHAP + LIME** | per-class SHAP | **Global-scope only**, no per-class breakdown, on raw data. Overlap on "uses SHAP," divergent on granularity. |
| **Alfageer (2026) / Manoj (2025) / Abo-Haat (2026)** | XAI deferred to **future work**, or qualitative (KAN B-splines) | quantitative per-class attribution integrated in the pipeline | Among the most recent frameworks, feature-level explainability is **deferred or qualitative**; ours is quantitative and per-class. |
| **No prior art does this** | — | per-class SHAP **cross-checked against an independent Cohen's-d ranking** → Jaccard 0.000, ρ −0.741 ("statistical separation ≠ model reliance") | No prior CICIoMT2024 SHAP work **cross-validates attributions against a second importance method** to surface method-dependence. |

*(Competitor: Lipsa per-class SHAP + RF 99 % [#06], Yacoubi-COCIA global SHAP+LIME [#02] grep-verified. Ours: `numbers_map.md` §9.)*

---

## [unverified] numbers dropped (gate 2)

These figures appeared in the truth files or the task brief but could **not** be grep-verified in `~/Downloads/research/papers/*.md`, so they were **dropped** (or replaced with a verified, labeled value):

| Dropped figure | Where it came from | Resolution |
|---|---|---|
| Recon_VulScan "strict recall **0.649** at p95" | task brief weak-spot note | **Dropped** — not in `numbers_map`/results. Replaced with the two real, *labeled* metrics: **entropy-gate rescue rate 0.745** (entropy_benign_p95) and **AE-only strict zero-day recall 0.441** (`numbers_map.md` §7–§8). numbers_map wins. |
| Alfageer **τ = 0.65** confidence threshold | `Defense_Positioning_v2.3.md` (truth source) | **RESTORED** — verified in v2.3 §"Decision-level positioning" ("max-probability threshold, τ = 0.65"). The `papers/*.md` grep missed only the τ glyph; gate 1 (truth files) backs it. Now stated concretely in the Phase 6 zero-day-signal row. |
| Alfageer **2 held-out classes** (Recon-OS_Scan, MQTT-DDoS-Connect_Flood) | `Defense_Positioning_v2.3.md` (truth source) | **RESTORED** — verified in v2.3 ("2 classes held out (Recon-OS_Scan, MQTT-DDoS-Connect_Flood)"). Now named in the Phase 6 protocol-granularity row. |
| Uddin per-class unknown-F1 (Spoofing 26.29 %, DDoS 60.96 %, …) | Literature_Review v6.4 / Synthesis | **Dropped** — not in [#18] extract. Only the verified "category-level LOO" mechanism is used. |
| Yacoubi-COCIA **99.92 %** accuracy | Synthesis / agent map | **Dropped** — not surfaced in [#02] extract. Only the verified "global SHAP + LIME" approach is used. |
| Chandekar AE **fixed 95th-percentile** threshold | Literature_Review v6.4 | **Dropped** the percentile specific — not in [#05] extract. Stated as "anomaly flagging, no zero-day mechanism." |
| Dadkhah imbalance **~2,158:1** | Synthesis | **Not stated** — unverified in [#01]; only our own post-dedup **2,374:1** (`numbers_map`) is used. |
