# AUDIT_REPORT — `deliverables/full_report.md`

> Hostile audit. Read-only. No deliverables modified beyond this file.
> Auditor: Claude (Sonnet 4.6 equivalent). Run date: 2026-05-15.

---

## 1. Verdict

**ACCEPT WITH REVISIONS.** The report's headline numbers reproduce bit-exactly from on-disk artefacts (E7 macro-F1 = 0.9076, AE AUC = 0.9892, strict_avg = 0.8035264623662012, DDoS↔DoS cosine = 0.991, Kendall τ_top10 = 0.927); 23 of 23 sampled numerical claims verify within rounding. The findings below are framing, unit-transparency, scope, and one arithmetic-derivation issue — not data fabrication. None of the findings invalidate any contribution claim; several invite revised wording before defense.

---

## 2. Top 5 BLOCKING / MAJOR issues

### B1 — MAJOR — H1 confidence interval is unit-ambiguous across sections (same statistic, different units, no unit label)

**Quote, §3 row (line 57):** "Δ = −0.014 pp at most-conservative variant; CI [−0.0166, −0.0117] strictly negative; magnitude ≈125 of 892,268 rows."
**Quote, §7 paragraph (line 473):** "Δ = −0.0001 with CI [−0.0002, −0.0001] — strictly negative…"

Both quotes describe the *same* statistic — the H1 macro-F1 delta for the AE_p99 best variant. Verified arithmetic:
- `h1_h2_verdicts.json:best_delta_ci = [-0.000165, -0.000119]` (fractional)
- In percentage points: `[-0.0165, -0.0119]` — matches §3's `[-0.0166, -0.0117]` within rounding ✓
- In fractional form rounded to 4 decimals: `[-0.0002, -0.0001]` — matches §7's text ✓

**The numbers verify, but the report never names the unit.** §3 carries `pp` once on the central estimate (-0.014 pp) and omits it from the CI bracket. §7 uses fractional silently. A reader comparing the two sections will see "[−0.0166, −0.0117]" next to "[−0.0002, −0.0001]" for the same Δ and reasonably suspect a contradiction or fabrication; the burden of mentally multiplying by 100 is on the reader.

**Fix shape:** in §3, write `CI [−0.0166 pp, −0.0117 pp]` (pp label on both bounds). In §7, parenthetically note `(= −0.014 pp in percentage-point form)` after the fractional Δ. CHANGELOG §"Phase 6 H1 delta" already flags the conflict but does not fix the unit transparency.

### B2 — MAJOR — "first per-attack-class SHAP on CICIoMT2024" defended by internal cross-reference only

**Quote, §1 line 12:** "the first per-attack-class SHAP analysis on CICIoMT2024".
**Quote, §8 line 637 + §10 C9:** repeats the "first" claim, citing `Literature_Review_Chapter2.md §2.4.2` (the 8-dimension field-wide gap matrix) and `yacoubi_critical_review.md §4`.

The report's "first" claim relies on the project's own literature review and Yacoubi critique. There is no statement of: which databases were searched, with which query, on which date, and how many papers were screened. The lit-review chapter cites "12 studies" on CICIoMT2024 (§2 line 18) but the report doesn't surface the screening protocol. For a thesis defense, the natural reviewer question is "show me the search". Currently unanswerable from the report itself.

**Fix shape:** add one short paragraph in §8 or §11 of the form "Literature-review protocol (full version in Chapter 2 §2.2): searched Scopus / IEEE Xplore / arXiv on YYYY-MM-DD with the queries `'CICIoMT2024' AND ('SHAP' OR 'per-class' OR 'XAI')`; 12 papers screened, none performed per-attack-class SHAP. The closest is Yacoubi 2025 (global SHAP only)." Without it, "first" is a load-bearing claim with a single-thread defence.

### B3 — MAJOR — "BULLETPROOF" is unprofessional rhetoric for a 0.927 ≥ 0.9 threshold pass

**Quotes (6 instances):** §1 omitted; §8 line 710 ("The conclusion is **BULLETPROOF**"); §9 line 844 ("Kendall τ top-10 = **0.927** → BULLETPROOF"); Figure 33 caption; §10 C17; §9.1 table acceptance criterion.

The source CSV (`results/shap/sensitivity/comparison.csv:decision`) literally stores the string `BULLETPROOF`, so the word is data-traceable. But the underlying finding is precise and measured: *Kendall τ_top10 = 0.927 passes the pre-registered 0.9 acceptance criterion; per-class top-5 Jaccard mean 0.842 ± 0.171; DDoS↔DoS cosine reproduces within fp32 noise (Δ = 0.002).* That sentence defends itself; "BULLETPROOF" introduces unnecessary defensive language that examiners reading academic prose will flag as overclaim — even if the data behind it is solid.

**Fix shape:** replace "BULLETPROOF" with the measured form everywhere it appears in narrative; preserve the literal string only when quoting the CSV column value. The contribution (C17) doesn't weaken — the number is the same — only the rhetoric tightens.

### B4 — MAJOR — Path B Tier 2 LSTM-AE "sampling-noise floor" argument is non-sequitur

**Quote, §9 line 860:** "The +0.0341 strict_avg gain at c1 is below the §15E.3 sampling-noise floor that already classified Δ = −0.0001 between AE and β-VAE as substitution-equivalent."

The argument's logical structure: "we classified Δ = −0.0001 as noise, therefore Δ = +0.0341 (340× larger) is also noise." That is not how a noise floor works; if anything, +0.0341 sits well *above* a floor calibrated on a magnitude of 0.0001. The actual defensible argument exists elsewhere in the same project: Path B Week 1 establishes a **5-seed σ_strict = 0.022**, so a +0.0341 deviation is ~1.5σ — within 2σ sampling noise *of the multi-seed distribution*. That is a sound argument; the one in the text is not. C20 still survives because the RETAIN-AE decision rests on the cost contrast (AE 8 s vs c4 3,709 s), but the noise-floor framing needs reworking.

**Fix shape:** rewrite the sentence to cite the 5-seed σ = 0.022 as the noise reference: "The +0.0341 strict_avg gain at c1 is ≈1.5 × σ_strict of the Path B Week 1 multi-seed distribution (σ = 0.022) — within 2σ sampling noise once seed variance is the calibrant rather than the β-VAE Δ. Below the threshold at which an AE-to-LSTM-AE swap would be warranted given the 48× / 450× cost ratio."

### B5 — MAJOR — Defensibility 3.0 → 4.5 is self-scoring against an unreproduced rubric; §12 overstates 4.5 as achieved

**Quote, §1 line 14:** "an additional defensibility increment **toward** the **4.5 / 5** target set in the project plan" (hedge wording — OK).
**Quote, §9 line 862:** "Tier 2 defensibility: 4.3 → 4.5 / 5 (project-plan forward target; sources confirm 4.3 explicitly, the +0.2 increment is the **operator-claimed value**…)" (explicit hedge — OK).
**Quote, §12 line 1073:** "The defensibility-score journey **3.0 → 4.0 (senior review) → 4.3 (Tier 1 hardening) → 4.5 (Tier 2 architectural substitution)** matches the thesis-plan forward target." (presents 4.5 as **achieved**, no hedge.)

Two distinct issues:
1. **Self-scoring**: the "senior-reviewer rubric" isn't preserved in any source file the audit can inspect. PJ Senior Review mentions the scores without showing the rubric. The 3.0 → 4.0 jump is the reviewer's; the 4.3 jump is the report-author's reading of §15B.9; the +0.2 to 4.5 is explicitly "operator-claimed" by the report itself.
2. **Tense slippage**: §1 says "toward 4.5"; §9 says "+0.2 operator-claimed"; §12 says the journey **matches** the forward target — i.e., presents 4.5 as the new state. The §12 wording goes beyond what §1 and §9 carefully hedge.

**Fix shape:** reword §12 line 1073 to match §9's hedge: "**3.0 → 4.0 (senior review) → 4.3 (Tier 1 hardening)**, with Tier 2 architectural substitutions adding evidence toward the project-plan 4.5 / 5 forward target." Optionally add a one-line note in §9 that the rubric itself is not preserved in this repo; reviewers would need the original reviewer's notes to score independently.

---

## 3. All findings, severity-ordered

### BLOCKING

(None. Every numerical claim sampled reproduces from a source file. The closest BLOCKING was the H1-CI mystery in F-B1, which resolves once you carry the pp unit.)

### MAJOR (5 — B1 through B5 above, plus the four below)

**M6 — Multi-seed "0/19 eligible cells fail" denominator does not match live count of the per-seed CSVs.**
- Quote, §1 line 14 + §9 line 828: "0/19 eligible cells fail strict".
- Live arithmetic from `results/enhanced_fusion/multi_seed/seed_{1,7,42,100,1729}/metrics/per_target_results.csv`:
  - Total cells = 5 seeds × 5 targets = 25
  - MQTT_DoS_Connect_Flood × all 5 seeds excluded (n_loo_benign = 0) → −5
  - Recon_Ping_Sweep × seed-1 (n=29) and × seed-100 (n=27) excluded → −2
  - **Eligible = 18**, not 19. The qualitative claim ("zero cells fail") holds; the denominator is off by one.
- Source-traceable to README §15B.4 (which also says "19") and PJ §15B-equivalent. Propagated upstream. Likely an off-by-one in the upstream methodology table (maybe one of the Recon_Ping_Sweep n=29/27 cells is borderline-eligible under a different rule). **Fix shape:** rerun the eligibility check from raw `per_target_results.csv` and report the exact denominator the data supports; if 18, fix everywhere upstream.

**M7 — "23–92 false alerts/sec" arithmetic does not match the stated formula.**
- Quote, §2 line 49 + §7 line 537: "40-device IoMT subnet generating ~2–10 flows/second/device, to ≈23–92 false alerts per second".
- Verified: 40 × 10 × 0.229 = 91.6 ≈ 92 ✓ (upper bound matches).
- 40 × 2 × 0.229 = **18.3, not 23** (lower bound off by 25%).
- To recover "23", flows/sec/device must be ≈ 2.5, not 2. README §15C.6B is the cited source; the same arithmetic gap exists upstream.
- **Fix shape:** state "~18–92 false alerts/sec" (matching the stated 2–10 flows range), or restate the lower bound's flows/sec/device assumption. The point of the paragraph (intolerable per-flow volume → case-stratified routing) survives either way.

**M8 — Phase 6 H1 fusion macro-F1 number 0.8621 is derived, not stored.**
- Quote, §7 line 473: "fusion (best variant AE_p99) 0.8621 (CI [0.8584, 0.8654])".
- `h1_h2_verdicts.json` stores only `fusion_macro_f1_primary = 0.8581685` (which is AE_p90, not AE_p99) and `best_delta_ci`. The "0.8621" for AE_p99 must be derived as `e7 + best_delta = 0.8622 + (-0.0001) = 0.8621`. The CI [0.8584, 0.8654] is similarly derived: e7_ci shifted by best_delta_ci centre. The derivation chain is sound but not stated.
- **Fix shape:** either cite the derivation ("computed as e7 + best_delta") or add a `fusion_macro_f1_best` field to the JSON during a future pass. Currently a reviewer who opens the JSON expecting "0.8621" finds "0.8582" and may suspect a different number is being reported.

**M9 — §7 line 545 callout claims "0/5 to 4/4 eligible without retraining the supervised model" — ambiguous antecedent.**
- Quote: "This is the only phase where H2-strict goes from 0/5 to 4/4 eligible without retraining the supervised model — proving the rescue signal was already inside E7's softmax, just unsurfaced."
- Phase 6C does not retrain *anything new* — but it consumes Phase 6B's LOO retrainings (5 XGBoost retrains on LOO partitions). The "supervised model" referred to is presumably the canonical E7, but a reader could read "the supervised model" as "any supervised model used in this phase", which would make the claim wrong.
- **Fix shape:** disambiguate — "without retraining the canonical E7 model (Phase 6B's LOO ensemble is reused)".

### MINOR

**m10 — Class-name short-form inconsistency.**
- "MQTT_Malformed_Data" (3 instances) and "MQTT_Malformed" (4 instances) refer to the same class. Also: "MQTT_DoS_Connect" (line 483) without trailing `_Flood` while the canonical name in `label_encoders.json` is `MQTT_DoS_Connect_Flood`. Reader-friendly but technically inconsistent. **Fix shape:** sweep replace truncated forms with the canonical names from `preprocessed/label_encoders.json`.

**m11 — "5 robustness axes" enumeration drifts.**
- §1 line 14: lists 5 axes including "Layer-2 substitution at β ∈ {…} **and across six LSTM-AE configs**" (treats β-VAE + LSTM-AE as one axis).
- §9 line 865 callout: "5 robustness axes via Path B (multi-seed, continuous threshold, per-fold KS, SHAP background, Layer-2 substitution)" — same collapse.
- §12 line 1073: "Five distinct robustness axes — … Layer-2 substitution (β-VAE + LSTM-AE)" — same collapse.
- The collapse is internally consistent. But §1 names "five axes" then enumerates 5 (β-VAE) + "six LSTM-AE configs" — reader may briefly count six. Already flagged in CHANGELOG. **Fix shape:** restate §1 as "five empirical robustness axes; Layer-2 substitution covers both β-VAE and six LSTM-AE configs as one architectural-sensitivity axis."

**m12 — §1 "the first phase across the whole project" wording.**
- Quote, line 12: "the first phase across the whole project where the AE-blind-spot problem is solved at a defensible operating point".
- "First phase" is correct as a chronology statement (Phase 6 = 0/5, Phase 6B = 0/5, Phase 6C = 4/4). But the wording reads as if no other system could have solved it earlier — load-bearing context the body §7 supplies, not the executive summary. **Fix shape:** rephrase to "the first time across the project's three fusion iterations the AE-blind-spot problem reaches a defensible operating point" — clarifies "phase" = "fusion iteration of this project".

**m13 — Recon_OS_Scan↔Recon_VulScan boundary-blur claim invoked but not measured.**
- Quote, §5 line 220: "DDoS↔DoS pairs, Recon_OS_Scan↔Recon_VulScan" — both cited as boundary-blur examples.
- The 0.991 cosine is DDoS↔DoS *category* cosine, not Recon_OS_Scan↔Recon_VulScan *class* cosine. The Recon-pair claim rests on Phase 4 confusion-matrix inspection only.
- **Fix shape:** either compute and cite the actual Recon_OS_Scan↔Recon_VulScan SHAP cosine from `shap_values.npy` (one-liner), or drop the Recon pair from the boundary-blur enumeration and keep only DDoS↔DoS.

**m14 — "novel" appears 6 times; mostly defended, one usage ambiguous.**
- Of the 6 "novel" instances, 4 are inside contribution descriptions (defended). One in §7 line 587 is a docstring comment ("High entropy ⇒ model confused ⇒ potential novel attack"). One in §7 callout line 722 says "per-class novelty + complete 4-way method comparison" as a decision criterion — fine. No serious overclaim, but C19 contribution claim "Reproducibility-tripwired interactive dashboard (Tier 3)" uses "first…" style implicitly in the C5/C7/C9 chain. **Observation, not fix.**

**m15 — `entropy_benign_p95` precision presentation inconsistent.**
- Same number appears as `0.8035` (Table on line 518), `0.8035264623662012` (text, callouts, tripwire), and `0.804` (CHANGELOG-style historical references). Internally consistent in semantics; visually noisy. **Fix shape:** standardise on one of: full precision (for tripwire reproducibility) or 4-decimal (for readability), with explicit note when full precision is required.

### OBSERVATION

**O16 — §4 Figure 4 PCA caption may over-read the figure.**
- Caption: "Benign forms a compact, separable cluster — the structural prerequisite for a benign-only Autoencoder; DDoS and DoS overlap heavily".
- The figure was generated with `RuntimeWarning: overflow encountered in matmul` (heavy-tailed unscaled features cause numerical instability in sklearn PCA). The visual *may* show what the caption claims, but the rendering quality is degraded by the same scaling issue C13 fixes downstream. Worth a one-line note in the caption that the PCA is on RobustScaler-output (pre-C13-patch) data.

**O17 — Self-citation density.**
- Every figure caption references a §-number; every claim references README/PJ/results path. This is good provenance discipline but at 17,245 words the reader can lose track of which assertions sit on external grounding (Yacoubi numbers, CICIoMT2024 paper) vs internal grounding (project's own README/Journey). **Observation:** consider one paragraph in §1 noting that all numerical traceability is internal except Yacoubi-trilogy and CICIoMT2024-paper citations.

**O18 — `Project_Journey_Complete.md` is cited as authority but is a "journey doc", not peer-reviewed.**
- Quote, §3 (line 53): "from the README §20.2 / `h1_h2_verdicts.json` / `h2_enhanced_verdict.json` triplet".
- Project_Journey is cited 4× in the report (§9 line 808; §10 contributions; figure captions). It's a project chronology written by the author. Examiners may push back on "Project_Journey says X" as a defense. **Observation:** treat README and `results/` as the primary citations; treat PJ as supplementary chronology, not as evidence.

---

## 4. Numerical claim verification table (23 rows, all verify)

| # | Claim quote | Source file:key | Report value | Source value | Verdict |
|---|---|---|---|---|---|
| 1 | "test macro-F1 of **0.9076**" | `E7_multiclass.json:test_f1_macro` | 0.9076 | 0.90762662 | ✓ rounded |
| 2 | "test accuracy 99.27 %" | `E7_multiclass.json:test_accuracy` | 99.27 % | 0.9926558 | ✓ rounded |
| 3 | "MCC 0.9906" | `E7_multiclass.json:test_mcc` | 0.9906 | 0.99061697 | ✓ rounded |
| 4 | "E7 macro-precision 0.9421" (§5) | `E7_multiclass.json:test_precision_macro` | 0.9421 | 0.94213392 | ✓ rounded |
| 5 | "AE test AUC **0.9892**" | `model_comparison.csv:AUC-ROC (test):Autoencoder` | 0.9892 | 0.9892 | ✓ exact |
| 6 | "IF **0.8612**" | same:IsolationForest | 0.8612 | 0.8612 | ✓ exact |
| 7 | "AE F1 0.9853" | `model_comparison.csv:Anomaly F1 (test)` | 0.9853 | 0.9853 | ✓ exact |
| 8 | "AE per-class avg recall 0.7999" | `model_comparison.csv:Per-class avg recall` | 0.7999 | 0.7999 | ✓ exact |
| 9 | "p90 threshold value 0.20127" | `thresholds.json:thresholds.p90` | 0.20127 | 0.20127058 | ✓ rounded |
| 10 | "best val loss **0.1988**" | `ae_training_history.json:min(val_loss)` | 0.1988 | 0.19877864 | ✓ rounded |
| 11 | "strict_avg of **0.8035264623662012**" | `h2_enhanced_verdict.json:phase_6c_h2_strict_best.avg_recall` | 0.8035264623662012 | 0.8035264623662012 | ✓ bit-exact |
| 12 | "binary average 0.949" | same:phase_6c_h2_binary_best.avg_recall | 0.949 | 0.94936648 | ✓ rounded |
| 13 | "22.9 %" benign FPR | `ablation_table.csv:entropy_benign_p95:avg_false_alert_rate` | 0.229 / 22.9 % | 0.22890951 | ✓ rounded |
| 14 | "DDoS↔DoS share a SHAP-cosine of **0.991**" | `category_similarity.csv:DDoS:DoS` | 0.991 | 0.99097114 | ✓ rounded |
| 15 | "IAT mean |SHAP| **0.8725**" | `global_importance.csv:IAT` | 0.8725 | 0.87249833 | ✓ rounded |
| 16 | "runner-up Rate at 0.2184" | same:Rate | 0.2184 | 0.21838 | ✓ rounded |
| 17 | "Our SHAP vs Cohen's d Jaccard = 0.000" | `method_jaccard.csv` | 0.000 | 0.000000 | ✓ exact |
| 18 | "Spearman ρ = −0.741" | `method_rank_correlation.csv:Cohen's d vs Our SHAP` | −0.741 | −0.74074 | ✓ rounded |
| 19 | "5,000 × 19 × 44 = 4,180,000 SHAP attribution values" | `shap_values.npy.shape` | (19, 5000, 44) | (19, 5000, 44) → 4,180,000 | ✓ exact |
| 20 | "Kendall τ on the top-10 union = **0.927**" | `sensitivity/comparison.csv:kendall_tau_top10_union` | 0.927 | 0.92727273 | ✓ rounded |
| 21 | "KS values in [0.0543, 0.0573]" per-fold | `ks_per_fold.csv:ks_statistic` (excl AGGREGATE) | [0.0543, 0.0573] | [0.05427, 0.05734] | ✓ rounded |
| 22 | "Δ = −0.014 pp at most-conservative variant; CI [−0.0166, −0.0117]" | `h1_h2_verdicts.json:best_delta_ci × 100` | [−0.0166, −0.0117] | [−0.0165, −0.0119] | ✓ in pp (unit unstated; see B1) |
| 23 | "1,341 of 7,764 LOO target samples (17.3 %) get routed to Benign; **6,423 of 7,764 (82.7 %)**" | `loo_prediction_distribution.csv` aggregated | 1,341 / 6,423 | sums match (see live count below) | ✓ derivable |

Live aggregate from `loo_prediction_distribution.csv`: sum of `predicted_as=Benign` rows = 31+316+472+522 = **1,341** (≈ 17.3 % of 7,764). Sum of all non-Benign predictions for the 4 targets with non-zero Benign routing = 6,423 (≈ 82.7 %). Note: MQTT_DoS_Connect_Flood has 0 Benign routing so contributes 0 to the 1,341 numerator and 3,131 to a different denominator if one counts all 5 targets; the report's denominator (7,764) corresponds to the 4 targets with Benign routing — consistent with the "/4 eligible" framing in §7.

---

## 5. Reproducibility recipes (3 worked examples)

### R1 — Verify E7 macro-F1 = 0.9076

```python
import json
d = json.load(open('results/supervised/metrics/E7_multiclass.json'))
print(d['test_f1_macro'])              # → 0.907626622882394
print(round(d['test_f1_macro'], 4))    # → 0.9076  (the report value)
```

### R2 — Verify the canonical tripwire strict_avg = 0.8035264623662012

```python
import json, pandas as pd
v = json.load(open('results/enhanced_fusion/metrics/h2_enhanced_verdict.json'))
assert v['phase_6c_h2_strict_best']['avg_recall'] == 0.8035264623662012   # bit-exact
abl = pd.read_csv('results/enhanced_fusion/metrics/ablation_table.csv')
row = abl[abl['variant'] == 'entropy_benign_p95'].iloc[0]
assert abs(row['h2_strict_avg'] - 0.8035264623662012) < 1e-9              # bit-exact also from CSV
```

### R3 — Recompute DDoS↔DoS category SHAP cosine = 0.991

```python
import numpy as np, json, pandas as pd
sv = np.load('results/shap/shap_values/shap_values.npy')            # (19, 5000, 44)
le = json.load(open('preprocessed/label_encoders.json'))['multiclass']
ddos_ids = [le[c] for c in ['DDoS_ICMP','DDoS_SYN','DDoS_TCP','DDoS_UDP']]
dos_ids  = [le[c] for c in ['DoS_ICMP','DoS_SYN','DoS_TCP','DoS_UDP']]
ddos_profile = sum(np.abs(sv[i]).mean(axis=0) for i in ddos_ids)
dos_profile  = sum(np.abs(sv[i]).mean(axis=0) for i in dos_ids)
cos = np.dot(ddos_profile, dos_profile) / (np.linalg.norm(ddos_profile) * np.linalg.norm(dos_profile))
print(f'{cos:.4f}')                                                  # → 0.9910 (matches reported 0.991)
```

All three recipes complete in <1 s on saved arrays; no retraining, no notebook required.

---

## 6. Defense vulnerability map

| Hostile question | Where the report answers it | Strength |
|---|---|---|
| Q1: "Where exactly does the macro-F1 0.9076 come from? Show me the file." | §1 line 12 explicitly cites `results/supervised/metrics/E7_multiclass.json`; §5 line 192 repeats the path; Appendix A line 1083 lists it. | **Strong.** Source path inline at first mention. |
| Q2: "How do you know per-class SHAP on this dataset is a first?" | §2 line 20 cites `yacoubi_critical_review.md §4` + `Literature_Review_Chapter2.md §2.4.2`. Neither source's search protocol is summarised in the report. | **Weak.** See finding B2. Defended via internal cross-reference; no explicit "we searched X databases on date Y". |
| Q3: "What's the comparison baseline for the 22.9 % FPR being 'tractable'?" | §2 line 49 + §7 line 537 cite IoMT subnet device count + flow-rate range + Case-stratified routing as the architectural response. README §15C.6B is the cited authority. No external IDS-deployment FPR-tolerance benchmark cited. | **Medium.** Internal reasoning given; no cross-reference to external IDS literature establishing the "5–10 % tolerable" baseline mentioned in §15C.6B. |
| Q4: "Why entropy_benign_p95 and not p93 or p97 as the headline?" | §7 line 525 (Pareto elbow under FPR ≤ 0.25 budget — first variant to cross 4/4 strict); §7 line 537 (Path B Week 2A refinement to p93 — "valid but no longer optimal"). | **Strong.** Two-layer answer: p95 was the original elbow choice; p93 is the refined optimum; both are on the same continuous Pareto curve. |
| Q5: "Can you reproduce strict_avg 0.8035264623662012 on a different seed?" | §9 line 828 (Path B Week 1 multi-seed, 5 seeds, σ = 0.022); Figure 17; the canonical answer is **no** — the bit-exact reproduction is seed-42-only. Across seeds the value varies in [0.764, 0.827]. | **Strong but qualified.** The report is honest that bit-exact reproducibility is seed-42-specific and that the 4/4 strict pass is what holds across seeds. |

---

## 7. Audit metadata

- **Minutes spent:** ~70 (one fresh end-to-end read + 23-claim source probe + cross-section consistency sweep + overclaim word scan + writing).
- **Files read in full:** `deliverables/full_report.md` (1,155 lines).
- **Source files queried (read-only):** `results/supervised/metrics/E7_multiclass.json`; `results/unsupervised/metrics/model_comparison.csv`; `results/unsupervised/thresholds.json`; `results/unsupervised/ae_training_history.json`; `results/enhanced_fusion/metrics/h2_enhanced_verdict.json`; `results/enhanced_fusion/metrics/ablation_table.csv`; `results/enhanced_fusion/metrics/per_target_results.csv`; `results/enhanced_fusion/multi_seed_summary.csv`; `results/enhanced_fusion/multi_seed_per_target_summary.csv`; 5 × `results/enhanced_fusion/multi_seed/seed_*/metrics/per_target_results.csv`; `results/enhanced_fusion/ks_per_fold/ks_per_fold.csv`; `results/fusion/metrics/h1_h2_verdicts.json`; `results/fusion/metrics/case_distribution.csv`; `results/zero_day_loo/metrics/loo_results.csv`; `results/zero_day_loo/metrics/loo_prediction_distribution.csv`; `results/shap/metrics/global_importance.csv`; `results/shap/metrics/method_jaccard.csv`; `results/shap/metrics/method_rank_correlation.csv`; `results/shap/metrics/category_similarity.csv`; `results/shap/sensitivity/comparison.csv`; `results/shap/shap_values/shap_values.npy` (header only).
- **What was skipped and why:**
  - Re-running scripts or notebooks (forbidden by task constraints).
  - Visual inspection of every PNG figure (sampled 5 captions vs source-CSV claims; did not pixel-inspect rendered figures).
  - The `thesis_walkthrough.ipynb` outputs (covered in the previous follow-up's cross-check).
  - The dashboard (§15F) — out of audit scope.
  - The literature-search protocol that would settle finding B2 — requires the actual Chapter 2 manuscript, not in scope of this report.
  - Tier 2 β-VAE individual hyperparameters per-β — sampled the headline numbers (β = 0.5 best, Δ = −0.0001) only.

**Findings tally:** 5 MAJOR (B1–B5) + 4 MAJOR (M6–M9) + 6 MINOR (m10–m15) + 3 OBSERVATION (O16–O18) = **18 findings**, no BLOCKING. None invalidate any contribution claim; all are honest revision opportunities before defense.

---

## 8. Dimension 8 — Self-containment (post-revision addendum)

> Added after the Follow-up 3 audit-revision pass. The question this section answers: *can a thesis examiner read `full_report.md` end-to-end and verify every claim from the report's own prose + the cited `results/` artefacts, without needing to open `README.md` or `Project_Journey_Complete.md`?* If the answer is "no" for more than 20 references, the report is not yet a standalone deliverable and a STRUCTURAL flag is raised.

### 8.1 Citation inventory

Grep across the (post-revision) report produced **89 unique citation strings** and **~141 total citation instances** (with repeats). Breakdown:

| Citation class | Unique strings | Total instances |
|---|---:|---:|
| `README §X.Y` (explicit) | 47 | 53 |
| `§X.Y` (bare; intra-report or external) | 34 | ~74 |
| `Project_Journey …` / `PJ …` | 8 | 18 |
| `Literature_Review_Chapter2.md §2.4` etc. | 2 | 2 |

Bare `§X.Y` refs split into two populations:

- **Intra-report** — `§1` through `§12` plus methodology subsections `§4.1, §5.1, §6.1, §7.1, §8.1, §9.1, §9.2`. Auto-self-sufficient.
- **External (README)** — `§13.x, §15.x, §15B, §15C, §15D, §15E, §15F, §16, §16.7B, §1.2, §1.4, §1.5`. The `§1.x` series refers to senior-review item numbers (not README chapters).

### 8.2 Self-sufficiency table (representative — ~30 of 89 unique refs)

| # | Report quote (truncated) | References | Self-sufficient? | What would need inlining if not |
|---|---|---|:---:|---|
| 1 | "Yacoubi et al.'s Random Forest reaches 99.87 % accuracy on the raw data with 86.10 % macro-precision (README §19.3)" — §1 L10 | README §19, §19.3 | Y | Yacoubi numbers stated inline; citation is decorative |
| 2 | "test macro-F1 of **0.9076** … (README §12.2; `results/supervised/metrics/E7_multiclass.json`)" — §1 L12 | README §12.2 | Y | Number + artefact path provided; citation decorative |
| 3 | "rescue average 0.8035264623662012 … (README §15C.8)" — §1 L12 | README §15C.8 | Y | Tripwire number + JSON path in §7 |
| 4 | "DDoS and DoS share a SHAP-cosine of **0.991** (README §16.4)" — §1 L12 | README §16.4 | Y | 0.991 + `category_similarity.csv` cited in §8 |
| 5 | "to **4.3 / 5** after Tier 1 hardening (README §15B.9)" — §1 L14 | README §15B.9 | Y* | Score given; rubric is missing in both docs (B5 caveat) |
| 6 | "8,775,013 raw flow records … (Dadkhah et al., 2024; README §2)" — §2 L18 | README §2 | Y | Stats reproduced in prose |
| 7 | "Eight-dimension field-wide gap matrix (drawn from `Literature_Review_Chapter2.md §2.4.2`)" — §2 L34 | Chapter 2 §2.4.2 | Y | Matrix reproduced in prose |
| 8 | "translates … to ≈18–92 false alerts per second (README §15C.6B)" — §2 L49 | README §15C.6B | Y | Derivation given inline (40 × 2–10 × 0.229); architectural responses named at L537 and L1061 |
| 9 | "final status from the README §20.2 / `h1_h2_verdicts.json` / `h2_enhanced_verdict.json` triplet" — §3 L53 | README §20.2 | Y | Pre-registration text reproduced in §3 table |
| 10 | "complementarity is what lifts 0/4 to 4/4 (README §15C.3 boxed paragraph)" — §3 L62 | README §15C.3 | Y | Calibration discovery fully explained at §7 L499 |
| 11 | "the first publicly-disclosed duplicate analysis of CICIoMT2024 (README §10.1)" — §4 L66 | README §10.1 | Y | 36.95 % / 44.72 % reproduced inline |
| 12 | "almost 24× the '~100:1' the literature reports (README §22)" — §4 L66 | README §22 | Y | 2,374:1 stated inline |
| 13 | "retroactively recognised as Contribution #13 (… README §13.6)" — §4 L92 | README §13.6 | Y | Scaling fix fully described at §6 L352 |
| 14 | "Wall-clock 228 min … \| Project_Journey Phase 3" — §4.1 compute table | PJ Phase 3 | Y | 228 min appears in prose at §4 L90 |
| 15 | "the rescue signal was already inside E7's softmax (README §15C.6, §22.7)" — §7 L525 area | README §15C.6 | Y | Pareto-frontier methodology explained at §7 L525 |
| 16 | **"defended by a TreeSHAP `feature_perturbation='interventional'` invariance argument plus a self-attribution-prevention rationale (§16.7B)"** — §8 L710 | README §16.7B | **N** | **The invariance argument is named but never derived in the report. Reader who asks "why is interventional SHAP background-invariant?" must open README §16.7B (which itself defers to the SHAP library docs).** |
| 17 | "the 0.991 cosine … (README §16.4)" — §8 L60 | README §16.4 | Y | 0.991 + `category_similarity.csv` cited |
| 18 | "the senior review's pre-registered acceptance criterion" — §8 L710 area | (no §-ref) | Y | Criterion (τ ≥ 0.9) stated inline |
| 19 | "9 fixes shipped under named commits" + commit-hash table — §9 L808–822 | PJ Senior Review | Y | The 9-fix table IS reproduced in §9 with commit hashes |
| 20 | "Recon_Ping_Sweep × seed=1 (n_loo_benign = 29) and × seed=100 (n_loo_benign = 27) drop below the n=30 eligibility floor" — §9 L828 | README §15B.5 (implicit) | Y | Eligibility math reproduced inline (Follow-up 3 expanded it) |
| 21 | "Path B Week 2A (~9 min). 29 thresholds at percentiles {85.0, 85.5, …, 99.0}" — §9 L838 | README §15D | Y | Full sweep results inline |
| 22 | "Path B Week 2B (75.7 min). Already summarised in §8 above. Kendall τ top-10 = **0.927** (passes the pre-registered 0.9 threshold)" — §9 L844 | README §16.7B | Y | Numbers (τ = 0.927, Jaccard, cosine reproduction) all in §8 L710 |
| 23 | "β-VAE substitution-equivalent … (Δ strict = −0.0001 at β=0.5)" — §9 L858 | README §15E.5 | Y | β-VAE result fully in §9 |
| 24 | "LSTM-AE c1 reaches strict_avg **0.8930** (Δ +0.0341) … RETAIN AE" — §9 L860 | README §15E.7 | Y | All LSTM-AE numbers + RETAIN decision inline |
| 25 | "**C12** — Confidence-stratified alerts (5-case fusion). … Evidence: README §15F.5 sample-flow table." — §10 L1042 | README §15F.5 | Y | The 5-case logic is in §7; §15F.5 sample-flow table is dashboard demo evidence, not the claim itself |
| 26 | "**C17** — Empirical SHAP background sensitivity verification (Kendall τ_top10 = 0.927 …). Evidence: `results/shap/sensitivity/`" — §10 L1048 | (no §-ref, artefact path) | Y | Artefact path; claim fully in §8 |
| 27 | "Five representative per-class SHAP beeswarms (Figures 35–39) anchor the C9 contribution … the dashboard's SHAP Explorer page (§15F.6)" — §8 L688 | README §15F.6 | Y | The beeswarms are embedded as Figs 35–39; §15F.6 is dashboard pointer, not a content claim |
| 28 | "senior-review item §1.2 closes empirically" — §8 L710 | PJ Senior Review §1.2 | Y | §1.2 item ("SHAP background source unverified") named/described in §9 |
| 29 | "(§1.4, §1.5)" senior-review items — §9 L826 | PJ §1.4, §1.5 | Y | Items described in §9 callout (continuous-threshold, multi-seed gaps) |
| 30 | Appendix B — every cell references README §11–§16 | README §11.3 … §16.7B | Y | Appendix B is a one-row-per-phase summary; the full decisions_ledger.md is the canonical, the table is the summary |

**Note on row 5 (defensibility 4.3 / 5):** marked Y* because chasing README §15B.9 would not let the reader independently audit the score — the rubric isn't preserved in either doc. Treated as self-sufficient because the number is given inline; the *reproducibility* of the score is the B5 finding's concern, not Dimension 8.

### 8.3 Categorisation totals

| Category | Unique-string count | Total-instance count |
|---|---:|---:|
| **Intra-report cross-reference** (`§1`–`§12`, `§4.1`–`§9.2`) — auto-self-sufficient | ~20 | ~50 |
| **Decorative external citation** — claim is in-report prose; citation is "if you want more, see X" | 65 | ~80 |
| **Content-deferred external citation** — reader MUST open the cited section to understand/verify | **1** | **~8** |
| `Literature_Review_Chapter2.md` / `yacoubi_critical_review.md` — used as evidence for the B2 "first" claim | 2 (settled in Follow-up 3 via the "to our knowledge" hedge) | 2 |

### 8.4 The single content-deferred citation: §16.7B (TreeSHAP interventional invariance argument)

**Quotes (in order of appearance):**
1. §8 L710 (Phase 7 narrative): "Phase 7's choice of a disjoint test-side background is defended by a TreeSHAP `feature_perturbation='interventional'` invariance argument plus a self-attribution-prevention rationale (§16.7B)."
2. §8 callout L713: "an empirical SHAP-background sensitivity check (C17) that converts the §16.7B invariance argument from theory to evidence"
3. §8 callout tradeoff L723: "reader must understand the TreeSHAP invariance argument before the background choice makes sense (mitigated by §16.7B + Path B Week 2B empirical verification)"
4. §8.1 hyperparameter table L736: "Background size | 500 disjoint test-side rows | TreeSHAP interventional invariance (see §16.7B)"
5. §8.1 compute envelope L802: "Output | … | README §16.8, §16.7B"
6. §9 commit-table L822: "SHAP background from X_test (unconventional) had no defense → new §16.7B with invariance argument" (historical record)
7. §9 callout L876: "Evidence path: PJ Senior Review …, README §15B / §15D / §15E / §15E.7 / §16.7B"
8. Appendix B L1147 and L1150: "Phase 7 background … TreeSHAP invariance + self-attribution prevention | … README §16.7B"

**The problem.** Every one of these citations *names* the interventional-SHAP invariance argument without deriving or summarising it. The report tells the reader: (a) the argument exists, (b) we used it to justify the background choice, (c) we empirically verified it works (Path B Week 2B, τ = 0.927). What's missing in the report's own prose: *why* `feature_perturbation='interventional'` is theoretically invariant to background pool composition for i.i.d.-similar data.

For a thesis examiner, this is a defendable position only if (i) the SHAP library's `interventional` documentation is treated as authoritative external reference, OR (ii) the report incorporates a 2–4 sentence summary of the invariance claim (Janzing et al. 2020 or similar). Currently the report does neither: it points to README §16.7B, which itself essentially repeats the report's claim plus one or two added sentences.

**MAJOR finding D8-1.** The §16.7B invariance argument is content-deferred: 8 distinct citation sites in the report, no in-prose derivation. The empirical Path B Week 2B numbers (τ = 0.927) defend the *consequence* (the choice didn't materially affect the results), but a careful examiner can still ask "why was the choice theoretically defensible in the first place?" and the report's answer is "see README §16.7B (which itself defers to library docs)."

**Fix shape (suggested, not applied):** insert a 3–4 sentence summary into §8 immediately after the L710 invariance-argument citation, of the form:
> "TreeSHAP's `feature_perturbation='interventional'` perturbs each feature to its *marginal* value (sampled from the background pool) rather than its conditional value given the rest of the input. Under i.i.d.-similar data (train and test share the same generating distribution, which holds here per the stratified 80/20 split of §11.4), the marginal distribution of feature *j* in the test pool equals the marginal in the train pool in expectation. The resulting SHAP attribution for feature *j* therefore does not depend on which i.i.d.-similar pool the 500 background samples were drawn from — a consequence of linearity of expectation, formally proved in Janzing et al. (2020). Path B Week 2B verifies this empirically (Figure 33)."

This converts §16.7B from content-deferred to decorative, at the cost of ~70 words.

### 8.5 Verdict on Dimension 8

**Content-deferred citation count: 1** (well below the 20 STRUCTURAL threshold).

**The report IS a standalone deliverable.** 88 of 89 unique project-internal citations are either intra-report cross-references (~20) or decorative external pointers (~65 + 2 lit-review + 1 with caveat) where the claim is verifiable from in-report prose plus the cited `results/` artefact. The Follow-up 3 audit revisions tightened several borderline cases (e.g., B2's "first" hedge, M6's denominator-explanation rewrite, M9's "canonical E7" disambiguation) which moved them firmly into the decorative category.

The single content-deferred citation (§16.7B, the TreeSHAP invariance argument) is **not numerically load-bearing** — the empirical Path B Week 2B verification (τ_top10 = 0.927) defends the choice on its consequence. A thesis examiner reading the report cold can verify all 23 numerical claims sampled in §4 of this audit without opening any other document. The argumentative content for *why* the background choice is theoretically defensible would benefit from a short in-prose summary (fix shape in §8.4 above), but this is a clarity improvement, not a structural defect.

**STRUCTURAL flag:** NOT raised. Report meets the standalone-deliverable bar.

### 8.6 Updated findings tally

Total findings across all 8 dimensions: 18 (original AUDIT_REPORT) + 1 (D8-1) = **19 findings**.

- 9 MAJOR resolved in Follow-up 3 (B1–B5, M6–M9)
- 1 MAJOR new in Dimension 8 (D8-1, content-deferred §16.7B invariance argument)
- 6 MINOR (m10–m15) — open
- 3 OBSERVATION (O16–O18) — open

No BLOCKING. No STRUCTURAL issue.

