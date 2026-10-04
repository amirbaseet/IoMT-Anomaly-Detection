# tau-Sweep — Confidence-Floor / Entropy-Ceiling Operating Characteristic

**Status: MEASURED.** All figures below are measured on the frozen thesis E7 model's existing predictions over the deduplicated thesis test set (published train/test boundary, 44-feature space). No training, no pcaps, no Track-A retrains. Saved softprob arrays were used directly — **no inference was run**; the frozen model was not touched.

> **Claim hygiene:** thresholds were characterised *post-hoc on the test set* as an operating-characteristic measurement. They were **not** used for model selection or tuning. The full curves are reported honestly in `tau_sweep_curve.csv` / `tau_sweep_results.json`; no tau was cherry-picked.

- Flows analysed: **892,268**  |  classes K = **19**
- Overall accuracy (all flows auto-decided, no floor): **0.992656**
- Entropy convention: raw Shannon natural-log `H=-Σ p·ln p`; normalized `H/ln(19)` ∈ [0,1] (matches `vae_fusion.compute_entropy`).
- Thesis Phase-6C threshold `ent_p95` = **0.394647** raw = **0.134031** normalized (P95 of 38,546 benign validation flows).

## Three named operating points

Columns: threshold · coverage · retained accuracy · retained macro-F1 · retained MCC · #deferred · deferred-set error rate. macro-F1/MCC computed over classes present in the retained set.

### Sweep 1 — confidence floor τ_c (auto-decide if max-prob ≥ τ_c)

| operating point | τ_c | coverage | ret-acc | ret-F1 | ret-MCC | #deferred | def-err |
|---|---|---|---|---|---|---|---|
| first ret-acc > 0.99  | 0.5 | 0.9992 | 0.993058 | 0.9116 | 0.9911 | 681 | 0.5345 |
| first ret-acc > 0.999 | 0.95 | 0.9774 | 0.999031 | 0.9688 | 0.9987 | 20,178 | 0.2829 |

### Sweep 2 — entropy ceiling τ_H (auto-decide if H/ln19 ≤ τ_H)

| operating point | τ_H | coverage | ret-acc | ret-F1 | ret-MCC | #deferred | def-err |
|---|---|---|---|---|---|---|---|
| first ret-acc > 0.99  | 0.9 | 1.0000 | 0.992656 | 0.9076 | 0.9906 | 0 | n/a |
| first ret-acc > 0.999 | 0.05 | 0.9749 | 0.999203 | 0.9711 | 0.9990 | 22,393 | 0.2617 |
| **thesis Phase-6C**   | 0.134031 | 0.9832 | 0.998588 | 0.9619 | 0.9982 | 14,951 | 0.3554 |

## Analyst-budget translation

The number a deployment actually sets: X% of flows auto-decided, the rest routed to Case-5 REVIEW. Deferrals normalised to a **1M flows/day** intake.

| operating point | % auto-decided | % to REVIEW | deferrals / 1M flows / day |
|---|---|---|---|
| Sweep 1 · ret-acc > 0.99 | 99.924% | 0.076% | 763 |
| Sweep 1 · ret-acc > 0.999 | 97.739% | 2.261% | 22,614 |
| Sweep 2 · ret-acc > 0.99 | 100.000% | 0.000% | 0 |
| Sweep 2 · ret-acc > 0.999 | 97.490% | 2.510% | 25,097 |
| Sweep 2 · thesis Phase-6C | 98.324% | 1.676% | 16,756 |

## Per-class deferral rate at each named operating point

Fraction of each class's flows routed to REVIEW. Rescue classes (Phase-6C zero-day targets) are **bold**.

| class | S1 acc>0.99 | S1 acc>0.999 | S2 acc>0.99 | S2 acc>0.999 | S2 thesis |
|---|---|---|---|---|---|
| **ARP_Spoofing** | 0.038 | 0.466 | 0.000 | 0.508 | 0.369 |
| Benign | 0.003 | 0.119 | 0.000 | 0.145 | 0.095 |
| DDoS_ICMP | 0.000 | 0.008 | 0.000 | 0.010 | 0.005 |
| DDoS_SYN | 0.001 | 0.010 | 0.000 | 0.011 | 0.007 |
| DDoS_TCP | 0.004 | 0.015 | 0.000 | 0.018 | 0.011 |
| DDoS_UDP | 0.000 | 0.001 | 0.000 | 0.001 | 0.000 |
| DoS_ICMP | 0.000 | 0.026 | 0.000 | 0.038 | 0.010 |
| DoS_SYN | 0.000 | 0.001 | 0.000 | 0.001 | 0.000 |
| DoS_TCP | 0.000 | 0.001 | 0.000 | 0.001 | 0.001 |
| DoS_UDP | 0.000 | 0.001 | 0.000 | 0.002 | 0.001 |
| MQTT_DDoS_Connect_Flood | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MQTT_DDoS_Publish_Flood | 0.016 | 0.768 | 0.000 | 0.819 | 0.668 |
| **MQTT_DoS_Connect_Flood** | 0.000 | 0.002 | 0.000 | 0.002 | 0.001 |
| MQTT_DoS_Publish_Flood | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **MQTT_Malformed_Data** | 0.035 | 0.420 | 0.000 | 0.468 | 0.318 |
| Recon_OS_Scan | 0.014 | 0.547 | 0.000 | 0.571 | 0.507 |
| **Recon_Ping_Sweep** | 0.059 | 0.568 | 0.000 | 0.615 | 0.479 |
| Recon_Port_Scan | 0.003 | 0.180 | 0.000 | 0.189 | 0.065 |
| **Recon_VulScan** | 0.074 | 0.525 | 0.000 | 0.576 | 0.452 |

## Deferred-set composition at the thesis Phase-6C threshold

τ_H = 0.134031 (normalized ent_p95). Deferred set size: **14,951** (1.676% of flows). Error rate it would have incurred if auto-decided: **0.3554** (vs retained 0.998588 accuracy).

Top classes dominating REVIEW:

| class | # deferred | share of deferred |
|---|---|---|
| MQTT_DDoS_Publish_Flood | 5,620 | 0.3759 |
| Benign | 3,557 | 0.2379 |
| Recon_OS_Scan | 1,492 | 0.0998 |
| Recon_Port_Scan | 1,269 | 0.0849 |
| **ARP_Spoofing** | 644 | 0.0431 |
| DDoS_SYN | 639 | 0.0427 |
| **MQTT_Malformed_Data** | 556 | 0.0372 |
| **Recon_VulScan** | 440 | 0.0294 |
| DDoS_UDP | 170 | 0.0114 |
| DoS_UDP | 135 | 0.0090 |

## Sanity checks

**Monotonicity.** Retained accuracy should not fall as the floor tightens.
- Sweep 1 (τ_c ↑): **0** non-monotone step(s).
- Sweep 2 (τ_H ↓): **1** non-monotone step(s).

**Investigation of the Sweep-2 step(s)** (brief requires it before reporting). Each flagged step is a benign **degenerate-bin / tie** artifact, not a real reversal:

| τ (looser→tighter) | Δacc | flows in band | errors in band | degenerate-bin? |
|---|---|---|---|---|
| 0.8 → 0.75 | -6.58e-08 | 8 | 0 | yes |

Reading: tightening τ_H across this step defers a handful of extreme-tail flows (coverage ≈ 99.9997%) that were **all correctly classified**. Removing correct predictions from the retained set lowers retained accuracy by a tie-level epsilon (~1e-7) while the retained error count is unchanged. The operating characteristic is monotone in substance; this is the expected quantisation wiggle at the very top of the entropy range where only single-digit flow counts separate thresholds.

**Phase-6C rescue reproduction @ thesis threshold.** At τ_H = 0.134031, **0.8109** of *all* E7 errors land in the deferred REVIEW set. Per rescue class — misclassifications that the entropy gate defers:

| rescue class | # misclassified | # deferred | misclass-deferral rate | class deferral rate |
|---|---|---|---|---|
| ARP_Spoofing | 506 | 350 | 0.692 | 0.369 |
| MQTT_DoS_Connect_Flood | 3 | 0 | 0.000 | 0.001 |
| MQTT_Malformed_Data | 300 | 270 | 0.900 | 0.318 |
| Recon_Ping_Sweep | 49 | 43 | 0.878 | 0.479 |
| Recon_VulScan | 650 | 340 | 0.523 | 0.452 |

If the rescue mechanism reproduces, a large share of each rescue class's *misclassifications* appears in the deferred set (they carry high entropy).

---

*Generated by `deliverables/scripts/07_tau_sweep.py`. Interpretation is deferred to the planning chat per the brief's stop point.*