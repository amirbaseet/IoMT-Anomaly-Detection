# CLAUDE CODE BRIEF — τ-Sweep (Tier-1, final item)

## Confidence-floor operating-characteristic analysis. Pure post-processing — no training, no new data.

You are running the last Tier-1 verification item for ShieldMind: characterizing the
accuracy-vs-coverage tradeoff of the fusion engine's confidence floor (the Phase-6C
entropy/confidence mechanism that routes uncertain flows to Case 5 REVIEW).

**Scope: analysis of the FROZEN E7 model's existing predictions only.** Do not retrain
anything. Do not touch pcaps. If saved per-flow prediction arrays exist from the thesis
runs, use them; if not, run inference ONCE with the frozen model on the deduplicated
thesis test set to produce them, then analyze.

**Golden rule (unchanged from Tracks A/B):** this is an operating-characteristic
measurement, not tuning. Report the full curve honestly; do not cherry-pick a τ that
flatters the model. All outputs tagged **Measured**.

---

## Inputs to locate (ask the user for paths; do not guess)

1. Frozen E7 model artifacts (XGBoost multi:softprob) — the thesis model, NOT any
   Track-A retrain.
2. The deduplicated thesis test set (the same split the README results were computed on
   — the published train/test file boundary, dedup-first).
3. If they exist: saved per-flow softmax probability arrays + true labels from Phase 6C.
   (If absent, produce them once via inference — log that this was done.)

---

## Method

For each test flow you need: softmax probability vector p, predicted class, true class,
confidence c = max(p), and normalized entropy H(p)/log(K) with K=19.

Sweep TWO thresholds independently (they are related but not identical mechanisms):

**Sweep 1 — confidence floor τ_c:** auto-decide flows with c ≥ τ_c; defer the rest.
τ_c ∈ {0.50, 0.55, ..., 0.95, 0.96, ..., 0.99, 0.995, 0.999} (fine steps near the top).

**Sweep 2 — entropy ceiling τ_H:** auto-decide flows with normalized entropy ≤ τ_H;
defer the rest. τ_H ∈ {0.05, 0.10, ..., 0.90} plus fine steps near the thesis's
operational threshold (locate it in README §Phase-6C and mark it on the curve).

At each threshold value report, for the AUTO-DECIDED (retained) set:
- coverage (fraction retained), overall accuracy, macro-F1, MCC
- and for the DEFERRED set: its size, its error rate had it been auto-decided
  (i.e., how many mistakes the deferral avoided), and its class composition
  (which classes dominate REVIEW — expect the Phase-6C rescue classes).

Also report per-class deferral rates at 3 named operating points:
- τ where retained accuracy first exceeds 0.99
- τ where retained accuracy first exceeds 0.999
- the thesis's existing Phase-6C threshold (for continuity with the README)

**Analyst-budget framing (required):** translate each operating point into
"X% of flows auto-decided, Y flows/day deferred per 1M flows/day" — the number a
deployment actually sets. This connects the sweep to the ShieldMind Case-5 REVIEW
workload story.

**Sanity checks:**
- The curve must be monotone-ish: retained accuracy should not decrease as τ_c rises.
  If it does at any step, investigate before reporting (ties/degenerate bins).
- Verify the Phase-6C rescue reproduces: at the thesis threshold, the previously
  rescued misclassifications should appear in the deferred set.

---

## Outputs → OUT_DIR/tausweep/

- `tau_sweep_results.json` — full curves (both sweeps), all metrics per threshold
- `tau_sweep_curve.csv` — tidy long-format table for plotting
- `tau_sweep_summary.md` — the 3 named operating points, per-class deferral at each,
  the analyst-budget translation, and the Phase-6C continuity check. Claim-hygiene:
  everything tagged Measured; note explicitly that thresholds were characterized on
  the test set post-hoc (operating-characteristic analysis), not used for model selection.

STOP after producing the summary. Interpretation happens in the planning chat.
