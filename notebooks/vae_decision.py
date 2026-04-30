#!/usr/bin/env python3
"""
Phase 6D Task 3 — VAE decision: SHIP / SHELVE / REPLACE-AE-ONLY
================================================================

Reads the per-β fusion ablation tables from vae_fusion.py and produces a
decision-ready CSV plus a one-paragraph summary. The framing decision
(SHIP / SHELVE / REPLACE-AE-ONLY) is recommended by this script but the
user makes the final call.

Decision rule (per Path B Phase 6D plan):
  SHIP            = some β beats §15D baseline by ≥ +0.02 strict_avg
                    AND avg_false_alert_rate ≤ 0.25 AND 4/4 strict pass
  REPLACE-AE-ONLY = VAE-only test AUC at some β ≥ Phase 5 AE test AUC (0.9892)
                    AND that β's best fusion variant matches §15D within ±0.005
                    (so VAE genuinely replaces AE without lifting the headline)
  SHELVE          = neither

Inputs:
  results/enhanced_fusion/vae_ablation/all_betas_ablation.csv
  results/enhanced_fusion/vae_ablation/per_beta_summary.json

Outputs:
  results/enhanced_fusion/vae_decision.csv          single-page decision table
  results/enhanced_fusion/vae_decision_summary.md   one-paragraph narrative
"""

# %% SECTION 0 — Imports
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "results" / "enhanced_fusion" / "vae_ablation"
DECISION_CSV = ROOT / "results" / "enhanced_fusion" / "vae_decision.csv"
DECISION_MD  = ROOT / "results" / "enhanced_fusion" / "vae_decision_summary.md"

OPERATIONAL_FPR_BUDGET = 0.25
SHIP_DELTA             = 0.02
H2_PASS_THRESHOLD      = 0.70
H2_STRICT_PASS_REQUIRED = 4   # /4 eligible targets
REPLACE_TIE_TOL        = 0.005   # |Δ strict_avg| ≤ 0.5pp counts as tied for REPLACE-AE-ONLY

# Identifiers carried over from vae_fusion.py.
BASELINE_VARIANT = "entropy_benign_p93_ae_p90"   # §15D anchor
AE_REFERENCE_P95 = "entropy_benign_p95_ae_p90"   # §15C anchor (Week 2A)
VAE_ONLY_VARIANT = "vae_p90"                      # raw VAE baseline (closest to AE-only)


def log(msg: str = "", t0: Optional[float] = None) -> None:
    elapsed = f" [+{time.time() - t0:6.1f}s]" if t0 is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)


# %% SECTION 1 — Load inputs

def load_inputs() -> tuple[pd.DataFrame, dict]:
    abl_csv = ABLATION_DIR / "all_betas_ablation.csv"
    if not abl_csv.exists():
        raise FileNotFoundError(f"Run vae_fusion.py first; missing {abl_csv}")
    df = pd.read_csv(abl_csv)
    log(f"loaded {abl_csv}: {len(df)} rows ({df['beta'].nunique()} βs × "
        f"{df['variant'].nunique()} variants)")

    summary_json = ABLATION_DIR / "per_beta_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"Run vae_fusion.py first; missing {summary_json}")
    with open(summary_json) as f:
        summary = json.load(f)
    log(f"loaded {summary_json}: betas={summary['betas']}, "
        f"phase5_ae_test_auc={summary['phase5_ae_test_auc']}")
    return df, summary


# %% SECTION 2 — Per-β best-variant selection

def _is_vae_conditioned(variant: str) -> bool:
    """True iff the variant uses the VAE channel (not the AE-only references)."""
    return ("vae_p" in variant) and (variant != BASELINE_VARIANT) \
        and (variant != AE_REFERENCE_P95)


def best_within_budget(df_beta: pd.DataFrame) -> Dict[str, float | str | int | None]:
    """Pick the best VAE-conditioned variant for one β under operational FPR budget.

    Ranking key (mirrors enhanced_fusion.py's "operationally usable" ranker):
      (strict_pass_int desc, strict_avg desc, binary_avg desc, fpr asc)
    """
    candidates = df_beta[
        df_beta["variant"].apply(_is_vae_conditioned)
        & (df_beta["avg_false_alert_rate"] <= OPERATIONAL_FPR_BUDGET)
    ].copy()
    if len(candidates) == 0:
        return {
            "best_variant": None,
            "best_strict_pass": None,
            "best_strict_avg": float("nan"),
            "best_binary_avg": float("nan"),
            "best_fpr": float("nan"),
            "n_candidates_under_budget": 0,
        }
    candidates = candidates.sort_values(
        by=["h2_strict_pass_int", "h2_strict_avg", "h2_binary_avg", "avg_false_alert_rate"],
        ascending=[False, False, False, True],
    )
    top = candidates.iloc[0]
    return {
        "best_variant":    str(top["variant"]),
        "best_strict_pass": str(top["h2_strict_pass"]),
        "best_strict_avg": float(top["h2_strict_avg"]),
        "best_binary_avg": float(top["h2_binary_avg"]),
        "best_fpr":        float(top["avg_false_alert_rate"]),
        "n_candidates_under_budget": int(len(candidates)),
    }


# %% SECTION 3 — Build decision CSV

def build_decision_table(df: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """One row per β + a §15D baseline row at the top for direct comparison."""
    phase5_ae_auc = float(summary["phase5_ae_test_auc"])
    betas = sorted(df["beta"].unique())

    # §15D baseline row (β-independent; same value at every β).
    baseline_subset = df[df["variant"] == BASELINE_VARIANT].iloc[0]
    rows: List[dict] = [{
        "row_kind":          "baseline_§15D",
        "beta":              "—",
        "variant":           BASELINE_VARIANT,
        "strict_pass":       str(baseline_subset["h2_strict_pass"]),
        "strict_avg":        float(baseline_subset["h2_strict_avg"]),
        "binary_avg":        float(baseline_subset["h2_binary_avg"]),
        "fpr":               float(baseline_subset["avg_false_alert_rate"]),
        "delta_strict_vs_baseline": 0.0,
        "delta_fpr_vs_baseline":    0.0,
        "fpr_within_budget": bool(baseline_subset["avg_false_alert_rate"] <= OPERATIONAL_FPR_BUDGET),
        "vae_only_p90_strict_avg": float("nan"),
        "vae_test_auc":      float("nan"),
        "vae_auc_vs_ae":     float("nan"),
        "n_candidates_under_budget": None,
        "ship_eligible":     False,
        "replace_eligible":  False,
        "decision_flag":     "BASELINE",
    }]

    baseline_strict = float(baseline_subset["h2_strict_avg"])
    baseline_fpr    = float(baseline_subset["avg_false_alert_rate"])

    for b in betas:
        df_b = df[df["beta"] == b]
        sel = best_within_budget(df_b)

        # VAE-only p90 strict_avg (β-specific) for REPLACE-AE-ONLY context.
        vae_only_row = df_b[df_b["variant"] == VAE_ONLY_VARIANT]
        vae_only_strict = float(vae_only_row["h2_strict_avg"].iloc[0]) if len(vae_only_row) else float("nan")

        # Per-β VAE test AUC pulled from per_beta_summary.json.
        beta_summary = next(
            (s for s in summary["summaries"] if abs(s["beta"] - b) < 1e-9),
            None,
        )
        vae_auc = float(beta_summary["vae_test_auc"]) if beta_summary else float("nan")
        auc_delta = vae_auc - phase5_ae_auc

        # SHIP eligibility: ≥ +0.02 strict_avg over baseline AND 4/4 strict pass AND under budget.
        ship_eligible = (
            sel["best_variant"] is not None
            and sel["best_strict_pass"] == f"{H2_STRICT_PASS_REQUIRED}/{H2_STRICT_PASS_REQUIRED}"
            and (sel["best_strict_avg"] - baseline_strict) >= SHIP_DELTA
            and sel["best_fpr"] <= OPERATIONAL_FPR_BUDGET
        )

        # REPLACE-AE-ONLY eligibility: best β's variant ties §15D AND VAE AUC ≥ AE AUC.
        replace_eligible = (
            sel["best_variant"] is not None
            and abs(sel["best_strict_avg"] - baseline_strict) <= REPLACE_TIE_TOL
            and vae_auc >= phase5_ae_auc
            and sel["best_strict_pass"] == f"{H2_STRICT_PASS_REQUIRED}/{H2_STRICT_PASS_REQUIRED}"
        )

        if ship_eligible:
            flag = "SHIP-CANDIDATE"
        elif replace_eligible:
            flag = "REPLACE-AE-ONLY-CANDIDATE"
        else:
            flag = "SHELVE-CANDIDATE"

        rows.append({
            "row_kind":          "vae_β",
            "beta":              float(b),
            "variant":           sel["best_variant"],
            "strict_pass":       sel["best_strict_pass"],
            "strict_avg":        sel["best_strict_avg"],
            "binary_avg":        sel["best_binary_avg"],
            "fpr":               sel["best_fpr"],
            "delta_strict_vs_baseline": (
                sel["best_strict_avg"] - baseline_strict
                if not np.isnan(sel["best_strict_avg"]) else float("nan")
            ),
            "delta_fpr_vs_baseline": (
                sel["best_fpr"] - baseline_fpr
                if not np.isnan(sel["best_fpr"]) else float("nan")
            ),
            "fpr_within_budget": (
                bool(sel["best_fpr"] <= OPERATIONAL_FPR_BUDGET)
                if not np.isnan(sel["best_fpr"]) else False
            ),
            "vae_only_p90_strict_avg": vae_only_strict,
            "vae_test_auc":      vae_auc,
            "vae_auc_vs_ae":     auc_delta,
            "n_candidates_under_budget": sel["n_candidates_under_budget"],
            "ship_eligible":     ship_eligible,
            "replace_eligible":  replace_eligible,
            "decision_flag":     flag,
        })

    return pd.DataFrame(rows)


# %% SECTION 4 — Recommendation logic

def recommend(decision_df: pd.DataFrame, phase5_ae_auc: float, baseline_strict: float) -> str:
    """Return one of {SHIP, SHELVE, REPLACE-AE-ONLY} based on per-β results."""
    vae_rows = decision_df[decision_df["row_kind"] == "vae_β"]
    if (vae_rows["ship_eligible"] == True).any():
        return "SHIP"
    if (vae_rows["replace_eligible"] == True).any():
        return "REPLACE-AE-ONLY"
    return "SHELVE"


def render_summary_md(decision_df: pd.DataFrame, recommendation: str,
                      phase5_ae_auc: float, baseline_strict: float,
                      baseline_fpr: float) -> str:
    """One-paragraph narrative with the headline numbers."""
    vae_rows = decision_df[decision_df["row_kind"] == "vae_β"].copy()
    valid = vae_rows.dropna(subset=["strict_avg"])
    if len(valid) == 0:
        return f"# VAE decision\n\nNo VAE-conditioned variant in any β satisfied the operational FPR budget.\n\n**Recommendation: {recommendation}**\n"

    best = valid.sort_values("strict_avg", ascending=False).iloc[0]
    best_beta = best["beta"]
    best_var  = best["variant"]
    best_strict = best["strict_avg"]
    best_fpr  = best["fpr"]
    delta = best_strict - baseline_strict

    # AUC headline (beta with best AUC).
    best_auc_row = vae_rows.sort_values("vae_test_auc", ascending=False).iloc[0]

    md_lines = [
        "# Phase 6D — VAE replacement decision summary",
        "",
        f"_Generated by `notebooks/vae_decision.py` from `all_betas_ablation.csv`._",
        "",
        "## Headline numbers",
        "",
        f"- §15D baseline (`{BASELINE_VARIANT}`): strict_avg = **{baseline_strict:.4f}**, "
        f"FPR = {baseline_fpr:.4f}, 4/4 strict pass.",
        f"- Phase 5 AE test AUC: **{phase5_ae_auc:.4f}** (anchor for REPLACE-AE-ONLY check).",
        "",
        f"### Best VAE-conditioned variant under FPR ≤ {OPERATIONAL_FPR_BUDGET}",
        f"- β = **{best_beta}**, variant = `{best_var}`",
        f"- strict_avg = **{best_strict:.4f}**, FPR = {best_fpr:.4f}, "
        f"strict_pass = {best['strict_pass']}",
        f"- vs §15D: **Δ strict_avg = {delta:+.4f}** (SHIP needs ≥ +{SHIP_DELTA})",
        "",
        f"### Best β by VAE-only test AUC",
        f"- β = **{best_auc_row['beta']}**, VAE test AUC = **{best_auc_row['vae_test_auc']:.4f}** "
        f"(Δ vs AE = {best_auc_row['vae_auc_vs_ae']:+.4f})",
        "",
        "## Per-β table",
        "",
        decision_df[[
            "row_kind", "beta", "variant", "strict_pass", "strict_avg",
            "fpr", "delta_strict_vs_baseline", "vae_test_auc", "vae_auc_vs_ae",
            "decision_flag",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)),
        "",
        "## Recommended decision",
        "",
        f"**{recommendation}**",
        "",
        "Decision rule recap:",
        f"- SHIP: some β beats baseline by ≥ +{SHIP_DELTA} strict_avg under FPR ≤ "
        f"{OPERATIONAL_FPR_BUDGET}, with 4/4 strict pass.",
        f"- REPLACE-AE-ONLY: best β's variant ties §15D within ±{REPLACE_TIE_TOL} strict_avg "
        f"AND that β's VAE-only AUC ≥ AE AUC ({phase5_ae_auc:.4f}).",
        "- SHELVE: neither.",
        "",
        "**The user (not this script) makes the framing decision** — this file documents the "
        "evidence and a recommendation. Override is appropriate when contextual judgement (e.g. "
        "complexity-vs-gain trade-off, latent-collapse caveats) outweighs the raw rule.",
    ]
    return "\n".join(md_lines) + "\n"


# %% SECTION 5 — Main

def main() -> int:
    t0 = time.time()
    log("=" * 76)
    log("Path B Phase 6D Task 3 — VAE decision (SHIP / SHELVE / REPLACE-AE-ONLY)")
    log("=" * 76)

    df, summary = load_inputs()

    baseline_subset = df[df["variant"] == BASELINE_VARIANT].iloc[0]
    baseline_strict = float(baseline_subset["h2_strict_avg"])
    baseline_fpr    = float(baseline_subset["avg_false_alert_rate"])
    phase5_ae_auc   = float(summary["phase5_ae_test_auc"])
    log(f"  §15D baseline: strict_avg={baseline_strict:.6f}, FPR={baseline_fpr:.4f}")
    log(f"  Phase 5 AE test AUC: {phase5_ae_auc:.4f}")

    decision_df = build_decision_table(df, summary)
    decision_df.to_csv(DECISION_CSV, index=False)
    log(f"\nwrote {DECISION_CSV} ({len(decision_df)} rows: 1 baseline + {df['beta'].nunique()} βs)")

    recommendation = recommend(decision_df, phase5_ae_auc, baseline_strict)
    log(f"\n{'=' * 76}")
    log(f"  RECOMMENDED DECISION: {recommendation}")
    log(f"{'=' * 76}\n")

    log("decision_df:")
    show = decision_df[[
        "row_kind", "beta", "variant", "strict_pass", "strict_avg", "fpr",
        "delta_strict_vs_baseline", "vae_test_auc", "vae_auc_vs_ae",
        "ship_eligible", "replace_eligible", "decision_flag",
    ]].copy()
    log("\n" + show.to_string(index=False))

    # One-page narrative.
    md = render_summary_md(decision_df, recommendation, phase5_ae_auc, baseline_strict, baseline_fpr)
    DECISION_MD.write_text(md)
    log(f"\nwrote {DECISION_MD} ({len(md)} chars)")
    log(f"DONE in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
