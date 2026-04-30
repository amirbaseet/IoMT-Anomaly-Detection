"""Cached data loaders + small derivations for the dashboard.

All loaders return df.copy() to avoid the @st.cache_data mutation footgun
(R5 mitigation in the plan): downstream `.loc` indexing must never mutate
the cached object.

Schemas verified live against actual CSV headers (not from memory):
- sweep_table.csv:      [percentile, ent_threshold, h2_strict_pass,
                         h2_strict_pass_int, h2_strict_evaluated,
                         h2_strict_avg, h2_binary_avg, avg_false_alert_rate]
- sweep_per_target.csv: [percentile, ent_threshold, target, n_target,
                         n_loo_benign, h2_strict_rescue_recall,
                         h2_binary_recall, false_alert_rate_benign]
"""

from __future__ import annotations

import datetime as _dt
import subprocess
from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

TEST_SET_SIZE = 892_268  # Source: README §3 preprocessing — Phase 3 70/15/15 split, post-deduplication

# §15D-published operating point.
PUBLISHED_PERCENTILE: Final[float] = 93.0
PUBLISHED_FPR_BUDGET: Final[float] = 0.25
PUBLISHED_CONFIDENCE_FLOOR: Final[float] = 0.7

# Recall thresholds from §15D acceptance criteria (badges + bar coloring).
PASS_RECALL: Final[float] = 0.80
BORDER_RECALL: Final[float] = 0.70

GITHUB_URL: Final[str] = "https://github.com/amirbaseet/IoMT-Anomaly-Detection"


@st.cache_data(ttl=None)
def load_sweep_table() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "enhanced_fusion/threshold_sweep/sweep_table.csv").copy()


@st.cache_data(ttl=None)
def load_sweep_per_target() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "enhanced_fusion/threshold_sweep/sweep_per_target.csv").copy()


@st.cache_data(ttl=None)
def load_multi_seed() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "enhanced_fusion/multi_seed_summary.csv").copy()


@st.cache_data(ttl=None)
def load_ks_per_fold() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "enhanced_fusion/ks_per_fold/ks_per_fold.csv").copy()


@st.cache_data(ttl=None)
def load_shap_comparison() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "shap/sensitivity/comparison.csv").copy()


@st.cache_data(ttl=None)
def load_vae_decision() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "enhanced_fusion/vae_decision.csv").copy()


def all_targets_at_percentile(percentile: float) -> pd.DataFrame:
    """Every target row at the given percentile (eligible + ineligible)."""
    df = load_sweep_per_target()
    return (
        df.loc[df["percentile"] == percentile]
        .sort_values("target")
        .reset_index(drop=True)
        .copy()
    )


def eligible_targets_at_percentile(percentile: float) -> pd.DataFrame:
    """Per-target rows at `percentile`, filtered on n_loo_benign > 0.

    CATCH 1 (data-driven eligibility): the badge set on Pages 1 and 4 is
    derived from this predicate, NOT a hard-coded list of 4 names. If a
    future multi-seed view drops a target's evaluable benign count to 0,
    it auto-disappears from the badge row without code change.
    """
    rows = all_targets_at_percentile(percentile)
    return rows.loc[rows["n_loo_benign"] > 0].reset_index(drop=True).copy()


def published_sweep_row() -> pd.Series:
    """The §15D-published p93.0 row of sweep_table.csv."""
    df = load_sweep_table()
    return df.loc[df["percentile"] == PUBLISHED_PERCENTILE].iloc[0].copy()


def sweep_row_at(percentile: float) -> pd.Series:
    """Snap to the nearest computed percentile and return that row.

    The sweep grid is finite (29 points, 0.5 pp steps). Off-grid requests
    snap to the closest computed row — Page 4 surfaces this honestly via
    a "snapped to p{x}" caption.
    """
    df = load_sweep_table()
    nearest_idx = (df["percentile"] - percentile).abs().idxmin()
    return df.loc[nearest_idx].copy()


def last_commit_timestamp() -> str:
    """ISO-8601 timestamp of the latest git commit.

    Falls back to this module's mtime on Streamlit Cloud sandboxes that
    lack a working git binary (R6 mitigation).
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        out = result.stdout.strip()
        if out:
            return out
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    mtime = Path(__file__).stat().st_mtime
    return _dt.datetime.fromtimestamp(mtime, tz=_dt.timezone.utc).isoformat()


# Robustness summary — published headline strings from README §15B / §15C.10 /
# §15D / §15E / §16.7B. Numbers verified live against the underlying CSVs:
#   σ=0.022   ← multi_seed_summary.csv:variant==entropy_benign_p95.h2_strict_avg_std=0.022541
#   τ=0.927   ← shap/sensitivity/comparison.csv.kendall_tau_top10_union=0.927273
#   Δ=-0.0001 ← vae_decision.csv:(row_kind==vae_β & beta==0.5).delta_strict_vs_baseline=-0.000148
# Rendered verbatim because each row's prose qualifier ("0/19 cells fail",
# "Monotone Pareto, p93.0 optimum", "uniform", "interchangeable") is an
# interpretive claim that lives in the README narrative, not the CSVs.
_ROBUSTNESS_ROWS: Final[list[dict[str, str]]] = [
    {"Robustness axis": "Multi-seed (5 seeds)",  "Headline number": "σ = 0.022, 0/19 cells fail",       "Section": "§15B"},
    {"Robustness axis": "Continuous threshold",  "Headline number": "Monotone Pareto, p93.0 optimum",   "Section": "§15D"},
    {"Robustness axis": "Per-fold KS",           "Headline number": "Spread = 0.003 (uniform)",         "Section": "§15C.10"},
    {"Robustness axis": "SHAP background",       "Headline number": "Kendall τ = 0.927",                "Section": "§16.7B"},
    {"Robustness axis": "Layer-2 substitution",  "Headline number": "VAE Δ = −0.0001 (interchangeable)", "Section": "§15E"},
]


def robustness_summary_rows() -> pd.DataFrame:
    return pd.DataFrame(_ROBUSTNESS_ROWS)
