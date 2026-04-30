"""Path B Tier 3 — Live Monitor (dashboard entry point).

Layout:
    1. Header (title + repo link + tagline)
    2. Status row — 4 metric cards (CSV-derived from sweep_table.csv)
    3. Per-target status badges — eligible LOO targets (CATCH 1, data-driven)
    4. Pareto frontier mini chart — published p93 / p95 highlighted
    5. Robustness summary — 5-row published headline table
    6. Footer — last-commit timestamp

All numbers are loaded from results/*.csv via the cached loaders in
components/data_loader.py — no recomputation, no external API calls.
"""

from __future__ import annotations

import streamlit as st

from components.data_loader import (
    GITHUB_URL,
    PUBLISHED_FPR_BUDGET,
    PUBLISHED_PERCENTILE,
    TEST_SET_SIZE,
    all_targets_at_percentile,
    eligible_targets_at_percentile,
    last_commit_timestamp,
    load_sweep_table,
    published_sweep_row,
    robustness_summary_rows,
)
from components.pareto_chart import pareto_mini
from components.status_indicators import render_target_badges

st.set_page_config(
    page_title="IoMT Hybrid IDS — Live Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. Header -----------------------------------------------------------
st.title("IoMT Hybrid IDS — Live Monitor")
st.markdown(
    f"Hybrid network intrusion detection for medical IoT · "
    f"[GitHub repo]({GITHUB_URL}) · "
    f"Path B Tier 1 + Tier 2 complete · Tier 3 (this dashboard) in progress."
)

st.divider()

# --- 2. Status row (4 metric cards) --------------------------------------
sweep = load_sweep_table()
p_row = published_sweep_row()
strict_avg = float(p_row["h2_strict_avg"])
fpr = float(p_row["avg_false_alert_rate"])
strict_pass = str(p_row["h2_strict_pass"])

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    label=f"Headline strict_avg (p{PUBLISHED_PERCENTILE:.1f})",
    value=f"{strict_avg:.3f}",
    delta="published §15D",
    delta_color="off",
)
c2.metric(
    label="Operational FPR",
    value=f"{fpr:.3f}",
    delta=f"budget ≤ {PUBLISHED_FPR_BUDGET:.2f}",
    delta_color="normal" if fpr <= PUBLISHED_FPR_BUDGET else "inverse",
)
c3.metric(
    label="H2-strict pass",
    value=strict_pass,
    delta="of eligible LOO targets",
    delta_color="off",
)
c4.metric(
    label="Test set size",
    value=f"{TEST_SET_SIZE:,}",
    delta="flows · 70/15/15 split",
    delta_color="off",
)

st.divider()

# --- 3. Per-target status badges (CATCH 1: data-driven eligibility) ------
st.subheader("Per-target status — eligible LOO targets")

eligible = eligible_targets_at_percentile(PUBLISHED_PERCENTILE)
all_at_p = all_targets_at_percentile(PUBLISHED_PERCENTILE)
ineligible_count = len(all_at_p) - len(eligible)

st.caption(
    f"Eligibility filter: `n_loo_benign > 0` (data-driven, never a hard-coded list). "
    f"At p{PUBLISHED_PERCENTILE:.1f}: {len(eligible)} of {len(all_at_p)} LOO targets are evaluable; "
    f"{ineligible_count} excluded automatically (no benign rescue rows)."
)
render_target_badges(eligible)
st.caption(
    "Hover any pill for exact recall and `n_loo_benign`. "
    "Green ≥ 0.80, yellow 0.70–0.80, red < 0.70 (per §15D acceptance criteria)."
)

st.divider()

# --- 4. Pareto frontier (mini) -------------------------------------------
st.subheader("Pareto frontier — continuous threshold sweep (§15D)")
st.plotly_chart(pareto_mini(sweep), use_container_width=True)
st.caption(
    "29 computed points spanning entropy percentiles p85.0 → p99.0 in 0.5-pp steps. "
    f"Published operating points p{PUBLISHED_PERCENTILE:.1f} (selected) and p95.0 highlighted. "
    "Interactive version on the Threshold Tuning page."
)

st.divider()

# --- 5. Robustness summary -----------------------------------------------
st.subheader("Robustness summary")
st.table(robustness_summary_rows())
st.caption(
    "Headline numbers from README §15B / §15C.10 / §15D / §15E / §16.7B. "
    "Section pointers reference the Project_Journey_Complete.md / README narrative for methodology."
)

st.divider()

# --- 6. Footer -----------------------------------------------------------
st.caption(
    f"Path B Tier 1 + Tier 2 complete; thesis defense pending · "
    f"Last commit: `{last_commit_timestamp()}`"
)
