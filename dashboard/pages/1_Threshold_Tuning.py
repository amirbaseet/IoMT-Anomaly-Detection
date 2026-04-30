"""Page 4 — Threshold Tuning (interactive Pareto exploration).

Sidebar sliders (CATCH 2):
- Entropy percentile (REAL): indexes sweep_table.csv. Snap-to-nearest
  computed percentile; metric cards + Pareto marker + per-target bars
  update live with computed numbers.
- Confidence floor (ANNOTATION): every sweep point was computed at
  confidence_floor = 0.70 (§15D). Slider > 0.70 hides the frontier
  (no measured points qualify) but does NOT recompute fusion. Metric
  cards stay live (entropy-driven only).
- FPR budget (ANNOTATION): draws a dashed vertical line at the chosen
  budget on the Pareto chart. Does NOT alter any computed metric.

URL query-param sync (?ent=&conf=&fpr=) restores slider state across
reloads. Snapshot JSON button captures the current operating point with
honesty fields (snapped_to, timestamp).
"""

from __future__ import annotations

import datetime as dt
import json

import plotly.graph_objects as go
import streamlit as st

from components.data_loader import (
    BORDER_RECALL,
    PASS_RECALL,
    PUBLISHED_CONFIDENCE_FLOOR,
    PUBLISHED_FPR_BUDGET,
    PUBLISHED_PERCENTILE,
    eligible_targets_at_percentile,
    load_sweep_table,
    sweep_row_at,
)
from components.pareto_chart import pareto_interactive
from components.status_indicators import BORDER, PASS, status_for_recall

st.set_page_config(
    page_title="IoMT IDS — Threshold Tuning",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state setup: parse URL once, snap entropy to grid, seed slider keys.
# ---------------------------------------------------------------------------
def _safe_float(qp: dict, key: str, default: float, *, lo: float, hi: float) -> float:
    try:
        v = float(qp.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _initialize_state() -> None:
    if st.session_state.get("page4_initialized"):
        return
    qp = st.query_params

    url_ent = _safe_float(qp, "ent", PUBLISHED_PERCENTILE, lo=85.0, hi=99.0)
    initial_ent = float(sweep_row_at(url_ent)["percentile"])  # snap to grid
    st.session_state.ent_user_input = url_ent
    st.session_state.ent_slider = initial_ent

    st.session_state.conf_slider = _safe_float(qp, "conf", PUBLISHED_CONFIDENCE_FLOOR, lo=0.5, hi=0.9)
    st.session_state.fpr_slider = _safe_float(qp, "fpr", PUBLISHED_FPR_BUDGET, lo=0.10, hi=0.40)

    st.session_state.page4_initialized = True


def _on_ent_change() -> None:
    """After a user drag, the slider's value IS the user input — no snap."""
    st.session_state.ent_user_input = st.session_state.ent_slider


def _reset_to_published() -> None:
    st.session_state.ent_slider = PUBLISHED_PERCENTILE
    st.session_state.ent_user_input = PUBLISHED_PERCENTILE
    st.session_state.conf_slider = PUBLISHED_CONFIDENCE_FLOOR
    st.session_state.fpr_slider = PUBLISHED_FPR_BUDGET


_initialize_state()

# ---------------------------------------------------------------------------
# Sidebar widgets
# ---------------------------------------------------------------------------
st.sidebar.header("Operating point")

ent_pct = st.sidebar.slider(
    "Entropy percentile (REAL)",
    min_value=85.0, max_value=99.0, step=0.5,
    key="ent_slider",
    on_change=_on_ent_change,
    help="Indexes the §15D continuous sweep. Live metric updates.",
)
ent_was_snapped = st.session_state.ent_user_input != ent_pct
if ent_was_snapped:
    st.sidebar.caption(
        f"⚠ Requested p{st.session_state.ent_user_input:.2f} → snapped to "
        f"p{ent_pct:.1f} (sweep grid: 0.5 pp steps)."
    )

conf_floor = st.sidebar.slider(
    "Confidence floor (annotation)",
    min_value=0.5, max_value=0.9, step=0.05,
    key="conf_slider",
    help="Visual-only filter; does not recompute fusion.",
)
fpr_budget = st.sidebar.slider(
    "FPR budget (annotation)",
    min_value=0.10, max_value=0.40, step=0.01,
    key="fpr_slider",
    help="Dashed vline on the Pareto chart; does not alter metrics.",
)

st.sidebar.caption(
    "⚠ Confidence and FPR-budget sliders are annotation-only — they "
    "highlight subsets of the entropy sweep but do NOT recompute fusion "
    "outcomes. The §15D continuous threshold sweep was computed for entropy "
    "at percentiles 85.0–99.0 (Δ=0.5pp); off-grid combinations require new "
    "ablation runs. See README §15C / §15D for methodology, or use the "
    "entropy slider above for live numerical updates."
)

st.sidebar.divider()
st.sidebar.button("↩ Reset to published p93", on_click=_reset_to_published, use_container_width=True)
snap_clicked = st.sidebar.button("📋 Generate snapshot JSON", use_container_width=True)

# ---------------------------------------------------------------------------
# Sync URL query-params (round-trip floats with explicit format)
# ---------------------------------------------------------------------------
st.query_params.update({
    "ent": f"{ent_pct:.1f}",
    "conf": f"{conf_floor:.2f}",
    "fpr": f"{fpr_budget:.2f}",
})

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Threshold Tuning")
st.caption("Best viewed on desktop ≥ 1024px wide.")

# ---------------------------------------------------------------------------
# Compute selected operating point (entropy-driven; CATCH 2: confidence + FPR
# budget sliders never feed into these numbers)
# ---------------------------------------------------------------------------
sweep = load_sweep_table()
sel = sweep_row_at(ent_pct)
strict_avg = float(sel["h2_strict_avg"])
fpr = float(sel["avg_false_alert_rate"])
strict_pass = str(sel["h2_strict_pass"])
strict_pass_int = int(sel["h2_strict_pass_int"])
ent_threshold = float(sel["ent_threshold"])

published = sweep_row_at(PUBLISHED_PERCENTILE)
delta_strict = strict_avg - float(published["h2_strict_avg"])
delta_fpr = fpr - float(published["avg_false_alert_rate"])

# ---------------------------------------------------------------------------
# Pareto frontier — interactive
# ---------------------------------------------------------------------------
st.subheader(f"Pareto frontier — selected p{ent_pct:.1f}")

fig = pareto_interactive(sweep, ent_pct, fpr_budget=fpr_budget)

confidence_excludes_all = conf_floor > PUBLISHED_CONFIDENCE_FLOOR
if confidence_excludes_all:
    # CATCH 2: every sweep point was computed at confidence=0.70. A slider value
    # above that excludes all measured points — hide the frontier trace but
    # leave the selected marker (entropy-driven) and FPR vline intact.
    fig.data[0].update(visible=False)
    st.caption(
        f"Confidence floor {conf_floor:.2f} > §15D published 0.70: "
        "Pareto frontier hidden — none of the 29 computed sweep points "
        "satisfy this floor (all were measured at confidence = 0.70). "
        "The selected marker remains live (entropy slider only)."
    )

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 4 live metric cards (entropy-driven)
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    label=f"strict_avg @ p{ent_pct:.1f}",
    value=f"{strict_avg:.3f}",
    delta=f"{delta_strict:+.3f} vs p{PUBLISHED_PERCENTILE:.1f}",
    delta_color="normal",
)
c2.metric(
    label=f"FPR (budget {fpr_budget:.2f})",
    value=f"{fpr:.3f}",
    delta=f"{delta_fpr:+.3f} vs p{PUBLISHED_PERCENTILE:.1f}",
    delta_color="normal" if fpr <= fpr_budget else "inverse",
)
strict_status = "PASS" if strict_pass_int == 4 else ("BORDERLINE" if strict_pass_int == 3 else "FAIL")
c3.metric(
    label="H2-strict pass",
    value=strict_pass,
    delta=strict_status,
    delta_color="normal" if strict_pass_int == 4 else ("off" if strict_pass_int == 3 else "inverse"),
)
c4.metric(
    label=f"Entropy threshold (raw)",
    value=f"{ent_threshold:.4f}",
    delta=f"p{ent_pct:.1f}",
    delta_color="off",
)

# ---------------------------------------------------------------------------
# Per-target rescue recall bars (entropy-driven; CATCH 1 eligibility filter)
# ---------------------------------------------------------------------------
st.subheader(f"Per-target rescue recall @ p{ent_pct:.1f}")
eligible = eligible_targets_at_percentile(ent_pct)

if eligible.empty:
    st.warning(
        f"No eligible LOO targets at p{ent_pct:.1f} (n_loo_benign > 0 filter)."
    )
else:
    recall_values = [float(v) for v in eligible["h2_strict_rescue_recall"]]
    bar_colors = [status_for_recall(v)[0] for v in recall_values]

    bar_fig = go.Figure(
        go.Bar(
            x=eligible["target"].tolist(),
            y=recall_values,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in recall_values],
            textposition="outside",
            cliponaxis=False,
            customdata=eligible[["n_loo_benign", "n_target"]].values,
            hovertemplate=(
                "%{x}<br>recall=%{y:.3f}<br>"
                "n_loo_benign=%{customdata[0]} · n_target=%{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )
    bar_fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        showlegend=False,
        xaxis=dict(title="LOO target"),
        yaxis=dict(
            title="H2-strict rescue recall",
            range=[0.0, 1.08],
            tickformat=".2f",
            dtick=0.10,
        ),
    )
    bar_fig.add_hline(
        y=PASS_RECALL, line_dash="dash", line_color=PASS, opacity=0.55,
        annotation_text=f"PASS ≥ {PASS_RECALL:.2f}",
        annotation_position="top right",
    )
    bar_fig.add_hline(
        y=BORDER_RECALL, line_dash="dash", line_color=BORDER, opacity=0.55,
        annotation_text=f"BORDERLINE ≥ {BORDER_RECALL:.2f}",
        annotation_position="top right",
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    st.caption(
        f"Eligibility filter: n_loo_benign > 0 (data-driven). "
        f"Showing {len(eligible)} eligible LOO target{'s' if len(eligible) != 1 else ''} "
        f"at p{ent_pct:.1f}."
    )

# ---------------------------------------------------------------------------
# Snapshot JSON
# ---------------------------------------------------------------------------
if snap_clicked:
    st.session_state.last_snapshot = {
        "entropy_percentile": round(float(ent_pct), 2),
        "entropy_threshold": round(ent_threshold, 6),
        "strict_avg": round(strict_avg, 4),
        "fpr": round(fpr, 4),
        "strict_pass": strict_pass,
        "confidence_floor": round(float(conf_floor), 2),
        "fpr_budget": round(float(fpr_budget), 2),
        "snapped_to": bool(ent_was_snapped),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

if "last_snapshot" in st.session_state:
    st.divider()
    st.subheader("Snapshot")
    st.code(json.dumps(st.session_state.last_snapshot, indent=2), language="json")
    st.caption(
        "Click the copy icon (top-right of the code block) to copy the snapshot JSON. "
        "`snapped_to=true` means the user-input percentile differed from the rendered "
        "grid value — typically when a URL preset (?ent=…) lands off-grid."
    )
