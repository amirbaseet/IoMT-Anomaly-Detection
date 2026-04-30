"""Pareto frontier plot helpers (used by Pages 1 + 4).

Plotly is preferred over matplotlib for interactivity and native Streamlit
reactive support — matplotlib redraws are slow and trigger flicker.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .status_indicators import BORDER, FAIL, NEUTRAL, PASS


def _strict_pass_color(p_int: int) -> str:
    """Color a Pareto point by its h2_strict_pass_int (0..4).

    Using .apply over .map(dict) avoids silent NaN colors if an unexpected
    integer ever appears in the column.
    """
    if p_int >= 4:
        return PASS
    if p_int >= 3:
        return BORDER
    return FAIL


def _base_scatter(df: pd.DataFrame) -> go.Figure:
    """Shared base trace: strict_avg vs operational FPR.

    Explicit axis ranges + tickformat are set from the data because Plotly's
    autorange heuristic can misbehave when customdata mixes float + string
    dtypes (here `percentile` is float and `h2_strict_pass` is a "4/4"-style
    string), which manifested as integer-spaced ticks like "−2, 0, 2, 4" on
    a [0, 1]-bounded recall axis. Pinning the range removes the ambiguity.
    """
    y_min = float(df["h2_strict_avg"].min())
    y_max = float(df["h2_strict_avg"].max())
    x_min = float(df["avg_false_alert_rate"].min())
    x_max = float(df["avg_false_alert_rate"].max())
    y_pad, x_pad = 0.04, 0.02

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["avg_false_alert_rate"],
            y=df["h2_strict_avg"],
            mode="lines+markers",
            line=dict(color=NEUTRAL, width=1.5),
            marker=dict(
                size=8,
                color=df["h2_strict_pass_int"].apply(_strict_pass_color).tolist(),
                line=dict(color="#222", width=0.5),
            ),
            customdata=df[["percentile", "h2_strict_pass"]].values,
            hovertemplate=(
                "p%{customdata[0]:.1f} · FPR=%{x:.3f} · "
                "strict_avg=%{y:.3f} · strict=%{customdata[1]}<extra></extra>"
            ),
            name="continuous sweep",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
        showlegend=False,
        hovermode="closest",
        xaxis=dict(
            title="Operational FPR (avg false alert rate)",
            range=[max(0.0, x_min - x_pad), x_max + x_pad],
            tickformat=".2f",
        ),
        yaxis=dict(
            title="H2-strict avg recall",
            range=[max(0.0, y_min - y_pad), min(1.0, y_max + y_pad)],
            tickformat=".2f",
            dtick=0.05,
        ),
    )
    return fig


def pareto_mini(
    df: pd.DataFrame,
    *,
    highlight: tuple[float, ...] = (93.0, 95.0),
) -> go.Figure:
    """Static Pareto for the Live Monitor — published p93/p95 highlighted."""
    fig = _base_scatter(df)
    hi = df[df["percentile"].isin(highlight)]
    fig.add_trace(
        go.Scatter(
            x=hi["avg_false_alert_rate"],
            y=hi["h2_strict_avg"],
            mode="markers+text",
            marker=dict(size=14, color=PASS, line=dict(color="#222", width=1.5)),
            text=[f"p{p:.1f}" for p in hi["percentile"]],
            textposition="top center",
            hoverinfo="skip",
            name="published",
        )
    )
    fig.update_layout(height=320)
    return fig


def pareto_interactive(
    df: pd.DataFrame,
    selected_pct: float,
    *,
    fpr_budget: float | None = None,
) -> go.Figure:
    """Interactive Pareto with a moving marker for the selected percentile.

    Used by Page 4 (Phase 3). The optional `fpr_budget` draws a dashed
    vertical line — annotation only, does NOT alter any computed metric
    (CATCH 2: confidence + FPR-budget sliders are annotation-only).
    """
    fig = _base_scatter(df)
    sel = df.loc[df["percentile"] == selected_pct]
    if not sel.empty:
        fig.add_trace(
            go.Scatter(
                x=sel["avg_false_alert_rate"],
                y=sel["h2_strict_avg"],
                mode="markers",
                marker=dict(
                    size=20,
                    color=FAIL,
                    line=dict(color="#222", width=1.5),
                    symbol="circle-open-dot",
                ),
                hoverinfo="skip",
                name="selected",
            )
        )
    if fpr_budget is not None:
        fig.add_vline(
            x=fpr_budget,
            line_dash="dash",
            line_color="#666",
            annotation_text=f"FPR budget = {fpr_budget:.2f}",
            annotation_position="top right",
        )
        # Extend x-range if the budget vline falls outside the data envelope
        # (slider goes to 0.40; data tops out around 0.32). Keeps the dashed
        # line + label visible across the full slider range.
        x_lo, x_hi = fig.layout.xaxis.range
        if fpr_budget > x_hi:
            fig.update_xaxes(range=[x_lo, fpr_budget + 0.02])
        elif fpr_budget < x_lo:
            fig.update_xaxes(range=[fpr_budget - 0.02, x_hi])
    fig.update_layout(height=420)
    return fig
