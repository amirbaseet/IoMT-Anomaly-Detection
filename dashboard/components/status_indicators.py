"""Color tokens + small UI components for status display.

Palette confirmed by user (matches existing visualizer aesthetic):
- PASS    #0F6E56 (green)   — recall ≥ 0.80
- BORDER  #854F0B (yellow)  — 0.70 ≤ recall < 0.80
- FAIL    #A32D2D (red)     — recall < 0.70
- NEUTRAL #E5E5E5            — track / muted lines
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import streamlit as st

from .data_loader import BORDER_RECALL, PASS_RECALL

PASS: Final[str] = "#0F6E56"
BORDER: Final[str] = "#854F0B"
FAIL: Final[str] = "#A32D2D"
NEUTRAL: Final[str] = "#E5E5E5"


def status_for_recall(recall: float) -> tuple[str, str]:
    """Map a recall value to (color, label) per §15D acceptance thresholds."""
    if recall >= PASS_RECALL:
        return PASS, "PASS"
    if recall >= BORDER_RECALL:
        return BORDER, "BORDERLINE"
    return FAIL, "FAIL"


def color_pill(label: str, color: str, *, tooltip: str | None = None) -> str:
    """Inline rounded badge as raw HTML (rendered with unsafe_allow_html=True)."""
    title_attr = f' title="{tooltip}"' if tooltip else ""
    return (
        f'<span{title_attr} style="display:inline-block;padding:4px 12px;'
        f'border-radius:14px;background:{color};color:#FFFFFF;font-weight:600;'
        f'font-size:0.85rem;margin:0 6px 6px 0;">{label}</span>'
    )


def render_target_badges(rows: pd.DataFrame) -> None:
    """One colored pill per row of an eligible-targets DataFrame.

    Tooltip on hover shows exact recall + n_loo_benign (CATCH 1: tooltip
    explicitly surfaces the eligibility predicate's input).
    """
    pills: list[str] = []
    for _, row in rows.iterrows():
        recall = float(row["h2_strict_rescue_recall"])
        color, _label = status_for_recall(recall)
        tooltip = (
            f"recall={recall:.3f} · n_loo_benign={int(row['n_loo_benign'])} "
            f"· n_target={int(row['n_target'])}"
        )
        pills.append(color_pill(str(row["target"]), color, tooltip=tooltip))
    st.markdown("".join(pills), unsafe_allow_html=True)
