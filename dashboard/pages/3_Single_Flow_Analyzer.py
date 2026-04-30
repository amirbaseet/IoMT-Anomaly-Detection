"""Page 2 — Single Flow Analyzer (Week 7 stub).

Week 6 ships an empty placeholder so the sidebar nav order is stable;
Week 7 will replace this body with the live single-flow pipeline.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="IoMT IDS — Single Flow Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Single Flow Analyzer")

st.info(
    "**Coming in Week 7.** This page will let you paste a single network "
    "flow's 44 features and see the full pipeline output: E7 prediction, "
    "AE flag, entropy score, fusion case (Case A / B / C), and SHAP top-5 "
    "contributing features."
)

st.markdown(
    "The intent is a defense-time \"what does the model say about *this* "
    "flow?\" probe — one row of CICIoMT2024 features in, full classification "
    "trace out. Available after Week 7's deployment commit."
)
