"""Page 3 — SHAP Explorer (Week 7 stub).

Week 6 ships an empty placeholder so the sidebar nav order is stable;
Week 7 will replace this body with the per-class SHAP signature browser.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="IoMT IDS — SHAP Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SHAP Explorer")

st.info(
    "**Coming in Week 7.** This page will let you browse per-class SHAP "
    "signatures across the 19 attack classes and visualize the §16.4 "
    "finding (DDoS ↔ DoS cosine = 0.991 — the boundary the global "
    "explainer cannot separate)."
)

st.markdown(
    "The intent is a feature-attribution view that surfaces which classes "
    "share signature space — useful for defense Q&A around \"why does the "
    "model conflate DDoS and DoS?\" Available after Week 7's deployment commit."
)
