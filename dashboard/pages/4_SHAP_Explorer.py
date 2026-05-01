"""Page 3 — SHAP Explorer (browse pre-computed §16 per-class signatures).

Layout (top to bottom):
1. Header + caption
2. Class selector dropdown (defaults to MQTT_DDoS_Connect_Flood, the
   §16 headline class). Pre-selectable from Page 2 via session_state.
3. Top-10 mean |SHAP| bar chart for the selected class.
4. DDoS↔DoS comparison panel — always visible. Each of 44 features as
   one point on a (DDoS, DoS) scatter plot with a y=x reference line and
   the 0.991 cosine in the title (Q4 = c).
5. All-class heatmap (collapsible, default collapsed) — 19 classes ×
   top-15 globally-ranked features.
6. Footnote with random_state, sensitivity Kendall τ, and SHA256 status.

This page reads ONLY from `results/shap/shap_values/shap_values.npy` and
metadata in `results/shap/{config.json, sensitivity/comparison.csv}`. No
runtime SHAP — that's Page 2's TreeExplainer code path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st

from components import shap_charts as sc
from components import shap_loader as sl

st.set_page_config(
    page_title="IoMT IDS — SHAP Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
COMPARISON_CSV: Final[Path] = (
    PROJECT_ROOT / "results/shap/sensitivity/comparison.csv"
)


@st.cache_data(ttl=None, show_spinner=False)
def _per_class_sample_counts() -> dict[str, int]:
    """Number of test rows per class in the SHAP subsample (n_samples
    caption — varies per class because Phase 7 used stratified subsampling
    with a 20-sample minimum)."""
    y = pd.read_csv(PROJECT_ROOT / "results/shap/shap_values/y_shap_subset.csv")
    return {str(k): int(v) for k, v in y["class_name"].value_counts().items()}


@st.cache_data(ttl=None, show_spinner=False)
def _sensitivity_summary() -> dict[str, float]:
    """Kendall τ and DDoS↔DoS cosines from the §16.7B sensitivity run."""
    row = pd.read_csv(COMPARISON_CSV).iloc[0]
    return {
        "kendall_top10": float(row["kendall_tau_top10_union"]),
        "kendall_full44": float(row["kendall_tau_full44"]),
        "cos_test_bg": float(row["ddos_dos_cosine_test_bg"]),
        "cos_train_bg": float(row["ddos_dos_cosine_train_bg"]),
        "decision": str(row["decision"]),
    }


# ---------------------------------------------------------------------------
# Tripwire: refuse to render if shap_values.npy hash mismatches.
# ---------------------------------------------------------------------------
sha_ok, observed = sl.verify_sha256()
if not sha_ok:
    st.error(
        f"**Reproducibility tripwire failed** — `shap_values.npy` SHA256 "
        f"does not match the committed reference. "
        f"Observed: `{observed[:16]}…`. Expected: `{sl.SHAP_VALUES_SHA256[:16]}…`. "
        f"Refusing to render — the file may be corrupted or a different SHAP "
        f"run was substituted. Restore from `results/shap/shap_values/` and "
        f"rerun if intentional."
    )
    st.stop()

meta = sl.shap_metadata()
class_names = meta["class_names"]
feature_names = meta["feature_names"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("SHAP Explorer")
st.caption("Pre-computed per-class feature attribution from §16.")

# ---------------------------------------------------------------------------
# Class selector — Page 2 may have pre-set the selection via session_state.
# ---------------------------------------------------------------------------
default_class = st.session_state.get("shap_selected_class", "MQTT_DDoS_Connect_Flood")
if default_class not in class_names:
    default_class = "MQTT_DDoS_Connect_Flood"

selected = st.selectbox(
    "Attack class",
    options=sorted(class_names),
    index=sorted(class_names).index(default_class),
    help="Defaults to MQTT_DDoS_Connect_Flood, the §16 headline class.",
)

# Persist whichever class the user just picked so cross-page navigation stays
# coherent (Page 2 footer → switch_page reads this on next visit).
st.session_state["shap_selected_class"] = selected

# ---------------------------------------------------------------------------
# Top-10 features for the selected class
# ---------------------------------------------------------------------------
counts = _per_class_sample_counts()
top10 = sl.top_k_features(selected, k=10)

st.plotly_chart(
    sc.per_class_bar(top10, class_name=selected, n_samples=counts.get(selected, 0)),
    use_container_width=True,
)
st.caption(
    f"Mean absolute SHAP value across {counts.get(selected, 0)} test samples of "
    f"class **{selected}**. Sourced from pre-computed `shap_values.npy`."
)

# ---------------------------------------------------------------------------
# DDoS↔DoS comparison panel (always visible)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("DDoS family ↔ DoS family — §16.4 headline finding")

ddos_sig = sl.signature_for_category("DDoS")
dos_sig = sl.signature_for_category("DoS")
cos_dd = sl.category_cosine("DDoS", "DoS")

st.plotly_chart(
    sc.ddos_vs_dos_scatter(ddos_sig, dos_sig, feature_names, cosine=cos_dd),
    use_container_width=True,
)
st.caption(
    f"Each point is one of the 44 features. Closeness to the dashed y=x line "
    f"shows how similarly the feature contributes across the two attack "
    f"families. Cosine similarity = **{cos_dd:.3f}** — the two families share "
    f"~{cos_dd * 100:.1f}% of their SHAP signature, the boundary the global "
    f"explainer cannot separate (§16.4)."
)

# ---------------------------------------------------------------------------
# All-class heatmap (collapsible)
# ---------------------------------------------------------------------------
st.divider()
with st.expander("All-class heatmap (19 classes × top-15 features)", expanded=False):
    st.plotly_chart(
        sc.class_heatmap(sl.per_class_signature(), class_names, feature_names),
        use_container_width=True,
    )
    st.caption(
        "Rows = 19 attack/benign classes; columns = top-15 features ranked by "
        "global mean |SHAP|. Use this to spot patterns — e.g., IAT dominates "
        "almost every class, while certain features cluster within attack "
        "families (DDoS_*, MQTT_*)."
    )

# ---------------------------------------------------------------------------
# Footnote
# ---------------------------------------------------------------------------
st.divider()
sens = _sensitivity_summary()
st.caption(
    f"SHAP values pre-computed during Phase 7 using `TreeExplainer` with "
    f"`random_state=42`, subsample n=5000, background n=500. Validated stable "
    f"to background sampling perturbations (§16.7B Kendall "
    f"τ\\_top10\\_union = {sens['kendall_top10']:.3f}, full-44 "
    f"τ = {sens['kendall_full44']:.3f}, decision = {sens['decision']}). "
    f"DDoS↔DoS cosine reproduces across backgrounds: test = "
    f"{sens['cos_test_bg']:.3f}, train = {sens['cos_train_bg']:.3f}."
)
