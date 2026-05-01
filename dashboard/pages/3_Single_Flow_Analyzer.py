"""Page 2 — Single Flow Analyzer.

Paste / upload / pick one network flow's 44 features, see the full pipeline:
    E7 prediction · softmax entropy · AE reconstruction · fusion case · SHAP

Key invariants:
- TF is imported lazily (only when the AE loads). Pages 1/3/4/5 stay TF-free.
- Reproducibility tripwires (E7 + AE) run on first model load. Page refuses
  to score if either fails.
- Runtime SHAP via TreeExplainer is computed for ONE flow at request time.
  Per-class signatures from §16 live on Page 3 — different code path.
- AE is gracefully optional: if TF env breaks, the AE card shows
  "unavailable" and the rest of the page still works.
- Fusion: §15D 5-case entropy_fusion (canonical source:
  notebooks/enhanced_fusion.py:499-512).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

# model_loader FIRST — on macOS, TF must claim libomp before pandas/numpy via
# data_loader, otherwise ae.predict deadlocks against xgboost. See model_loader
# docstring §1a.
from components import model_loader as ml

import numpy as np
import streamlit as st

from components import flow_input as fi
from components import shap_charts as sc
from components import shap_loader as sl
from components.data_loader import (
    PUBLISHED_PERCENTILE,
    load_sweep_table,
    sweep_row_at,
)
from components.status_indicators import BORDER, FAIL, PASS

st.set_page_config(
    page_title="IoMT IDS — Single Flow Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENTROPY_THRESHOLDS_PATH: Final[Path] = (
    PROJECT_ROOT / "results/enhanced_fusion/signals/entropy_thresholds.json"
)
AE_THRESHOLD_P90: Final[float] = 0.20127058029174805  # from unsupervised/config.json
BENIGN_CLASS_IDX: Final[int] = 1                       # global_class_map["Benign"]


@st.cache_data(ttl=None, show_spinner=False)
def _entropy_thresholds() -> dict[str, float]:
    return json.loads(ENTROPY_THRESHOLDS_PATH.read_text())


@st.cache_data(ttl=None, show_spinner=False)
def _shap_config() -> dict:
    return json.loads((PROJECT_ROOT / "results/shap/config.json").read_text())


def _classify_case(
    *,
    pred_idx: int,
    entropy: float,
    ae_mse: float | None,
    ent_threshold: float,
) -> tuple[int, str, str]:
    """Apply the canonical §15D 5-case entropy_fusion partition to one flow.

    Source of truth: ``entropy_fusion`` at notebooks/enhanced_fusion.py:499-512.
    Mirrors that function's np.where chain exactly:

        sup_attack   = (E7 prediction != Benign)
        high_entropy = (entropy > ent_threshold)
        ae_anomaly   = (AE recon MSE > AE_THRESHOLD_P90)

        Case 1: sup_attack &  ae_anomaly & ¬high_entropy  → Confirmed Alert
        Case 2: ¬sup_attack &  ae_anomaly                 → Zero-Day Warning
        Case 2:  high_entropy &  ae_anomaly               → Zero-Day Warning
        Case 5:  high_entropy & ¬ae_anomaly               → Uncertain Alert
        Case 3:  sup_attack & ¬ae_anomaly & ¬high_entropy → Low-Confidence Alert
        Case 4: otherwise                                 → Clear

    A flow that is sup_attack & ae_anomaly & high_entropy is *promoted* to
    Case 2 — the entropy signal escalates a "confirmed" verdict to a
    "zero-day" one, on the theory that high model uncertainty stacked on top
    of an anomaly signal suggests a novel attack rather than a familiar one.

    AE-unavailable degradation: when the TF env regression makes the AE
    unavailable, ``ae_mse is None`` and ``ae_anomaly`` collapses to False
    uniformly. The partition then reduces to {3, 4, 5} only — Cases 1 and 2
    become unreachable. The AE card on the page already surfaces "unavailable"
    in that mode, so no per-case override is needed here.

    Concretely, the four sample fixtures degrade as follows when AE drops out:
        Recon_Ping_Sweep    : Case 1 → 3   (AE confirmation lost)
        ARP_Spoofing        : Case 2 → 5   (entropy still escalates → review)
        Benign              : Case 2 → 4   (AE rescue lost — silent false negative)
        MQTT_Malformed_Data : Case 5 → 5   (entropy-only verdict; unchanged)

    The Benign Case 2 → Case 4 collapse is the load-bearing one: a flow that
    AE *rescued* from a false-benign becomes a silent false negative when AE
    drops out. This is partition-mechanical, not a partition bug — but it
    deserves a Model Card limitation note in the next README update.

    Severity colors (alarm-priority ordering, not signal-confidence):
        Case 1 (confirmed attack)   → FAIL (red)
        Case 2 (zero-day warning)   → FAIL (red)
        Case 3 (low-confidence)     → BORDER (yellow)
        Case 4 (clear)              → PASS (green)
        Case 5 (uncertain alert)    → BORDER (yellow) — operator review
    """
    sup_attack = (pred_idx != BENIGN_CLASS_IDX)
    high_entropy = (entropy > ent_threshold)
    ae_anomaly = (ae_mse is not None) and (ae_mse > AE_THRESHOLD_P90)

    if sup_attack and ae_anomaly and not high_entropy:
        return 1, "Confirmed Alert — E7 attack + AE anomaly agree", FAIL
    if not sup_attack and ae_anomaly:
        return 2, "Zero-Day Warning — AE flags an E7-benign flow", FAIL
    if high_entropy and ae_anomaly:
        return 2, "Zero-Day Warning — high entropy + AE anomaly agree", FAIL
    if high_entropy and not ae_anomaly:
        return 5, "Uncertain Alert — high entropy, AE clean (operator review)", BORDER
    if sup_attack and not ae_anomaly and not high_entropy:
        return 3, "Low-Confidence Alert — E7 attack but AE clean", BORDER
    return 4, "Clear — E7 benign, AE clean, low entropy", PASS


def _assert_classify_case_canonical() -> None:
    """Defensive partition-logic self-check, run once at module load.

    Validates that ``_classify_case`` produces the canonical case numbers
    for four hardcoded signal triples spanning Cases 1, 2, and 5. Same
    defensive pattern as Week 7's E7 + SHA256 tripwires — failure raises
    AssertionError before any UI renders, so Page 2 cannot ship a partition
    that has drifted from notebooks/enhanced_fusion.py.

    The threshold (0.1) is synthetic, chosen to bracket the four sample-flow
    entropies (0.0022, 0.0041, 0.5212, 1.1802) cleanly. This is a logic-only
    check and is intentionally decoupled from sweep_table.csv regeneration.
    """
    THR = 0.1
    fixtures = [
        # (pred_idx, entropy, ae_mse, expected_case, name)
        (3, 0.0022, 0.4266, 1, "Recon_Ping_Sweep"),
        (3, 0.5212, 0.2809, 2, "ARP_Spoofing"),
        (1, 0.0041, 0.2993, 2, "Benign"),
        (1, 1.1802, 0.1937, 5, "MQTT_Malformed_Data"),
    ]
    for pred, ent, ae, expected, name in fixtures:
        actual, _, _ = _classify_case(
            pred_idx=pred, entropy=ent, ae_mse=ae, ent_threshold=THR,
        )
        assert actual == expected, (
            f"_classify_case partition drift: {name} expected Case {expected}, "
            f"got Case {actual}. Canonical source: "
            f"notebooks/enhanced_fusion.py:499-512."
        )


_assert_classify_case_canonical()


def _entropy_marker_chart(entropy: float, ent_thr: dict[str, float]) -> "object":
    """Static-marker entropy bar (Q3 = I): published p90/p95/p97/p99 markers + dot."""
    import plotly.graph_objects as go
    pts = [(name.removeprefix("ent_"), val) for name, val in ent_thr.items()]
    pts.sort(key=lambda kv: kv[1])
    x_max = max(max(v for _, v in pts), entropy) * 1.15

    fig = go.Figure()
    # Track
    fig.add_shape(type="line", x0=0, x1=x_max, y0=0, y1=0,
                  line=dict(color="#cccccc", width=2))
    # Threshold markers
    for name, val in pts:
        fig.add_shape(type="line", x0=val, x1=val, y0=-0.15, y1=0.15,
                      line=dict(color="#666666", width=1, dash="dash"))
        fig.add_annotation(x=val, y=0.25, text=name, showarrow=False,
                           font=dict(size=10, color="#666666"))
    # Current-value dot
    fig.add_trace(go.Scatter(
        x=[entropy], y=[0],
        mode="markers",
        marker=dict(size=14, color=PASS if entropy <= pts[0][1] else FAIL,
                    line=dict(color="#222", width=1.5)),
        hovertemplate=f"entropy = {entropy:.4f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=20), height=110, showlegend=False,
        xaxis=dict(title="softmax entropy (nats)", range=[0, x_max], tickformat=".2f"),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
    )
    return fig


def _ae_marker_chart(ae_mse: float | None) -> "object":
    """Static-marker AE recon bar (parallel to entropy bar; uses p90)."""
    import plotly.graph_objects as go
    if ae_mse is None:
        return None
    x_max = max(AE_THRESHOLD_P90 * 2, ae_mse * 1.15)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, x1=x_max, y0=0, y1=0,
                  line=dict(color="#cccccc", width=2))
    fig.add_shape(type="line", x0=AE_THRESHOLD_P90, x1=AE_THRESHOLD_P90,
                  y0=-0.15, y1=0.15,
                  line=dict(color="#666666", width=1, dash="dash"))
    fig.add_annotation(x=AE_THRESHOLD_P90, y=0.25, text="p90 = 0.201",
                       showarrow=False, font=dict(size=10, color="#666666"))
    fig.add_trace(go.Scatter(
        x=[ae_mse], y=[0], mode="markers",
        marker=dict(size=14,
                    color=PASS if ae_mse <= AE_THRESHOLD_P90 else FAIL,
                    line=dict(color="#222", width=1.5)),
        hovertemplate=f"recon MSE = {ae_mse:.4f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=20), height=110, showlegend=False,
        xaxis=dict(title="AE reconstruction MSE", range=[0, x_max], tickformat=".3f"),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
    )
    return fig


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Single Flow Analyzer")
st.caption("Paste a flow's 44 features, see the full pipeline output.")

# ---------------------------------------------------------------------------
# Reproducibility tripwires (run once per Streamlit session).
# Order matters on macOS: prime_tf() must run BEFORE any xgboost predict,
# otherwise the libomp threadpool collision deadlocks ae.predict (see
# model_loader.py docstring §1a).
# ---------------------------------------------------------------------------
with st.spinner("Verifying model integrity ..."):
    ml.prime_tf()
    e7_ok, e7_diff = ml.verify_e7_tripwire()

if not e7_ok:
    st.error(
        f"**Reproducibility tripwire failed** — E7 produced predictions that "
        f"differ from the committed reference by max |Δ| = {e7_diff:.2e} "
        f"(tolerance {ml.E7_TRIPWIRE_TOL:.0e}). The dashboard will not score "
        f"user input until this is investigated. The loaded model may differ "
        f"from the one used to produce the published §16 results."
    )
    st.stop()

# AE tripwire is informational only — failure degrades AE card, doesn't block.
ae_ok, ae_fresh, ae_ref = ml.verify_ae_tripwire()

# ---------------------------------------------------------------------------
# Input mode selector
# ---------------------------------------------------------------------------
st.subheader("Input")
mode = st.radio(
    "Mode",
    options=("Paste CSV row", "Upload CSV file", "Use sample flow"),
    index=0,
    horizontal=True,
    label_visibility="collapsed",
)

validation: fi.ValidationResult
if mode == "Paste CSV row":
    text = st.text_area(
        f"Paste a comma-separated row of {fi.N_FEATURES} numeric values "
        f"(feature order: see Page 3's class signature legend, or hover the "
        f"feature names below)",
        height=110,
        placeholder="3958.44, 1.83, -0.42, ...",
    )
    validation = fi.parse_paste_row(text)
elif mode == "Upload CSV file":
    uploaded = st.file_uploader(
        f"Upload a CSV with header row containing all {fi.N_FEATURES} expected columns",
        type=("csv",),
    )
    if uploaded is None:
        validation = fi.ValidationResult(False, "red", "Upload a CSV to score.", None)
    else:
        validation = fi.parse_csv_upload(uploaded)
else:
    sample_choice = st.selectbox(
        "Pre-selected demo flows from X_test.npy",
        options=list(fi.SAMPLE_FLOWS.keys()),
        index=0,
    )
    validation = fi.load_sample_flow(sample_choice)

# Validation feedback
if validation.severity == "green":
    st.success(validation.message)
elif validation.severity == "yellow":
    st.warning(validation.message)
else:
    st.error(validation.message)

# Score button: disabled if validation fails. Streamlit reruns the page on
# any subsequent widget click (e.g., the navigation button at the bottom of
# this page), and `st.button()` returns True only on the rerun it was clicked
# — so we persist the scored input in session_state and key the pipeline-
# render gate off that, not the transient `score_clicked`.
score_clicked = st.button(
    "🔍 Validate & Score",
    type="primary",
    disabled=not validation.ok,
)

if score_clicked and validation.ok:
    st.session_state["page2_scored_values"] = np.asarray(validation.values, dtype=np.float32)

# ---------------------------------------------------------------------------
# Pipeline output (renders if any score has been performed in this session)
# ---------------------------------------------------------------------------
scored_values = st.session_state.get("page2_scored_values")
if scored_values is None:
    st.stop()

result = ml.score_flow(scored_values)

shap_cfg = _shap_config()
class_names = shap_cfg["class_names"]
feat_names = shap_cfg["feature_names"]
pred_class = class_names[result["pred_class_idx"]]
pred_conf = float(result["proba"][result["pred_class_idx"]])

# Snap-to-published-grid for entropy threshold display
ent_thr = _entropy_thresholds()
ent_p93 = float(sweep_row_at(PUBLISHED_PERCENTILE)["ent_threshold"])
case_n, case_label, case_color = _classify_case(
    pred_idx=result["pred_class_idx"],
    entropy=result["entropy"],
    ae_mse=result["ae_mse"],
    ent_threshold=ent_p93,
)
ent_elevated = result["entropy"] > ent_p93

st.divider()
st.subheader("Pipeline output")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    label="E7 prediction",
    value=pred_class,
    delta=f"confidence {pred_conf:.3f}",
    delta_color="off",
)

ent_status = "below p93" if result["entropy"] <= ent_p93 else "above p93"
c2.metric(
    label="Softmax entropy",
    value=f"{result['entropy']:.4f}",
    delta=f"{ent_status} ({ent_p93:.4f})",
    delta_color="normal" if result["entropy"] <= ent_p93 else "inverse",
)

if result["ae_available"]:
    ae_status = "below p90" if result["ae_mse"] <= AE_THRESHOLD_P90 else "above p90"
    c3.metric(
        label="AE reconstruction MSE",
        value=f"{result['ae_mse']:.4f}",
        delta=f"{ae_status} ({AE_THRESHOLD_P90:.4f})",
        delta_color="normal" if result["ae_mse"] <= AE_THRESHOLD_P90 else "inverse",
    )
else:
    c3.metric(
        label="AE reconstruction MSE",
        value="unavailable",
        delta="see Model Card §Limitations",
        delta_color="off",
    )

c4.metric(
    label=f"Fusion case",
    value=f"Case {case_n}",
    delta=case_label,
    delta_color=("normal" if case_color == PASS else
                 "inverse" if case_color == FAIL else "off"),
)

# Entropy advisory — reinforces the entropy-aware case verdict for high-
# entropy flows. Pre-Week-7.1 this fired *instead* of routing entropy through
# the partition; now the partition itself uses entropy (canonical 5-case
# entropy_fusion), so this banner is contextual reinforcement of the Case 2
# entropy-promotion or Case 5 uncertain-alert verdict above — not a separate
# signal. Banner trigger condition (ent_elevated) intentionally unchanged.
if ent_elevated:
    st.warning(
        f"⚠ §15D entropy-rescue branch context: this flow's entropy "
        f"{result['entropy']:.4f} exceeds the p93 threshold ({ent_p93:.4f}), "
        f"which contributed to the Case {case_n} verdict above. The threshold "
        f"sweep on Page 4 explores how this boundary moves under different "
        f"operating points."
    )

# Entropy + AE static-marker bars
b1, b2 = st.columns(2)
with b1:
    st.plotly_chart(
        _entropy_marker_chart(result["entropy"], ent_thr),
        use_container_width=True,
    )
    st.caption(
        f"Static §15D entropy thresholds: "
        f"p90={ent_thr['ent_p90']:.3f} · p95={ent_thr['ent_p95']:.3f} · "
        f"p97={ent_thr['ent_p97']:.3f} · p99={ent_thr['ent_p99']:.3f}. "
        f"Dot = current flow."
    )

with b2:
    ae_chart = _ae_marker_chart(result["ae_mse"])
    if ae_chart is not None:
        st.plotly_chart(ae_chart, use_container_width=True)
        if not ae_ok:
            st.caption(
                f"⚠ AE tripwire deviation: fresh mean MSE = {ae_fresh:.4f} "
                f"vs committed reference {ae_ref:.4f}. AE numbers may not match §15D."
            )
        else:
            st.caption(f"Phase 5 AE recon p90 threshold = {AE_THRESHOLD_P90:.4f}.")
    else:
        st.info(
            "**AE component unavailable in this Python environment** — "
            "see Model Card §Limitations. E7, entropy, fusion, and SHAP cards "
            "above are unaffected."
        )

# ---------------------------------------------------------------------------
# SHAP explanation (runtime TreeExplainer, top-5)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("SHAP explanation")

with st.spinner("Computing SHAP for this flow ..."):
    explainer = ml.get_e7_explainer()
    sv_full = explainer.shap_values(scored_values.reshape(1, -1))

# shap 0.51 + xgboost 3.x returns either (1, 44, 19) or list-of-arrays.
sv_arr = np.asarray(sv_full)
if sv_arr.ndim == 3:
    # shape (1, 44, 19) → take row 0, class column for pred_class_idx
    sv_for_pred = sv_arr[0, :, result["pred_class_idx"]]
elif sv_arr.ndim == 2:
    # already (44, 19) — same as above sliced
    sv_for_pred = sv_arr[:, result["pred_class_idx"]]
else:
    raise RuntimeError(f"unexpected SHAP shape {sv_arr.shape}")

st.plotly_chart(
    sc.runtime_shap_bar(
        sv_for_pred,
        scored_values,
        feat_names,
        pred_class=pred_class,
        top_k=5,
    ),
    use_container_width=True,
)
st.caption(
    f"These features pushed the model toward predicting **{pred_class}**. "
    f"Larger bars = stronger contribution. Red bars push toward `{pred_class}`; "
    f"blue bars push away."
)
st.caption(
    "**Note:** SHAP values shown are computed at runtime using `TreeExplainer` "
    "on the E7 XGBoost model. Per-class signatures from §16 are pre-computed "
    "and shown on Page 3 (SHAP Explorer)."
)

# ---------------------------------------------------------------------------
# Closest matching pre-computed §16 signature.
# Cross-page contract: `st.session_state["shap_selected_class"]` is read by
# Page 3's class dropdown to pre-select the matched class on navigation.
# ---------------------------------------------------------------------------
st.divider()
closest_name, closest_cos = sl.closest_class(sv_for_pred, feat_names)
left, right = st.columns([3, 2])
with left:
    st.caption(
        f"Closest matching pre-computed §16 signature: **{closest_name}** "
        f"(cosine similarity = {closest_cos:.3f}). "
        f"Open Page 3 for the full signature."
    )
with right:
    if st.button("→ Open Page 3 with this class pre-selected", use_container_width=True):
        st.session_state["shap_selected_class"] = closest_name
        st.switch_page("pages/4_SHAP_Explorer.py")
