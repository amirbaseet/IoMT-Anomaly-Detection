"""SHAP visualization helpers — Plotly only.

Page 2 (`runtime_shap_bar`): per-flow TreeExplainer output. Sign-aware.
Page 3 (`per_class_bar`, `ddos_vs_dos_scatter`, `class_heatmap`): pre-computed
mean |SHAP| signatures from §16. Magnitude-only (no sign).

Design rule: this module never opens model files. It accepts already-computed
SHAP arrays + feature names + the predicted class, and returns Plotly figures.
That keeps Page 2 and Page 3 sharing one charting layer regardless of where
their SHAP values come from (runtime TreeExplainer vs pre-computed §16 values).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import plotly.graph_objects as go

# Sign-aware palette: red = pushes prediction toward `pred_class`,
# blue = pushes away. Matches the spec's "red/blue split for sign".
POS: Final[str] = "#A32D2D"  # FAIL red — coincident with status_indicators palette
NEG: Final[str] = "#1F4E79"  # navy blue

# Magnitude palette for Page 3 (no sign — pre-computed |SHAP|).
MAG: Final[str] = "#0F6E56"   # PASS green
MAG_DDOS: Final[str] = "#A32D2D"  # red for DDoS family
MAG_DOS: Final[str] = "#1F4E79"   # blue for DoS family


def runtime_shap_bar(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    *,
    pred_class: str,
    top_k: int = 5,
) -> go.Figure:
    """Top-k SHAP contributions for a single user-input flow.

    Args:
        shap_values: shape (44,). SHAP values for `pred_class` against this
                     specific flow. Sign matters: positive → toward pred_class,
                     negative → away.
        feature_values: shape (44,). The actual (raw) feature values supplied
                     by the user — surfaced in the hover so reviewers can see
                     "which feature value generated this contribution."
        feature_names: list of 44 feature names in the canonical order.
        pred_class: predicted class string (used only for the chart title).
        top_k: number of features to render. Default 5 matches the spec.

    Returns:
        A Plotly horizontal bar chart with red/blue split, sorted by |SHAP|
        descending so the strongest contributor is at the top.
    """
    if shap_values.shape != (len(feature_names),):
        raise ValueError(
            f"shap_values shape {shap_values.shape} != ({len(feature_names)},)"
        )
    if feature_values.shape != shap_values.shape:
        raise ValueError("feature_values shape must match shap_values shape")

    abs_shap = np.abs(shap_values)
    order = np.argsort(abs_shap)[::-1][:top_k]
    # Reverse so the strongest is rendered at the top of the horizontal bar
    # (Plotly draws axis-bottom-to-top by default).
    order = order[::-1]

    sel_shap = shap_values[order]
    sel_feat = feature_values[order]
    sel_names = [feature_names[i] for i in order]
    colors = [POS if v >= 0 else NEG for v in sel_shap]

    fig = go.Figure(
        go.Bar(
            x=sel_shap,
            y=sel_names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in sel_shap],
            textposition="outside",
            cliponaxis=False,
            customdata=np.column_stack([sel_feat]),
            hovertemplate=(
                "%{y}<br>SHAP=%{x:+.4f}<br>"
                "feature value=%{customdata[0]:.4g}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Top {top_k} SHAP contributors → {pred_class}",
            font=dict(size=14),
        ),
        margin=dict(l=10, r=40, t=40, b=10),
        height=max(220, 40 * top_k + 80),
        showlegend=False,
        xaxis=dict(
            title="SHAP value (raw, log-odds space)",
            zeroline=True,
            zerolinecolor="#222",
            zerolinewidth=1,
        ),
        yaxis=dict(title=""),
    )
    return fig


def per_class_bar(
    feature_value_pairs: list[tuple[str, float]],
    *,
    class_name: str,
    n_samples: int,
) -> go.Figure:
    """Top-k mean |SHAP| bars for one pre-computed class signature (Page 3).

    Args:
        feature_value_pairs: list of (feature_name, mean_abs_shap), already
                             sorted descending by magnitude (output of
                             `shap_loader.top_k_features`).
        class_name: class string for the chart title.
        n_samples: number of test samples this signature was averaged over —
                   surfaced in the caption per spec ("Mean absolute SHAP value
                   across [N] test samples of this class.").
    """
    # Reverse for bottom-to-top rendering of the strongest at the top
    pairs = list(reversed(feature_value_pairs))
    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=MAG,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>mean |SHAP| = %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Top {len(pairs)} mean |SHAP| features for {class_name}",
            font=dict(size=14),
        ),
        margin=dict(l=10, r=40, t=40, b=10),
        height=max(280, 32 * len(pairs) + 80),
        showlegend=False,
        xaxis=dict(
            title=f"mean |SHAP| (n={n_samples} sampled rows)",
            zeroline=True,
            zerolinecolor="#222",
            zerolinewidth=1,
        ),
        yaxis=dict(title=""),
    )
    return fig


def ddos_vs_dos_scatter(
    ddos_signature: np.ndarray,
    dos_signature: np.ndarray,
    feature_names: list[str],
    *,
    cosine: float,
) -> go.Figure:
    """Each of the 44 features as one point on a (DDoS_mean_abs_shap,
    DoS_mean_abs_shap) scatter plot. y=x reference line + cosine in title.

    The diagonal-ness IS the cosine result — points hugging the y=x line
    visually justifies the high category-level cosine, which is the §16.4
    headline finding ("DDoS↔DoS heterogeneity").
    """
    if ddos_signature.shape != dos_signature.shape:
        raise ValueError("DDoS/DoS signatures must have same shape")
    if ddos_signature.shape != (len(feature_names),):
        raise ValueError(
            f"signature shape {ddos_signature.shape} != ({len(feature_names)},)"
        )

    # Annotate the 8 features with the strongest combined contribution so
    # reviewers can read the labels without scanning all 44 dots.
    combined = ddos_signature + dos_signature
    label_idx = set(np.argsort(combined)[::-1][:8].tolist())
    text = [feature_names[i] if i in label_idx else "" for i in range(len(feature_names))]

    axis_max = max(float(ddos_signature.max()), float(dos_signature.max())) * 1.08

    fig = go.Figure()
    # y=x reference line — the visual anchor for the cosine claim
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=axis_max, y1=axis_max,
        line=dict(color="#888", width=1, dash="dash"),
    )
    fig.add_trace(
        go.Scatter(
            x=ddos_signature,
            y=dos_signature,
            mode="markers+text",
            marker=dict(
                size=10,
                color=MAG,
                line=dict(color="#222", width=0.6),
            ),
            text=text,
            textposition="top right",
            textfont=dict(size=10, color="#222"),
            customdata=feature_names,
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "DDoS mean |SHAP| = %{x:.4f}<br>"
                "DoS mean |SHAP| = %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=(
                f"DDoS family ↔ DoS family — per-feature mean |SHAP| "
                f"(cosine = {cosine:.3f})"
            ),
            font=dict(size=14),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        height=480,
        showlegend=False,
        xaxis=dict(title="DDoS category mean |SHAP|", range=[0, axis_max], tickformat=".2f"),
        yaxis=dict(title="DoS category mean |SHAP|", range=[0, axis_max], tickformat=".2f", scaleanchor="x", scaleratio=1),
    )
    return fig


def class_heatmap(
    per_class_imp: np.ndarray,
    class_names: list[str],
    feature_names: list[str],
    *,
    top_k_features: int = 15,
) -> go.Figure:
    """All-class heatmap (default collapsed on Page 3).

    Rows = 19 classes; columns = the top-k features ranked by global mean
    |SHAP|. Color intensity = mean |SHAP| for (class, feature).
    """
    if per_class_imp.shape != (len(class_names), len(feature_names)):
        raise ValueError(
            f"per_class_imp shape {per_class_imp.shape} != "
            f"({len(class_names)}, {len(feature_names)})"
        )
    global_imp = per_class_imp.mean(axis=0)
    top_idx = np.argsort(global_imp)[::-1][:top_k_features]
    z = per_class_imp[:, top_idx]
    cols = [feature_names[i] for i in top_idx]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=cols,
            y=class_names,
            colorscale="Viridis",
            colorbar=dict(title="mean |SHAP|", thickness=12),
            hovertemplate="%{y} · %{x}<br>mean |SHAP| = %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=20, t=20, b=80),
        height=520,
        xaxis=dict(title="", tickangle=-45),
        yaxis=dict(title=""),
    )
    return fig
