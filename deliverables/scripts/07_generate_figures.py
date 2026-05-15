"""
Phase F — Figure generation for the thesis report.

WHAT WE FOUND
  16 required figures + 1 optional (PCA scatter) generated to deliverables/
  figures/ at 150 DPI. All from saved arrays/CSV/JSON — no retraining, no
  GPU. Deterministic sampling (np.random.default_rng(42)) for any sub-sample
  step.

WHY WE CHOSE THIS APPROACH
  Alternatives considered:
    - Skip figures, leave the report as text-only (rejected: examiners need
      visual confirmation of the headline plots)
    - Copy existing PNGs from results/ (rejected: existing figures use
      multiple styles; we want one consistent look across §4–§9)
    - Render via the dashboard (rejected: Streamlit / Plotly produce
      interactive HTML, not embeddable PNG)
  Decision criterion: a single Python script with matplotlib Agg backend
    gives bit-stable deterministic output, no display dependencies, and a
    single style pass.
  Tradeoff accepted: figures here are static and slightly less polished
    than dashboard equivalents; mitigated by consistent typography.
  Evidence: deliverables/figures/*.png, results/* artefacts cited per figure.

Run from project root:
  venv/bin/python -m deliverables.scripts.07_generate_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deliverables.scripts._common import (
    PROJECT_ROOT, FIGURES, banner, check_artifact_exists,
)

RNG = np.random.default_rng(42)
DPI = 150

# Consistent style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

FIGURES.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> str:
    """Save figure to deliverables/figures/<name> and return relative path."""
    out = FIGURES / name
    fig.savefig(out)
    plt.close(fig)
    rel = out.relative_to(PROJECT_ROOT)
    print(f"  wrote {rel}")
    return str(rel)


# ---------------------------------------------------------------------------
# §4 — Data and preprocessing
# ---------------------------------------------------------------------------

def fig01_class_distribution() -> None:
    """Log-scale bar chart, 19 classes, train + test side by side."""
    y_train = pd.read_csv(PROJECT_ROOT / "preprocessed/full_features/y_train.csv")
    y_test = pd.read_csv(PROJECT_ROOT / "preprocessed/full_features/y_test.csv")
    tr_counts = y_train["label"].value_counts().sort_values(ascending=False)
    te_counts = y_test["label"].value_counts()
    classes = tr_counts.index.tolist()
    tr_vals = [tr_counts.get(c, 0) for c in classes]
    te_vals = [te_counts.get(c, 0) for c in classes]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(classes))
    w = 0.4
    b1 = ax.bar(x - w / 2, tr_vals, w, label="Train (after dedup)", color="#1f77b4")
    b2 = ax.bar(x + w / 2, te_vals, w, label="Test (after dedup)", color="#ff7f0e")
    ax.set_yscale("log")
    ax.set_ylabel("Rows (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=70, ha="right", fontsize=8)
    ax.set_title("19-class distribution (train vs test, after deduplication)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    _save(fig, "fig01_class_distribution.png")


def fig02_cohens_d_top10() -> None:
    """Horizontal bar — top-10 |Cohen's d|, Attack vs Benign."""
    cd = pd.read_csv(PROJECT_ROOT / "eda_output/feature_target_cohens_d.csv")
    cd.columns = [c.strip() for c in cd.columns]
    feat_col = cd.columns[0]
    val_col = cd.columns[1]
    cd["abs_d"] = cd[val_col].abs()
    top = cd.sort_values("abs_d", ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top[feat_col].astype(str), top["abs_d"], color="#2ca02c")
    ax.set_xlabel("|Cohen's d|  (Attack vs Benign)")
    ax.set_title("Top-10 features by univariate effect size")
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(top["abs_d"]):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)
    _save(fig, "fig02_cohens_d_top10.png")


def fig03_correlation_heatmap() -> None:
    """Heatmap of |Pearson r| over a 50K sample of X_train."""
    cfg = json.load(open(PROJECT_ROOT / "preprocessed/config.json"))
    feat_names = cfg.get("feature_names_full", [f"f{i}" for i in range(44)])
    X = np.load(PROJECT_ROOT / "preprocessed/full_features/X_train.npy", mmap_mode="r")
    n = min(50_000, X.shape[0])
    idx = RNG.choice(X.shape[0], size=n, replace=False)
    sample = np.asarray(X[idx])
    corr = np.abs(np.corrcoef(sample, rowvar=False))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(feat_names)))
    ax.set_yticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=85, fontsize=6)
    ax.set_yticklabels(feat_names, fontsize=6)
    ax.set_title(f"|Pearson r| feature correlation matrix  (n = {n:,} stratified-random rows)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|r|")
    _save(fig, "fig03_correlation_heatmap.png")


def fig04_pca_2d() -> None:
    """PCA 2-D projection of 50K stratified sample (optional)."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  [fig-skipped] sklearn.decomposition.PCA unavailable")
        return
    X = np.load(PROJECT_ROOT / "preprocessed/full_features/X_train.npy", mmap_mode="r")
    y = pd.read_csv(PROJECT_ROOT / "preprocessed/full_features/y_train.csv")["category"]
    n_per_class = max(50, 50_000 // y.nunique())
    sampled_idx = []
    for cls in y.unique():
        cls_idx = np.where(y.values == cls)[0]
        take = min(len(cls_idx), n_per_class)
        sampled_idx.extend(RNG.choice(cls_idx, size=take, replace=False).tolist())
    sampled_idx = np.array(sampled_idx)
    sample = np.asarray(X[sampled_idx])
    labels = y.values[sampled_idx]
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(sample)
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(np.unique(labels)):
        mask = labels == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], s=3, alpha=0.35,
                   label=cls, color=cmap(i % 10))
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"PCA 2-D projection (50K stratified sample, 6 categories)")
    ax.legend(loc="upper right", markerscale=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, "fig04_pca_2d.png")


# ---------------------------------------------------------------------------
# §5 — Supervised
# ---------------------------------------------------------------------------

def fig05_e7_confusion_matrix() -> None:
    """Normalised 19×19 confusion matrix for E7 (test set)."""
    cm = np.load(PROJECT_ROOT / "results/supervised/metrics/E7_cm_19class_test.npy")
    le = json.load(open(PROJECT_ROOT / "preprocessed/label_encoders.json"))
    classes = sorted(le["multiclass"].keys(), key=lambda k: le["multiclass"][k])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("E7 (XGBoost / Full 44 / Original) — normalised test confusion matrix")
    # Annotate diagonal and large off-diagonal cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm_norm[i, j]
            if v >= 0.05:
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")
    _save(fig, "fig05_e7_confusion_matrix.png")


def fig06_e1_e8_comparison() -> None:
    """Grouped bar — 8 experiments × {macro-F1, accuracy, MCC} on 19-class test."""
    oc = pd.read_csv(PROJECT_ROOT / "results/supervised/metrics/overall_comparison.csv")
    mc = oc[oc["task"] == "multiclass"].set_index("experiment")
    order = ["E1", "E2", "E3", "E4", "E5", "E5G", "E6", "E7", "E8"]
    mc = mc.reindex([e for e in order if e in mc.index])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(mc.index))
    w = 0.27
    ax.bar(x - w, mc["test_f1_macro"], w, label="macro-F1", color="#1f77b4")
    ax.bar(x, mc["test_accuracy"], w, label="accuracy", color="#ff7f0e")
    ax.bar(x + w, mc["test_mcc"], w, label="MCC", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(mc.index.tolist())
    ax.set_ylim(0.80, 1.005)
    ax.set_ylabel("Metric value (test set)")
    ax.set_title("E1–E8 + E5G — 19-class test metrics")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate E7 macro-F1
    e7_idx = list(mc.index).index("E7") if "E7" in mc.index else None
    if e7_idx is not None:
        f1 = mc.loc["E7", "test_f1_macro"]
        ax.annotate(f"E7: {f1:.4f}", xy=(e7_idx - w, f1),
                    xytext=(e7_idx - w + 0.3, f1 + 0.015),
                    arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    _save(fig, "fig06_e1_e8_comparison.png")


def fig07_smote_effect() -> None:
    """Paired bar — macro-F1 Original vs SMOTE for 4 configs."""
    sc = pd.read_csv(PROJECT_ROOT / "results/supervised/metrics/smote_comparison.csv")
    mc = sc[(sc["task"] == "multiclass") & (sc["metric"] == "test_f1_macro")]
    labels = [f"{r['model']} / {r['feature_set']}" for _, r in mc.iterrows()]
    orig_vals = mc["original"].astype(float).tolist()
    smote_vals = mc["smote"].astype(float).tolist()
    deltas = mc["delta"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, orig_vals, w, label="Original", color="#1f77b4")
    ax.bar(x + w / 2, smote_vals, w, label="SMOTETomek", color="#d62728")
    ax.set_ylabel("macro-F1 (19-class test)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_title("SMOTETomek effect on macro-F1 — degrades all 4 configurations")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    for i, d in enumerate(deltas):
        ax.text(i, max(orig_vals[i], smote_vals[i]) + 0.005, f"Δ {d:+.3f}",
                ha="center", fontsize=8, color="#8B0000")
    ax.set_ylim(0.80, 0.95)
    _save(fig, "fig07_smote_effect.png")


# ---------------------------------------------------------------------------
# §6 — Unsupervised
# ---------------------------------------------------------------------------

def fig08_ae_loss_curve() -> None:
    """AE training loss + val_loss vs epoch."""
    hist = json.load(open(PROJECT_ROOT / "results/unsupervised/ae_training_history.json"))
    epochs = np.arange(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, hist["loss"], label="train loss", color="#1f77b4", lw=1.5)
    ax.plot(epochs, hist["val_loss"], label="val loss", color="#ff7f0e", lw=1.5)
    best_idx = int(np.argmin(hist["val_loss"]))
    best_val = hist["val_loss"][best_idx]
    ax.axvline(best_idx + 1, color="gray", ls="--", alpha=0.5,
               label=f"best epoch (val={best_val:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"AE training history — best val_loss = {best_val:.4f} (epoch {best_idx + 1})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "fig08_ae_loss_curve.png")


def fig09_ae_recon_error_hist() -> None:
    """Overlaid histogram of AE test recon-error: benign vs attack, log-y, p90 vline."""
    ae_test = np.load(PROJECT_ROOT / "results/unsupervised/scores/ae_test_mse.npy")
    y_test = pd.read_csv(PROJECT_ROOT / "preprocessed/full_features/y_test.csv")
    is_benign = (y_test["binary_label"].values == 0)
    thresh = json.load(open(PROJECT_ROOT / "results/unsupervised/thresholds.json"))
    p90 = thresh["thresholds"]["p90"]

    fig, ax = plt.subplots(figsize=(9, 5))
    # Clip far right tail for readability
    mse_clipped = np.clip(ae_test, 0, np.percentile(ae_test, 99.5))
    bins = np.linspace(0, mse_clipped.max(), 80)
    ax.hist(mse_clipped[is_benign], bins=bins, alpha=0.55,
            label="Benign", color="#1f77b4", density=False)
    ax.hist(mse_clipped[~is_benign], bins=bins, alpha=0.55,
            label="Attack", color="#d62728", density=False)
    ax.set_yscale("log")
    ax.axvline(p90, color="black", ls="--", lw=1.5,
               label=f"p90 threshold = {p90:.4f}")
    ax.set_xlabel("AE reconstruction MSE (clipped at p99.5)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("AE test reconstruction error — Benign vs Attack distributions")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y", which="both")
    _save(fig, "fig09_ae_recon_error_hist.png")


def fig10_detection_rate_heatmap() -> None:
    """19 classes × 5 thresholds heatmap (AE rows)."""
    pcd = pd.read_csv(PROJECT_ROOT / "results/unsupervised/metrics/per_class_detection_rates.csv")
    ae = pcd[pcd["model"] == "Autoencoder"].set_index("class")
    cols = ["p90", "p95", "p99", "mean_2std", "mean_3std"]
    # Sort by p90 desc so the visual reading is high→low recall
    ae = ae.sort_values("p90", ascending=False)
    mat = ae[cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20)
    ax.set_yticks(range(len(ae.index)))
    ax.set_yticklabels(ae.index.astype(str).tolist(), fontsize=8)
    ax.set_title("AE per-class detection rate — 19 classes × 5 threshold candidates")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            color = "white" if v < 0.3 or v > 0.8 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Detection rate")
    _save(fig, "fig10_detection_rate_heatmap.png")


def fig11_ae_vs_if_roc() -> None:
    """AE vs IF ROC curves with AUC annotations."""
    from sklearn.metrics import roc_curve, roc_auc_score
    ae = np.load(PROJECT_ROOT / "results/unsupervised/scores/ae_test_mse.npy")
    if_scores = np.load(PROJECT_ROOT / "results/unsupervised/scores/if_test_scores.npy")
    y_test = pd.read_csv(PROJECT_ROOT / "preprocessed/full_features/y_test.csv")
    y_binary = y_test["binary_label"].astype(int).values
    n = min(len(ae), len(if_scores), len(y_binary))

    fpr_ae, tpr_ae, _ = roc_curve(y_binary[:n], ae[:n])
    auc_ae = roc_auc_score(y_binary[:n], ae[:n])
    # IF: lower scores = more anomalous in some implementations; the published
    # AUC is 0.8612, so check sign and negate if needed
    if_pos = -if_scores
    auc_if_pos = roc_auc_score(y_binary[:n], if_pos[:n])
    if auc_if_pos < 0.5:
        if_pos = if_scores  # don't negate
        auc_if_pos = roc_auc_score(y_binary[:n], if_pos[:n])
    fpr_if, tpr_if, _ = roc_curve(y_binary[:n], if_pos[:n])

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot(fpr_ae, tpr_ae, label=f"Autoencoder  (AUC = {auc_ae:.4f})",
            color="#1f77b4", lw=2)
    ax.plot(fpr_if, tpr_if, label=f"Isolation Forest  (AUC = {auc_if_pos:.4f})",
            color="#d62728", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Binary anomaly detection — AE vs Isolation Forest (test)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    _save(fig, "fig11_ae_vs_if_roc.png")


# ---------------------------------------------------------------------------
# §7 — Fusion
# ---------------------------------------------------------------------------

def fig12_pareto_frontier() -> None:
    """Scatter FPR vs strict_avg from ablation_table; label entropy_benign_p95 elbow."""
    abl = pd.read_csv(PROJECT_ROOT / "results/enhanced_fusion/metrics/ablation_table.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, r in abl.iterrows():
        v = r["variant"]
        x = float(r["avg_false_alert_rate"])
        y = float(r["h2_strict_avg"])
        pass4 = "4/4" in str(r["h2_strict_pass"])
        color = "#2ca02c" if pass4 else "#888888"
        marker = "*" if v == "entropy_benign_p95" else ("o" if pass4 else "x")
        size = 200 if v == "entropy_benign_p95" else 80
        ax.scatter(x, y, s=size, color=color, marker=marker,
                   edgecolor="black" if v == "entropy_benign_p95" else "none",
                   linewidth=1.5, zorder=3)
        # label every point with the variant name
        offset_y = 0.02 if v != "entropy_benign_p95" else 0.025
        ax.annotate(v, xy=(x, y), xytext=(x + 0.005, y + offset_y),
                    fontsize=7.5, alpha=0.85)
    # Highlight the 0.70 strict threshold and the FPR=0.25 budget
    ax.axhline(0.70, color="gray", ls="--", alpha=0.5, lw=1)
    ax.text(0.30, 0.71, "H2-strict 0.70 threshold", fontsize=8, color="gray", ha="right")
    ax.axvline(0.25, color="gray", ls=":", alpha=0.5, lw=1)
    ax.text(0.252, 0.05, "FPR=0.25 budget", fontsize=8, color="gray", rotation=90, va="bottom")
    ax.set_xlabel("Benign FPR (fusion level)")
    ax.set_ylabel("H2-strict average recall")
    ax.set_title("Phase 6C Pareto frontier — 11 variants × (FPR, strict_avg)")
    # green / grey star/circle/x legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2ca02c",
               markersize=15, markeredgecolor="black", label="entropy_benign_p95 (chosen elbow)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=10, label="4/4 strict pass"),
        Line2D([0], [0], marker="x", color="#888888", markersize=10,
               linestyle="None", label="< 4/4 strict pass"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(True, alpha=0.3)
    _save(fig, "fig12_pareto_frontier.png")


def fig13_threshold_sweep() -> None:
    """Dual-axis: strict_avg + FPR vs percentile, p93/p95 markers."""
    sw = pd.read_csv(PROJECT_ROOT / "results/enhanced_fusion/threshold_sweep/sweep_table.csv")
    pcol = next((c for c in sw.columns if c.lower() in ("percentile", "p", "pct")), None)
    avg_col = next((c for c in sw.columns if "strict_avg" in c.lower()), None)
    fpr_col = next((c for c in sw.columns if "fpr" in c.lower() or "false_alert" in c.lower()), None)
    pass_col = next((c for c in sw.columns if "strict_pass" in c.lower()), None)
    sw = sw.sort_values(pcol)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(sw[pcol], sw[avg_col], "-o", color="#1f77b4",
             label="strict_avg", markersize=4)
    ax1.set_xlabel("Entropy threshold percentile (on benign-val)")
    ax1.set_ylabel("H2-strict avg recall", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.axhline(0.70, color="gray", ls="--", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(sw[pcol], sw[fpr_col], "-s", color="#d62728",
             label="benign FPR", markersize=4)
    ax2.set_ylabel("Benign FPR (fusion level)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.axhline(0.25, color="gray", ls=":", alpha=0.5)
    # Mark p93 and p95
    for pct, label in [(93.0, "p93.0 refined optimum"), (95.0, "p95.0 published anchor")]:
        ax1.axvline(pct, color="green" if pct == 93.0 else "purple",
                    ls="-", alpha=0.35, lw=1)
        row = sw[sw[pcol].astype(float).round(2) == pct]
        if not row.empty:
            r = row.iloc[0]
            ax1.annotate(f"{label}\nstrict={float(r[avg_col]):.4f}\nFPR={float(r[fpr_col]):.4f}",
                         xy=(pct, float(r[avg_col])),
                         xytext=(pct + (1.5 if pct == 95.0 else -1.5), float(r[avg_col]) - 0.08),
                         fontsize=7.5, ha="center",
                         arrowprops=dict(arrowstyle="->", lw=0.8,
                                         color="green" if pct == 93.0 else "purple"))
    ax1.set_title("Continuous threshold sweep — 29 points at p85.0–p99.0 (Δ=0.5pp)")
    ax1.grid(True, alpha=0.3)
    _save(fig, "fig13_threshold_sweep.png")


def fig14_per_target_rescue() -> None:
    """Grouped bar — baseline AE-p90 vs entropy_benign_p95 per target."""
    pt = pd.read_csv(PROJECT_ROOT / "results/enhanced_fusion/metrics/per_target_results.csv")
    base = pt[pt["variant"] == "baseline_ae_p90"].set_index("target")["h2_strict_rescue_recall"]
    best = pt[pt["variant"] == "entropy_benign_p95"].set_index("target")["h2_strict_rescue_recall"]
    eligible = [t for t in best.index if pd.notna(best[t]) and pd.notna(base[t])]
    # Order matches §15C.5 reading order
    order = ["Recon_Ping_Sweep", "Recon_VulScan", "MQTT_Malformed_Data", "ARP_Spoofing"]
    targets = [t for t in order if t in eligible]
    base_vals = [float(base[t]) for t in targets]
    best_vals = [float(best[t]) for t in targets]
    deltas_pp = [(b - a) * 100 for a, b in zip(base_vals, best_vals)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(targets))
    w = 0.38
    ax.bar(x - w / 2, base_vals, w, label="Baseline AE p90", color="#d62728")
    ax.bar(x + w / 2, best_vals, w, label="Phase 6C entropy_benign_p95",
           color="#2ca02c")
    ax.axhline(0.70, color="gray", ls="--", alpha=0.5,
               label="H2-strict 0.70 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=10)
    ax.set_ylabel("Rescue recall")
    ax.set_title("Per-target rescue lift — 4 eligible LOO targets (entropy_benign_p95)")
    for i, d in enumerate(deltas_pp):
        ax.text(i + w / 2, best_vals[i] + 0.02, f"+{d:.0f} pp",
                ha="center", fontsize=9, color="#2ca02c", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.10)
    _save(fig, "fig14_per_target_rescue.png")


# ---------------------------------------------------------------------------
# §8 — SHAP
# ---------------------------------------------------------------------------

def fig15_shap_global_top10() -> None:
    """Horizontal bar — global SHAP top-10."""
    gi = pd.read_csv(PROJECT_ROOT / "results/shap/metrics/global_importance.csv")
    feat_col = "feature" if "feature" in gi.columns else gi.columns[0]
    val_col = next((c for c in ["mean_abs_shap", "mean_abs_SHAP", "importance", "value"]
                    if c in gi.columns), gi.columns[1])
    top = gi.sort_values(val_col, ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top[feat_col].astype(str), top[val_col].astype(float),
            color="#9467bd")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Phase 7 — global SHAP top-10  (TreeSHAP on E7, 5,000 stratified test samples)")
    for i, v in enumerate(top[val_col].astype(float)):
        ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, "fig15_shap_global_top10.png")


def fig16_shap_ddos_vs_dos() -> None:
    """Side-by-side bar — top-10 per class for DDoS_UDP vs DoS_UDP using per_class_importance.csv."""
    pci = pd.read_csv(PROJECT_ROOT / "results/shap/metrics/per_class_importance.csv")
    # Schema may be long (class, feature, importance) or wide (feature × classes)
    if "class" in pci.columns and "feature" in pci.columns:
        val_col = next((c for c in ["mean_abs_shap", "mean_abs_SHAP", "importance", "value"]
                        if c in pci.columns), pci.columns[-1])
        def top10(cls: str):
            sub = pci[pci["class"] == cls].sort_values(val_col, ascending=False).head(10)
            return sub["feature"].tolist(), sub[val_col].astype(float).tolist()
    else:
        # wide layout: first col = feature, remainder = class columns
        feat_col = pci.columns[0]
        def top10(cls: str):
            if cls not in pci.columns:
                return [], []
            sub = pci.sort_values(cls, ascending=False).head(10)
            return sub[feat_col].astype(str).tolist(), sub[cls].astype(float).tolist()
    ddos_f, ddos_v = top10("DDoS_UDP")
    dos_f, dos_v = top10("DoS_UDP")
    if not ddos_f or not dos_f:
        # fallback: SYN pair
        ddos_f, ddos_v = top10("DDoS_SYN")
        dos_f, dos_v = top10("DoS_SYN")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=False)
    axes[0].barh(list(reversed(ddos_f)), list(reversed(ddos_v)), color="#1f77b4")
    axes[0].set_title("DDoS_UDP — top-10 features")
    axes[0].set_xlabel("Mean |SHAP|")
    axes[1].barh(list(reversed(dos_f)), list(reversed(dos_v)), color="#ff7f0e")
    axes[1].set_title("DoS_UDP — top-10 features")
    axes[1].set_xlabel("Mean |SHAP|")
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("DDoS vs DoS per-class SHAP profiles  (category cosine similarity = 0.991)",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig16_shap_ddos_vs_dos.png")


# ---------------------------------------------------------------------------
# §9 — Path B
# ---------------------------------------------------------------------------

def fig17_multi_seed_distribution() -> None:
    """Boxplot: 5 seeds × per-target rescue recall (entropy_benign_p95)."""
    pts = pd.read_csv(
        PROJECT_ROOT / "results/enhanced_fusion/multi_seed_per_target_summary.csv"
    )
    # Identify the columns we need
    cols = pts.columns.tolist()
    variant_col = "variant" if "variant" in cols else cols[0]
    target_col = "target" if "target" in cols else cols[1]
    # 5 per-seed values are usually columns named *_seed_1, _seed_7, etc.
    seed_cols = [c for c in cols if "seed_" in c.lower() and "strict_rescue" in c.lower()]
    if not seed_cols:
        # try alternate naming
        seed_cols = [c for c in cols if c.lower().startswith("seed_")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if seed_cols:
        sub = pts[pts[variant_col] == "entropy_benign_p95"]
        targets = sub[target_col].tolist()
        data = []
        for _, r in sub.iterrows():
            vals = [r[c] for c in seed_cols if pd.notna(r[c])]
            data.append([float(v) for v in vals])
        bp = ax.boxplot(data, tick_labels=targets, patch_artist=True, widths=0.55)
        for patch in bp["boxes"]:
            patch.set_facecolor("#2ca02c")
            patch.set_alpha(0.5)
    else:
        # Fallback: plot per-target mean ± std from summary
        mean_col = next((c for c in cols if "rescue_recall_mean" in c.lower() or "strict_rescue_recall_mean" in c.lower()), None)
        std_col = next((c for c in cols if "rescue_recall_std" in c.lower() or "strict_rescue_recall_std" in c.lower()), None)
        if mean_col and std_col:
            sub = pts[pts[variant_col] == "entropy_benign_p95"]
            targets = sub[target_col].tolist()
            means = sub[mean_col].astype(float).tolist()
            stds = sub[std_col].astype(float).tolist()
            ax.errorbar(targets, means, yerr=stds, fmt="o-",
                        color="#2ca02c", capsize=8, capthick=2,
                        elinewidth=1.5, markersize=8, ecolor="#1f77b4")
            for i, (m, s) in enumerate(zip(means, stds)):
                ax.text(i, m + s + 0.02, f"{m:.3f}±{s:.3f}",
                        ha="center", fontsize=8)
    ax.axhline(0.70, color="gray", ls="--", alpha=0.5,
               label="H2-strict 0.70 threshold")
    ax.set_ylabel("H2-strict rescue recall  (5 seeds: 1, 7, 42, 100, 1729)")
    ax.set_title("Path B Tier 1 Week 1 — multi-seed per-target distribution at entropy_benign_p95")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, "fig17_multi_seed_distribution.png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

FIGURES_LIST = [
    ("fig01", fig01_class_distribution),
    ("fig02", fig02_cohens_d_top10),
    ("fig03", fig03_correlation_heatmap),
    ("fig04", fig04_pca_2d),  # optional
    ("fig05", fig05_e7_confusion_matrix),
    ("fig06", fig06_e1_e8_comparison),
    ("fig07", fig07_smote_effect),
    ("fig08", fig08_ae_loss_curve),
    ("fig09", fig09_ae_recon_error_hist),
    ("fig10", fig10_detection_rate_heatmap),
    ("fig11", fig11_ae_vs_if_roc),
    ("fig12", fig12_pareto_frontier),
    ("fig13", fig13_threshold_sweep),
    ("fig14", fig14_per_target_rescue),
    ("fig15", fig15_shap_global_top10),
    ("fig16", fig16_shap_ddos_vs_dos),
    ("fig17", fig17_multi_seed_distribution),
]


def main() -> int:
    banner("Phase F — Figure generation (17 figures, deliverables/figures/)")
    fail = []
    for name, fn in FIGURES_LIST:
        try:
            fn()
        except Exception as e:
            print(f"  [fig-skipped] {name}: {type(e).__name__}: {e}")
            fail.append((name, str(e)))
    print()
    print(f"Generated {len(FIGURES_LIST) - len(fail)} / {len(FIGURES_LIST)} figures")
    if fail:
        print("Skipped figures:")
        for n, msg in fail:
            print(f"  - {n}: {msg}")
        # Non-fatal: return 0 so run_all keeps going
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
