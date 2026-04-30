#!/usr/bin/env python3
"""
Path B Week 2B — Task 3: SHAP Background Sensitivity Check
==========================================================

Closes senior review item §1.2: Phase 7 used a 500-sample uniform-random
background drawn from a *test-disjoint* slice of `X_test`. The senior review
accepted the test-side defense (TreeSHAP `feature_perturbation="interventional"`
is invariant to background source for i.i.d.-similar data) but suggested
empirical verification.

This script reruns TreeSHAP on the **same** 5,000-sample explained set
(`X_shap_subset.npy`) using the **same** sampling protocol as Phase 7
(`shap_analysis.py:280` — `np.random.default_rng(RANDOM_STATE + 1)`,
uniform `.choice(..., size=500, replace=False)`) — *only the source pool
changes from X_test (test-disjoint) to X_train*. This is the apples-to-apples
comparison: same N, same seed, same selector, same explainer arguments —
only the marginal distribution of the background shifts.

It then compares the new SHAP attributions against the Phase 7 baseline
(`results/shap/shap_values/shap_values.npy`):

  - **Global Kendall τ** between the two full 44-feature rank lists derived
    from `mean |shap|` (and a separately-reported τ over the top-10 union).
  - **Per-class top-5 Jaccard** for all 19 classes (how many of the original
    Phase 7 per-class top-5 features remain in the new top-5).
  - **Category cosine matrix** — DDoS↔DoS and the full 5×5 attack-category
    cosine table; expects DDoS↔DoS to reproduce 0.991 within ±0.01.

Decision rule (Kendall τ over top-10 union):
  τ ≥ 0.9     → BULLETPROOF  (test-side defense empirically confirmed)
  0.7 ≤ τ < 0.9 → DEFENSIBLE (write a paragraph in §16.7B)
  τ < 0.7     → RERUN        (rerun Phase 7 with train-drawn background)

This task does NOT retrain anything; only TreeSHAP is recomputed with a
different background. Phase 7's saved attributions are NOT modified.

Inputs (read-only):
  preprocessed/full_features/X_train.npy
  results/supervised/models/E7_xgb_full_original.pkl
  results/shap/shap_values/X_shap_subset.npy   (FIXED explained set; 5000 rows)
  results/shap/shap_values/shap_values.npy     (Phase 7 baseline; (19, 5000, 44))

Outputs:
  results/shap/sensitivity/
    comparison.csv               # summary metrics (1 row)
    global_top10_ranks.csv       # side-by-side rank table over top-10 union
    per_class_jaccard.csv        # 19 rows: class, top5_old, top5_new, jaccard
    category_cosine.csv          # 5×5 cosine matrices for both backgrounds
    shap_values_train_bg.npy     # new (19, 5000, 44) attribution tensor
    top10_rank_comparison.png
    per_class_jaccard.png

Smoke gate:
    SMOKE=1 python notebooks/shap_sensitivity.py
    (X_shap[:100] + 50-sample background; should complete in <2 min)

Full run (~70 min on M4, no GPU):
    caffeinate -dimsu python -u notebooks/shap_sensitivity.py 2>&1 \
      | tee results/shap/sensitivity/run.log
"""

# %% SECTION 0 — Imports
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9


# %% SECTION 1 — Configuration

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = ROOT / "preprocessed"
SUPERVISED_DIR = ROOT / "results" / "supervised"
SHAP_DIR = ROOT / "results" / "shap"
OUTPUT_DIR = SHAP_DIR / "sensitivity"

MODEL_PATH = SUPERVISED_DIR / "models" / "E7_xgb_full_original.pkl"
X_TRAIN_PATH = PREPROCESSED_DIR / "full_features" / "X_train.npy"
X_SHAP_PATH = SHAP_DIR / "shap_values" / "X_shap_subset.npy"
SHAP_BASELINE_PATH = SHAP_DIR / "shap_values" / "shap_values.npy"

# Phase 7 SHAP parameters (matching shap_analysis.py:60-66) — kept identical
# so the only changing variable is the background source pool.
RANDOM_STATE = 42
SHAP_BACKGROUND_N = 500
SHAP_MODEL_OUTPUT = "raw"
SHAP_FEATURE_PERTURBATION = "interventional"

# Decision-rule cutoffs.
TAU_BULLETPROOF = 0.9
TAU_DEFENSIBLE = 0.7

# Class / feature names — copied verbatim from shap_analysis.py:69-98
# (these are hardcoded in Phase 7, not loaded from a file).
FEATURE_NAMES: List[str] = [
    "IAT", "Rate", "Header_Length", "Tot sum", "Min", "Max", "Covariance",
    "Variance", "Duration", "ack_count", "syn_count", "fin_count",
    "rst_count", "Srate", "AVG", "Std", "Tot size", "Number", "Magnitue",
    "Radius", "Weight", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "HTTP", "HTTPS", "DNS", "TCP",
    "DHCP", "ARP", "ICMP", "Protocol Type", "UDP", "IPv", "LLC",
    "Telnet", "SMTP", "SSH", "IRC", "IGMP",
]

CLASS_NAMES: List[str] = [
    "ARP_Spoofing", "Benign", "DDoS_ICMP", "DDoS_SYN", "DDoS_TCP",
    "DDoS_UDP", "DoS_ICMP", "DoS_SYN", "DoS_TCP", "DoS_UDP",
    "MQTT_DDoS_Connect_Flood", "MQTT_DDoS_Publish_Flood",
    "MQTT_DoS_Connect_Flood", "MQTT_DoS_Publish_Flood",
    "MQTT_Malformed_Data", "Recon_OS_Scan", "Recon_Ping_Sweep",
    "Recon_Port_Scan", "Recon_VulScan",
]

CATEGORIES: dict = {
    "DDoS":     ["DDoS_ICMP", "DDoS_SYN", "DDoS_TCP", "DDoS_UDP"],
    "DoS":      ["DoS_ICMP", "DoS_SYN", "DoS_TCP", "DoS_UDP"],
    "MQTT":     ["MQTT_DDoS_Connect_Flood", "MQTT_DDoS_Publish_Flood",
                 "MQTT_DoS_Connect_Flood", "MQTT_DoS_Publish_Flood",
                 "MQTT_Malformed_Data"],
    "Recon":    ["Recon_OS_Scan", "Recon_Ping_Sweep",
                 "Recon_Port_Scan", "Recon_VulScan"],
    "Spoofing": ["ARP_Spoofing"],
}

SMOKE = bool(int(os.environ.get("SMOKE", "0")))


def log(msg: str, t0: float | None = None) -> None:
    elapsed = f" [+{time.time() - t0:7.1f}s]" if t0 is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)


# %% SECTION 2 — SHAP normalization (copied from shap_analysis.py:310-348)


def normalize_shap_output(raw, n_classes: int, n_samples: int, n_features: int) -> np.ndarray:
    """Normalize SHAP output across versions to (n_classes, n_samples, n_features).

    Copied verbatim from shap_analysis.py to ensure both the Phase 7 baseline
    and the new train-bg run produce identically-shaped tensors before
    comparison.
    """
    if hasattr(raw, "values"):
        vals = raw.values
    else:
        vals = raw
    if isinstance(vals, list):
        return np.stack(vals, axis=0).astype(np.float32)
    if isinstance(vals, np.ndarray):
        if vals.ndim == 3:
            if vals.shape == (n_samples, n_features, n_classes):
                return np.transpose(vals, (2, 0, 1)).astype(np.float32)
            if vals.shape == (n_classes, n_samples, n_features):
                return vals.astype(np.float32)
            if vals.shape == (n_samples, n_classes, n_features):
                return np.transpose(vals, (1, 0, 2)).astype(np.float32)
        elif vals.ndim == 2:
            raise ValueError(f"Got 2D SHAP output {vals.shape}; expected multiclass.")
    raise ValueError(
        f"Unexpected SHAP output: type={type(vals)}, "
        f"shape={getattr(vals, 'shape', None)}"
    )


# %% SECTION 3 — Background sampling (apples-to-apples with Phase 7)


def build_train_background(X_train: np.ndarray, n_bg: int, seed_base: int) -> np.ndarray:
    """Build the 500-sample background using Phase 7's exact sampling protocol —
    `np.random.default_rng(seed_base + 1)`, uniform `.choice(..., replace=False)` —
    but drawn from `np.arange(len(X_train))` instead of the test-disjoint pool.
    Source pool is the *only* change from `shap_analysis.py:278-283`.
    """
    rng_bg = np.random.default_rng(seed_base + 1)  # 42+1 = 43, identical to Phase 7
    bg_indices = rng_bg.choice(np.arange(len(X_train)), size=n_bg, replace=False)
    X_bg = X_train[bg_indices].astype(np.float32, copy=False)
    return X_bg


# %% SECTION 4 — Comparison metrics


def global_rank_kendall_tau(
    shap_old: np.ndarray, shap_new: np.ndarray, feature_names: List[str],
) -> tuple[float, float, pd.DataFrame]:
    """Kendall's τ between global feature rankings derived from mean |shap|.

    Returns (tau_full44, tau_top10_union, side_by_side_table).
      - tau_full44: τ over all 44 feature ranks.
      - tau_top10_union: τ over the union of the two top-10 sets, computed
        on each list's rank within that union (the load-bearing decision metric).
    """
    # Global importance per feature: mean over (class, sample) of |shap|.
    imp_old = np.abs(shap_old).mean(axis=(0, 1))  # (44,)
    imp_new = np.abs(shap_new).mean(axis=(0, 1))  # (44,)

    rank_old = pd.Series(imp_old, index=feature_names).rank(ascending=False, method="min")
    rank_new = pd.Series(imp_new, index=feature_names).rank(ascending=False, method="min")

    # Full-44 τ.
    tau_full, _ = kendalltau(rank_old.values, rank_new.values)

    # Top-10 union τ — restricted to features that appear in either top-10.
    top10_old = set(rank_old.nsmallest(10).index)
    top10_new = set(rank_new.nsmallest(10).index)
    union = sorted(top10_old | top10_new)
    sub_old = rank_old.loc[union].values
    sub_new = rank_new.loc[union].values
    tau_top10, _ = kendalltau(sub_old, sub_new)

    side_by_side = pd.DataFrame({
        "feature": union,
        "rank_test_bg": rank_old.loc[union].astype(int).values,
        "rank_train_bg": rank_new.loc[union].astype(int).values,
        "in_top10_test_bg": [f in top10_old for f in union],
        "in_top10_train_bg": [f in top10_new for f in union],
        "mean_abs_shap_test_bg": imp_old[[feature_names.index(f) for f in union]],
        "mean_abs_shap_train_bg": imp_new[[feature_names.index(f) for f in union]],
    }).sort_values("rank_test_bg").reset_index(drop=True)

    return float(tau_full), float(tau_top10), side_by_side


def per_class_top5_jaccard(
    shap_old: np.ndarray, shap_new: np.ndarray,
    feature_names: List[str], class_names: List[str], k: int = 5,
) -> pd.DataFrame:
    """For each class, compute Jaccard between top-k features under the two
    backgrounds. Returns a 19-row dataframe."""
    rows = []
    for c_idx, name in enumerate(class_names):
        imp_old_c = np.abs(shap_old[c_idx]).mean(axis=0)  # (44,)
        imp_new_c = np.abs(shap_new[c_idx]).mean(axis=0)
        top_old_idx = np.argsort(imp_old_c)[::-1][:k]
        top_new_idx = np.argsort(imp_new_c)[::-1][:k]
        top_old = set(feature_names[i] for i in top_old_idx)
        top_new = set(feature_names[i] for i in top_new_idx)
        intersection = top_old & top_new
        union = top_old | top_new
        jaccard = len(intersection) / len(union) if union else float("nan")
        rows.append({
            "class": name,
            "top5_test_bg": ", ".join(feature_names[i] for i in top_old_idx),
            "top5_train_bg": ", ".join(feature_names[i] for i in top_new_idx),
            "jaccard": float(jaccard),
            "n_intersection": len(intersection),
        })
    return pd.DataFrame(rows)


def category_cosine_matrix(
    shap_arr: np.ndarray, class_names: List[str], categories: dict,
) -> pd.DataFrame:
    """Build a 5×5 cosine-similarity matrix of category-level mean |shap|
    profiles. Mirrors shap_analysis.py:720-764."""
    per_class_imp = np.abs(shap_arr).mean(axis=1)  # (n_classes, n_features)
    cat_rows = {}
    for cat_name, members in categories.items():
        idx = [class_names.index(m) for m in members]
        cat_rows[cat_name] = per_class_imp[idx].mean(axis=0)
    cat_df = pd.DataFrame(cat_rows).T  # (n_cats, n_features)
    sim = cosine_similarity(cat_df.values)
    return pd.DataFrame(sim, index=cat_df.index, columns=cat_df.index)


# %% SECTION 5 — Plots


def plot_top10_rank_comparison(side_df: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side bar chart of ranks under the two backgrounds for the
    union of top-10 features."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    n = len(side_df)
    x = np.arange(n)
    width = 0.4

    ax.bar(x - width / 2, side_df["rank_test_bg"].values,
           width=width, color="#1f77b4", edgecolor="#0d3a66",
           label="Phase 7 baseline (test-disjoint background)")
    ax.bar(x + width / 2, side_df["rank_train_bg"].values,
           width=width, color="#ff7f0e", edgecolor="#7a3a00",
           label="Train-drawn background (Week 2B)")

    for i, row in side_df.iterrows():
        ax.text(i - width / 2, row["rank_test_bg"] + 0.5, str(int(row["rank_test_bg"])),
                ha="center", va="bottom", fontsize=8, color="#0d3a66")
        ax.text(i + width / 2, row["rank_train_bg"] + 0.5, str(int(row["rank_train_bg"])),
                ha="center", va="bottom", fontsize=8, color="#7a3a00")

    ax.set_xticks(x)
    ax.set_xticklabels(side_df["feature"].tolist(), rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Global rank (lower is more important)")
    ax.set_title(
        "SHAP background sensitivity — global top-10 ranks under two backgrounds\n"
        f"Same 5,000-sample explained set; same seed + protocol; only background source pool changes (Path B Week 2B)"
    )
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log(f"  saved → {out_path}")


def plot_per_class_jaccard(jacc_df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of Jaccard top-5 stability per class."""
    df = jacc_df.sort_values("jaccard").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    bars = ax.barh(df["class"].values, df["jaccard"].values,
                   color="#2ca02c", edgecolor="#1a5d1a", height=0.7)
    mean_jacc = df["jaccard"].mean()
    ax.axvline(mean_jacc, color="#c0392b", linestyle="--", linewidth=1.4,
               label=f"mean = {mean_jacc:.3f}")
    ax.axvline(0.6, color="#666", linestyle=":", linewidth=1.0,
               label="0.60 (typical 'stable' threshold)")
    for bar, v in zip(bars, df["jaccard"].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_xlabel("Jaccard similarity (top-5 features under each background)")
    ax.set_title(
        "Per-class top-5 SHAP feature stability — train-drawn vs test-drawn background\n"
        f"Path B Week 2B — same 5,000-sample explained set, only background source changes"
    )
    ax.set_xlim(0, 1.08)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    log(f"  saved → {out_path}")


# %% SECTION 6 — Main


def decide(tau_top10: float) -> str:
    if tau_top10 >= TAU_BULLETPROOF:
        return "BULLETPROOF"
    if tau_top10 >= TAU_DEFENSIBLE:
        return "DEFENSIBLE"
    return "RERUN"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("=" * 78)
    log("Path B Week 2B — Task 3: SHAP Background Sensitivity Check")
    log(f"  SMOKE={int(SMOKE)}  output_dir={OUTPUT_DIR}")
    log("=" * 78)

    # ---- Load Phase 7 baseline + fixed explained set + model + train data ----
    log("Loading Phase 7 baseline SHAP tensor ...")
    shap_old = np.load(SHAP_BASELINE_PATH)  # (19, 5000, 44)
    log(f"  Phase 7 SHAP: shape={shap_old.shape}, dtype={shap_old.dtype}")
    if shap_old.shape != (len(CLASS_NAMES), 5000, len(FEATURE_NAMES)):
        raise RuntimeError(
            f"Phase 7 baseline shape unexpected: {shap_old.shape}; "
            f"expected ({len(CLASS_NAMES)}, 5000, {len(FEATURE_NAMES)})"
        )

    log("Loading X_shap (fixed explained set) ...")
    X_shap = np.load(X_SHAP_PATH).astype(np.float32, copy=False)
    log(f"  X_shap: shape={X_shap.shape}")
    if X_shap.shape != (5000, len(FEATURE_NAMES)):
        raise RuntimeError(f"X_shap shape unexpected: {X_shap.shape}")

    log(f"Loading XGBoost model: {MODEL_PATH.name}")
    model = joblib.load(MODEL_PATH)

    log("Loading X_train (only used to draw the background) ...")
    X_train = np.load(X_TRAIN_PATH).astype(np.float32, copy=False)
    log(f"  X_train: shape={X_train.shape}, dtype={X_train.dtype}")

    # ---- Apples-to-apples background: same N, same seed, same selector,
    # ----                              ONLY source pool changes (X_train).
    bg_n = 50 if SMOKE else SHAP_BACKGROUND_N
    log(f"Building train-drawn background  (N={bg_n}, seed_base={RANDOM_STATE}) ...")
    X_bg = build_train_background(X_train, n_bg=bg_n, seed_base=RANDOM_STATE)
    log(f"  background: shape={X_bg.shape}  (source pool: X_train, uniform random)")
    # Free X_train — we no longer need it (~635 MB).
    del X_train

    # ---- Smoke-mode shrinks the explained set, NOT the comparison logic ----
    if SMOKE:
        X_shap_used = X_shap[:100].copy()
        shap_old_used = shap_old[:, :100, :].copy()
        log(f"[SMOKE] explained set reduced to {X_shap_used.shape[0]} rows")
    else:
        X_shap_used = X_shap
        shap_old_used = shap_old
    del X_shap, shap_old

    # ---- TreeSHAP recompute with the new background ----
    log("Constructing TreeExplainer (interventional, model_output=raw) ...")
    explainer = shap.TreeExplainer(
        model,
        data=X_bg,
        feature_perturbation=SHAP_FEATURE_PERTURBATION,
        model_output=SHAP_MODEL_OUTPUT,
    )
    log(f"  expected_value (base): {np.array(explainer.expected_value).round(4)}")

    log(f"Computing SHAP values on X_shap (shape={X_shap_used.shape}) — slow step ...")
    t_shap = time.time()
    try:
        raw_shap = explainer(X_shap_used)
        api_used = "modern (explainer(X) → Explanation)"
    except Exception as exc:  # pragma: no cover  - defensive fallback
        log(f"  Modern API failed ({exc}); falling back to legacy shap_values()")
        raw_shap = explainer.shap_values(X_shap_used)
        api_used = "legacy (shap_values)"
    shap_new = normalize_shap_output(
        raw_shap, n_classes=len(CLASS_NAMES),
        n_samples=X_shap_used.shape[0], n_features=X_shap_used.shape[1],
    )
    log(
        f"  SHAP recompute done in {time.time() - t_shap:.1f}s "
        f"(api={api_used})  shape={shap_new.shape}",
        t0=t0,
    )

    # Save the new attribution tensor — useful for downstream re-analysis.
    if not SMOKE:
        np.save(OUTPUT_DIR / "shap_values_train_bg.npy", shap_new)
        log(f"  saved → {OUTPUT_DIR / 'shap_values_train_bg.npy'}")

    # ---- Comparisons ----
    log("Computing global Kendall τ (full 44 + top-10 union) ...")
    tau_full, tau_top10, side_df = global_rank_kendall_tau(
        shap_old_used, shap_new, FEATURE_NAMES,
    )
    log(f"  τ (full 44 features)    = {tau_full:.4f}")
    log(f"  τ (top-10 union)        = {tau_top10:.4f}  → decision: {decide(tau_top10)}")

    log("Computing per-class top-5 Jaccard for all 19 classes ...")
    jacc_df = per_class_top5_jaccard(shap_old_used, shap_new, FEATURE_NAMES, CLASS_NAMES)
    n_above_06 = int((jacc_df["jaccard"] >= 0.6).sum())
    log(
        f"  per-class Jaccard: mean={jacc_df['jaccard'].mean():.4f}  "
        f"std={jacc_df['jaccard'].std():.4f}  "
        f"≥0.6 in {n_above_06}/{len(jacc_df)} classes"
    )

    log("Computing category cosine matrices ...")
    cos_old = category_cosine_matrix(shap_old_used, CLASS_NAMES, CATEGORIES)
    cos_new = category_cosine_matrix(shap_new, CLASS_NAMES, CATEGORIES)
    ddos_dos_old = float(cos_old.loc["DDoS", "DoS"])
    ddos_dos_new = float(cos_new.loc["DDoS", "DoS"])
    delta = abs(ddos_dos_old - ddos_dos_new)
    log(f"  DDoS↔DoS cosine: test-bg={ddos_dos_old:.4f}  train-bg={ddos_dos_new:.4f}  |Δ|={delta:.4f}")

    # ---- Save outputs ----
    side_df.to_csv(OUTPUT_DIR / "global_top10_ranks.csv", index=False)
    jacc_df.to_csv(OUTPUT_DIR / "per_class_jaccard.csv", index=False)

    # Stack the two cosine matrices into one wide CSV for easy comparison.
    cos_old_named = cos_old.copy()
    cos_old_named.index = [f"test_bg::{i}" for i in cos_old_named.index]
    cos_new_named = cos_new.copy()
    cos_new_named.index = [f"train_bg::{i}" for i in cos_new_named.index]
    pd.concat([cos_old_named, cos_new_named]).to_csv(
        OUTPUT_DIR / "category_cosine.csv",
    )

    summary = pd.DataFrame([{
        "explained_set_n": X_shap_used.shape[0],
        "background_n": bg_n,
        "background_seed_base": RANDOM_STATE,
        "kendall_tau_full44": tau_full,
        "kendall_tau_top10_union": tau_top10,
        "decision": decide(tau_top10),
        "per_class_jaccard_mean": float(jacc_df["jaccard"].mean()),
        "per_class_jaccard_std": float(jacc_df["jaccard"].std()),
        "per_class_jaccard_min": float(jacc_df["jaccard"].min()),
        "n_classes_jaccard_ge_0_6": n_above_06,
        "ddos_dos_cosine_test_bg": ddos_dos_old,
        "ddos_dos_cosine_train_bg": ddos_dos_new,
        "ddos_dos_cosine_abs_delta": delta,
    }])
    summary.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    log(f"  saved → {OUTPUT_DIR / 'comparison.csv'}")
    log("\n" + summary.T.to_string(header=False))
    log("")

    if not SMOKE:
        plot_top10_rank_comparison(side_df, OUTPUT_DIR / "top10_rank_comparison.png")
        plot_per_class_jaccard(jacc_df, OUTPUT_DIR / "per_class_jaccard.png")
    else:
        log("[SMOKE] skipping plots")

    log(f"DONE in {time.time() - t0:.1f}s  (decision: {decide(tau_top10)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
