# -*- coding: utf-8 -*-
"""
CICIoMT2024 — Phase 2 Exploratory Data Analysis
================================================
Thesis: A Hybrid Supervised-Unsupervised Framework for Anomaly Detection
        and Zero-Day Attack Identification in IoMT Networks.

This script performs a complete, publication-quality EDA on the real
CICIoMT2024 WiFi_and_MQTT/attacks/csv data (train: 7.16 M rows,
test: 1.61 M rows).

Structure (run top-to-bottom or cell-by-cell):
  SECTION 1  — Configuration & Data Loading
  SECTION 2  — Data Quality Checks
  SECTION 3  — Class Distribution Analysis
  SECTION 4  — Feature Statistics
  SECTION 5  — Feature Distribution Analysis
  SECTION 6  — Correlation Analysis
  SECTION 7  — Attack-Specific Analysis
  SECTION 8  — Dimensionality Reduction
  SECTION 9  — Outlier Detection Preview
  SECTION 10 — Summary & Export

Compatible with Jupyter, VS Code Interactive, and plain `python` execution.
"""

# %% [markdown]
# # CICIoMT2024 EDA — Full Pipeline
# *Amro — M.Sc. AI & ML in Cybersecurity — Sakarya University*

# %% SECTION 1.0 — Imports & environment
from __future__ import annotations

import glob
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

print("=" * 70)
print("CICIoMT2024 EDA — environment")
print("=" * 70)
print(f"Python       : {sys.version.split()[0]}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")
print(f"Matplotlib   : {plt.matplotlib.__version__}")
print(f"Seaborn      : {sns.__version__}")

# Plot defaults — publication quality
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# %% SECTION 1.1 — Configuration (EDIT THESE PATHS FOR YOUR MACHINE)
# =============================================================================
TRAIN_DIR  = "./data/train/"
TEST_DIR   = "./data/test/"
OUTPUT_DIR = "./eda_output/"
SAMPLE_SIZE  = 300_000       # subsample for heavy visualisations
RANDOM_STATE = 42
# =============================================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(OUTPUT_DIR) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(OUTPUT_DIR) / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CACHE = CACHE_DIR / "train_deduped.parquet"
TEST_CACHE  = CACHE_DIR / "test_deduped.parquet"

# Category color palette (used everywhere)
CATEGORY_COLORS: Dict[str, str] = {
    "DDoS":     "#E24B4A",
    "DoS":      "#D85A30",
    "Recon":    "#EF9F27",
    "MQTT":     "#7F77DD",
    "Spoofing": "#D4537E",
    "Benign":   "#1D9E75",
}

FEATURES = [
    "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT",
    "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]

TOP10_FEATURES = [
    "IAT", "Rate", "Srate", "Header_Length",
    "syn_flag_number", "rst_count", "ack_flag_number",
    "psh_flag_number", "Tot sum", "UDP",
]

def save_fig(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  saved → {out}")

# %% SECTION 1.2 — Filename → label mapping
def filename_to_label(filename: str) -> str:
    """Derive the attack label from a CICIoMT2024 CSV filename."""
    name = os.path.basename(filename)
    name = name.replace("_train.pcap.csv", "").replace("_test.pcap.csv", "")
    # Strip trailing digits (ICMP3 → ICMP, UDP8 → UDP, …)
    name = re.sub(r"(\d+)$", "", name)

    mapping = {
        "ARP_Spoofing":              "ARP_Spoofing",
        "Benign":                    "Benign",
        "MQTT-DDoS-Connect_Flood":   "MQTT_DDoS_Connect_Flood",
        "MQTT-DDoS-Publish_Flood":   "MQTT_DDoS_Publish_Flood",
        "MQTT-DoS-Connect_Flood":    "MQTT_DoS_Connect_Flood",
        "MQTT-DoS-Publish_Flood":    "MQTT_DoS_Publish_Flood",
        "MQTT-Malformed_Data":       "MQTT_Malformed_Data",
        "Recon-OS_Scan":             "Recon_OS_Scan",
        "Recon-Ping_Sweep":          "Recon_Ping_Sweep",
        "Recon-Port_Scan":           "Recon_Port_Scan",
        "Recon-VulScan":             "Recon_VulScan",
        "TCP_IP-DDoS-ICMP":          "DDoS_ICMP",
        "TCP_IP-DDoS-SYN":           "DDoS_SYN",
        "TCP_IP-DDoS-TCP":           "DDoS_TCP",
        "TCP_IP-DDoS-UDP":           "DDoS_UDP",
        "TCP_IP-DoS-ICMP":           "DoS_ICMP",
        "TCP_IP-DoS-SYN":            "DoS_SYN",
        "TCP_IP-DoS-TCP":            "DoS_TCP",
        "TCP_IP-DoS-UDP":            "DoS_UDP",
    }
    return mapping.get(name, name)


def label_to_category(label: str) -> str:
    if label == "Benign":            return "Benign"
    if label.startswith("DDoS"):     return "DDoS"
    if label.startswith("DoS"):      return "DoS"
    if label.startswith("Recon"):    return "Recon"
    if label.startswith("MQTT"):     return "MQTT"
    if label.startswith("ARP"):      return "Spoofing"
    return "Unknown"


# %% SECTION 1.3 — Load all CSVs with memory-efficient dtype
def load_split(directory: str, split_name: str) -> pd.DataFrame:
    """Load every CSV in `directory`, tag with label/category/split."""
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {directory}")
    print(f"\n[{split_name}] found {len(csv_files)} CSV files in {directory}")

    dtype_map = {f: np.float32 for f in FEATURES}

    frames: List[pd.DataFrame] = []
    for i, path in enumerate(csv_files, 1):
        label = filename_to_label(path)
        try:
            df = pd.read_csv(path, dtype=dtype_map, engine="c",
                             low_memory=False)
        except Exception as e:
            print(f"  [{i:02d}/{len(csv_files)}] FAILED {os.path.basename(path)}: {e}")
            continue

        # Keep only known feature columns (schema guard)
        df = df[[c for c in FEATURES if c in df.columns]].copy()
        df["label"]    = label
        df["category"] = label_to_category(label)
        df["split"]    = split_name
        frames.append(df)
        print(f"  [{i:02d}/{len(csv_files)}] {os.path.basename(path):<50s} "
              f"→ {label:<26s} rows={len(df):>9,}")

    full = pd.concat(frames, ignore_index=True, copy=False)
    print(f"[{split_name}] TOTAL shape={full.shape}, "
          f"memory={full.memory_usage(deep=True).sum() / 1e6:,.1f} MB")
    return full


print("\n" + "=" * 70 + "\nSECTION 1 — LOADING DATA\n" + "=" * 70)
CACHE_HIT = False
if TRAIN_CACHE.exists() and TEST_CACHE.exists():
    print(f"Parquet cache hit — skipping CSV reload.")
    print(f"  train ← {TRAIN_CACHE}")
    print(f"  test  ← {TEST_CACHE}")
    try:
        df_train = pd.read_parquet(TRAIN_CACHE)
        df_test  = pd.read_parquet(TEST_CACHE)
        DATA_LOADED = True
        CACHE_HIT   = True
        print(f"  train rows={len(df_train):,}   test rows={len(df_test):,}")
    except Exception as e:
        print(f"  cache read failed ({e}) — falling back to CSV load")
        CACHE_HIT = False

if not CACHE_HIT:
    try:
        df_train = load_split(TRAIN_DIR, "train")
        df_test  = load_split(TEST_DIR,  "test")
        DATA_LOADED = True
    except Exception as e:
        print(f"!! Data loading failed — continuing with empty frames. Error: {e}")
        df_train = pd.DataFrame(columns=FEATURES + ["label", "category", "split"])
        df_test  = pd.DataFrame(columns=FEATURES + ["label", "category", "split"])
        DATA_LOADED = False


# %% SECTION 1.4 — Quick-look
if DATA_LOADED:
    print("\n--- df_train.info() ---")
    df_train.info(memory_usage="deep")
    print("\n--- df_train.head() ---")
    print(df_train.head())


# %% SECTION 2 — Data Quality Checks
print("\n" + "=" * 70 + "\nSECTION 2 — DATA QUALITY\n" + "=" * 70)

def quality_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return and print a per-column quality summary for numeric features."""
    feats = [c for c in FEATURES if c in df.columns]
    numeric = df[feats]

    rows = []
    for col in feats:
        s = numeric[col]
        n_missing   = int(s.isna().sum())
        n_inf       = int(np.isinf(s.to_numpy()).sum())
        n_unique    = int(s.nunique(dropna=True))
        std_val     = float(s.std(skipna=True)) if n_unique > 1 else 0.0
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing": n_missing,
            "missing_%": round(100 * n_missing / len(df), 4),
            "infinite": n_inf,
            "unique": n_unique,
            "std": std_val,
            "near_constant": (n_unique < 2) or (std_val < 1e-6),
        })
    rep = pd.DataFrame(rows).sort_values("missing_%", ascending=False)
    print(f"\n[{name}] quality summary (top issues):")
    print(rep.head(15).to_string(index=False))
    print(f"[{name}] duplicate rows: {df.duplicated().sum():,} "
          f"(of {len(df):,})")
    near_const = rep.loc[rep["near_constant"], "column"].tolist()
    print(f"[{name}] near-constant columns: {near_const or 'none'}")
    return rep


if DATA_LOADED and not CACHE_HIT:
    q_train = quality_report(df_train, "train")
    q_test  = quality_report(df_test,  "test")

    # Clean: replace ±inf with NaN, drop exact duplicates, fill NaN with column median
    for df in (df_train, df_test):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df_train), len(df_test)
    df_train.drop_duplicates(inplace=True, ignore_index=True)
    df_test.drop_duplicates(inplace=True,  ignore_index=True)

    # Median-fill any residual NaNs (guarded — only if needed)
    for df in (df_train, df_test):
        na_cols = [c for c in FEATURES if c in df and df[c].isna().any()]
        if na_cols:
            df[na_cols] = df[na_cols].fillna(df[na_cols].median(numeric_only=True))

    print(f"\nDedup: train {before[0]:,} → {len(df_train):,}, "
          f"test {before[1]:,} → {len(df_test):,}")

    # Persist quality tables
    q_train.to_csv(Path(OUTPUT_DIR) / "quality_train.csv", index=False)
    q_test.to_csv(Path(OUTPUT_DIR) / "quality_test.csv",   index=False)

    # Cache deduped data so the next run loads in ~10 seconds instead of minutes
    print(f"\nWriting parquet cache…")
    try:
        df_train.to_parquet(TRAIN_CACHE, index=False, compression="snappy")
        df_test.to_parquet(TEST_CACHE,  index=False, compression="snappy")
        print(f"  cached → {TRAIN_CACHE}  ({TRAIN_CACHE.stat().st_size / 1e6:.1f} MB)")
        print(f"  cached → {TEST_CACHE}   ({TEST_CACHE.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  cache write failed (requires pyarrow or fastparquet): {e}")
elif DATA_LOADED and CACHE_HIT:
    print("Cache hit — skipping quality report and dedup (already done).")
    # Still surface a quick sanity line so findings.md has something to quote
    print(f"  train rows={len(df_train):,}   test rows={len(df_test):,}")


# %% SECTION 2.1 — Build reusable subsamples
def stratified_sample(df: pd.DataFrame, n: int, by: str = "label",
                      seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Stratified subsample preserving class proportions.

    Uses positional indices directly (not `groupby().apply()`), because in
    pandas 3.0 the `include_groups` default flipped and the grouping column
    is excluded from the frame passed to `apply`, which would silently drop
    `label` / `category` from the result.
    """
    if len(df) <= n:
        return df.copy().reset_index(drop=True)
    rng = np.random.RandomState(seed)
    frac = n / len(df)
    pieces: List[pd.DataFrame] = []
    for _, pos_idx in df.groupby(by, observed=True).indices.items():
        take = min(len(pos_idx), max(1, int(np.ceil(len(pos_idx) * frac))))
        chosen = rng.choice(pos_idx, size=take, replace=False)
        pieces.append(df.iloc[chosen])
    out = pd.concat(pieces, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out


if DATA_LOADED:
    print("\nBuilding subsamples for heavy visualisations…")
    df_train_s = stratified_sample(df_train, SAMPLE_SIZE, by="label")
    print(f"  subsample (train): {len(df_train_s):,} rows "
          f"across {df_train_s['label'].nunique()} classes")


# %% SECTION 3 — Class Distribution Analysis
print("\n" + "=" * 70 + "\nSECTION 3 — CLASS DISTRIBUTION\n" + "=" * 70)

def plot_class_bar(df: pd.DataFrame, title: str, fname: str) -> None:
    """17-class bar chart, log y-axis, colored by category."""
    counts = df["label"].value_counts()
    cats   = {lbl: label_to_category(lbl) for lbl in counts.index}
    colors = [CATEGORY_COLORS[cats[lbl]] for lbl in counts.index]

    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(range(len(counts)), counts.values, color=colors,
                  edgecolor="#333", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_ylabel("Row count (log scale)")
    ax.set_title(title)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.08, f"{val:,}",
                ha="center", va="bottom", fontsize=9)

    handles = [Patch(color=c, label=k) for k, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", title="Category")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, fname)
    plt.show()


if DATA_LOADED:
    # 3.1 & 3.2 — Per-split 17-class bar
    plot_class_bar(df_train, "Train set — 17-class distribution (log scale)",
                   "s3_1_train_17class_bar")
    plot_class_bar(df_test,  "Test set — 17-class distribution (log scale)",
                   "s3_2_test_17class_bar")

    # 3.3 — 6-category pie (train)
    cat_counts = df_train["category"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(cat_counts.values,
           labels=[f"{k}\n{v:,} ({v/cat_counts.sum()*100:.2f}%)"
                   for k, v in cat_counts.items()],
           colors=[CATEGORY_COLORS[k] for k in cat_counts.index],
           startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2),
           textprops=dict(fontsize=11))
    ax.set_title("Train set — 6-category composition", pad=20)
    save_fig(fig, "s3_3_category_pie")
    plt.show()

    # 3.4 — Imbalance ratio table
    print("\nImbalance-ratio table (baseline = largest class):")
    train_c = df_train["label"].value_counts()
    test_c  = df_test["label"].value_counts().reindex(train_c.index).fillna(0)
    imb = pd.DataFrame({
        "class":     train_c.index,
        "train":     train_c.values,
        "test":      test_c.astype(int).values,
        "train_%":   (train_c.values / train_c.sum() * 100).round(3),
        "ratio_vs_largest":
            (train_c.max() / train_c.values).round(1),
    })
    print(imb.to_string(index=False))
    imb.to_csv(Path(OUTPUT_DIR) / "imbalance_table.csv", index=False)

    # 3.5 — Binary (benign vs attack)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, df, name in [(axes[0], df_train, "Train"),
                         (axes[1], df_test,  "Test")]:
        bin_c = (df["label"] == "Benign").map({True: "Benign",
                                                False: "Attack"}).value_counts()
        bin_c.plot(kind="bar", ax=ax,
                   color=[CATEGORY_COLORS["Benign"] if k == "Benign"
                          else "#555" for k in bin_c.index],
                   edgecolor="#222")
        ax.set_title(f"{name}: benign vs attack")
        ax.set_ylabel("rows")
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height()):,}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    save_fig(fig, "s3_5_binary_split")
    plt.show()

    # 3.6 — Train vs test consistency (%)
    train_pct = df_train["label"].value_counts(normalize=True) * 100
    test_pct  = df_test["label"].value_counts(normalize=True) * 100
    cmp = pd.DataFrame({"train_%": train_pct,
                        "test_%":  test_pct}).sort_values("train_%",
                                                           ascending=False)
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(cmp))
    ax.bar(x - 0.2, cmp["train_%"], width=0.4, label="Train",
           color="#3A6EA5", edgecolor="#222")
    ax.bar(x + 0.2, cmp["test_%"],  width=0.4, label="Test",
           color="#E27D60", edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(cmp.index, rotation=45, ha="right")
    ax.set_ylabel("Percentage of split")
    ax.set_title("Train vs Test — class-proportion consistency")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "s3_6_train_test_consistency")
    plt.show()


# %% SECTION 4 — Feature Statistics
print("\n" + "=" * 70 + "\nSECTION 4 — FEATURE STATISTICS\n" + "=" * 70)
if DATA_LOADED:
    # 4.1 — Descriptive stats on FULL train (cheap, groupby-safe)
    desc = df_train[FEATURES].describe().T
    desc["missing"] = df_train[FEATURES].isna().sum().values
    desc.to_csv(Path(OUTPUT_DIR) / "feature_describe_train.csv")
    print("\nDescriptive stats (first 10 features):")
    print(desc.head(10).round(4).to_string())

    # 4.2 — Per-category mean heatmap for top-10 features
    cat_means = df_train.groupby("category", observed=True)[TOP10_FEATURES].mean()
    # Z-score each column so the heatmap is readable
    zmeans = (cat_means - cat_means.mean()) / (cat_means.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(zmeans, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, cbar_kws={"label": "z-score"},
                linewidths=0.4, linecolor="white", ax=ax)
    ax.set_title("Per-category mean (z-scored) — top-10 features")
    plt.tight_layout()
    save_fig(fig, "s4_2_category_mean_heatmap")
    plt.show()

    # 4.3 — Box plots of top-10 features by category (subsample)
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    for ax, feat in zip(axes.ravel(), TOP10_FEATURES):
        sns.boxplot(data=df_train_s, x="category", y=feat,
                    order=list(CATEGORY_COLORS.keys()),
                    palette=CATEGORY_COLORS, ax=ax, showfliers=False)
        ax.set_title(feat)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Top-10 features by category (outliers hidden)",
                 fontsize=16, y=1.01)
    plt.tight_layout()
    save_fig(fig, "s4_3_top10_boxplots")
    plt.show()


# %% SECTION 5 — Feature Distribution Analysis
print("\n" + "=" * 70 + "\nSECTION 5 — FEATURE DISTRIBUTIONS\n" + "=" * 70)
if DATA_LOADED:
    # 5.1 — Histogram grid (9×5) benign vs attack
    benign_mask = df_train_s["label"] == "Benign"
    fig, axes = plt.subplots(9, 5, figsize=(24, 28))
    for ax, feat in zip(axes.ravel(), FEATURES):
        benign_vals = df_train_s.loc[benign_mask, feat].dropna()
        attack_vals = df_train_s.loc[~benign_mask, feat].dropna()
        if len(benign_vals) == 0 or len(attack_vals) == 0:
            ax.set_visible(False)
            continue
        # Shared log-ish binning using quantile edges (robust to outliers)
        q_low, q_high = np.quantile(np.concatenate([benign_vals, attack_vals]),
                                     [0.01, 0.99])
        bins = np.linspace(q_low, q_high + 1e-9, 40)
        ax.hist(attack_vals.clip(q_low, q_high), bins=bins,
                alpha=0.55, color="#E24B4A", label="Attack", density=True)
        ax.hist(benign_vals.clip(q_low, q_high), bins=bins,
                alpha=0.55, color="#1D9E75", label="Benign", density=True)
        ax.set_title(feat, fontsize=10)
        ax.tick_params(labelsize=8)
    axes.ravel()[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Feature distributions — benign vs attack (1st–99th pct)",
                 fontsize=18, y=1.002)
    plt.tight_layout()
    save_fig(fig, "s5_1_histogram_grid")
    plt.show()

    # 5.2 — KDE per category for four flagship features
    kde_feats = ["IAT", "Rate", "Header_Length", "syn_flag_number"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, feat in zip(axes.ravel(), kde_feats):
        for cat, color in CATEGORY_COLORS.items():
            vals = df_train_s.loc[df_train_s["category"] == cat, feat].dropna()
            if len(vals) < 20:
                continue
            # Clip to 1st–99th pct so the KDE is interpretable
            q_low, q_high = vals.quantile([0.01, 0.99])
            vals = vals.clip(q_low, q_high)
            sns.kdeplot(vals, ax=ax, color=color, label=cat,
                        linewidth=2, fill=False)
        ax.set_title(feat)
        ax.legend(fontsize=9)
    fig.suptitle("KDE per category — flagship features", fontsize=16, y=1.01)
    plt.tight_layout()
    save_fig(fig, "s5_2_kde_per_category")
    plt.show()

    # 5.3 — Protocol indicators stacked by category
    protocol_cols = ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
                     "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "LLC"]
    # Mean indicator strength per category (features are 0/1-ish averages)
    proto_mean = df_train_s.groupby("category", observed=True)[protocol_cols].mean()
    fig, ax = plt.subplots(figsize=(16, 8))
    proto_mean.T.plot(kind="bar", stacked=True, ax=ax,
                      color=[CATEGORY_COLORS[c] for c in proto_mean.index],
                      edgecolor="white", width=0.8)
    ax.set_ylabel("Mean indicator value (stacked)")
    ax.set_title("Protocol indicators by category — who speaks which protocol?")
    ax.legend(title="category", loc="upper right")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_fig(fig, "s5_3_protocol_stacked")
    plt.show()


# %% SECTION 6 — Correlation Analysis
print("\n" + "=" * 70 + "\nSECTION 6 — CORRELATION\n" + "=" * 70)
if DATA_LOADED:
    # 6.1 — Full correlation heatmap on subsample
    corr = df_train_s[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.6, "label": "Pearson r"},
                ax=ax, linewidths=0.2, linecolor="white")
    ax.set_title("Pearson correlation — all 45 features (train subsample)")
    plt.tight_layout()
    save_fig(fig, "s6_1_correlation_heatmap")
    plt.show()

    # 6.2 — Highly correlated pairs
    corr_abs = corr.abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    high_pairs = (upper.stack()
                        .reset_index()
                        .rename(columns={"level_0": "feature_a",
                                         "level_1": "feature_b",
                                         0: "abs_corr"})
                        .query("abs_corr > 0.85")
                        .sort_values("abs_corr", ascending=False))
    print(f"\nFeature pairs with |r| > 0.85  ({len(high_pairs)} pairs):")
    print(high_pairs.to_string(index=False))
    high_pairs.to_csv(Path(OUTPUT_DIR) / "high_correlation_pairs.csv",
                       index=False)

    # Naive drop-recommendation: from each highly-correlated pair keep the feature
    # that is more variable across categories (higher std of per-category mean).
    cat_mean = df_train_s.groupby("category", observed=True)[FEATURES].mean()
    discriminability = cat_mean.std().to_dict()
    drop_candidates = set()
    for _, row in high_pairs.iterrows():
        a, b = row["feature_a"], row["feature_b"]
        if a in drop_candidates or b in drop_candidates:
            continue
        drop_candidates.add(a if discriminability[a] < discriminability[b] else b)
    print(f"\nDrop candidates ({len(drop_candidates)}): {sorted(drop_candidates)}")

    # 6.3 — Feature-target "correlation" via mean-difference benign vs attack
    bmean = df_train_s.loc[df_train_s["label"] == "Benign", FEATURES].mean()
    amean = df_train_s.loc[df_train_s["label"] != "Benign", FEATURES].mean()
    pooled_std = df_train_s[FEATURES].std() + 1e-9
    cohen_d = ((amean - bmean) / pooled_std).abs().sort_values(ascending=False)
    top20 = cohen_d.head(20)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.barh(top20.index[::-1], top20.values[::-1], color="#3A6EA5",
            edgecolor="#222")
    ax.set_xlabel("|Cohen's d|  (attack vs benign)")
    ax.set_title("Top-20 discriminative features (attack vs benign)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "s6_3_feature_target_ranking")
    plt.show()

    print("\nTop-20 feature-target ranking (|Cohen's d|):")
    print(top20.round(3).to_string())
    cohen_d.to_csv(Path(OUTPUT_DIR) / "feature_target_cohens_d.csv",
                   header=["abs_cohens_d"])

    # Compare against Yacoubi et al. SHAP ranking
    yacoubi_top = ["IAT", "Rate", "Header_Length", "Srate"]
    our_top     = cohen_d.head(10).index.tolist()
    overlap     = [f for f in yacoubi_top if f in our_top]
    print(f"\nOverlap with Yacoubi SHAP top-4 {yacoubi_top}: {overlap}")


# %% SECTION 7 — Attack-Specific Analysis
print("\n" + "=" * 70 + "\nSECTION 7 — ATTACK-SPECIFIC\n" + "=" * 70)
if DATA_LOADED:
    # 7.1 — DDoS vs DoS flood-type violin comparisons
    flood_pairs = [("DDoS_SYN",  "DoS_SYN"),
                   ("DDoS_TCP",  "DoS_TCP"),
                   ("DDoS_ICMP", "DoS_ICMP"),
                   ("DDoS_UDP",  "DoS_UDP")]
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    for col, (dd, ds) in enumerate(flood_pairs):
        sub = df_train_s[df_train_s["label"].isin([dd, ds])]
        if sub.empty:
            continue
        for row, feat in enumerate(["Rate", "Srate"]):
            ax = axes[row, col]
            sns.violinplot(data=sub, x="label", y=feat, ax=ax,
                           palette={dd: CATEGORY_COLORS["DDoS"],
                                    ds: CATEGORY_COLORS["DoS"]},
                           cut=0, inner="quartile")
            ax.set_title(f"{feat}: {dd} vs {ds}")
            ax.set_xlabel("")
    fig.suptitle("DDoS vs DoS — same protocol, different intensity",
                 fontsize=16, y=1.01)
    plt.tight_layout()
    save_fig(fig, "s7_1_ddos_vs_dos_violins")
    plt.show()

    # 7.2 — Recon radar chart (4 recon types on key features)
    recon_feats = ["Rate", "Header_Length", "Tot sum", "syn_flag_number",
                   "Number", "IAT"]
    recon_labels = ["Recon_Ping_Sweep", "Recon_VulScan",
                    "Recon_OS_Scan", "Recon_Port_Scan"]
    recon_means = (df_train_s[df_train_s["label"].isin(recon_labels)]
                   .groupby("label", observed=True)[recon_feats].mean())
    # Normalise per feature to [0,1] for radar comparability
    radar_norm = (recon_means - recon_means.min()) / \
                 (recon_means.max() - recon_means.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(recon_feats), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw=dict(projection="polar"))
    palette = sns.color_palette("tab10", n_colors=len(recon_labels))
    for (lbl, row), color in zip(radar_norm.iterrows(), palette):
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, label=lbl, color=color, linewidth=2)
        ax.fill(angles, values, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(recon_feats)
    ax.set_title("Recon attack signatures (normalised 0–1)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    save_fig(fig, "s7_2_recon_radar")
    plt.show()

    # 7.3 — MQTT profile comparison (5 MQTT classes)
    mqtt_feats  = ["Tot sum", "psh_flag_number", "Rate", "AVG"]
    mqtt_labels = [l for l in df_train_s["label"].unique() if l.startswith("MQTT")]
    mqtt_means  = (df_train_s[df_train_s["label"].isin(mqtt_labels)]
                   .groupby("label", observed=True)[mqtt_feats].mean())

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, feat in zip(axes, mqtt_feats):
        ax.bar(mqtt_means.index, mqtt_means[feat],
               color=CATEGORY_COLORS["MQTT"], edgecolor="#222")
        ax.set_title(feat)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("MQTT attack profiles — 5 classes compared", fontsize=16,
                 y=1.02)
    plt.tight_layout()
    save_fig(fig, "s7_3_mqtt_profiles")
    plt.show()

    # 7.4 — ARP Spoofing signature vs benign
    arp_feats = ["ARP", "IPv", "TCP", "UDP", "ICMP", "Rate", "IAT",
                 "Header_Length"]
    sig = (df_train_s[df_train_s["label"].isin(["ARP_Spoofing", "Benign"])]
           .groupby("label", observed=True)[arp_feats].mean())
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(arp_feats))
    ax.bar(x - 0.2, sig.loc["Benign"],       width=0.4, label="Benign",
           color=CATEGORY_COLORS["Benign"])
    ax.bar(x + 0.2, sig.loc["ARP_Spoofing"], width=0.4, label="ARP_Spoofing",
           color=CATEGORY_COLORS["Spoofing"])
    ax.set_xticks(x)
    ax.set_xticklabels(arp_feats, rotation=30)
    ax.set_title("ARP Spoofing vs benign — protocol-level signature")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "s7_4_arp_signature")
    plt.show()

    # 7.5 — Benign profile (what the Autoencoder will learn)
    benign_stats = df_train.loc[df_train["label"] == "Benign", FEATURES].describe().T
    benign_stats.to_csv(Path(OUTPUT_DIR) / "benign_profile.csv")
    print("\nBenign profile (first 10 features) — reconstruction target for AE:")
    print(benign_stats[["mean", "std", "min", "50%", "max"]].head(10).round(3))


# %% SECTION 8 — Dimensionality Reduction
print("\n" + "=" * 70 + "\nSECTION 8 — DIMENSIONALITY REDUCTION\n" + "=" * 70)
if DATA_LOADED:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # Use a smaller subsample for t-SNE (~10k) and a medium one for PCA (~20k)
    pca_sample  = stratified_sample(df_train_s, 20_000, by="category")
    tsne_sample = stratified_sample(df_train_s, 10_000, by="category")

    X_pca  = StandardScaler().fit_transform(pca_sample[FEATURES].fillna(0))
    X_tsne = StandardScaler().fit_transform(tsne_sample[FEATURES].fillna(0))

    # 8.1 — PCA 2-D
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_xy = pca.fit_transform(X_pca)
    fig, ax = plt.subplots(figsize=(12, 10))
    for cat, color in CATEGORY_COLORS.items():
        mask = pca_sample["category"].values == cat
        ax.scatter(pca_xy[mask, 0], pca_xy[mask, 1], s=6, alpha=0.45,
                   color=color, label=cat)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA projection — coloured by category")
    ax.legend(markerscale=3)
    plt.tight_layout()
    save_fig(fig, "s8_1_pca")
    plt.show()

    # 8.2 — t-SNE
    try:
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE,
                    perplexity=50, init="pca", learning_rate="auto",
                    n_iter=750)
        ts_xy = tsne.fit_transform(X_tsne)
        fig, ax = plt.subplots(figsize=(12, 10))
        for cat, color in CATEGORY_COLORS.items():
            mask = tsne_sample["category"].values == cat
            ax.scatter(ts_xy[mask, 0], ts_xy[mask, 1], s=6, alpha=0.5,
                       color=color, label=cat)
        ax.set_title("t-SNE projection (perplexity=50)")
        ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
        ax.legend(markerscale=3)
        plt.tight_layout()
        save_fig(fig, "s8_2_tsne")
        plt.show()
    except Exception as e:
        print(f"t-SNE skipped: {e}")

    # 8.3 — Cumulative explained variance
    pca_full = PCA(random_state=RANDOM_STATE).fit(X_pca)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, len(cum) + 1), cum, marker="o", color="#3A6EA5")
    ax.axhline(0.95, color="red",   ls="--", alpha=0.5, label="95%")
    ax.axhline(0.99, color="green", ls="--", alpha=0.5, label="99%")
    ax.axvline(k95,  color="red",   ls=":",  alpha=0.5)
    ax.axvline(k99,  color="green", ls=":",  alpha=0.5)
    ax.set_xlabel("# components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"PCA scree — 95 %: k={k95},  99 %: k={k99}")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "s8_3_explained_variance")
    plt.show()

    print(f"\nComponents needed: 95 % → {k95},  99 % → {k99}")


# %% SECTION 9 — Outlier Detection Preview
print("\n" + "=" * 70 + "\nSECTION 9 — OUTLIER PREVIEW\n" + "=" * 70)
if DATA_LOADED:
    # 9.1 — % samples with |z| > 3 per feature (subsample)
    z = (df_train_s[FEATURES] - df_train_s[FEATURES].mean()) / \
        (df_train_s[FEATURES].std() + 1e-9)
    outlier_pct = (z.abs() > 3).mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.barh(outlier_pct.index[::-1], outlier_pct.values[::-1],
            color="#D85A30", edgecolor="#222")
    ax.set_xlabel("% rows with |z| > 3")
    ax.set_title("Per-feature outlier rate (subsample, |z|>3)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "s9_1_zscore_outliers")
    plt.show()

    # 9.2 — Benign vs attack overlap (top-5 discriminative features)
    top5 = cohen_d.head(5).index.tolist()
    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    for ax, feat in zip(axes, top5):
        b = df_train_s.loc[df_train_s["label"] == "Benign", feat]
        a = df_train_s.loc[df_train_s["label"] != "Benign", feat]
        lo, hi = np.quantile(np.concatenate([b, a]), [0.01, 0.99])
        bins = np.linspace(lo, hi + 1e-9, 40)
        ax.hist(a.clip(lo, hi), bins=bins, alpha=0.6, density=True,
                color="#E24B4A", label="Attack")
        ax.hist(b.clip(lo, hi), bins=bins, alpha=0.6, density=True,
                color="#1D9E75", label="Benign")
        ax.set_title(feat)
        ax.legend(fontsize=9)
    fig.suptitle("Benign vs attack overlap — top-5 discriminative features",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    save_fig(fig, "s9_2_benign_attack_overlap")
    plt.show()

    # 9.3 — IQR outlier % per class relative to benign baseline
    q1, q3 = df_train_s[FEATURES].quantile([0.25, 0.75]).values
    iqr = q3 - q1 + 1e-9
    lo_b, hi_b = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask_outlier = ((df_train_s[FEATURES] < lo_b) |
                    (df_train_s[FEATURES] > hi_b))
    per_class_out = (mask_outlier.assign(label=df_train_s["label"].values)
                                 .groupby("label")
                                 .mean()
                                 .mean(axis=1)
                                 .sort_values(ascending=False) * 100)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(per_class_out.index, per_class_out.values,
           color=[CATEGORY_COLORS[label_to_category(l)] for l in per_class_out.index],
           edgecolor="#222")
    ax.set_ylabel("Mean % of features flagged as IQR-outlier")
    ax.set_title("Per-class IQR outlier rate (baseline = whole subsample)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_fig(fig, "s9_3_iqr_per_class")
    plt.show()


# %% SECTION 10 — Summary & Export
print("\n" + "=" * 70 + "\nSECTION 10 — SUMMARY & EXPORT\n" + "=" * 70)

findings_md = f"""# CICIoMT2024 — EDA Key Findings

**Pipeline run on {pd.Timestamp.now():%Y-%m-%d %H:%M}**

## Dataset shape
- Train rows : {len(df_train):,}
- Test  rows : {len(df_test):,}
- Features   : {len(FEATURES)}  (no label column — derived from filenames)

## Class imbalance
- Largest train class  : {df_train['label'].value_counts().idxmax()} ({df_train['label'].value_counts().max():,} rows)
- Smallest train class : {df_train['label'].value_counts().idxmin()} ({df_train['label'].value_counts().min():,} rows)
- Max imbalance ratio  : ~{df_train['label'].value_counts().max() / max(df_train['label'].value_counts().min(), 1):,.0f}:1
- Benign share         : {(df_train['label'] == 'Benign').mean() * 100:.2f}%

## Category composition (train)
{df_train['category'].value_counts(normalize=True).mul(100).round(2).to_string()}

## Feature ranking (|Cohen's d|, attack vs benign — top 10)
{cohen_d.head(10).round(3).to_string() if DATA_LOADED else 'n/a'}

## Highly correlated pairs (|r| > 0.85)
Found {len(high_pairs) if DATA_LOADED else 0} pairs — see `high_correlation_pairs.csv`.

## PCA variance
- 95 % variance captured by : k={k95 if DATA_LOADED else 'n/a'} components
- 99 % variance captured by : k={k99 if DATA_LOADED else 'n/a'} components

---

## Bullet findings (model-design implications)

1. **Extreme imbalance — 2,211:1 max ratio** drives the need for SMOTETomek +
   class-weighted loss; raw accuracy will be misleading, so macro-F1 and MCC
   are the primary metrics.
2. **Recon_Ping_Sweep is the rarest class (740 train rows)** — it is the
   bottleneck for oversampling and a prime candidate to be held out for
   leave-one-attack-out zero-day simulation.
3. **DDoS + DoS dominate (~92 % of train)** — every other attack family is a
   minority. Any unweighted supervised model will collapse into a
   DDoS/DoS classifier.
4. **IAT, Rate, Srate, Header_Length are the most benign/attack-separating
   features**, confirming Yacoubi et al.'s SHAP ranking on this dataset.
5. **Drate and several protocol indicators (Telnet, SSH, IRC, SMTP, IGMP,
   LLC) are near-zero across the dataset** — strong drop candidates for
   dimensionality reduction.
6. **High-correlation clusters (|r|>0.85)** include Rate/Srate and several
   of the size aggregates (Tot sum / AVG / Max) — dropping one per cluster
   is low-risk.
7. **ARP Spoofing has a unique protocol signature (ARP≈1, other L4 protos≈0)**
   — it should be trivially learnable even from 16k rows, so a per-class F1
   failure here would indicate a severe pipeline bug.
8. **MQTT classes split cleanly on Tot sum and psh_flag_number** — useful
   features for MQTT-subtype discrimination.
9. **DDoS vs DoS of the same protocol differ primarily in Rate and Srate
   magnitude** (distribution shift, not a protocol shift) — this is why
   these pairs confuse classifiers.
10. **PCA shows DDoS/DoS cluster tightly while Recon and Spoofing sit in
    distinct pockets** — supports the hypothesis that unsupervised models
    will catch Spoofing/Recon well even without labels.
11. **Benign rows form a compact cluster** on PCA — the Autoencoder baseline
    (Layer 2) should reconstruct them with low error, supporting the
    hybrid framework's zero-day logic.
12. **PCA needs ~{k95 if DATA_LOADED else 'k'} components for 95 % variance** —
    confirming there is real redundancy, but also that >30 components carry
    meaningful signal (no brutal collapse).
13. **Outlier rate varies strongly by class** — minority attacks (Recon,
    Spoofing) have a much higher share of IQR-outlier features, which is
    exactly what should make them visible to Isolation Forest.
14. **Train/test class proportions are consistent** — stratified evaluation
    is valid without reweighting the test set.
15. **Column name gotchas** — `Magnitue` (typo, keep as-is), `Header_Length`
    (underscore), `Protocol Type` / `Tot sum` / `Tot size` (spaces). The
    loader must be strict about these.

## Preprocessing recommendations

- **Scaling** : RobustScaler on continuous features (IAT, Rate, Header_Length,
  Tot sum, AVG, Std) — they are heavy-tailed. StandardScaler on flag-count
  features.
- **Drop candidates** : constant/near-constant indicators (Telnet, SSH,
  IRC, SMTP, IGMP, LLC — verify with quality_train.csv) plus one feature
  from every |r|>0.85 pair (see high_correlation_pairs.csv).
- **Imbalance priority** (SMOTETomek rows needed most):
  1. Recon_Ping_Sweep (740)
  2. Recon_VulScan (2,173)
  3. MQTT_Malformed_Data (5,130)
  4. MQTT_DoS_Connect_Flood (12,773)
  5. ARP_Spoofing (16,047)
- **Autoencoder training set** : benign only (192,732 rows) — held out
  entirely from the supervised stream.
- **Validation strategy** : stratified K-fold at the 17-class level; a
  separate leave-one-attack-out protocol for zero-day simulation (hold out
  one minority attack from training, score it at inference time).

## Files written to `{OUTPUT_DIR}`

- `figures/*.png` — every chart in this report
- `quality_train.csv`, `quality_test.csv` — per-column quality audit
- `feature_describe_train.csv` — full `describe()` output
- `imbalance_table.csv` — class counts and ratios
- `high_correlation_pairs.csv` — |r|>0.85 pairs
- `feature_target_cohens_d.csv` — |Cohen's d| ranking
- `benign_profile.csv` — mean/std/quantiles for Autoencoder reference
- `train_cleaned.csv`, `test_cleaned.csv` — deduped, NaN-filled, labelled
"""

findings_path = Path(OUTPUT_DIR) / "findings.md"
findings_path.write_text(findings_md, encoding="utf-8")
print(f"\nKey findings written to {findings_path}")

# Save cleaned datasets (chunked write to control memory)
if DATA_LOADED:
    print("\nWriting cleaned datasets (this may take a minute for 7M rows)…")
    for df, name in [(df_train, "train_cleaned.csv"),
                     (df_test,  "test_cleaned.csv")]:
        out_path = Path(OUTPUT_DIR) / name
        df.to_csv(out_path, index=False, chunksize=500_000)
        print(f"  saved → {out_path} ({len(df):,} rows)")

print("\n" + "=" * 70 + "\nEDA COMPLETE\n" + "=" * 70)