"""
Verify thesis duplicate counts & investigate the Riyadi/Akkal 5,119 figure.
Writes results to stdout; does NOT modify any existing files.
"""
from __future__ import annotations
import glob, os, re, sys
import numpy as np
import pandas as pd
from pathlib import Path

TRAIN_DIR = "./data/train/"
TEST_DIR  = "./data/test/"

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


def filename_to_label(filename: str) -> str:
    name = os.path.basename(filename)
    name = name.replace("_train.pcap.csv", "").replace("_test.pcap.csv", "")
    name = re.sub(r"(\d+)$", "", name)
    mapping = {
        "ARP_Spoofing": "ARP_Spoofing",
        "Benign": "Benign",
        "MQTT-DDoS-Connect_Flood": "MQTT_DDoS_Connect_Flood",
        "MQTT-DDoS-Publish_Flood": "MQTT_DDoS_Publish_Flood",
        "MQTT-DoS-Connect_Flood": "MQTT_DoS_Connect_Flood",
        "MQTT-DoS-Publish_Flood": "MQTT_DoS_Publish_Flood",
        "MQTT-Malformed_Data": "MQTT_Malformed_Data",
        "Recon-OS_Scan": "Recon_OS_Scan",
        "Recon-Ping_Sweep": "Recon_Ping_Sweep",
        "Recon-Port_Scan": "Recon_Port_Scan",
        "Recon-VulScan": "Recon_VulScan",
        "TCP_IP-DDoS-ICMP": "DDoS_ICMP",
        "TCP_IP-DDoS-SYN": "DDoS_SYN",
        "TCP_IP-DDoS-TCP": "DDoS_TCP",
        "TCP_IP-DDoS-UDP": "DDoS_UDP",
        "TCP_IP-DoS-ICMP": "DoS_ICMP",
        "TCP_IP-DoS-SYN": "DoS_SYN",
        "TCP_IP-DoS-TCP": "DoS_TCP",
        "TCP_IP-DoS-UDP": "DoS_UDP",
    }
    return mapping.get(name, name)


def load_split(directory: str, split_name: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    dtype_map = {f: np.float32 for f in FEATURES}
    frames = []
    for path in csv_files:
        label = filename_to_label(path)
        df = pd.read_csv(path, dtype=dtype_map, engine="c", low_memory=False)
        df = df[[c for c in FEATURES if c in df.columns]].copy()
        df["label"] = label
        frames.append(df)
    full = pd.concat(frames, ignore_index=True, copy=False)
    print(f"[{split_name}] loaded {len(full):,} rows from {len(csv_files)} files")
    return full


print("=" * 72)
print("TASK 1: REPRODUCE THESIS DUPLICATE COUNTS")
print("=" * 72)

df_train = load_split(TRAIN_DIR, "train")
df_test  = load_split(TEST_DIR,  "test")

n_train_raw = len(df_train)
n_test_raw  = len(df_test)

# ---- 1a: Exact duplicates over ALL columns (45 features + label) ----
# This matches the EDA code: df.duplicated() with no subset
dup_train_all = df_train.duplicated().sum()
dup_test_all  = df_test.duplicated().sum()
pct_train_all = 100.0 * dup_train_all / n_train_raw
pct_test_all  = 100.0 * dup_test_all / n_test_raw

print(f"\n--- Duplicates: ALL columns (45 features + label) ---")
print(f"Train: {n_train_raw:,} raw → {dup_train_all:,} duplicates → {pct_train_all:.2f}%")
print(f"Test:  {n_test_raw:,}  raw → {dup_test_all:,}  duplicates → {pct_test_all:.2f}%")
print(f"Train after dedup: {n_train_raw - dup_train_all:,}")
print(f"Test  after dedup: {n_test_raw  - dup_test_all:,}")

# ---- 1b: Exact duplicates over FEATURES only (no label) ----
dup_train_feat = df_train.duplicated(subset=FEATURES).sum()
dup_test_feat  = df_test.duplicated(subset=FEATURES).sum()
pct_train_feat = 100.0 * dup_train_feat / n_train_raw
pct_test_feat  = 100.0 * dup_test_feat / n_test_raw

print(f"\n--- Duplicates: FEATURES only (45 cols, no label) ---")
print(f"Train: {dup_train_feat:,} duplicates ({pct_train_feat:.2f}%)")
print(f"Test:  {dup_test_feat:,}  duplicates ({pct_test_feat:.2f}%)")

# ---- 1c: NaN / Inf counts ----
print(f"\n--- NaN/Inf row counts ---")
for name, df in [("Train", df_train), ("Test", df_test)]:
    feat_cols = [c for c in FEATURES if c in df.columns]
    nan_rows = df[feat_cols].isna().any(axis=1).sum()
    inf_rows = np.isinf(df[feat_cols].select_dtypes(include=[np.number]).to_numpy()).any(axis=1).sum()
    print(f"{name}: NaN rows = {nan_rows:,}, Inf rows = {inf_rows:,}")

print()
print("=" * 72)
print("TASK 2: HUNT FOR THE 5,119 NUMBER")
print("=" * 72)

# ---- 2a: Cross-split duplicates (rows in BOTH train and test) ----
# Check on features only (label may differ between splits for same flow)
train_feat_only = df_train[FEATURES]
test_feat_only  = df_test[FEATURES]

# Use hashing for efficiency on large datasets
train_hashes = pd.util.hash_pandas_object(train_feat_only, index=False)
test_hashes  = pd.util.hash_pandas_object(test_feat_only, index=False)

train_hash_set = set(train_hashes.unique())
test_hash_set  = set(test_hashes.unique())
cross_overlap_hashes = train_hash_set & test_hash_set

# Count test rows whose feature-hash appears in train
cross_dup_test = test_hashes.isin(cross_overlap_hashes).sum()
cross_dup_train = train_hashes.isin(cross_overlap_hashes).sum()

print(f"\n--- 2a: Cross-split duplicates (features only) ---")
print(f"Unique feature-hashes in train: {len(train_hash_set):,}")
print(f"Unique feature-hashes in test:  {len(test_hash_set):,}")
print(f"Feature-hashes in BOTH splits:  {len(cross_overlap_hashes):,}")
print(f"Train rows with feature-hash also in test: {cross_dup_train:,}")
print(f"Test rows with feature-hash also in train: {cross_dup_test:,}")

# Also check with label included
train_with_label = df_train[FEATURES + ["label"]]
test_with_label  = df_test[FEATURES + ["label"]]
train_hash_wl = pd.util.hash_pandas_object(train_with_label, index=False)
test_hash_wl  = pd.util.hash_pandas_object(test_with_label, index=False)
cross_wl = set(train_hash_wl.unique()) & set(test_hash_wl.unique())
cross_dup_test_wl = test_hash_wl.isin(cross_wl).sum()
print(f"With label: feature+label hashes in BOTH: {len(cross_wl):,}")
print(f"Test rows with feature+label hash also in train: {cross_dup_test_wl:,}")

# ---- 2b: Per-class duplicate counts ----
print(f"\n--- 2b: Per-class duplicate counts (train, all cols) ---")
per_class_dups = []
for label, grp in df_train.groupby("label"):
    n_grp = len(grp)
    n_dup = grp.duplicated().sum()
    per_class_dups.append((label, n_grp, n_dup, 100.0 * n_dup / n_grp if n_grp else 0))
per_class_dups.sort(key=lambda x: x[2])
print(f"{'Label':<30s} {'Total':>10s} {'Dups':>10s} {'%':>8s}")
for label, total, dups, pct in per_class_dups:
    print(f"{label:<30s} {total:>10,} {dups:>10,} {pct:>7.2f}%")

# ---- 2c: Per-FILE duplicate counts (individual CSVs) ----
print(f"\n--- 2c: Per-file duplicate counts (train) ---")
per_file_dups = []
for path in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv"))):
    fname = os.path.basename(path)
    df_f = pd.read_csv(path, dtype={f: np.float32 for f in FEATURES}, engine="c", low_memory=False)
    df_f = df_f[[c for c in FEATURES if c in df_f.columns]]
    n = len(df_f)
    d = df_f.duplicated().sum()
    per_file_dups.append((fname, n, d))

per_file_dups.sort(key=lambda x: x[2])
print(f"{'File':<55s} {'Rows':>8s} {'Dups':>8s}")
for fname, rows, dups in per_file_dups:
    print(f"{fname:<55s} {rows:>8,} {dups:>8,}")

# Look for files where dups ≈ 5119
close_to_5119 = [(f, r, d) for f, r, d in per_file_dups if abs(d - 5119) < 500]
if close_to_5119:
    print(f"\n  ** Files with dups near 5,119: {close_to_5119}")

# ---- 2d: NaN row count (exact match to 5,119?) ----
print(f"\n--- 2d: NaN/Inf diagnostics ---")
for name, df in [("Train", df_train), ("Test", df_test)]:
    feat_cols = [c for c in FEATURES if c in df.columns]
    nan_per_row = df[feat_cols].isna().sum(axis=1)
    inf_per_row = np.isinf(df[feat_cols].select_dtypes(include=[np.number]).to_numpy()).sum(axis=1)
    total_problem = ((nan_per_row > 0) | (inf_per_row > 0)).sum()
    print(f"{name}: rows with any NaN or Inf = {total_problem:,}")

# Combined
combined_feats = pd.concat([df_train[FEATURES], df_test[FEATURES]], ignore_index=True)
nan_combined = combined_feats.isna().any(axis=1).sum()
inf_combined = np.isinf(combined_feats.select_dtypes(include=[np.number]).to_numpy()).any(axis=1).sum()
print(f"Combined: NaN rows = {nan_combined:,}, Inf rows = {inf_combined:,}")
print(f"Combined: NaN+Inf rows = {((combined_feats.isna().any(axis=1)) | (pd.DataFrame(np.isinf(combined_feats.select_dtypes(include=[np.number]).to_numpy()), columns=combined_feats.select_dtypes(include=[np.number]).columns).any(axis=1))).sum():,}")

# ---- 2e: DDoS-only subset duplicates (Akkal scope) ----
print(f"\n--- 2e: DDoS-only subset duplicates ---")
ddos_train = df_train[df_train["label"].str.startswith("DDoS")]
ddos_test  = df_test[df_test["label"].str.startswith("DDoS")]
ddos_all   = pd.concat([ddos_train, ddos_test], ignore_index=True)
ddos_dup   = ddos_all.duplicated().sum()
ddos_dup_feat = ddos_all.duplicated(subset=FEATURES).sum()
print(f"DDoS-only rows (train+test): {len(ddos_all):,}")
print(f"DDoS-only duplicates (all cols): {ddos_dup:,}")
print(f"DDoS-only duplicates (features only): {ddos_dup_feat:,}")

# ---- 2f: Unique duplicate count (how many unique rows appear >1 time) ----
print(f"\n--- 2f: Unique rows that are duplicated ---")
for name, df in [("Train", df_train), ("Test", df_test)]:
    mask = df.duplicated(keep=False)
    n_involved = mask.sum()
    n_unique_patterns = df[mask].drop_duplicates().shape[0]
    print(f"{name}: {n_involved:,} rows involved in duplication "
          f"({n_unique_patterns:,} unique patterns repeated)")

print()
print("=" * 72)
print("TASK 4: POOLED DUPLICATE RATE (NAEEM COMPARISON)")
print("=" * 72)

df_pooled = pd.concat([df_train, df_test], ignore_index=True)
n_pooled = len(df_pooled)
dup_pooled = df_pooled.duplicated().sum()
pct_pooled = 100.0 * dup_pooled / n_pooled
n_after_pooled_dedup = n_pooled - dup_pooled

print(f"Pooled raw rows:    {n_pooled:,}")
print(f"Pooled duplicates:  {dup_pooled:,} ({pct_pooled:.2f}%)")
print(f"After pooled dedup: {n_after_pooled_dedup:,}")
print(f"Row reduction:      {pct_pooled:.2f}%")
print(f"(Naeem reports ~55% row reduction)")

# Also check features-only pooled
dup_pooled_feat = df_pooled.duplicated(subset=FEATURES).sum()
pct_pooled_feat = 100.0 * dup_pooled_feat / n_pooled
print(f"\nPooled duplicates (features-only): {dup_pooled_feat:,} ({pct_pooled_feat:.2f}%)")

print("\n" + "=" * 72)
print("DONE — all counts verified from raw CSVs")
print("=" * 72)
