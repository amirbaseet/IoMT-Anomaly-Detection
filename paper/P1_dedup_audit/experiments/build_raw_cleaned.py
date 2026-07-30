"""
C1 step 1 — build the NON-deduplicated ("raw") counterpart of eda_output/*_cleaned.csv.

The thesis pipeline consumes eda_output/{train,test}_cleaned.csv, which
notebooks/ciciomt2024_eda.py produced by: load 72 CSVs as float32 -> +-inf -> NaN
-> drop_duplicates -> median-fill. This script reproduces every step EXCEPT
drop_duplicates, so the C1 ablation differs from the published arm in duplicates
alone (paper/P1_dedup_audit/EXPERIMENT_C1.md).

Faithfulness self-test (the reason this script is trustworthy): on the same pass
it ALSO computes what the deduplicated counts would be and asserts they equal the
canonical 4,515,080 / 892,268 (numbers_map.md Section 2). If the replication had
drifted from the EDA script, those two numbers would not reproduce.

Outputs -> eda_output_raw/{train,test}_cleaned.csv  (+ build_report.json)
Reads   -> data/{train,test}/*.csv  (never written to)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TRAIN_DIR = REPO / "data" / "train"
TEST_DIR = REPO / "data" / "test"
OUT_DIR = REPO / "eda_output_raw"

# Canonical anchors (deliverables/numbers_map.md Section 2)
RAW_ROWS = {"train": 7_160_831, "test": 1_614_182}
DEDUP_ROWS = {"train": 4_515_080, "test": 892_268}

# Verbatim from notebooks/ciciomt2024_eda.py:104-113 — order is load-bearing (INV-01).
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

LABEL_MAP = {
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


def filename_to_label(filename: str) -> str:
    """Verbatim behaviour of ciciomt2024_eda.py:127-155."""
    name = os.path.basename(filename)
    name = name.replace("_train.pcap.csv", "").replace("_test.pcap.csv", "")
    name = re.sub(r"(\d+)$", "", name)
    return LABEL_MAP.get(name, name)


def label_to_category(label: str) -> str:
    """Verbatim behaviour of ciciomt2024_eda.py:158-165."""
    if label == "Benign":
        return "Benign"
    if label.startswith("DDoS"):
        return "DDoS"
    if label.startswith("DoS"):
        return "DoS"
    if label.startswith("Recon"):
        return "Recon"
    if label.startswith("MQTT"):
        return "MQTT"
    if label.startswith("ARP"):
        return "Spoofing"
    return "Unknown"


def verify_feature_list() -> None:
    """Guard against silent drift: our FEATURES must equal the EDA script's."""
    src = (REPO / "notebooks" / "ciciomt2024_eda.py").read_text(encoding="utf-8")
    block = re.search(r"^FEATURES = \[(.*?)^\]", src, re.S | re.M)
    if block is None:
        sys.exit("Could not locate FEATURES in ciciomt2024_eda.py — refusing to guess.")
    theirs = re.findall(r'"([^"]+)"', block.group(1))
    if theirs != FEATURES:
        sys.exit(
            "FEATURES drifted from ciciomt2024_eda.py.\n"
            f"  theirs ({len(theirs)}): {theirs}\n"
            f"  ours   ({len(FEATURES)}): {FEATURES}"
        )
    print(f"  feature list matches the EDA script ({len(FEATURES)} columns)")


def load_split(directory: Path, split_name: str) -> pd.DataFrame:
    """Load every CSV as float32 and tag label/category/split (EDA lines 169-199)."""
    csv_files = sorted(glob.glob(str(directory / "*.csv")))
    if not csv_files:
        sys.exit(f"No CSV files under {directory}")
    print(f"[{split_name}] {len(csv_files)} CSV files")

    dtype_map = {f: np.float32 for f in FEATURES}
    frames = []
    for i, path in enumerate(csv_files, 1):
        label = filename_to_label(path)
        df = pd.read_csv(path, dtype=dtype_map, engine="c", low_memory=False)
        df = df[[c for c in FEATURES if c in df.columns]].copy()
        df["label"] = label
        df["category"] = label_to_category(label)
        df["split"] = split_name
        frames.append(df)
        if i % 10 == 0 or i == len(csv_files):
            print(f"  [{i:02d}/{len(csv_files)}] {os.path.basename(path)} -> {label}")

    full = pd.concat(frames, ignore_index=True)
    del frames
    print(f"[{split_name}] loaded {len(full):,} rows x {full.shape[1]} cols")
    return full


def build(split_name: str, directory: Path) -> dict:
    df = load_split(directory, split_name)

    if len(df) != RAW_ROWS[split_name]:
        sys.exit(
            f"{split_name}: loaded {len(df):,} rows, expected {RAW_ROWS[split_name]:,} "
            "(numbers_map.md Section 2). Refusing to continue on a different substrate."
        )

    # Clean step 1 — +-inf -> NaN (EDA line 280). Applied to BOTH arms.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Faithfulness self-test — what the deduplicated count WOULD be (EDA lines 283-284).
    n_dedup = int(len(df.drop_duplicates()))
    ok = n_dedup == DEDUP_ROWS[split_name]
    print(f"[{split_name}] self-test: dedup would give {n_dedup:,} "
          f"(canonical {DEDUP_ROWS[split_name]:,}) -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(
            f"{split_name}: replication self-test FAILED. This script does not reproduce "
            "the published deduplicated row count, so its non-deduplicated output cannot "
            "be trusted as a matched control arm. Investigate before continuing."
        )

    # Clean step 2 — median-fill (EDA line 290). Medians come from THIS arm's own data:
    # the raw arm is 'what if the pipeline had never deduplicated', end to end.
    na_cols = df.columns[df.isna().any()].tolist()
    n_filled = int(df[na_cols].isna().sum().sum()) if na_cols else 0
    if na_cols:
        df[na_cols] = df[na_cols].fillna(df[na_cols].median(numeric_only=True))
    print(f"[{split_name}] median-filled {n_filled:,} cells across {len(na_cols)} columns")

    # NOTE: no drop_duplicates — that omission IS the experiment.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split_name}_cleaned.csv"
    df.to_csv(out_path, index=False, chunksize=500_000)
    size_gb = out_path.stat().st_size / 1e9
    print(f"[{split_name}] wrote {out_path} ({len(df):,} rows, {size_gb:.2f} GB)")

    return {
        "split": split_name,
        "rows_written": int(len(df)),
        "rows_expected_raw": RAW_ROWS[split_name],
        "dedup_selftest_count": n_dedup,
        "dedup_selftest_canonical": DEDUP_ROWS[split_name],
        "dedup_selftest_pass": ok,
        "duplicates_retained": int(len(df) - n_dedup),
        "nan_cells_filled": n_filled,
        "nan_columns": na_cols,
        "output_path": str(out_path),
        "output_bytes": int(out_path.stat().st_size),
    }


def main() -> None:
    print("=" * 70)
    print("C1 step 1 — building the NON-deduplicated cleaned CSVs")
    print("=" * 70)
    verify_feature_list()

    report = {"repo": str(REPO), "features": len(FEATURES), "splits": []}
    for split_name, directory in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        report["splits"].append(build(split_name, directory))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2))
    print("\nSelf-test passed for both splits; raw arm inputs are ready.")
    print(f"Report -> {OUT_DIR / 'build_report.json'}")


if __name__ == "__main__":
    main()
