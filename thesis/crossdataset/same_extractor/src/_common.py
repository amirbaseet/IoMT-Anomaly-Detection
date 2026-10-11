"""Shared helpers for the same-extractor experiment (steps 01-04).

Run every step as:  <venv>/bin/python -B -I src/NN_name.py
  -B  no __pycache__ is written (neither here nor in iomt-pcap-experiments)
  -I  isolated mode: the script directory is not on sys.path, so each step
      inserts SRC_DIR and EXP_DIR explicitly via `bootstrap()`.

All wall-clock values are UTC.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SAME_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SAME_DIR, "results")
THESIS_REPO = os.path.abspath(os.path.join(SAME_DIR, "..", "..", ".."))
EXP_DIR = os.path.expanduser("~/code/iomt-pcap-experiments")
SSD = "/Volumes/PortableSSD"

FEATURES_IOMT = os.path.join(EXP_DIR, "features_identity")
FEATURES_IOT = os.path.join(SSD, "features_ciciot23")
IOT_MANIFEST = os.path.join(FEATURES_IOT, "manifest.jsonl")
IOMT_ZIP_MAIN = os.path.join(SSD, "IOMT", "CICIoMT2024 (1).zip")
IOMT_ZIP_ALT = os.path.join(SSD, "IOMT", "CICIoMT2024_pcaps_archive.zip")
IOMT_ZIP_BROKEN = os.path.join(SSD, "IOMT", "CICIoMT2024_pcaps_archive (1).zip")
DEDUP_REPORT = os.path.join(EXP_DIR, "track_a_out", "dedup_report.json")
STAGE2A = os.path.join(EXP_DIR, "stage2a_extract_features.py")
STAGE2C = os.path.join(EXP_DIR, "stage2c_dedup.py")

SEEDS = (42, 43, 44)          # root INV-04 / thesis/crossdataset TRAP 4; steps 01-03 use none
MAX_RESULT_BYTES = 5 * 1024 * 1024   # rule 3

# Pair table (rule: families with pcaps on BOTH sides + Benign). Each side is a list of
# dataset-native classes. CICIoMT2024 DoS and DDoS stay separate rows; CICIoT2023 has
# DoS-{SYN,TCP,UDP} but DDoS-ICMP, so a pair row is NOT a claim that DoS == DDoS.
PAIRS = {
    "SYN":       {"iomt": ["DoS_SYN", "DDoS_SYN"],   "iot": ["DoS-SYN_Flood"]},
    "TCP":       {"iomt": ["DoS_TCP", "DDoS_TCP"],   "iot": ["DoS-TCP_Flood"]},
    "UDP":       {"iomt": ["DoS_UDP", "DDoS_UDP"],   "iot": ["DoS-UDP_Flood"]},
    "ICMP":      {"iomt": ["DoS_ICMP", "DDoS_ICMP"], "iot": ["DDoS-ICMP_Flood"]},
    "port_scan": {"iomt": ["Recon_Port_Scan"],       "iot": ["Recon-PortScan"]},
    "os_scan":   {"iomt": ["Recon_OS_Scan"],         "iot": ["Recon-OSScan"]},
    "ping_sweep": {"iomt": ["Recon_Ping_Sweep"],     "iot": ["Recon-PingSweep"]},
    "vuln_scan": {"iomt": ["Recon_VulScan"],         "iot": ["VulnerabilityScan"]},
    "arp":       {"iomt": ["ARP_Spoofing"],          "iot": ["MITM-ArpSpoofing"]},
    "Benign":    {"iomt": ["Benign"],                "iot": ["Benign"]},
}


def bootstrap():
    """Make SRC_DIR and EXP_DIR importable under `python -I` (read-only imports)."""
    for p in (SRC_DIR, EXP_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, _cache={}):
    path = os.path.abspath(path)
    if path in _cache:
        return _cache[path]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    _cache[path] = h.hexdigest()
    return _cache[path]


def _git(*args):
    out = subprocess.run(["git", "-C", THESIS_REPO, *args], capture_output=True, text=True)
    return out.stdout.rstrip("\n")


def lib_versions():
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "nfstream", "sklearn", "pyarrow"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "?")
        except Exception:
            versions[name] = None
    return versions


def provenance(script_file, input_paths, *, seeds=(), inputs_note=None, hash_inputs=True):
    """Everything rule 4 asks for, except the per-number fields (those sit with each number)."""
    status = [ln for ln in _git("status", "--porcelain").splitlines()
              if "thesis/crossdataset/same_extractor/" not in ln]
    inputs = []
    for p in input_paths:
        rec = {"path": p, "size_bytes": os.path.getsize(p)}
        if hash_inputs:
            rec["sha256"] = sha256_file(p)
        inputs.append(rec)
    return {
        "run_utc": utc_now(),
        "script": {"path": os.path.abspath(script_file), "sha256": sha256_file(script_file)},
        "helper": {"path": os.path.abspath(__file__), "sha256": sha256_file(__file__)},
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "libs": lib_versions(),
        "thesis_repo": {
            "path": THESIS_REPO,
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "tree_dirty": bool(status),
            "dirty_entries_excluding_same_extractor": status,
        },
        "seeds": list(seeds),
        "inputs_note": inputs_note,
        "inputs": inputs,
    }


def write_json(name, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    text = json.dumps(payload, indent=2, default=str, allow_nan=True)
    if len(text.encode("utf-8")) > MAX_RESULT_BYTES:
        raise SystemExit(f"{name} exceeds the 5 MB result limit")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite existing result {path} (rule 8: never delete)")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def write_text(name, text):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    if len(text.encode("utf-8")) > MAX_RESULT_BYTES:
        raise SystemExit(f"{name} exceeds the 5 MB result limit")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite existing result {path} (rule 8: never delete)")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def model_input_columns(columns):
    """The 59 model inputs: every column except NON_FEATURE_COLS (imported, never retyped) and label."""
    bootstrap()
    from stage2c_dedup import NON_FEATURE_COLS  # noqa: WPS433 (import after bootstrap, by design)
    return [c for c in columns if c not in NON_FEATURE_COLS and c != "label"]


def iot_manifest():
    with open(IOT_MANIFEST) as fh:
        return [json.loads(line) for line in fh if line.strip()]
