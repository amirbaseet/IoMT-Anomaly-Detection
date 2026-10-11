"""Step 01 - version check: does the CURRENT extractor + nfstream 6.6.0 reproduce the STORED
CICIoMT2024 features_identity output?

Venv: ~/code/iomt-pcap-experiments/venv (the only one with nfstream).
Run:  ~/code/iomt-pcap-experiments/venv/bin/python -B -I src/01_version_check.py --scratch <dir>

Two pcaps are re-extracted from the readable zip with the current stage2a `extract_one`:
Recon-Ping_Sweep_test (ICMP-ish, tiny) and Recon-Port_Scan_test (TCP-heavy).
Comparison per file: row count, column list, three medians, and the max absolute difference
over the numeric columns after sorting both sides by flow key. The fresh frame is written with
to_csv and read back, the same path the stored file took, so float formatting cannot fake a diff.
Also reports, per file, how many numeric columns are all-zero on BOTH sides (a match there
proves nothing), and which of the 72 stored captures have a pcap on disk.
"""
import argparse
import importlib.util
import json
import os
import sys
import zipfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

C.bootstrap()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TARGETS = ["Recon-Ping_Sweep_test.pcap", "Recon-Port_Scan_test.pcap"]
MEDIAN_COLS = ["bidirectional_packets", "bidirectional_bytes", "bidirectional_duration_ms"]
KEY_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "vlan_id", "tunnel_id",
            "bidirectional_first_seen_ms"]


def zip_member(zf, basename):
    hits = [i for i in zf.infolist() if os.path.basename(i.filename) == basename]
    if len(hits) != 1:
        raise SystemExit(f"{basename}: expected 1 member, found {len(hits)}")
    return hits[0]


def extract_member(zf, info, out_dir):
    os.makedirs(out_dir, exist_ok=False)  # new, empty dir; never reuse (untrusted download)
    dest = os.path.join(out_dir, os.path.basename(info.filename))
    with zf.open(info) as src, open(dest, "wb") as dst:
        for chunk in iter(lambda: src.read(1 << 22), b""):
            dst.write(chunk)
    return dest


def load_extract_one():
    spec = importlib.util.spec_from_file_location("stage2a", C.STAGE2A)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_one


def sort_frame(df, feature_cols):
    order = [c for c in KEY_COLS if c in df.columns]
    order += [c for c in feature_cols if c not in order]
    order += [c for c in df.columns if c not in order and c not in ("id",)] + ["id"]
    return df.sort_values(order, kind="mergesort", na_position="last").reset_index(drop=True)


def compare(stored, fresh, feature_cols):
    out = {"rows_stored": int(len(stored)), "rows_fresh": int(len(fresh)),
           "columns_stored": int(stored.shape[1]), "columns_fresh": int(fresh.shape[1]),
           "column_list_identical_and_same_order": list(stored.columns) == list(fresh.columns),
           "column_set_identical": set(stored.columns) == set(fresh.columns),
           "columns_only_in_stored": sorted(set(stored.columns) - set(fresh.columns)),
           "columns_only_in_fresh": sorted(set(fresh.columns) - set(stored.columns))}
    out["medians"] = {c: {"stored": float(stored[c].median()), "fresh": float(fresh[c].median())}
                      for c in MEDIAN_COLS}
    num_cols = [c for c in stored.select_dtypes("number").columns if c in fresh.columns]
    zero_both = [c for c in num_cols if (stored[c] == 0).all() and (fresh[c] == 0).all()]
    out["numeric_columns"] = len(num_cols)
    out["numeric_feature_columns"] = len([c for c in num_cols if c in feature_cols])
    out["all_zero_on_both_sides"] = {
        "n_numeric_columns": len(zero_both),
        "n_model_input_columns": len([c for c in zero_both if c in feature_cols]),
        "columns": zero_both,
    }
    if len(stored) != len(fresh) or not out["column_set_identical"]:
        sk = stored[KEY_COLS].astype(str).agg("|".join, axis=1).value_counts()
        fk = fresh[KEY_COLS].astype(str).agg("|".join, axis=1).value_counts()
        out["elementwise_compared"] = False
        out["flow_key_multiset"] = {
            "keys_only_in_stored": int((sk.sub(fk, fill_value=0) > 0).sum()),
            "keys_only_in_fresh": int((fk.sub(sk, fill_value=0) > 0).sum()),
        }
        return out
    s, f = sort_frame(stored, feature_cols), sort_frame(fresh[stored.columns], feature_cols)
    out["elementwise_compared"] = True
    obj_cols = [c for c in s.columns if c not in num_cols]
    out["object_columns_mismatching_rows"] = {
        c: int((s[c].fillna("<NA>") != f[c].fillna("<NA>")).sum()) for c in obj_cols}
    both_nan = s[num_cols].isna() & f[num_cols].isna()
    one_nan = s[num_cols].isna() ^ f[num_cols].isna()
    diff = (s[num_cols].astype("float64") - f[num_cols].astype("float64")).abs().mask(both_nan, 0.0)
    out["nan_mismatch_cells"] = int(one_nan.to_numpy().sum())
    per_col = diff.max()
    feat = [c for c in num_cols if c in feature_cols]
    nonfeat = [c for c in num_cols if c not in feature_cols]
    out["max_abs_diff"] = {
        "model_input_columns": float(per_col[feat].max()) if feat else None,
        "non_feature_numeric_columns": float(per_col[nonfeat].max()) if nonfeat else None,
    }
    out["columns_with_nonzero_diff"] = {c: float(v) for c, v in per_col.items() if v > 0}
    out["rows_with_any_diff"] = int((diff > 0).any(axis=1).sum())
    return out


def coverage(stored_manifest):
    """Which of the stored captures have a pcap in a readable zip (by basename)."""
    stored = sorted(m["capture"] for m in stored_manifest)
    names, per_zip = {}, {}
    for label, path in (("main", C.IOMT_ZIP_MAIN), ("alt", C.IOMT_ZIP_ALT)):
        with zipfile.ZipFile(path) as zf:
            found = {os.path.basename(i.filename): i.CRC for i in zf.infolist()
                     if i.filename.lower().endswith(".pcap") and "WiFI_and_MQTT" in i.filename}
        per_zip[label] = {"path": path, "wifi_mqtt_pcaps": len(found)}
        for n, crc in found.items():
            names.setdefault(n, {})[label] = crc
    on_disk = [s for s in stored if s in names]
    crc_conflicts = [n for n, v in names.items() if len(set(v.values())) > 1]
    return {"stored_captures": len(stored), "stored_with_pcap_on_disk": len(on_disk),
            "stored_without_pcap_on_disk": len(stored) - len(on_disk),
            "missing": [s for s in stored if s not in names],
            "same_basename_different_crc_between_zips": crc_conflicts,
            "zips": per_zip,
            "unreadable_zip": {"path": C.IOMT_ZIP_BROKEN, "error": "BadZipFile: File is not a zip file (checked 2026-10-10)"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True, help="scratch dir for extracted pcaps and fresh CSVs")
    args = ap.parse_args()

    stored_manifest = json.load(open(os.path.join(C.FEATURES_IOMT, "_manifest.json")))
    extract_one = load_extract_one()
    prov_inputs, per_file = [], {}
    with zipfile.ZipFile(C.IOMT_ZIP_MAIN) as zf, zipfile.ZipFile(C.IOMT_ZIP_ALT) as zf_alt:
        for base in TARGETS:
            info = zip_member(zf, base)
            alt = zip_member(zf_alt, base)
            out_dir = os.path.join(args.scratch, "zip_extract_" + base.replace(".pcap", ""))
            pcap = extract_member(zf, info, out_dir)
            stored_csv = os.path.join(C.FEATURES_IOMT, base + ".csv")
            fresh = extract_one(pcap)
            fresh["src_capture"] = base
            fresh_csv = os.path.join(out_dir, "fresh_" + base + ".csv")
            fresh.to_csv(fresh_csv, index=False)
            stored_df, fresh_df = pd.read_csv(stored_csv), pd.read_csv(fresh_csv)
            feature_cols = C.model_input_columns(stored_df.columns)
            assert len(feature_cols) == 59, len(feature_cols)
            prov_inputs += [stored_csv, pcap]
            per_file[base] = {
                "zip_member": info.filename, "zip_member_crc32": f"{info.CRC:08x}",
                "same_member_in_alt_zip_crc32": f"{alt.CRC:08x}",
                "pcap_extracted_to": pcap, "fresh_csv_written_to": fresh_csv,
                "stored_csv": stored_csv,
                **compare(stored_df, fresh_df, feature_cols),
            }
            print(base, {k: per_file[base][k] for k in ("rows_stored", "rows_fresh", "max_abs_diff") if k in per_file[base]})

    payload = {
        "what": ("Does extract_one() of the current stage2a script with nfstream 6.6.0 reproduce the stored "
                 "features_identity CSV of the same CICIoMT2024 pcap? Numbers are per-pcap, measured on ALL "
                 "flows of that pcap (rows_stored / rows_fresh)."),
        "stored_run_nfstream_version": "NOT RECORDED (features_identity/_manifest.json has no version field)",
        "zip_used": C.IOMT_ZIP_MAIN, "zip_size_bytes": os.path.getsize(C.IOMT_ZIP_MAIN),
        "zip_hash_policy": "21 GB zip not hashed; path, size and member CRC32 recorded, extracted pcap sha256 in inputs",
        "median_columns": MEDIAN_COLS, "flow_key_columns_for_sorting": KEY_COLS,
        "pcaps": per_file,
        "coverage_of_stored_captures": coverage(stored_manifest),
        "model_input_columns_n": 59,
    }
    prov = C.provenance(__file__, prov_inputs, seeds=(),
                        inputs_note="stored feature CSV + extracted pcap per target; zip not hashed")
    path = C.write_json("01_version_check.json", {"meta": prov, **payload})
    print("wrote", path)


if __name__ == "__main__":
    main()
