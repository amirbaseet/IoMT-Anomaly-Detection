"""Step 03 - family counts, raw and deduplicated, both NFStream flow tables; benign capture audit.

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/03_family_counts.py

Dedup = exactly stage2c_dedup.py: key = the 59 model-input columns (float64), keep="first", GLOBAL
within a dataset (across classes), files loaded in sorted(basename) order. Consequence stated in the
result: a feature vector shared by two classes is kept for whichever file sorts first, so the
per-class deduplicated counts depend on file order. The within-label variant (key + label) is
reported next to it as the order-free sensitivity.

CONTROL (must pass before any count is trusted): the CICIoMT2024 run must reproduce
track_a_out/dedup_report.json -> 7,357,600 raw / 4,567,803 deduplicated and every per-class `kept`.

Labelling in both pipelines = the capture's filename, not a per-flow decision:
  CICIoMT2024: stage2c_dedup.load_all L51/L56 calls filename_to_class (stage2_device_disjoint.py L44)
               and assigns df["label"] = cls to EVERY flow of the file.
  CICIoT2023 : run_ciciot23.py takes the class from the TSV (ciciot23_files.tsv) per pcap and writes it
               into manifest.jsonl; this script uses that class for every flow of the file.
Benign background traffic inside an attack capture therefore carries the attack label in both tables.
"""
import csv
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

CONTIGUITY_GAP_S = 60.0   # MY threshold for "time-contiguous chunk"; raw gaps are reported so it can be re-judged
TS_FIRST, TS_LAST = "bidirectional_first_seen_ms", "bidirectional_last_seen_ms"


def iomt_jobs():
    from stage2_device_disjoint import EXCLUDE_SUBSTR, filename_to_class
    jobs = []
    for f in sorted(os.listdir(C.FEATURES_IOMT)):
        if f.endswith(".csv") and not any(s in f for s in EXCLUDE_SUBSTR):
            cls = filename_to_class(f)
            if cls is not None:
                jobs.append((os.path.join(C.FEATURES_IOMT, f), cls))
    return jobs


def iot_jobs():
    jobs = [(os.path.join(C.FEATURES_IOT, m["file"] + ".csv"), m["class"], m) for m in C.iot_manifest()
            if m.get("status") == "ok"]
    jobs.sort(key=lambda j: os.path.basename(j[0]))
    return [(p, c) for p, c, _ in jobs], {os.path.basename(p): m for p, c, m in jobs}


def ms_to_utc(ms):
    return pd.to_datetime(int(ms), unit="ms", utc=True).isoformat()


def load_dataset(name, jobs):
    """One float64 frame of the 59 inputs + per-row file index; per-file stats on the side."""
    cols, frames, file_idx, per_file, benign_macs = None, [], [], [], {}
    for i, (path, cls) in enumerate(jobs):
        head = pd.read_csv(path, nrows=0)
        this_cols = C.model_input_columns(head.columns)
        assert len(this_cols) == 59, (path, len(this_cols))
        cols = cols or this_cols
        assert this_cols == cols, f"column order differs in {path}"
        df = pd.read_csv(path, usecols=cols + [TS_FIRST, TS_LAST, "src_mac"],
                         dtype={c: "float64" for c in cols})
        macs = set(df["src_mac"].dropna())
        per_file.append({
            "dataset": name, "file": os.path.basename(path), "class": cls, "rows": int(len(df)),
            "first_flow_utc": ms_to_utc(df[TS_FIRST].min()), "last_flow_utc": ms_to_utc(df[TS_LAST].max()),
            "first_flow_ms": int(df[TS_FIRST].min()), "last_flow_ms": int(df[TS_LAST].max()),
            "distinct_src_mac": len(macs),
        })
        if cls == "Benign":
            benign_macs[os.path.basename(path)] = macs
        frames.append(df[cols])
        file_idx.append(np.full(len(df), i, dtype=np.int16))
        print(f"  [{name} {i + 1}/{len(jobs)}] {os.path.basename(path)} {len(df):,} rows", flush=True)
    X = pd.concat(frames, ignore_index=True)
    del frames
    return X, np.concatenate(file_idx), per_file, benign_macs, cols


def dedup_counts(X, file_idx, per_file):
    n = len(X)
    classes = sorted({p["class"] for p in per_file})
    label_code = np.array([classes.index(per_file[i]["class"]) for i in range(len(per_file))])[file_idx]
    dup = X.duplicated(keep="first").to_numpy()
    X["__label"] = label_code.astype("float64")   # in place: a copy would double the 4 GB frame
    dup_lab = X.duplicated(keep="first").to_numpy()
    out = {}
    for ci, cls in enumerate(classes):
        m = label_code == ci
        out[cls] = {"rows_raw": int(m.sum()), "rows_dedup_global": int((m & ~dup).sum()),
                    "rows_dedup_within_label": int((m & ~dup_lab).sum())}
    per_file_kept = np.bincount(file_idx[~dup], minlength=len(per_file))
    return out, {"raw": n, "dedup_global": int((~dup).sum()), "dedup_within_label": int((~dup_lab).sum())}, per_file_kept


def benign_audit(files, macs_by_file, pcap_sizes):
    rows = sorted(files, key=lambda r: r["first_flow_ms"])
    out, group = [], 0
    for k, r in enumerate(rows):
        rec = {**r, "source_pcap_size_bytes": pcap_sizes.get(r["file"])}
        if k:
            prev = rows[k - 1]
            gap = (r["first_flow_ms"] - prev["last_flow_ms"]) / 1000.0
            a, b = macs_by_file[prev["file"]], macs_by_file[r["file"]]
            rec["gap_s_from_previous_file_last_flow"] = gap
            rec["mac_jaccard_with_previous"] = len(a & b) / len(a | b) if a | b else None
            if abs(gap) > CONTIGUITY_GAP_S:
                group += 1
        rec["contiguity_group"] = group
        out.append(rec)
    union = set().union(*macs_by_file.values()) if macs_by_file else set()
    return {"files_in_time_order": out, "contiguity_rule": f"consecutive files are one chunk-chain if |first_flow - previous last_flow| <= {CONTIGUITY_GAP_S:g} s (my threshold, flagged)",
            "independent_benign_captures_under_rule": group + 1 if out else 0,
            "distinct_src_mac_in_benign_union": len(union)}


def zip_sizes():
    sizes = {}
    with zipfile.ZipFile(C.IOMT_ZIP_MAIN) as zf:
        for i in zf.infolist():
            if i.filename.endswith(("Benign_train.pcap", "Benign_test.pcap")):
                sizes[os.path.basename(i.filename) + ".csv"] = i.file_size
    return sizes


def main():
    ctrl = json.load(open(C.DEDUP_REPORT))
    j_iomt, (j_iot, iot_meta) = iomt_jobs(), iot_jobs()
    result = {}
    all_inputs = [p for p, _ in j_iomt] + [p for p, _ in j_iot]
    print(f"PRE-FLIGHT: read ~{sum(os.path.getsize(p) for p in all_inputs) / 2**30:.1f} GiB CSV; "
          f"peak RAM ~ 2 x (rows x 59 x 8 B) per dataset; writes < 5 MB; scratch none", flush=True)

    for name, jobs in (("CICIoMT2024", j_iomt), ("CICIoT2023", j_iot)):
        X, fidx, per_file, benign_macs, cols = load_dataset(name, jobs)
        counts, totals, kept = dedup_counts(X, fidx, per_file)
        for i, pf in enumerate(per_file):
            pf["rows_dedup_global"] = int(kept[i])
        result[name] = {"cols": cols, "per_class": counts, "totals": totals, "per_file": per_file,
                        "benign_macs": benign_macs, "n_files": len(jobs)}
        del X

    # control against the stored stage2c report (CICIoMT2024 only)
    c = result["CICIoMT2024"]
    control = {"expected_raw": ctrl["n_flows_raw"], "expected_dedup": ctrl["n_flows_deduped"],
               "got_raw": c["totals"]["raw"], "got_dedup": c["totals"]["dedup_global"],
               "per_class_mismatches": {k: {"expected_kept": v["kept"], "got_kept": c["per_class"].get(k, {}).get("rows_dedup_global")}
                                        for k, v in ctrl["per_class"].items()
                                        if c["per_class"].get(k, {}).get("rows_dedup_global") != v["kept"]}}
    control["passed"] = (control["expected_raw"] == control["got_raw"] and control["expected_dedup"] == control["got_dedup"]
                         and not control["per_class_mismatches"])
    # manifest vs CSV rows for CICIoT2023
    manifest_row_mismatch = {pf["file"]: {"manifest": iot_meta[pf["file"]]["flows"], "csv": pf["rows"]}
                             for pf in result["CICIoT2023"]["per_file"] if iot_meta[pf["file"]]["flows"] != pf["rows"]}
    assert result["CICIoMT2024"]["cols"] == result["CICIoT2023"]["cols"]

    pairs, benign_share = {}, {}
    for tag, key in (("CICIoMT2024", "iomt"), ("CICIoT2023", "iot")):
        r = result[tag]
        tot = r["totals"]
        b = r["per_class"].get("Benign", {})
        benign_share[tag] = {k: {"benign": b.get(f, 0), "table": tot[t], "share": b.get(f, 0) / tot[t]}
                             for k, f, t in (("raw", "rows_raw", "raw"), ("dedup_global", "rows_dedup_global", "dedup_global"),
                                             ("dedup_within_label", "rows_dedup_within_label", "dedup_within_label"))}
        for pair, sides in C.PAIRS.items():
            cl = sides[key]
            files = sum(1 for pf in r["per_file"] if pf["class"] in cl)
            pairs.setdefault(pair, {})[tag] = {
                "classes": cl, "files": files,
                "rows_raw": sum(r["per_class"].get(x, {}).get("rows_raw", 0) for x in cl),
                "rows_dedup_global": sum(r["per_class"].get(x, {}).get("rows_dedup_global", 0) for x in cl),
                "rows_dedup_within_label": sum(r["per_class"].get(x, {}).get("rows_dedup_within_label", 0) for x in cl),
                "per_class": {x: r["per_class"].get(x) for x in cl},
            }
    in_pairs = {"CICIoMT2024": {x for s in C.PAIRS.values() for x in s["iomt"]},
                "CICIoT2023": {x for s in C.PAIRS.values() for x in s["iot"]}}
    outside = {t: {k: v for k, v in result[t]["per_class"].items() if k not in in_pairs[t]} for t in result}

    pcap_sizes_iot = {fn: meta["size_bytes"] for fn, meta in iot_meta.items()}
    benign = {
        "CICIoMT2024": benign_audit([p for p in result["CICIoMT2024"]["per_file"] if p["class"] == "Benign"],
                                    result["CICIoMT2024"]["benign_macs"], zip_sizes()),
        "CICIoT2023": benign_audit([p for p in result["CICIoT2023"]["per_file"] if p["class"] == "Benign"],
                                   result["CICIoT2023"]["benign_macs"], pcap_sizes_iot),
    }
    for t in benign:
        for r in benign[t]["files_in_time_order"]:
            r.pop("first_flow_ms", None), r.pop("last_flow_ms", None)

    payload = {
        "meta": C.provenance(__file__, all_inputs, seeds=(),
                             inputs_note=("CICIoMT2024 tables: nfstream version NOT RECORDED for the stored run (see step 01); "
                                          "CICIoT2023 tables: nfstream 6.6.0, stage2a sha256 241e5654.., per manifest.jsonl")),
        "what": ("Flows per dataset, files per dataset, raw and deduplicated counts per class and per pair, benign share, "
                 "and a benign-capture audit. Rows measured on: every flow of every listed file (profiling captures "
                 "excluded for CICIoMT2024 by stage2c's EXCLUDE_SUBSTR)."),
        "dedup": {"key": "the 59 model-input columns, float64", "keep": "first", "scope": "global within a dataset (as stage2c_dedup.py)",
                  "file_order": "sorted(csv basename)",
                  "order_caveat": "cross-class duplicate vectors are attributed to the first file in sort order; per-class dedup_global counts depend on it. dedup_within_label is order-free."},
        "control_stage2c_reproduction": control,
        "manifest_vs_csv_row_mismatches_ciciot2023": manifest_row_mismatch,
        "model_input_columns": result["CICIoMT2024"]["cols"],
        "table_totals": {t: result[t]["totals"] for t in result},
        "n_files": {t: result[t]["n_files"] for t in result},
        "benign_share_in_table": benign_share,
        "pairs": pairs,
        "classes_not_in_any_pair": outside,
        "labelling_of_benign_background_in_attack_captures": {
            "CICIoMT2024": "label = filename class for every flow (stage2c_dedup.py L51,L56 via stage2_device_disjoint.filename_to_class L44); background benign flows inside an attack pcap are labelled as that attack. The dataset README (WiFi_and_MQTT attacks, 1 page) says nothing about benign background.",
            "CICIoT2023": "label = class of the pcap from ciciot23_files.tsv -> manifest.jsonl (run_ciciot23.py), applied to every flow of the file; same consequence. How the OFFICIAL CICIoT2023 CSVs label such flows was not checked here.",
        },
        "benign_captures": benign,
        "per_file": {t: result[t]["per_file"] for t in result},
    }
    path = C.write_json("03_family_counts.json", payload)

    rows = []
    for t in result:
        for cls, v in sorted(result[t]["per_class"].items()):
            rows.append({"dataset": t, "class": cls, "files": sum(p["class"] == cls for p in result[t]["per_file"]), **v})
    out_csv = os.path.join(C.RESULTS_DIR, "03_family_counts.csv")
    with open(out_csv, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("CONTROL passed:", control["passed"], control)
    print("wrote", path, "and", out_csv)
    if not control["passed"]:
        raise SystemExit("CONTROL FAILED - do not trust the counts")


if __name__ == "__main__":
    main()
