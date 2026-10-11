"""Step 02 - column audit of the 59 model-input columns, BENIGN flows only, both datasets.

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/02_column_audit.py

Rows measured on: ALL raw (not deduplicated) benign flows of each table.
  CICIoMT2024 : features_identity/Benign_{train,test}.pcap.csv   (class Benign by filename_to_class)
  CICIoT2023  : features_ciciot23/BenignTraffic{,1,2,3}.pcap.csv (class Benign by manifest.jsonl)

Ratio rule (stated, because a zero median makes a plain ratio undefined):
  ratio = |median_iot| / |median_iomt|                              when both medians are non-zero
  both medians 0            -> ratio 1.0,  status BOTH_ZERO  (not flagged)
  exactly one median is 0   -> ratio undefined, status ONE_ZERO (flagged, counted SEPARATELY)
  flag RATIO_GE_10X         -> ratio >= 10 or ratio <= 0.1 (both medians non-zero)
  a sign difference between non-zero medians is noted in `sign_differs`
Sort order: most divergent first = ONE_ZERO, then by |log10(ratio)| descending, BOTH_ZERO last.
"""
import csv
import math
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

C.bootstrap()
import pandas as pd  # noqa: E402


def benign_files():
    from stage2_device_disjoint import EXCLUDE_SUBSTR, filename_to_class
    iomt = [os.path.join(C.FEATURES_IOMT, f) for f in sorted(os.listdir(C.FEATURES_IOMT))
            if f.endswith(".csv") and not any(s in f for s in EXCLUDE_SUBSTR)
            and filename_to_class(f) == "Benign"]
    iot = [os.path.join(C.FEATURES_IOT, m["file"] + ".csv") for m in C.iot_manifest()
           if m["class"] == "Benign"]
    return iomt, iot


def load(paths):
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def classify(med_iomt, med_iot):
    if med_iomt == 0 and med_iot == 0:
        return 1.0, "BOTH_ZERO", False
    if med_iomt == 0 or med_iot == 0:
        return None, "ONE_ZERO", False
    ratio = abs(med_iot) / abs(med_iomt)
    return ratio, ("RATIO_GE_10X" if ratio >= 10 or ratio <= 0.1 else "OK"), (med_iomt < 0) != (med_iot < 0)


def sort_key(row):
    if row["status"] == "ONE_ZERO":
        return (0, 0.0)
    if row["status"] == "BOTH_ZERO":
        return (2, 0.0)
    return (1, -abs(math.log10(row["ratio_iot_over_iomt"])))


def main():
    iomt_files, iot_files = benign_files()
    iomt, iot = load(iomt_files), load(iot_files)
    cols_iomt, cols_iot = C.model_input_columns(iomt.columns), C.model_input_columns(iot.columns)
    assert len(cols_iomt) == 59 and cols_iomt == cols_iot, (len(cols_iomt), len(cols_iot))
    cols = cols_iomt
    non_numeric = [c for c in cols if not (pd.api.types.is_numeric_dtype(iomt[c]) and pd.api.types.is_numeric_dtype(iot[c]))]
    assert not non_numeric, non_numeric

    rows = []
    for c in cols:
        m_a, m_b = float(iomt[c].median()), float(iot[c].median())
        ratio, status, sign_differs = classify(m_a, m_b)
        rows.append({
            "column": c, "median_iomt": m_a, "median_iot": m_b, "ratio_iot_over_iomt": ratio,
            "status": status, "flag_ge_10x": status in ("RATIO_GE_10X", "ONE_ZERO"),
            "sign_differs": sign_differs,
            "direction": ("" if ratio is None or status in ("OK", "BOTH_ZERO") and ratio == 1.0 else
                          "iot_larger" if ratio > 1 else "iomt_larger"),
            "zero_share_iomt": float((iomt[c] == 0).mean()), "zero_share_iot": float((iot[c] == 0).mean()),
            "nan_share_iomt": float(iomt[c].isna().mean()), "nan_share_iot": float(iot[c].isna().mean()),
        })
    rows.sort(key=sort_key)

    csv_text_path = os.path.join(C.RESULTS_DIR, "02_column_audit.csv")
    if os.path.exists(csv_text_path):
        raise SystemExit(f"refusing to overwrite {csv_text_path}")
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(csv_text_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    counts = {s: sum(r["status"] == s for r in rows) for s in ("OK", "RATIO_GE_10X", "ONE_ZERO", "BOTH_ZERO")}
    inputs = iomt_files + iot_files
    payload = {
        "meta": C.provenance(__file__, inputs, seeds=(),
                             inputs_note=("stored CICIoMT2024 run: nfstream version NOT RECORDED (step 01 re-extracts 2 "
                                          "pcaps to test this); CICIoT2023 run: nfstream 6.6.0 per manifest.jsonl")),
        "what": ("Benign median of each of the 59 model-input columns in each dataset, their ratio "
                 "(iot/iomt), zero share, and a flag where medians differ by >= 10x."),
        "rows_measured_on": {"iomt_benign_flows_raw": int(len(iomt)), "iot_benign_flows_raw": int(len(iot)),
                             "iomt_files": [os.path.basename(p) for p in iomt_files],
                             "iot_files": [os.path.basename(p) for p in iot_files],
                             "deduplicated": False},
        "rule": __doc__.split("Ratio rule")[1].split('"""')[0].strip(),
        "n_model_input_columns": len(cols),
        "model_input_columns": cols,
        "status_counts": counts,
        "n_flagged_for_review": sum(r["flag_ge_10x"] for r in rows),
        "table_csv": "02_column_audit.csv",
        "table": rows,
    }
    print(len(iomt), len(iot), counts)
    print("wrote", C.write_json("02_column_audit.json", payload), "and", csv_text_path)


if __name__ == "__main__":
    main()
