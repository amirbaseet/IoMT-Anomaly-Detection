"""Step 11 - ICMP-flood indicator shares, no model. Added after the background audit (README, 'ICMP decision').

Venv: ~/code/IoMT-Project/venv   Run: ~/code/IoMT-Project/venv/bin/python -B -I src/11_icmp_shares.py
(a) RELEASED CSVs, all rows (raw, NOT deduplicated): for the ICMP-flood classes and Benign (contrast) in both datasets, the share of rows
    whose `ICMP` indicator column is >= 0.5. CICIoMT2024: official train + test CSVs, class from the thesis filename_to_class.
    CICIoT2023: the 63 members of MERGED_CSV.zip, class from the Label column (thesis normalize_ciciot2023).
(b) NFStream flow tables, the ICMP family and Benign, BEFORE dedup: share of flows with protocol == 1 (all rows of the family's files).
    The after-dedup share is not recomputed here; it is read from results/07_background_audit.json (3.18% / 4.56%).
"""
import json
import os
import sys
import time
import zipfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

C.bootstrap()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("step05", os.path.join(C.SRC_DIR, "05_detection_arms.py"))
S5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S5)
THESIS_X = os.path.join(C.THESIS_REPO, "thesis", "crossdataset")
ZIP = os.path.join(C.SSD, "CICIOT23", "csv", "MERGED_CSV.zip")
ICMP_FAM, BEN = "ICMP", "Benign"


def share(n_ind, n):
    return {"n_rows": int(n), "n_icmp_ge_0.5": int(n_ind), "share": float(n_ind / n) if n else None}


def csv_iomt(config):
    from stage2_device_disjoint import filename_to_class
    cls_of = {c: f for f, v in C.PAIRS.items() for c in v["iomt"]}
    acc = {}
    for d in (config.CICIOMT2024_TRAIN_DIR, config.CICIOMT2024_TEST_DIR):
        for f in sorted(os.listdir(d)):
            if not f.endswith(".csv"):
                continue
            cls = filename_to_class(os.path.join(d, f))
            fam = cls_of.get(cls)
            if fam not in (ICMP_FAM, BEN):
                continue
            ind = pd.read_csv(os.path.join(d, f), usecols=["ICMP"])["ICMP"].to_numpy()
            a = acc.setdefault(cls, [0, 0])
            a[0] += int((ind >= 0.5).sum()); a[1] += len(ind)
            print("  iomt", f, len(ind), flush=True)
    return acc


def csv_iot(config, labels):
    want = {labels.normalize_ciciot2023(c): c for f in (ICMP_FAM, BEN) for c in C.PAIRS[f]["iot"]}
    zf = zipfile.ZipFile(ZIP)
    members = sorted(i.filename for i in zf.infolist() if i.filename.endswith(".csv") and "/._" not in i.filename)
    acc, t0 = {c: [0, 0] for c in want.values()}, time.time()
    for mi, m in enumerate(members):
        with zf.open(m) as fh:
            for ch in pd.read_csv(fh, usecols=["ICMP", config.CICIOT2023_LABEL_COL], dtype={"ICMP": np.float64, config.CICIOT2023_LABEL_COL: str}, chunksize=500_000):
                lab = ch[config.CICIOT2023_LABEL_COL].map(labels.normalize_ciciot2023)
                ind = ch["ICMP"].to_numpy()
                for k, name in want.items():
                    msk = (lab == k).to_numpy()
                    acc[name][0] += int((ind[msk] >= 0.5).sum()); acc[name][1] += int(msk.sum())
        print(f"  iot [{mi + 1}/{len(members)}] {m} {time.time() - t0:.0f}s", flush=True)
    return acc


def nfstream_pre_dedup(which):
    out = {}
    for path, cls in S5.jobs(which):
        fam = S5.CLASS2FAM[which].get(cls)
        if fam not in (ICMP_FAM, BEN):
            continue
        p = pd.read_csv(path, usecols=["protocol"])["protocol"].to_numpy()
        a = out.setdefault(fam, {"n": 0, "n1": 0, "files": {}})
        a["n"] += len(p); a["n1"] += int((p == 1).sum())
        a["files"][os.path.basename(path)] = share((p == 1).sum(), len(p))
    return {f: {**share(a["n1"], a["n"]), "files": a["files"]} for f, a in out.items()}


def main():
    sys.path.insert(0, os.path.join(THESIS_X, "src"))
    import config  # noqa: E402,F811
    import labels  # noqa: E402,F811
    t0 = time.time()
    res = {"a_released_csv_icmp_indicator": {}, "b_nfstream_protocol1_before_dedup": {}}
    for w, fn in (("iomt", lambda: csv_iomt(config)), ("iot", lambda: csv_iot(config, labels))):
        acc = fn()
        res["a_released_csv_icmp_indicator"][w] = {c: share(a[0], a[1]) for c, a in acc.items()}
        print("(a)", w, res["a_released_csv_icmp_indicator"][w], flush=True)
    r7 = json.load(open(os.path.join(C.RESULTS_DIR, "07_background_audit.json")))
    for w in ("iomt", "iot"):
        res["b_nfstream_protocol1_before_dedup"][w] = nfstream_pre_dedup(w)
        res["b_nfstream_protocol1_before_dedup"][w]["after_dedup_from_step07"] = {"share": r7["datasets"][w]["floods"]["ICMP"]["share_matching"], "n_rows": r7["datasets"][w]["floods"]["ICMP"]["n_survivors"]}
        print("(b)", w, {f: v["share"] if "share" in v else v for f, v in res["b_nfstream_protocol1_before_dedup"][w].items()}, flush=True)
    payload = {"meta": C.provenance(__file__, [os.path.join(C.RESULTS_DIR, "07_background_audit.json")], seeds=(), inputs_note="released CSVs (all rows, raw) and step-03 NFStream tables (raw)", hash_inputs=False),
               "label": "added after the background audit", **res}
    print("wrote", C.write_json("11_icmp_shares.json", payload), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
