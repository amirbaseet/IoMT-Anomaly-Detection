"""Step 07 - background audit of attack-labelled flows. Definitions: README "Changed after seeing arms 1-4", section B. No model.

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/07_background_audit.py

Basis = the experiment's own: step-03 NFStream flow tables, the 59-column key, global dedup over all classes (files in sorted-name
order, first kept), pair families only. CONTROL: survivor counts per family equal results/03_family_counts.json.
Floods (SYN, TCP, UDP, ICMP): share of surviving rows whose `protocol` matches the attack (ICMP=1, SYN/TCP=6, UDP=17).
Scans and ARP: per capture file, the dominant RAW-flow src_mac (most flows of the file before dedup); share of the file's surviving
rows that involve it as src_mac, as dst_mac and as either; whether it appears in any benign capture (raw flows) as src_mac / dst_mac.
"""
import importlib.util
import json
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

C.bootstrap()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

spec = importlib.util.spec_from_file_location("step05", os.path.join(C.SRC_DIR, "05_detection_arms.py"))
S5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S5)
FAMS = S5.FAMS
FLOOD_PROTO = {"SYN": 6, "TCP": 6, "UDP": 17, "ICMP": 1}
SCAN_ARP = ["port_scan", "os_scan", "ping_sweep", "vuln_scan", "arp"]
WHICH = {"iomt": "CICIoMT2024", "iot": "CICIoT2023"}


class MacIds:
    def __init__(self):
        self.reg = {}

    def ids(self, series):
        codes, uniques = pd.factorize(series.fillna("<NA>").astype(str))
        table = np.array([self.reg.setdefault(u, len(self.reg)) for u in uniques], dtype=np.int32)
        return table[codes]

    def name(self, i):
        return next(k for k, v in self.reg.items() if v == i)


def audit(which, step03):
    jb = S5.jobs(which)
    reg = MacIds()
    frames, src, dst, fid, cols = [], [], [], [], None
    for i, (path, cls) in enumerate(jb):
        head = pd.read_csv(path, nrows=0)
        this = C.model_input_columns(head.columns)
        cols = cols or this
        assert this == cols and len(this) == 59
        df = pd.read_csv(path, usecols=cols + ["src_mac", "dst_mac"], dtype={c: "float64" for c in cols})
        frames.append(df[cols]); src.append(reg.ids(df["src_mac"])); dst.append(reg.ids(df["dst_mac"]))
        fid.append(np.full(len(df), i, dtype=np.int16))
        print(f"  [{which} {i + 1}/{len(jb)}] {os.path.basename(path)} {len(df):,}", flush=True)
    X = pd.concat(frames, ignore_index=True); del frames
    src, dst, fid = np.concatenate(src), np.concatenate(dst), np.concatenate(fid)
    dup = X.duplicated(keep="first").to_numpy()
    proto = X["protocol"].to_numpy(); del X
    classes = [c for _, c in jb]
    files = [os.path.basename(p) for p, _ in jb]
    fam_of_file = np.array([FAMS.index(S5.CLASS2FAM[which][c]) if c in S5.CLASS2FAM[which] else -1 for c in classes], dtype=np.int8)
    fam = fam_of_file[fid]
    keep = (~dup) & (fam >= 0)
    counts = {FAMS[k]: int((keep & (fam == k)).sum()) for k in range(len(FAMS))}
    exp = {f: step03["pairs"][f][WHICH[which]]["rows_dedup_global"] for f in FAMS}
    control = {"got": counts, "expected_step03": exp, "passed": counts == exp}
    assert control["passed"], control

    ben_files = [i for i in range(len(files)) if fam_of_file[i] == FAMS.index("Benign")]
    ben_raw = np.isin(fid, ben_files)
    ben_src, ben_dst = set(np.unique(src[ben_raw]).tolist()), set(np.unique(dst[ben_raw]).tolist())

    out = {"control": control, "floods": {}, "scans_arp": {}}
    for f, want in FLOOD_PROTO.items():
        m = keep & (fam == FAMS.index(f))
        vals, cnt = np.unique(proto[m], return_counts=True)
        by_file = {}
        for i in np.where(fam_of_file == FAMS.index(f))[0]:
            mm = keep & (fid == i)
            by_file[files[i]] = {"n_survivors": int(mm.sum()), "share_matching": float((proto[mm] == want).mean()) if mm.any() else None}
        out["floods"][f] = {"attack_protocol": want, "n_survivors": int(m.sum()), "share_matching": float((proto[m] == want).mean()),
                            "protocol_counts": {str(int(v)): int(c) for v, c in zip(vals, cnt)}, "by_file": by_file}
    for f in SCAN_ARP:
        k = FAMS.index(f)
        tot = {"n": 0, "src": 0, "dst": 0, "either": 0}
        by_file = {}
        for i in np.where(fam_of_file == k)[0]:
            raw = fid == i
            top = np.bincount(src[raw]).argmax()
            raw_share = float((src[raw] == top).mean())
            mm = keep & raw
            n = int(mm.sum())
            s, d = int((src[mm] == top).sum()), int((dst[mm] == top).sum())
            e = int(((src[mm] == top) | (dst[mm] == top)).sum())
            by_file[files[i]] = {"dominant_raw_src_mac": reg.name(int(top)), "dominant_share_of_raw_flows_as_src": raw_share, "n_raw": int(raw.sum()), "n_survivors": n,
                                 "survivor_share_src": s / n if n else None, "survivor_share_dst": d / n if n else None, "survivor_share_either": e / n if n else None,
                                 "dominant_in_benign_as_src": bool(int(top) in ben_src), "dominant_in_benign_as_dst": bool(int(top) in ben_dst)}
            for key, v in (("n", n), ("src", s), ("dst", d), ("either", e)):
                tot[key] += v
        out["scans_arp"][f] = {"n_survivors": tot["n"], "pooled_share_src": tot["src"] / tot["n"], "pooled_share_dst": tot["dst"] / tot["n"],
                               "pooled_share_either": tot["either"] / tot["n"], "by_file": by_file}
    out["benign_capture_macs"] = {"n_distinct_src_mac": len(ben_src), "n_distinct_dst_mac": len(ben_dst)}
    return out


def main():
    step03 = json.load(open(os.path.join(C.RESULTS_DIR, "03_family_counts.json")))
    t0 = time.time()
    res = {w: audit(w, step03) for w in ("iomt", "iot")}
    for w, r in res.items():
        print(f"\n== {WHICH[w]} (control passed: {r['control']['passed']})")
        for f, v in r["floods"].items():
            print(f"  flood {f:5s} proto {v['attack_protocol']:>2}: {v['share_matching']:.4f} of {v['n_survivors']:,} survivors; counts {v['protocol_counts']}")
        for f, v in r["scans_arp"].items():
            print(f"  {f:10s} survivors {v['n_survivors']:,}: involve dominant MAC src {v['pooled_share_src']:.3f} dst {v['pooled_share_dst']:.3f} either {v['pooled_share_either']:.3f}")
            for fn, b in v["by_file"].items():
                print(f"      {fn}: {b['dominant_raw_src_mac']} (raw src share {b['dominant_share_of_raw_flows_as_src']:.3f}) either {b['survivor_share_either']:.3f} "
                      f"in benign as src={b['dominant_in_benign_as_src']} dst={b['dominant_in_benign_as_dst']}")
    payload = {"meta": C.provenance(__file__, [os.path.join(C.RESULTS_DIR, "03_family_counts.json")], seeds=(), inputs_note="step-03 NFStream feature CSVs (CICIoMT2024 nfstream version not recorded; CICIoT2023 6.6.0)"),
               "settings": "README.md 'Changed after seeing arms 1-4', section B", "datasets": res}
    print("wrote", C.write_json("07_background_audit.json", payload), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
