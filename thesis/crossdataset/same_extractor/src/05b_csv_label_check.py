"""Step 05b - pre-check before arm 5: how do the official CICIoT2023 CSVs label benign background in attack captures?

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/05b_csv_label_check.py

What is possible: the official rows carry NO timestamps, addresses, ports or file names, and the 63 Merged files mix
captures, so a row cannot be traced to a pcap. What is measurable is INDIRECT: compare (a) the protocol composition of
an attack pcap's NFStream flows (our table, labelled by file name) with (b) the protocol composition of the official
rows carrying that attack's label. If the pcap holds background traffic of a protocol the attack cannot produce
(e.g. UDP/TCP in a ping sweep) and the official rows of that class show none of it, the official labelling removed or
re-labelled the background; if they show a comparable share, it stayed under the attack label. This is inference, not proof.

Deviation from the plan text: `read_member` (inverse_full_run.py) returns only a binary y and a float32 X, so the
per-class label is lost; this script streams the same zip members with the same label normalisation
(`labels.normalize_ciciot2023`, TRAP 2) and keeps the class.
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
sys.path.insert(0, os.path.join(C.THESIS_REPO, "thesis", "crossdataset", "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ZIP = os.path.join(C.SSD, "CICIOT23", "csv", "MERGED_CSV.zip")
FRAC = ["TCP", "UDP", "ICMP", "ARP", "DNS", "DHCP", "HTTP", "HTTPS", "SSH", "Telnet", "SMTP", "IRC", "IGMP", "LLC"]
CHUNK = 250_000
# official label (normalised) -> our feature files of that class
CHECK = {"RECON-PINGSWEEP": ["Recon-PingSweep.pcap.csv"], "MITM-ARPSPOOFING": ["MITM-ArpSpoofing.pcap.csv", "MITM-ArpSpoofing1.pcap.csv"],
         "RECON-OSSCAN": ["Recon-OSScan.pcap.csv"], "RECON-PORTSCAN": ["Recon-PortScan.pcap.csv"],
         "VULNERABILITYSCAN": ["VulnerabilityScan.pcap.csv"], "BENIGN": ["BenignTraffic.pcap.csv", "BenignTraffic1.pcap.csv", "BenignTraffic2.pcap.csv", "BenignTraffic3.pcap.csv"]}
PROTO_NAME = {1: "ICMP", 6: "TCP", 17: "UDP", 0: "0(non-IP, e.g. ARP)"}


def official_rows():
    import labels
    zf = zipfile.ZipFile(ZIP)
    members = sorted(i.filename for i in zf.infolist() if i.filename.endswith(".csv") and "/._" not in i.filename)
    acc = {}
    t0 = time.time()
    for n, m in enumerate(members):
        with zf.open(m) as fh:
            for ch in pd.read_csv(fh, usecols=FRAC + ["Label"], dtype={**{c: "float32" for c in FRAC}, "Label": str}, chunksize=CHUNK):
                lab = ch["Label"].map(labels.normalize_ciciot2023)
                for name, g in ch[FRAC].groupby(lab.to_numpy()):
                    a = acc.setdefault(name, {"rows": 0, **{f"sum_{c}": 0.0 for c in FRAC}, **{f"pos_{c}": 0 for c in FRAC}, "mostly_icmp": 0, "mostly_tcp": 0, "mostly_udp": 0})
                    a["rows"] += len(g)
                    for c in FRAC:
                        a[f"sum_{c}"] += float(g[c].sum()); a[f"pos_{c}"] += int((g[c] > 0).sum())
                    a["mostly_icmp"] += int((g["ICMP"] >= 0.5).sum()); a["mostly_tcp"] += int((g["TCP"] >= 0.5).sum()); a["mostly_udp"] += int((g["UDP"] >= 0.5).sum())
        print(f"  [{n + 1}/{len(members)}] {m} {time.time() - t0:.0f}s", flush=True)
    return members, acc


def our_flows(files):
    frames = [pd.read_csv(os.path.join(C.FEATURES_IOT, f), usecols=["protocol", "bidirectional_packets"]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    by = df.groupby("protocol")["bidirectional_packets"].agg(["size", "sum"])
    tot_f, tot_p = int(by["size"].sum()), int(by["sum"].sum())
    return {"flows": tot_f, "packets": tot_p,
            "by_protocol": {PROTO_NAME.get(int(p), str(int(p))): {"flows": int(r["size"]), "flow_share": float(r["size"] / tot_f),
                                                                    "packets": int(r["sum"]), "packet_share": float(r["sum"] / tot_p)} for p, r in by.iterrows()}}


def main():
    members, acc = official_rows()
    off = {}
    for lab, a in acc.items():
        n = a["rows"]
        off[lab] = {"rows": n, "mean_fraction": {c: a[f"sum_{c}"] / n for c in FRAC}, "share_rows_with_positive": {c: a[f"pos_{c}"] / n for c in FRAC},
                    "share_rows_mostly_icmp": a["mostly_icmp"] / n, "share_rows_mostly_tcp": a["mostly_tcp"] / n, "share_rows_mostly_udp": a["mostly_udp"] / n}
    ours = {lab: our_flows(fs) for lab, fs in CHECK.items()}
    payload = {"meta": C.provenance(__file__, [ZIP], seeds=(), inputs_note="official CICIoT2023 MERGED_CSV.zip; our side = step-02/03 NFStream feature CSVs (6.6.0)", hash_inputs=False),
               "zip_members_read": len(members), "official_labels_found": sorted(acc),
               "official_rows_total": int(sum(a["rows"] for a in acc.values())),
               "method_note": "indirect comparison, see script docstring; the official CSV has no timestamp/address/port/file-name column",
               "official_by_label": {k: off[k] for k in CHECK if k in off}, "our_nfstream_by_class": ours,
               "official_labels_missing": [k for k in CHECK if k not in off]}
    print("wrote", C.write_json("05_csv_label_check.json", payload))


if __name__ == "__main__":
    main()
