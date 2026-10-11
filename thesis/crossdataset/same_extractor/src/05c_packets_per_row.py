"""Step 05c - derived from results/05_csv_label_check.json only: NFStream packets of each pcap class / official CSV rows of that label.
If the official rows are windows of a fixed number of packets covering the WHOLE capture, this ratio is the same for every class,
background included. Run: ~/code/IoMT-Project/venv/bin/python -B -I src/05c_packets_per_row.py
"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

src = os.path.join(C.RESULTS_DIR, "05_csv_label_check.json")
r = json.load(open(src))
out = {k: {"official_rows": r["official_by_label"][k]["rows"], "nfstream_packets_of_class_files": r["our_nfstream_by_class"][k]["packets"],
           "packets_per_official_row": r["our_nfstream_by_class"][k]["packets"] / r["official_by_label"][k]["rows"]} for k in r["official_by_label"]}
payload = {"meta": C.provenance(__file__, [src], seeds=()), "what": "packets_per_official_row = NFStream bidirectional_packets summed over the class's feature files / official CSV rows with that label; "
           "caveat: NFStream drops packets it cannot assign to a flow, and our files for a class may be a subset of the official class", "by_label": out}
print("wrote", C.write_json("05c_packets_per_row.json", payload))
print({k: round(v["packets_per_official_row"], 2) for k, v in out.items()})
