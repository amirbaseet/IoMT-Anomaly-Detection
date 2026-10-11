"""Append the 'ICMP decision' results to ../README.md from results/10_macro4.json and 11_icmp_shares.json (nothing typed). Refuses to run twice.
Run: ~/code/IoMT-Project/venv/bin/python -B -I src/render_readme_icmp.py
"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

README = os.path.join(C.SAME_DIR, "README.md")
HEAD = "## ICMP decision: 4-family macro and ICMP indicator shares (added after the background audit)"
R = lambda n: json.load(open(os.path.join(C.RESULTS_DIR, n)))  # noqa: E731
ms = lambda v, d=3: "n/a" if v is None else f"{v['mean']:.{d}f} ± {(v['sd'] or 0):.{d}f}"  # noqa: E731


def main():
    t = open(README).read()
    assert HEAD not in t, "already appended"
    m, s = R("10_macro4.json"), R("11_icmp_shares.json")
    NF, CS = m["nfstream"], m["csv"]
    L = ["", HEAD, "",
         "Owner ruling: the pre-declared 5-family headline (SYN, TCP, UDP, ICMP, OS scan) stays. The 4-family macro (SYN, TCP, UDP, OS scan; ICMP excluded) is added beside it. No model was fitted: `src/10_macro4.py` reads `05_detection_arms.json` and `06_like_for_like.json`; `src/11_icmp_shares.py` reads the released CSVs and the NFStream tables. Mean ± sd (ddof=1) over seeds 42/43/44. No family was dropped for small support in any entry.", "",
         "| task | 5-family headline | **4-family macro** (added) |", "|---|---|---|"]
    for k, n in (("arm1_within_iomt", "arm 1 within CICIoMT2024 (NFStream)"), ("arm2_within_iot", "arm 2 within CICIoT2023 (NFStream)"),
                 ("arm3_iomt_to_iot", "arm 3 CICIoMT2024 → CICIoT2023 (NFStream)"), ("arm4_iot_to_iomt", "arm 4 CICIoT2023 → CICIoMT2024 (NFStream)")):
        L.append(f"| {n} | {ms(NF[k]['macro5_common_stored'])} | **{ms(NF[k]['macro4'])}** |")
    for fs, fn in (("shared38", "CSV 38"), ("shared31_drop7", "CSV 31")):
        for tk, n in (("base_iomt", "CICIoMT2024 official train → test"), ("base_iot", "CICIoT2023 held-out files (random split)"),
                      ("arm5a_iomt_to_iot", "5a CICIoMT2024 → CICIoT2023"), ("arm5b_iot_to_iomt", "5b CICIoT2023 → CICIoMT2024")):
            L.append(f"| {fn}: {n} | {ms(CS[fs][tk]['macro5_common_stored'])} | **{ms(CS[fs][tk]['macro4'])}** |")
    L += ["", "**Per-family AUC vs all benign, mean over 3 seeds** (SYN / TCP / UDP / OS scan):", "", "| direction | NFStream (arm 3 / arm 4) | CSV 38 (5a / 5b) | CSV 31 (5a / 5b) |", "|---|---|---|---|"]
    f4 = ("SYN", "TCP", "UDP", "os_scan")
    row = lambda e: " / ".join(f"{e['per_family_auc'][f]['mean']:.3f}" for f in f4)  # noqa: E731
    for lab, arm, tk in (("CICIoMT2024 → CICIoT2023", "arm3_iomt_to_iot", "arm5a_iomt_to_iot"), ("CICIoT2023 → CICIoMT2024", "arm4_iot_to_iomt", "arm5b_iot_to_iomt")):
        L.append(f"| {lab} | {row(NF[arm])} | {row(CS['shared38'][tk])} | {row(CS['shared31_drop7'][tk])} |")
    a, b = s["a_released_csv_icmp_indicator"], s["b_nfstream_protocol1_before_dedup"]
    L += ["", "**(a) Released CSVs, all rows (raw, not deduplicated): share of rows with the `ICMP` indicator ≥ 0.5**", "", "| dataset | class | rows | share ICMP ≥ 0.5 |", "|---|---|---|---|"]
    for w, wn in (("iomt", "CICIoMT2024"), ("iot", "CICIoT2023")):
        for c, v in a[w].items():
            L.append(f"| {wn} | {c} | {v['n_rows']:,} | {100 * v['share']:.4f}% |")
    L += ["", "**(b) NFStream flow tables, BEFORE dedup: share of flows with `protocol` = 1** (after-dedup value from `07_background_audit.json` beside it):", "", "| dataset | family | flows before dedup | protocol 1 before dedup | protocol 1 after dedup |", "|---|---|---|---|---|"]
    for w, wn in (("iomt", "CICIoMT2024"), ("iot", "CICIoT2023")):
        for fam in ("ICMP", "Benign"):
            v = b[w][fam]
            aft = f"{100 * b[w]['after_dedup_from_step07']['share']:.2f}% of {b[w]['after_dedup_from_step07']['n_rows']:,}" if fam == "ICMP" else "not computed"
            L.append(f"| {wn} | {fam} | {v['n_rows']:,} | {100 * v['share']:.2f}% | {aft} |")
    open(README, "a").write("\n".join(L) + "\n")
    print("appended", len(L), "lines")


if __name__ == "__main__":
    main()
