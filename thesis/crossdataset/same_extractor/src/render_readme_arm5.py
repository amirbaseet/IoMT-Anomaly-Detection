"""Append the arm 5 / audit / sensitivity RESULT tables to ../README.md. Every number is read from results/*.json (nothing typed).

Not a numbered step. Refuses to run twice (the section heading is the guard).
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/render_readme_arm5.py
"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

README = os.path.join(C.SAME_DIR, "README.md")
HEAD = "## Results after the declaration: arm 5, background audit, two sensitivities"
FAMS = ["SYN", "TCP", "UDP", "ICMP", "port_scan", "os_scan", "ping_sweep", "vuln_scan", "arp", "Benign"]
NAMES = {"base_iomt": "CICIoMT2024 official train → test", "base_iot": "CICIoT2023 19 of 63 files held out (random, not capture-disjoint)",
         "arm5a_iomt_to_iot": "5a CICIoMT2024 → CICIoT2023", "arm5b_iot_to_iomt": "5b CICIoT2023 → CICIoMT2024"}
SETS = {"shared38": "shared 38 (primary)", "shared31_drop7": "shared 31 (drop 7, secondary)"}


def load(n):
    return json.load(open(os.path.join(C.RESULTS_DIR, n)))


def ms(v, d=3):
    return "n/a" if v is None else f"{v['mean']:.{d}f} ± {(v['sd'] or 0):.{d}f}"


def main():
    t = open(README).read()
    assert HEAD not in t, "section already appended"
    r6, r7, r8, r5 = load("06_like_for_like.json"), load("07_background_audit.json"), load("08_sensitivities.json"), load("05_detection_arms.json")
    lc = json.load(open(os.path.join(C.THESIS_REPO, "thesis", "crossdataset", "artifacts", "label_counts.json")))
    L = ["", HEAD, "",
         "Run 2026-10-11 (UTC dates in the result files), after the declaration above. All `post-hoc`. Scripts: `src/06_like_for_like.py`, `src/07_background_audit.py`, `src/08_sensitivities.py`; results `results/06_like_for_like.json`, `07_background_audit.json`, `08_sensitivities.json`.", "",
         "### Arm 5 — official CSV features, same family subset", "",
         "Basis: deduplicated CSV rows, the 10 `PAIRS` classes only, training capped at 20,000 per family / 40,000 benign, test uncapped, mean ± sd over seeds 42/43/44, thesis XGB params. **Headline = common macro** (SYN, TCP, UDP, ICMP, OS scan).", "",
         "| feature set | task | test flows (benign), seed 42 | **common macro AUC** | per-arm macro | pooled AUC | balanced acc @0.5 | benign recall @0.5 | overlap share | non-overlap pooled AUC |", "|---|---|---|---|---|---|---|---|---|---|"]
    for fs, fl in SETS.items():
        for tk, tn in NAMES.items():
            s, e = r6["summary_over_seeds"][fs][tk], r6["per_seed"][fs][tk]["42"]
            ov = ms(s["overlap_share_full_table"], 4) if "overlap_share_full_table" in s else "n/a (dedup)"
            no = ms(s["non_overlap_pooled_auc"]) if "non_overlap_pooled_auc" in s else "n/a"
            L.append(f"| {fl} | {tn} | {e['n_test']:,} ({e['n_benign']:,}) | **{ms(s['macro_common'])}** | {ms(s['macro_per_arm'])} | {ms(s['pooled_auc'])} | {ms(s['balanced_accuracy'])} | {ms(s['benign_recall'])} | {ov} | {no} |")
    n_iot = [r6["per_seed"]["shared38"]["base_iot"][str(s)]["n_test"] for s in C.SEEDS]
    L += ["", f"The `base_iot` test set differs per seed (test rows {n_iot[0]:,} / {n_iot[1]:,} / {n_iot[2]:,} for seeds 42 / 43 / 44); the other three tasks have a fixed test set.", "",
          "**Raw → deduplicated rows per family** (dedup key = the 38 shared columns, global within the dataset, first kept; the CICIoT2023 raw counts also equal the official label counts in `thesis/crossdataset/artifacts/label_counts.json`: "
          + ("**yes, all 10**" if all(r6["dedup"]["iot"]["raw_by_family"][f] == lc[k] for f, k in zip(FAMS, ["DOS-SYN_FLOOD", "DOS-TCP_FLOOD", "DOS-UDP_FLOOD", "DDOS-ICMP_FLOOD", "RECON-PORTSCAN", "RECON-OSSCAN", "RECON-PINGSWEEP", "VULNERABILITYSCAN", "MITM-ARPSPOOFING", "BENIGN"])) else "**NO, check**") + ").", "",
          "| family | CICIoMT2024 raw → dedup | CICIoT2023 raw → dedup |", "|---|---|---|"]
    for f in FAMS:
        a, b = r6["dedup"]["iomt"], r6["dedup"]["iot"]
        L.append(f"| {f} | {a['raw_by_family'][f]:,} → {a['dedup_by_family'][f]:,} | {b['raw_by_family'][f]:,} → {b['dedup_by_family'][f]:,} |")
    c = r6["dedup"]["iomt"]["dedup_control_hash_vs_exact"]
    L += ["", f"Controls: for CICIoMT2024 the hash dedup and `DataFrame.duplicated` mark the identical rows ({c['dup_exact']:,} duplicates, masks identical: {c['identical_masks']}). Non-finite cells left after dedup: CICIoMT2024 {r6['dedup']['iomt']['n_nonfinite_cells_after_dedup']}, CICIoT2023 {r6['dedup']['iot']['n_nonfinite_cells_after_dedup']} (filled with the training-sample median). Rows in all classes after dedup, CICIoT2023: {r6['dedup']['iot']['rows_dedup_all_classes']:,} of {r6['dedup']['iot']['rows_raw_all_members']:,}.", "",
          "**Same direction, NFStream vs CSV features** (common macro, mean ± sd over 3 seeds; the test sets differ, so this is a comparison of settings, not of one model on one test set):", "",
          "| direction | NFStream 59 features (arm 3/4) | CSV shared 38 (arm 5) | CSV shared 31 (arm 5, secondary) |", "|---|---|---|---|"]
    for lab, arm, tk in (("CICIoMT2024 → CICIoT2023", "arm3_iomt_to_iot", "arm5a_iomt_to_iot"), ("CICIoT2023 → CICIoMT2024", "arm4_iot_to_iomt", "arm5b_iot_to_iomt")):
        L.append(f"| {lab} | {ms(r5['summary_over_seeds'][arm]['macro_common'])} | {ms(r6['summary_over_seeds']['shared38'][tk]['macro_common'])} | {ms(r6['summary_over_seeds']['shared31_drop7'][tk]['macro_common'])} |")
    L += ["", "### Background audit (no model; NFStream flows, 59-column dedup basis, pair families)", "",
          f"Control: survivor counts per family equal `03_family_counts.json` in both datasets (passed: {all(d['control']['passed'] for d in r7['datasets'].values())}).", "",
          "**Floods — share of surviving rows whose `protocol` matches the attack** (ICMP = 1, SYN/TCP = 6, UDP = 17):", "", "| family | CICIoMT2024 matching / survivors | CICIoT2023 matching / survivors |", "|---|---|---|"]
    for f in ("SYN", "TCP", "UDP", "ICMP"):
        a, b = r7["datasets"]["iomt"]["floods"][f], r7["datasets"]["iot"]["floods"][f]
        L.append(f"| {f} (protocol {a['attack_protocol']}) | {100 * a['share_matching']:.2f}% of {a['n_survivors']:,} | {100 * b['share_matching']:.2f}% of {b['n_survivors']:,} |")
    L += ["", "**Scans and ARP — per capture file: dominant raw-flow `src_mac` and share of the file's surviving rows that involve it** (either = as `src_mac` or `dst_mac`):", "",
          "| dataset | family | capture file | dominant raw `src_mac` | its share of RAW flows as src | survivors | survivors involving it (either) | in a benign capture as src / dst |", "|---|---|---|---|---|---|---|---|"]
    for w, wn in (("iomt", "CICIoMT2024"), ("iot", "CICIoT2023")):
        for f, v in r7["datasets"][w]["scans_arp"].items():
            for fn, b in v["by_file"].items():
                L.append(f"| {wn} | {f} | {fn} | `{b['dominant_raw_src_mac']}` | {100 * b['dominant_share_of_raw_flows_as_src']:.1f}% | {b['n_survivors']:,} | {100 * b['survivor_share_either']:.1f}% | {b['dominant_in_benign_as_src']} / {b['dominant_in_benign_as_dst']} |")
    L += ["", "### Sensitivity (a): arms 1–4, flood attack TEST rows limited to protocol-matching rows (3 seeds)", "",
          f"Control: the unrestricted re-fit reproduces `05_detection_arms.json` (max abs difference over pooled AUC, both macros, balanced accuracy, benign recall, all 12 fits: **{r8['control_max_abs_diff']:g}**).", "",
          "| arm | flood macro (4 floods), all rows → matching rows | common macro, all → matching | ICMP AUC, all → matching | ICMP test rows, seed 42: all → matching |", "|---|---|---|---|---|"]
    an = {"arm1_within_iomt": "1 within CICIoMT2024", "arm2_within_iot": "2 within CICIoT2023", "arm3_iomt_to_iot": "3 CICIoMT2024 → CICIoT2023", "arm4_iot_to_iomt": "4 CICIoT2023 → CICIoMT2024"}
    for arm, s in r8["a_protocol_matched_flood_test_rows"]["summary_over_seeds"].items():
        p = r8["a_protocol_matched_flood_test_rows"]["per_seed"][arm]["42"]
        sm = "".join(f" (SMALL_SUPPORT on {k})" for k, v in p["restricted"]["flood_small_support"].items() if v)
        L.append(f"| {an[arm]} | {ms(s['flood_macro_unrestricted'])} → {ms(s['flood_macro_restricted'])} | {ms(s['macro_common_unrestricted'])} → {ms(s['macro_common_restricted'])} | "
                 f"{ms(s['flood_auc_unrestricted']['ICMP'])} → {ms(s['flood_auc_restricted']['ICMP'])}{sm} | {p['flood_rows']['ICMP']['n_test']:,} → {p['flood_rows']['ICMP']['n_protocol_matching']:,} |")
    L += ["", "Per-family flood AUC (all rows → matching rows), mean over 3 seeds:", "", "| arm | SYN | TCP | UDP | ICMP |", "|---|---|---|---|---|"]
    for arm, s in r8["a_protocol_matched_flood_test_rows"]["summary_over_seeds"].items():
        L.append(f"| {an[arm]} | " + " | ".join(f"{s['flood_auc_unrestricted'][f]['mean']:.3f} → {s['flood_auc_restricted'][f]['mean']:.3f}" for f in ("SYN", "TCP", "UDP", "ICMP")) + " |")
    L += ["", "### Sensitivity (b): arms 2 and 4, seed 42, training on the four flood families + benign only", "",
          "| arm | training rows (flood-only / current) | benign recall @0.5, current → flood-only | balanced acc | pooled AUC | common macro | per-arm macro |", "|---|---|---|---|---|---|---|"]
    for arm, b in r8["b_flood_only_training_seed42"].items():
        c0, c1 = b["current_step05"], b["flood_only"]
        L.append(f"| {an[arm]} | {b['train_rows_flood_only']:,} / {b['train_rows_current']:,} | **{c0['benign_recall']:.4f} → {c1['benign_recall']:.4f}** | {c0['balanced_accuracy']:.4f} → {c1['balanced_accuracy']:.4f} | "
                 f"{c0['pooled_auc']:.4f} → {c1['pooled_auc']:.4f} | {c0['macro_common']:.4f} → {c1['macro_common']:.4f} | {c0['macro_per_arm']:.4f} → {c1['macro_per_arm']:.4f} |")
    open(README, "a").write("\n".join(L) + "\n")
    print("appended", len(L), "lines")


if __name__ == "__main__":
    main()
