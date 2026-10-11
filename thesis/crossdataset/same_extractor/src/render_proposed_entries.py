"""Render ../proposed_numbers_map_entries.md from results/*.json (no number is typed by hand).

Not a numbered step: it only reads result files. Status is always "post-hoc" (rule: do not edit
deliverables/numbers_map.md; these rows are PROPOSALS).
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/render_proposed_entries.py
"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

OUT = os.path.join(C.SAME_DIR, "proposed_numbers_map_entries.md")
STATUS = "post-hoc"


def load(name):
    with open(os.path.join(C.RESULTS_DIR, name)) as fh:
        return json.load(fh)


def pct(x):
    return f"{100 * x:.2f}%"


def main():
    r1, r2, r3 = load("01_version_check.json"), load("02_column_audit.json"), load("03_family_counts.json")
    rows = []

    def add(number, what, measured_on, source, seeds="n/a (no randomness)"):
        rows.append((number, what, measured_on, seeds, STATUS, source))

    for base, v in r1["pcaps"].items():
        src = "results/01_version_check.json -> pcaps." + base
        on = f"{v['rows_stored']} stored / {v['rows_fresh']} fresh flows of {base}; nfstream 6.6.0 vs stored run (version not recorded)"
        add(f"{v['max_abs_diff']['model_input_columns']:g}", "max abs diff over the 59 model-input columns, stored vs fresh extraction, sorted by flow key", on, src)
        add(f"{v['max_abs_diff']['non_feature_numeric_columns']:g}", "max abs diff over non-feature numeric columns (only `id` differs)", on, src)
        z = v["all_zero_on_both_sides"]
        add(f"{z['n_model_input_columns']} of 59", "model-input columns that are all-zero on both sides (match there proves nothing)", on, src)
    c = r2["status_counts"]
    on2 = f"{r2['rows_measured_on']['iomt_benign_flows_raw']:,} CICIoMT2024 and {r2['rows_measured_on']['iot_benign_flows_raw']:,} CICIoT2023 raw benign flows"
    src2 = "results/02_column_audit.json -> status_counts"
    add(str(c["RATIO_GE_10X"]), "model-input columns whose benign medians differ >= 10x (both medians non-zero)", on2, src2)
    add(str(c["ONE_ZERO"]), "model-input columns where exactly one benign median is zero (flagged separately)", on2, src2)
    add(str(c["BOTH_ZERO"]), "model-input columns where both benign medians are zero", on2, src2)
    add(str(c["OK"]), "model-input columns within 10x with both medians non-zero", on2, src2)
    ctrl = r3["control_stage2c_reproduction"]
    add(f"{ctrl['got_raw']:,} -> {ctrl['got_dedup']:,}", "CONTROL: CICIoMT2024 raw -> deduplicated, reproduces stage2c dedup_report.json" + ("" if ctrl["passed"] else " (FAILED)"),
        "all 72 stored CICIoMT2024 captures, 59-column key", "results/03_family_counts.json -> control_stage2c_reproduction")
    for t, tot in r3["table_totals"].items():
        on = f"{r3['n_files'][t]} files of {t}, 59-column key, global dedup, files in sorted-name order"
        add(f"{tot['raw']:,}", f"{t} flows, raw", on, "results/03_family_counts.json -> table_totals")
        add(f"{tot['dedup_global']:,}", f"{t} flows, deduplicated (global, stage2c method)", on, "results/03_family_counts.json -> table_totals")
        bs = r3["benign_share_in_table"][t]
        add(pct(bs["raw"]["share"]), f"{t} benign share of the table, raw", on, "results/03_family_counts.json -> benign_share_in_table")
        add(pct(bs["dedup_global"]["share"]), f"{t} benign share of the table, deduplicated (global)", on, "results/03_family_counts.json -> benign_share_in_table")
        b = r3["benign_captures"][t]
        add(str(b["independent_benign_captures_under_rule"]), f"{t} independent benign captures under the stated contiguity rule ({b['contiguity_rule']})",
            f"{len(b['files_in_time_order'])} benign files, first/last flow timestamps", "results/03_family_counts.json -> benign_captures")
        add(str(b["distinct_src_mac_in_benign_union"]), f"{t} distinct src_mac across benign flows", f"{len(b['files_in_time_order'])} benign files, raw", "results/03_family_counts.json -> benign_captures")

    r4 = load("04_origin_control.json")
    for key, label in (("origin", "origin classifier, CICIoMT2024 vs CICIoT2023 deduplicated benign"),
                       ("reference_cicIoMT2024_train_vs_test", "reference: CICIoMT2024 Benign_train vs Benign_test"),
                       ("reference_cicIoT2023_first_vs_second_half", "reference: CICIoT2023 benign first vs second half")):
        e = r4["results"][key]
        o = e["summary"]["over_seeds"]
        on = f"{e['rows']:,} deduplicated benign flows, classes {e['class_sizes_total']}, {e['groups_total']} src_mac groups, 5 grouped folds"
        add(f"{o['auc_mean']:.3f} +/- {o['auc_sd']:.3f}", f"AUC, {label}", on, f"results/04_origin_control.json -> results.{key}", seeds="42, 43, 44")
        add(f"{o['bal_acc_mean']:.3f} +/- {o['bal_acc_sd']:.3f}", f"balanced accuracy at 0.5, {label}", on, f"results/04_origin_control.json -> results.{key}", seeds="42, 43, 44")

    r4b = load("04b_followup.json")
    pn, rr = r4b["permutation_null_origin"], r4b["rerun_ref_iomt_train_vs_test"]
    onp = f"{pn['n_valid']} permutations of the dataset label over 90 src_mac groups (23 / 67), fold seed 42"
    srcp = "results/04b_followup.json -> permutation_null_origin"
    add(f"{pn['null_auc_mean']:.3f} (sd {pn['null_auc_sd']:.3f})", "group-permutation null AUC of the origin task, mean and sd", onp, srcp, seeds="permutation seeds 10000+i; fold seed 42")
    add(f"{pn['null_p2_5']:.3f} to {pn['null_p97_5']:.3f}", "group-permutation null AUC, 2.5th to 97.5th percentile", onp, srcp, seeds="permutation seeds 10000+i; fold seed 42")
    add(f"{pn['observed_auc_seed42_mean_of_folds']:.3f}; null values >= it: {pn['n_null_ge_observed_seed42']} of {pn['n_valid']}", "observed origin AUC, fold seed 42, against the null", onp, srcp, seeds="42")
    add(f"{rr['auc_mean_over_seeds']:.3f} +/- {rr['auc_sd_over_seeds']:.3f}", "AUC, CICIoMT2024 Benign_train vs Benign_test, GroupKFold(shuffle) re-run (replaces the seed-invariant folds of step 04)",
        "10,540 deduplicated benign flows, 23 src_mac groups, 5 folds, 3 different partitions", "results/04b_followup.json -> rerun_ref_iomt_train_vs_test", seeds="42, 43, 44")

    r5 = load("05_detection_arms.json")
    names = {"arm1_within_iomt": "arm 1 within CICIoMT2024", "arm2_within_iot": "arm 2 within CICIoT2023",
             "arm3_iomt_to_iot": "arm 3 CICIoMT2024 -> CICIoT2023", "arm4_iot_to_iomt": "arm 4 CICIoT2023 -> CICIoMT2024"}
    for arm, sm in r5["summary_over_seeds"].items():
        e0 = r5["per_seed"][arm]["42"]
        ns = [r5["per_seed"][arm][s]["n_test"] for s in ("42", "43", "44")]
        n_txt = f"{e0['n_test']:,}" if len(set(ns)) == 1 else f"{ns[0]:,} / {ns[1]:,} / {ns[2]:,} (seeds 42 / 43 / 44; the held-out captures differ per seed)"
        on5 = (f"{n_txt} deduplicated test flows ({e0['n_benign']:,} benign), training capped at "
               f"{r5['caps']['attack_per_family']:,} per family / {r5['caps']['benign']:,} benign, 59 NFStream inputs, thesis XGB params")
        src5 = f"results/05_detection_arms.json -> summary_over_seeds.{arm}"
        def ms(k, sm=sm):
            v = sm[k]
            return f"{v['mean']:.4f} +/- {v['sd']:.4f}"
        add(ms("macro_common"), f"{names[arm]}: HEADLINE common-family macro AUC (SYN, TCP, UDP, ICMP, OS scan; each family vs all benign)", on5, src5, seeds="42, 43, 44")
        add(ms("macro_per_arm"), f"{names[arm]}: per-arm macro AUC (non-small-support families of this arm; secondary)", on5, src5, seeds="42, 43, 44")
        add(ms("pooled_auc"), f"{names[arm]}: pooled AUC (flood-dominated)", on5, src5, seeds="42, 43, 44")
        add(ms("balanced_accuracy"), f"{names[arm]}: pooled balanced accuracy at 0.5", on5, src5, seeds="42, 43, 44")
        add(ms("benign_recall"), f"{names[arm]}: benign recall at 0.5", on5, src5, seeds="42, 43, 44")
        if "overlap_share_full_table" in sm:
            add(ms("overlap_share_full_table"), f"{names[arm]}: share of test vectors also in the full training-side deduplicated table", on5, src5, seeds="42, 43, 44")
            add(ms("non_overlap_pooled_auc"), f"{names[arm]}: pooled AUC on the non-overlapping test rows", on5, src5, seeds="42, 43, 44")
    for arm, e in r5["sensitivity_uncapped_seed42"].items():
        c = r5["per_seed"][arm]["42"]
        on5 = f"{e['n_test']:,} test flows, UNCAPPED training ({e['train_rows']:,} rows), seed 42 only"
        src5 = f"results/05_detection_arms.json -> sensitivity_uncapped_seed42.{arm}"
        add(f"{e['macro_common']:.4f} (capped, same seed: {c['macro_common']:.4f})", f"{names[arm]}: common-family macro AUC, uncapped training", on5, src5, seeds="42")
        add(f"{e['pooled_auc']:.4f} (capped, same seed: {c['pooled_auc']:.4f})", f"{names[arm]}: pooled AUC, uncapped training", on5, src5, seeds="42")

    r6, r7, r8 = load("06_like_for_like.json"), load("07_background_audit.json"), load("08_sensitivities.json")
    tn = {"base_iomt": "CICIoMT2024 official train->test baseline", "base_iot": "CICIoT2023 19-of-63 files held-out baseline (random, not capture-disjoint)",
          "arm5a_iomt_to_iot": "arm 5a CICIoMT2024 -> CICIoT2023", "arm5b_iot_to_iomt": "arm 5b CICIoT2023 -> CICIoMT2024"}
    for fs, fn in (("shared38", "CSV shared 38 features"), ("shared31_drop7", "CSV shared 31 features (7 dropped, chosen with CICIoMT2024 test labels)")):
        for tk, name in tn.items():
            s = r6["summary_over_seeds"][fs][tk]
            e0 = r6["per_seed"][fs][tk]["42"]
            on6 = f"{fn}; {e0['n_test']:,} deduplicated test rows at seed 42 ({e0['n_benign']:,} benign), training capped at {r6['caps']['attack_per_family']:,} per family / {r6['caps']['benign']:,} benign, thesis XGB params"
            src6 = f"results/06_like_for_like.json -> summary_over_seeds.{fs}.{tk}"
            for k, lab in (("macro_common", "HEADLINE common-family macro AUC"), ("macro_per_arm", "per-arm macro AUC"), ("pooled_auc", "pooled AUC"),
                           ("balanced_accuracy", "balanced accuracy at 0.5"), ("benign_recall", "benign recall at 0.5")):
                v = s[k]
                add(f"{v['mean']:.4f} +/- {v['sd']:.4f}", f"{name}: {lab}", on6, src6, seeds="42, 43, 44")
    for d, dn in (("iomt", "CICIoMT2024"), ("iot", "CICIoT2023")):
        fl = r7["datasets"][d]["floods"]
        for f in ("SYN", "TCP", "UDP", "ICMP"):
            add(pct(fl[f]["share_matching"]), f"{dn} {f} flood: share of surviving rows whose protocol matches the attack (protocol {fl[f]['attack_protocol']})",
                f"{fl[f]['n_survivors']:,} deduplicated NFStream flows, 59-column key", f"results/07_background_audit.json -> datasets.{d}.floods.{f}")
        sc = r7["datasets"][d]["scans_arp"]
        for f, v in sc.items():
            add(pct(v["pooled_share_either"]), f"{dn} {f}: share of surviving rows involving the capture's dominant raw src_mac (src or dst; files pooled)",
                f"{v['n_survivors']:,} deduplicated NFStream flows in {len(v['by_file'])} capture file(s); the dominant MAC also appears in a benign capture in every file", f"results/07_background_audit.json -> datasets.{d}.scans_arp.{f}")
    r10, r11 = load("10_macro4.json"), load("11_icmp_shares.json")
    lab10 = "4-family macro AUC (SYN, TCP, UDP, OS scan; ICMP excluded; added after the background audit)"
    for k, v in r10["nfstream"].items():
        add(f"{v['macro4']['mean']:.4f} +/- {v['macro4']['sd']:.4f}", f"{k}: {lab10}", "no model fitted; computed from results/05_detection_arms.json per-family AUCs", f"results/10_macro4.json -> nfstream.{k}", seeds="42, 43, 44")
    for fs, d in r10["csv"].items():
        for tk, v in d.items():
            add(f"{v['macro4']['mean']:.4f} +/- {v['macro4']['sd']:.4f}", f"CSV {fs} {tk}: {lab10}", "no model fitted; computed from results/06_like_for_like.json per-family AUCs", f"results/10_macro4.json -> csv.{fs}.{tk}", seeds="42, 43, 44")
    for w, wn in (("iomt", "CICIoMT2024"), ("iot", "CICIoT2023")):
        for c, v in r11["a_released_csv_icmp_indicator"][w].items():
            add(pct(v["share"]), f"{wn} released CSV, class {c}: share of rows with ICMP indicator >= 0.5", f"{v['n_rows']:,} raw rows, not deduplicated", f"results/11_icmp_shares.json -> a_released_csv_icmp_indicator.{w}.{c}")
        for fam in ("ICMP", "Benign"):
            v = r11["b_nfstream_protocol1_before_dedup"][w][fam]
            add(pct(v["share"]), f"{wn} NFStream {fam} family: share of flows with protocol 1, BEFORE dedup", f"{v['n_rows']:,} raw flows", f"results/11_icmp_shares.json -> b_nfstream_protocol1_before_dedup.{w}.{fam}")

    nm = {"arm1_within_iomt": "arm 1", "arm2_within_iot": "arm 2", "arm3_iomt_to_iot": "arm 3", "arm4_iot_to_iomt": "arm 4"}
    for arm, s in r8["a_protocol_matched_flood_test_rows"]["summary_over_seeds"].items():
        on8 = "flood attack TEST rows limited to protocol-matching rows; benign test rows and training unchanged; models re-fitted, unrestricted control diff " + f"{r8['control_max_abs_diff']:g}"
        src8 = f"results/08_sensitivities.json -> a_protocol_matched_flood_test_rows.summary_over_seeds.{arm}"
        for k, lab in (("flood_macro_unrestricted", "flood macro (SYN, TCP, UDP, ICMP) AUC, all rows"), ("flood_macro_restricted", "flood macro AUC, protocol-matching flood rows only"),
                       ("macro_common_restricted", "common-family macro AUC, protocol-matching flood rows only")):
            add(f"{s[k]['mean']:.4f} +/- {s[k]['sd']:.4f}", f"{nm[arm]} sensitivity: {lab}", on8, src8, seeds="42, 43, 44")
    for arm, b in r8["b_flood_only_training_seed42"].items():
        src8 = f"results/08_sensitivities.json -> b_flood_only_training_seed42.{arm}"
        on8 = f"{b['flood_only']['n_test']:,} test flows ({b['flood_only']['n_benign']:,} benign); training = SYN, TCP, UDP, ICMP + benign only ({b['train_rows_flood_only']:,} rows)"
        add(f"{b['flood_only']['benign_recall']:.4f} (current, all families in training: {b['current_step05']['benign_recall']:.4f})", f"{nm[arm]} sensitivity: benign recall at 0.5, flood-only training", on8, src8, seeds="42")

    lines = ["# Proposed numbers_map entries (same-extractor experiment)", "",
             "PROPOSALS only. `deliverables/numbers_map.md` was not edited. Every row is generated by "
             "`src/render_proposed_entries.py` from `results/*.json`; status is `post-hoc` for all rows.",
             "Per-pair and per-class counts are not headline numbers: see `results/03_family_counts.csv`.", "",
             "| number | what it is | measured on | seeds | status | source |", "|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in rows]
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
