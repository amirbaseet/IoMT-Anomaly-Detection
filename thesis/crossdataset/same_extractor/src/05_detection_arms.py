"""Step 05 - detection arms 1-4 (+ uncapped sensitivity of arms 3-4). Plan: README "Detection stage".

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/05_detection_arms.py --scratch <dir>

Everything below was fixed in the README before the first run. Binary benign(0)/attack(1), deduplicated rows,
the 59 model inputs, thesis XGB via models._xgb (config.XGB_PARAMS_BASE) with random_state=seed, no class weights.
Training cap: 20,000 per family, 40,000 benign (read from config.ATTACK_CAP / BENIGN_CAP), drawn with the seed.
Test rows uncapped. Seeds 42, 43, 44. Arm 5 is NOT run here.

Cache (scratch only, never in the repo): per dataset, the kept rows (pair families, global-dedup survivors) as .npy.
CONTROL: kept counts per family must equal results/03_family_counts.json.
"""
import argparse
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
from sklearn.metrics import roc_auc_score  # noqa: E402

TS = "bidirectional_first_seen_ms"
FAMS = ["SYN", "TCP", "UDP", "ICMP", "port_scan", "os_scan", "ping_sweep", "vuln_scan", "arp", "Benign"]
ATTACK_FAMS = FAMS[:-1]
COMMON = ["SYN", "TCP", "UDP", "ICMP", "os_scan"]       # owner decision: not small-support in either dataset
SMALL_TOTAL, SMALL_TEST = 1000, 300
EMBARGO_MS = 60_000.0                                   # step-03 contiguity constant (author's choice)
CLASS2FAM = {"iomt": {c: f for f, v in C.PAIRS.items() for c in v["iomt"]},
             "iot": {c: f for f, v in C.PAIRS.items() for c in v["iot"]}}
HASH_KEYS = ("0123456789123456", "fedcba9876543210")


def load_config_and_models():
    sys.path.insert(0, os.path.join(C.THESIS_REPO, "thesis", "crossdataset", "src"))
    import config  # noqa: WPS433
    import models  # noqa: WPS433
    return config, models


def jobs(which):
    if which == "iomt":
        from stage2_device_disjoint import EXCLUDE_SUBSTR, filename_to_class
        out = []
        for f in sorted(os.listdir(C.FEATURES_IOMT)):
            if f.endswith(".csv") and not any(s in f for s in EXCLUDE_SUBSTR):
                cls = filename_to_class(f)
                if cls is not None:
                    out.append((os.path.join(C.FEATURES_IOMT, f), cls))
        return out
    j = [(os.path.join(C.FEATURES_IOT, m["file"] + ".csv"), m["class"]) for m in C.iot_manifest() if m.get("status") == "ok"]
    return sorted(j, key=lambda x: os.path.basename(x[0]))


def row_hashes(X):
    """Two independent 64-bit pandas row hashes over the 59 float64 values (128 bits in total)."""
    out = np.empty((len(X), 2), dtype=np.uint64)
    for s in range(0, len(X), 500_000):
        df = pd.DataFrame(X[s:s + 500_000])
        for k, key in enumerate(HASH_KEYS):
            out[s:s + 500_000, k] = pd.util.hash_pandas_object(df, index=False, hash_key=key).to_numpy()
    return out


def build_cache(which, cache_dir, step03):
    """Load every file (dedup is global over ALL classes), keep pair-family survivors, save .npy."""
    meta_path = os.path.join(cache_dir, "meta.json")
    if os.path.exists(meta_path):
        return json.load(open(meta_path))
    os.makedirs(cache_dir, exist_ok=True)
    jb = jobs(which)
    frames, ts, mac, fid, cols = [], [], [], [], None
    for i, (path, cls) in enumerate(jb):
        head = pd.read_csv(path, nrows=0)
        this = C.model_input_columns(head.columns)
        assert len(this) == 59
        cols = cols or this
        assert this == cols
        df = pd.read_csv(path, usecols=cols + [TS, "src_mac"], dtype={c: "float64" for c in cols})
        frames.append(df[cols]); ts.append(df[TS].to_numpy()); mac.append(df["src_mac"].fillna("<NA>").to_numpy().astype(str))
        fid.append(np.full(len(df), i, dtype=np.int16))
        print(f"  [{which} {i + 1}/{len(jb)}] {os.path.basename(path)} {len(df):,}", flush=True)
    X = pd.concat(frames, ignore_index=True); del frames
    fid = np.concatenate(fid); ts = np.concatenate(ts); mac = np.concatenate(mac)
    dup = X.duplicated(keep="first").to_numpy()
    files = [os.path.basename(p) for p, _ in jb]
    classes = [c for _, c in jb]
    fam_of_file = np.array([FAMS.index(CLASS2FAM[which][c]) if c in CLASS2FAM[which] else -1 for c in classes], dtype=np.int8)
    fam = fam_of_file[fid]
    keep = (~dup) & (fam >= 0)
    Xk = X.to_numpy()[keep]; del X
    assert np.isfinite(Xk).all(), "NaN/inf present in the 59 inputs: the planned train-side median imputer would be needed"
    np.save(os.path.join(cache_dir, "X.npy"), Xk)
    np.save(os.path.join(cache_dir, "fam.npy"), fam[keep]); np.save(os.path.join(cache_dir, "fid.npy"), fid[keep])
    np.save(os.path.join(cache_dir, "ts.npy"), ts[keep]); np.save(os.path.join(cache_dir, "mac.npy"), mac[keep])
    np.save(os.path.join(cache_dir, "hash.npy"), row_hashes(Xk))
    counts = {FAMS[k]: int((fam[keep] == k).sum()) for k in range(len(FAMS))}
    exp = {f: step03["pairs"][f][{"iomt": "CICIoMT2024", "iot": "CICIoT2023"}[which]]["rows_dedup_global"] for f in FAMS if f in step03["pairs"]}
    control = {"got": counts, "expected_step03": exp, "passed": counts == exp}
    meta = {"files": files, "classes": classes, "cols": cols, "control": control, "rows_raw_all_classes": int(len(fid)),
            "rows_kept": int(keep.sum())}
    json.dump(meta, open(meta_path, "w"))
    if not control["passed"]:
        raise SystemExit(f"dedup control failed for {which}: {control}")
    return meta


def open_cache(which, cache_dir):
    g = lambda n: np.load(os.path.join(cache_dir, n + ".npy"), mmap_mode="r")  # noqa: E731
    return {k: g(k) for k in ("X", "fam", "fid", "ts", "hash")}


def drop_embargo(ts, side, a, b, cut):
    """Rows of masks a/b within EMBARGO_MS of `cut` (a ts, b ts) are removed from the split."""
    side[(a | b) & (np.abs(ts - cut) <= EMBARGO_MS)] = -1


def split_arm1(d, files):
    """CICIoMT2024: official train/test captures; benign = Benign_train vs Benign_test time split with embargo."""
    fn = np.array(files)[np.asarray(d["fid"])]
    side = np.where(np.char.find(fn.astype(str), "_test.pcap") >= 0, 1, 0).astype(np.int8)
    fam, ts = np.asarray(d["fam"]), np.asarray(d["ts"])
    ben = fam == FAMS.index("Benign")
    t_tr, t_te = ts[ben & (side == 0)], ts[ben & (side == 1)]
    first_is_train = t_tr.min() < t_te.min()
    cut = t_te.min() if first_is_train else t_tr.min()
    info = {"benign_train_ts": [float(t_tr.min()), float(t_tr.max())], "benign_test_ts": [float(t_te.min()), float(t_te.max())], "cut_ms": float(cut)}
    drop_embargo(ts, side, ben & (side == (0 if first_is_train else 1)), ben & (side == (1 if first_is_train else 0)), cut)
    return side, info


def split_arm2(d, files, seed):
    """CICIoT2023: >=2-capture families split by file (~30% of files, >=1), single-capture families and benign by time."""
    fam, ts, fid = np.asarray(d["fam"]), np.asarray(d["ts"]), np.asarray(d["fid"])
    side = np.zeros(len(fam), dtype=np.int8)
    rng = np.random.default_rng(seed)
    info = {}
    for k, name in enumerate(FAMS):
        m = fam == k
        fids = sorted(set(fid[m].tolist()))
        if len(fids) >= 2 and name != "Benign":
            n_te = max(1, round(0.3 * len(fids)))
            te = set(rng.permutation(fids)[:n_te].tolist())
            side[m & np.isin(fid, list(te))] = 1
            info[name] = {"mode": "file", "n_files": len(fids), "test_files": sorted(files[i] for i in te)}
        else:
            cut = np.percentile(ts[m], 70)
            side[m & (ts > cut)] = 1
            drop_embargo(ts, side, m & (ts <= cut), m & (ts > cut), cut)
            info[name] = {"mode": "time", "n_files": len(fids), "cut_ms": float(cut)}
    return side, info


def capped_train_idx(fam, idx_train, seed, cap_attack, cap_benign):
    """Per-family draw without replacement from the training rows; None caps = keep everything."""
    rng = np.random.default_rng(seed)
    sel = []
    for k, name in enumerate(FAMS):
        pool = idx_train[fam[idx_train] == k]
        cap = cap_benign if name == "Benign" else cap_attack
        if cap is not None and len(pool) > cap:
            pool = rng.choice(pool, size=cap, replace=False)
        sel.append(np.sort(pool))
    return np.concatenate(sel)


def fit_predict(models, d_tr, tr_idx, d_te, te_idx, seed):
    y = (np.asarray(d_tr["fam"])[tr_idx] != FAMS.index("Benign")).astype(np.int32)
    m = models._xgb("binary:logistic", "logloss")
    m.set_params(random_state=seed)
    m.fit(np.asarray(d_tr["X"][tr_idx]), y)
    score = np.empty(len(te_idx), dtype=np.float32)
    for s in range(0, len(te_idx), 400_000):
        score[s:s + 400_000] = m.predict_proba(np.asarray(d_te["X"][te_idx[s:s + 400_000]]))[:, 1]
    return score, m.get_params()


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else None


def evaluate(fam_te, score, total_by_fam, keep_mask=None):
    """Pooled and per-family metrics on the test side; keep_mask restricts to non-overlapping rows."""
    if keep_mask is not None:
        fam_te, score = fam_te[keep_mask], score[keep_mask]
    ben = fam_te == FAMS.index("Benign")
    y = (~ben).astype(int)
    n_ben = int(ben.sum())
    out = {"n_test": int(len(y)), "n_benign": n_ben, "n_attack": int((~ben).sum()),
           "pooled_auc": auc(y, score),
           "benign_recall": float((score[ben] < 0.5).mean()) if n_ben else None,
           "attack_recall_pooled": float((score[~ben] >= 0.5).mean()) if (~ben).any() else None}
    out["balanced_accuracy"] = (0.5 * (out["benign_recall"] + out["attack_recall_pooled"])
                                if n_ben and (~ben).any() else None)
    fams = {}
    for k, name in enumerate(ATTACK_FAMS):
        m = fam_te == k
        n = int(m.sum())
        small = total_by_fam.get(name, 0) < SMALL_TOTAL or n < SMALL_TEST
        sel = m | ben
        fams[name] = {"n_test": n, "small_support": bool(small),
                      "auc_vs_all_benign": auc(y[sel], score[sel]) if n and n_ben else None,
                      "recall_at_0.5": float((score[m] >= 0.5).mean()) if n else None}
    out["per_family"] = fams
    out["macro_common"] = _macro([fams[f]["auc_vs_all_benign"] for f in COMMON])
    out["macro_common_missing"] = [f for f in COMMON if fams[f]["auc_vs_all_benign"] is None]
    arm = [f for f in ATTACK_FAMS if not fams[f]["small_support"] and fams[f]["auc_vs_all_benign"] is not None]
    out["macro_per_arm"] = _macro([fams[f]["auc_vs_all_benign"] for f in arm])
    out["macro_per_arm_families"] = arm
    return out


def _macro(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def overlap(h_test, h_train_full, h_train_cap):
    """Boolean per test row: its 128-bit row hash occurs in the training table."""
    def key(h):
        return h[:, 0].astype(np.uint64), h[:, 1].astype(np.uint64)
    def mark(h_tr, h_te):
        a = pd.MultiIndex.from_arrays(key(h_tr))
        return pd.MultiIndex.from_arrays(key(h_te)).isin(a)
    return mark(h_train_full, h_test), mark(h_train_cap, h_test)


def summarise(per_seed, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in per_seed if r.get(k) is not None]
        out[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n_seeds": len(v)} if v else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()
    config, models = load_config_and_models()
    cap_a, cap_b = int(config.ATTACK_CAP), int(config.BENIGN_CAP)
    step03 = json.load(open(os.path.join(C.RESULTS_DIR, "03_family_counts.json")))
    t0 = time.time()
    meta = {w: build_cache(w, os.path.join(args.scratch, f"cache_{w}"), step03) for w in ("iomt", "iot")}
    print("CACHE", {w: m["control"]["passed"] for w, m in meta.items()}, f"{time.time() - t0:.0f}s", flush=True)
    D = {w: open_cache(w, os.path.join(args.scratch, f"cache_{w}")) for w in meta}
    totals = {w: {f: int((np.asarray(D[w]["fam"]) == k).sum()) for k, f in enumerate(FAMS)} for w in D}
    results = {"arm1_within_iomt": {}, "arm2_within_iot": {}, "arm3_iomt_to_iot": {}, "arm4_iot_to_iomt": {}}
    draws = {k: {} for k in results}
    ben = FAMS.index("Benign")

    for seed in C.SEEDS:
        # arm 1 / arm 2: within dataset
        for arm, w, splitter in (("arm1_within_iomt", "iomt", lambda s: split_arm1(D["iomt"], meta["iomt"]["files"])),
                                 ("arm2_within_iot", "iot", lambda s: split_arm2(D["iot"], meta["iot"]["files"], s))):
            side, sinfo = splitter(seed)
            fam = np.asarray(D[w]["fam"])
            tr_all, te = np.where(side == 0)[0], np.where(side == 1)[0]
            tr = capped_train_idx(fam, tr_all, seed, cap_a, cap_b)
            draws[arm][seed] = int(pd.util.hash_array(tr.astype(np.int64)).sum() % (1 << 62))
            score, params = fit_predict(models, D[w], tr, D[w], te, seed)
            ev = evaluate(fam[te], score, totals[w])
            ev.update(seed=seed, train_rows=int(len(tr)), train_rows_by_family={FAMS[k]: int((fam[tr] == k).sum()) for k in range(len(FAMS))},
                      test_rows_by_family={FAMS[k]: int((fam[te] == k).sum()) for k in range(len(FAMS))}, dropped_by_embargo=int((side == -1).sum()),
                      split=sinfo if seed == 42 or arm == "arm2_within_iot" else "same as seed 42")
            results[arm][str(seed)] = ev
            print(arm, seed, {k: ev[k] for k in ("pooled_auc", "macro_common", "macro_per_arm", "balanced_accuracy", "benign_recall")}, f"{time.time() - t0:.0f}s", flush=True)
        # arms 3 / 4: cross dataset
        for arm, a, b in (("arm3_iomt_to_iot", "iomt", "iot"), ("arm4_iot_to_iomt", "iot", "iomt")):
            fa, fb = np.asarray(D[a]["fam"]), np.asarray(D[b]["fam"])
            tr = capped_train_idx(fa, np.arange(len(fa)), seed, cap_a, cap_b)
            draws[arm][seed] = int(pd.util.hash_array(tr.astype(np.int64)).sum() % (1 << 62))
            te = np.arange(len(fb))
            score, params = fit_predict(models, D[a], tr, D[b], te, seed)
            h_full, h_te = np.asarray(D[a]["hash"]), np.asarray(D[b]["hash"])
            ov_full, ov_cap = overlap(h_te, h_full, np.asarray(D[a]["hash"][tr]))
            ev = evaluate(fb, score, totals[b])
            ev["overlap_full_table"] = {"share_all": float(ov_full.mean()), "share_by_family": {FAMS[k]: float(ov_full[fb == k].mean()) if (fb == k).any() else None for k in range(len(FAMS))}}
            ev["overlap_capped_sample"] = {"share_all": float(ov_cap.mean()), "share_by_family": {FAMS[k]: float(ov_cap[fb == k].mean()) if (fb == k).any() else None for k in range(len(FAMS))}}
            ev["non_overlap_vs_full_table"] = evaluate(fb, score, totals[b], keep_mask=~ov_full)
            ev.update(seed=seed, train_rows=int(len(tr)), train_rows_by_family={FAMS[k]: int((fa[tr] == k).sum()) for k in range(len(FAMS))},
                      test_rows_by_family={FAMS[k]: int((fb == k).sum()) for k in range(len(FAMS))})
            results[arm][str(seed)] = ev
            print(arm, seed, {k: ev[k] for k in ("pooled_auc", "macro_common", "macro_per_arm", "balanced_accuracy", "benign_recall")},
                  "overlap", round(ev["overlap_full_table"]["share_all"], 4), f"{time.time() - t0:.0f}s", flush=True)
    for arm, d in draws.items():
        assert len(set(d.values())) == len(C.SEEDS), f"{arm}: seeds produced identical training draws (or splits)"

    # sensitivity: arms 3-4, UNCAPPED training, seed 42 only
    sens = {}
    for arm, a, b in (("arm3_iomt_to_iot", "iomt", "iot"), ("arm4_iot_to_iomt", "iot", "iomt")):
        fa, fb = np.asarray(D[a]["fam"]), np.asarray(D[b]["fam"])
        tr = np.arange(len(fa))
        score, _ = fit_predict(models, D[a], tr, D[b], np.arange(len(fb)), 42)
        h_full, h_te = np.asarray(D[a]["hash"]), np.asarray(D[b]["hash"])
        ov_full, _ = overlap(h_te, h_full, h_full[:1])
        ev = evaluate(fb, score, totals[b])
        ev["overlap_full_table"] = {"share_all": float(ov_full.mean())}
        ev["non_overlap_vs_full_table"] = evaluate(fb, score, totals[b], keep_mask=~ov_full)
        ev.update(seed=42, train_rows=int(len(tr)), note="UNCAPPED training; same test rows as the capped arm")
        sens[arm] = ev
        print("SENS", arm, {k: ev[k] for k in ("pooled_auc", "macro_common", "macro_per_arm", "balanced_accuracy", "benign_recall")}, f"{time.time() - t0:.0f}s", flush=True)

    keys = ["pooled_auc", "balanced_accuracy", "benign_recall", "macro_common", "macro_per_arm"]
    summary = {}
    for arm, per in results.items():
        rows = list(per.values())
        summary[arm] = summarise(rows, keys)
        if "overlap_full_table" in rows[0]:
            summary[arm]["overlap_share_full_table"] = summarise([{"v": r["overlap_full_table"]["share_all"]} for r in rows], ["v"])["v"]
            summary[arm]["overlap_share_capped_sample"] = summarise([{"v": r["overlap_capped_sample"]["share_all"]} for r in rows], ["v"])["v"]
            summary[arm]["non_overlap_pooled_auc"] = summarise([{"v": r["non_overlap_vs_full_table"]["pooled_auc"]} for r in rows], ["v"])["v"]
            summary[arm]["non_overlap_macro_common"] = summarise([{"v": r["non_overlap_vs_full_table"]["macro_common"]} for r in rows], ["v"])["v"]

    payload = {"meta": C.provenance(__file__, [], seeds=C.SEEDS, inputs_note="inputs = step-03 feature CSVs via caches in scratch; CICIoMT2024 nfstream version NOT RECORDED; CICIoT2023 6.6.0", hash_inputs=False),
               "plan": "README.md 'Detection stage'; changes before any run in its section 7",
               "caps": {"attack_per_family": cap_a, "benign": cap_b, "source": "thesis/crossdataset/src/config.py ATTACK_CAP / BENIGN_CAP"},
               "xgb_params_used": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in params.items() if k in config.XGB_PARAMS_BASE or k in ("objective", "eval_metric")},
               "model_input_columns": meta["iomt"]["cols"], "dedup_controls": {w: m["control"] for w, m in meta.items()},
               "totals_by_family_dedup": totals, "small_support_rule": {"total_below": SMALL_TOTAL, "test_side_below": SMALL_TEST},
               "common_families": COMMON, "embargo_ms": EMBARGO_MS, "overlap_method": "128-bit key = two pandas hash_pandas_object row hashes (different hash_key) over the 59 float64 values; not a byte compare",
               "seed_draw_fingerprints_differ": {a: sorted(d.values()) for a, d in draws.items()},
               "summary_over_seeds": summary, "per_seed": results, "sensitivity_uncapped_seed42": sens}
    print("wrote", C.write_json("05_detection_arms.json", payload), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
