"""Step 06 - arm 5: like-for-like transfer on the OFFICIAL CSV features, same family subset. Settings: README "Changed after seeing arms 1-4", section A.

Venv: ~/code/IoMT-Project/venv
Run:  ~/code/IoMT-Project/venv/bin/python -B -I src/06_like_for_like.py --scratch <dir>

Rows: CICIoMT2024 data/train + data/test CSVs (class from the filename, MQTT excluded after dedup), CICIoT2023 the 63 members of
MERGED_CSV.zip (class from the Label column). Dedup within each dataset on the 38 shared columns, BEFORE the cap, global over all
classes, files in sorted-name order, first kept. The key is a 128-bit hash of the float64 values; for CICIoMT2024 it is checked
against DataFrame.duplicated (exact). Feature sets: shared 38 (primary), shared minus the 7 of Addendum 4 (secondary, read from
inverse_full_drop_scale.json). Model, cap, seeds and metrics are those of step 05 (imported from it).
Tasks per feature set: base_iomt (official train -> test), base_iot (19 of 63 merged files held out by seed; NOT capture-disjoint),
arm5a iomt->iot, arm5b iot->iomt.
"""
import argparse
import importlib.util
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

spec = importlib.util.spec_from_file_location("step05", os.path.join(C.SRC_DIR, "05_detection_arms.py"))
S5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S5)
FAMS, CLASS2FAM = S5.FAMS, S5.CLASS2FAM
THESIS_X = os.path.join(C.THESIS_REPO, "thesis", "crossdataset")
ZIP = os.path.join(C.SSD, "CICIOT23", "csv", "MERGED_CSV.zip")
CHUNK = 250_000
HOLDOUT_FRACTION = 0.3


def thesis_imports():
    sys.path.insert(0, os.path.join(THESIS_X, "src"))
    import config  # noqa: WPS433
    import labels  # noqa: WPS433
    import models  # noqa: WPS433
    return config, labels, models


def feature_sets():
    shared = json.load(open(os.path.join(THESIS_X, "shared_features.json")))["shared_features"]
    dropped = json.load(open(os.path.join(THESIS_X, "inverse_full_drop_scale.json")))["dropped_features"]
    assert len(shared) == 38 and len(dropped) == 7 and set(dropped) <= set(shared)
    return {"shared38": shared, "shared31_drop7": [f for f in shared if f not in dropped]}, dropped


def hashes(X64):
    return S5.row_hashes(X64)


def first_occurrence_dup(H):
    return pd.DataFrame({"a": H[:, 0], "b": H[:, 1]}).duplicated(keep="first").to_numpy()


def counts_by_family(fam):
    return {FAMS[k]: int((fam == k).sum()) for k in range(len(FAMS))}


def to_float32(X64):
    X = X64.astype(np.float32)
    X[np.isinf(X)] = np.nan
    return X


def build_iomt(cache_dir, shared, config):
    meta_path = os.path.join(cache_dir, "meta.json")
    if os.path.exists(meta_path):
        return json.load(open(meta_path))
    from stage2_device_disjoint import filename_to_class
    os.makedirs(cache_dir, exist_ok=True)
    paths = sorted([os.path.join(d, f) for d in (config.CICIOMT2024_TRAIN_DIR, config.CICIOMT2024_TEST_DIR)
                    for f in os.listdir(d) if f.endswith(".csv")], key=os.path.basename)
    frames, fam_l, side_l = [], [], []
    for i, p in enumerate(paths):
        cls = filename_to_class(p)
        assert cls is not None, f"unmapped CICIoMT2024 file {p}"
        fam = FAMS.index(CLASS2FAM["iomt"][cls]) if cls in CLASS2FAM["iomt"] else -1
        df = pd.read_csv(p, usecols=shared, dtype=np.float64)[shared]
        frames.append(df.to_numpy())
        fam_l.append(np.full(len(df), fam, dtype=np.int8))
        side_l.append(np.full(len(df), 1 if "_test.pcap" in os.path.basename(p) else 0, dtype=np.int8))
        print(f"  [iomt {i + 1}/{len(paths)}] {os.path.basename(p)} {len(df):,} -> {cls}", flush=True)
    X64 = np.concatenate(frames); del frames
    fam, side = np.concatenate(fam_l), np.concatenate(side_l)
    dup_hash = first_occurrence_dup(hashes(X64))
    dup_exact = pd.DataFrame(X64).duplicated(keep="first").to_numpy()
    control = {"dup_hash": int(dup_hash.sum()), "dup_exact": int(dup_exact.sum()), "identical_masks": bool((dup_hash == dup_exact).all())}
    assert control["identical_masks"], f"hash dedup differs from exact dedup: {control}"
    keep = (~dup_exact) & (fam >= 0)
    X = to_float32(X64[keep])
    n_nan = int(np.isnan(X).sum())
    np.save(os.path.join(cache_dir, "X.npy"), X); np.save(os.path.join(cache_dir, "fam.npy"), fam[keep])
    np.save(os.path.join(cache_dir, "side.npy"), side[keep])
    raw_all = {FAMS[k]: int((fam == k).sum()) for k in range(len(FAMS))}
    meta = {"n_files": len(paths), "rows_raw_all_files": int(len(fam)), "rows_raw_non_pair_files": int((fam < 0).sum()),
            "raw_by_family": raw_all, "dedup_by_family": counts_by_family(fam[keep]), "dedup_control_hash_vs_exact": control,
            "n_nonfinite_cells_after_dedup": n_nan, "rows_kept": int(keep.sum())}
    json.dump(meta, open(meta_path, "w"))
    return meta


def build_iot(cache_dir, shared, config, labels):
    meta_path = os.path.join(cache_dir, "meta.json")
    if os.path.exists(meta_path):
        return json.load(open(meta_path))
    os.makedirs(cache_dir, exist_ok=True)
    lab2fam = {labels.normalize_ciciot2023(c): FAMS.index(f) for f, v in C.PAIRS.items() for c in v["iot"]}
    assert len(lab2fam) == 10, lab2fam
    zf = zipfile.ZipFile(ZIP)
    members = sorted(i.filename for i in zf.infolist() if i.filename.endswith(".csv") and "/._" not in i.filename)
    dtypes = {**{f: np.float64 for f in shared}, config.CICIOT2023_LABEL_COL: str}
    H, FAM, FID, GIDX, XK = [], [], [], [], []
    n_total, t0 = 0, time.time()
    for mi, m in enumerate(members):
        with zf.open(m) as fh:
            for ch in pd.read_csv(fh, usecols=shared + [config.CICIOT2023_LABEL_COL], dtype=dtypes, chunksize=CHUNK):
                lab = ch[config.CICIOT2023_LABEL_COL].map(labels.normalize_ciciot2023)
                fam = lab.map(lab2fam).fillna(-1).to_numpy().astype(np.int8)
                X64 = ch[shared].to_numpy()
                H.append(hashes(X64)); FAM.append(fam); FID.append(np.full(len(ch), mi, dtype=np.int16))
                sel = np.where(fam >= 0)[0]
                GIDX.append(n_total + sel); XK.append(to_float32(X64[sel]))
                n_total += len(ch)
        print(f"  [iot {mi + 1}/{len(members)}] {m} total {n_total:,} {time.time() - t0:.0f}s", flush=True)
    H, FAM, FID = np.concatenate(H), np.concatenate(FAM), np.concatenate(FID)
    G, XK = np.concatenate(GIDX), np.concatenate(XK)
    dup = first_occurrence_dup(H); del H
    keep = ~dup[G]
    X, fam, fid = XK[keep], FAM[G][keep], FID[G][keep]
    np.save(os.path.join(cache_dir, "X.npy"), X); np.save(os.path.join(cache_dir, "fam.npy"), fam); np.save(os.path.join(cache_dir, "fid.npy"), fid)
    meta = {"members": members, "rows_raw_all_members": int(n_total), "rows_raw_non_pair_labels": int((FAM < 0).sum()),
            "raw_by_family": counts_by_family(FAM), "dedup_by_family": counts_by_family(fam), "rows_dedup_all_classes": int((~dup).sum()),
            "n_nonfinite_cells_after_dedup": int(np.isnan(X).sum()), "rows_kept": int(len(fam)),
            "dedup_method": "128-bit hash of float64 values, first occurrence, members in sorted order (not byte compare)"}
    json.dump(meta, open(meta_path, "w"))
    return meta


def load_cache(cache_dir, names):
    return {n: np.load(os.path.join(cache_dir, n + ".npy"), mmap_mode="r") for n in names}


def fit_predict_imp(models, Xtr, ytr, Xte, seed):
    """Thesis protocol: median of the TRAIN sample fills NaN (TRAP 1); chunked predict."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
    fill = lambda A: np.where(np.isnan(A), med[None, :], A).astype(np.float32)  # noqa: E731
    m = models._xgb("binary:logistic", "logloss")
    m.set_params(random_state=seed)
    m.fit(fill(Xtr), ytr)
    score = np.empty(len(Xte), dtype=np.float32)
    for s in range(0, len(Xte), 400_000):
        score[s:s + 400_000] = m.predict_proba(fill(np.asarray(Xte[s:s + 400_000])))[:, 1]
    return score, int(np.isnan(Xtr).sum())


def overlap_mask(Xtr_all, Xte):
    a = pd.MultiIndex.from_arrays(list(hashes(np.asarray(Xtr_all, dtype=np.float64)).T))
    b = pd.MultiIndex.from_arrays(list(hashes(np.asarray(Xte, dtype=np.float64)).T))
    return b.isin(a)


def run_task(models, cap_a, cap_b, task, seed, D, cols_idx, totals):
    """Returns (evaluation dict, draw fingerprint)."""
    ben = FAMS.index("Benign")
    if task == "base_iomt":
        tr_all, te = np.where(D["iomt"]["side"] == 0)[0], np.where(D["iomt"]["side"] == 1)[0]
        a, b = "iomt", "iomt"
    elif task == "base_iot":
        fid = np.asarray(D["iot"]["fid"])
        members = np.arange(int(fid.max()) + 1)
        test_m = np.random.default_rng(seed).permutation(members)[:round(HOLDOUT_FRACTION * len(members))]
        is_te = np.isin(fid, test_m)
        tr_all, te = np.where(~is_te)[0], np.where(is_te)[0]
        a, b = "iot", "iot"
    elif task == "arm5a_iomt_to_iot":
        a, b = "iomt", "iot"; tr_all, te = np.arange(len(D["iomt"]["fam"])), np.arange(len(D["iot"]["fam"]))
    else:
        a, b = "iot", "iomt"; tr_all, te = np.arange(len(D["iot"]["fam"])), np.arange(len(D["iomt"]["fam"]))
    fa, fb = np.asarray(D[a]["fam"]), np.asarray(D[b]["fam"])
    tr = S5.capped_train_idx(fa, tr_all, seed, cap_a, cap_b)
    fingerprint = int(pd.util.hash_array(tr.astype(np.int64)).sum() % (1 << 62)) + (int(np.sum(te[:1000])) if task == "base_iot" else 0)
    Xtr = np.asarray(D[a]["X"][tr])[:, cols_idx]
    ytr = (fa[tr] != ben).astype(np.int32)
    Xte_src = D[b]["X"]
    # test matrix restricted to the feature set, row-selected lazily in chunks via a small wrapper
    class _View:
        def __init__(self): self.n = len(te)
        def __len__(self): return self.n
        def __getitem__(self, sl): return np.asarray(Xte_src[te[sl]])[:, cols_idx]
    score, n_nan_train = fit_predict_imp(models, Xtr, ytr, _View(), seed)
    ev = S5.evaluate(fb[te], score, totals[b])
    if a != b:
        ov = overlap_mask(np.asarray(D[a]["X"])[:, cols_idx], _View()[slice(None)])
        ev["overlap_full_table"] = {"share_all": float(ov.mean()), "share_by_family": {FAMS[k]: float(ov[fb[te] == k].mean()) if (fb[te] == k).any() else None for k in range(len(FAMS))}}
        ev["non_overlap_vs_full_table"] = S5.evaluate(fb[te], score, totals[b], keep_mask=~ov)
    ev.update(seed=seed, train_rows=int(len(tr)), n_nan_cells_in_train_sample=n_nan_train,
              train_rows_by_family=counts_by_family(fa[tr]), test_rows_by_family=counts_by_family(fb[te]))
    return ev, fingerprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    args = ap.parse_args()
    config, labels, models = thesis_imports()
    cap_a, cap_b = int(config.ATTACK_CAP), int(config.BENIGN_CAP)
    sets, dropped = feature_sets()
    shared = sets["shared38"]
    t0 = time.time()
    meta = {"iomt": build_iomt(os.path.join(args.scratch, "cache_csv_iomt"), shared, config),
            "iot": build_iot(os.path.join(args.scratch, "cache_csv_iot"), shared, config, labels)}
    print("RAW -> DEDUP per family", f"{time.time() - t0:.0f}s")
    for w in meta:
        print(f"  {w}: all files raw {meta[w].get('rows_raw_all_files', meta[w].get('rows_raw_all_members')):,}")
        for f in FAMS:
            print(f"    {f:10s} {meta[w]['raw_by_family'][f]:>12,} -> {meta[w]['dedup_by_family'][f]:>12,}")
    D = {"iomt": load_cache(os.path.join(args.scratch, "cache_csv_iomt"), ("X", "fam", "side")),
         "iot": load_cache(os.path.join(args.scratch, "cache_csv_iot"), ("X", "fam", "fid"))}
    totals = {w: counts_by_family(np.asarray(D[w]["fam"])) for w in D}
    tasks = ("base_iomt", "base_iot", "arm5a_iomt_to_iot", "arm5b_iot_to_iomt")
    results = {fs: {t: {} for t in tasks} for fs in sets}
    for fs, cols in sets.items():
        cols_idx = np.array([shared.index(c) for c in cols])
        draws = {t: {} for t in tasks}
        for seed in C.SEEDS:
            for t in tasks:
                ev, fp = run_task(models, cap_a, cap_b, t, seed, D, cols_idx, totals)
                results[fs][t][str(seed)] = ev; draws[t][seed] = fp
                print(fs, t, seed, {k: round(ev[k], 4) if ev[k] is not None else None for k in ("pooled_auc", "macro_common", "macro_per_arm", "balanced_accuracy", "benign_recall")},
                      f"{time.time() - t0:.0f}s", flush=True)
        for t, d in draws.items():
            assert len(set(d.values())) == len(C.SEEDS), f"{fs}/{t}: seeds produced identical training draws"
    keys = ["pooled_auc", "balanced_accuracy", "benign_recall", "macro_common", "macro_per_arm"]
    summary = {}
    for fs in results:
        summary[fs] = {}
        for t, per in results[fs].items():
            rows = list(per.values())
            summary[fs][t] = S5.summarise(rows, keys)
            if "overlap_full_table" in rows[0]:
                summary[fs][t]["overlap_share_full_table"] = S5.summarise([{"v": r["overlap_full_table"]["share_all"]} for r in rows], ["v"])["v"]
                summary[fs][t]["non_overlap_pooled_auc"] = S5.summarise([{"v": r["non_overlap_vs_full_table"]["pooled_auc"]} for r in rows], ["v"])["v"]
                summary[fs][t]["non_overlap_macro_common"] = S5.summarise([{"v": r["non_overlap_vs_full_table"]["macro_common"]} for r in rows], ["v"])["v"]
    payload = {"meta": C.provenance(__file__, [os.path.join(THESIS_X, "shared_features.json"), os.path.join(THESIS_X, "inverse_full_drop_scale.json")], seeds=C.SEEDS,
                                   inputs_note=f"official CSVs: CICIoMT2024 data/train+data/test; CICIoT2023 {ZIP} (size {os.path.getsize(ZIP)} bytes, not hashed)", hash_inputs=True),
               "settings": "README.md 'Changed after seeing arms 1-4', section A", "caps": {"attack_per_family": cap_a, "benign": cap_b},
               "feature_sets": sets, "dropped_features_source": "thesis/crossdataset/inverse_full_drop_scale.json -> dropped_features", "dropped_features": dropped,
               "holdout_base_iot": {"rule": f"round({HOLDOUT_FRACTION} x 63) merged members held out, drawn with default_rng(seed); random split of merged files, NOT capture-disjoint"},
               "dedup": {w: {k: v for k, v in m.items() if k not in ("members",)} for w, m in meta.items()},
               "imputation": "inf -> NaN at load; NaN filled with the median of the capped training sample (thesis TRAP 1)",
               "small_support_rule": {"total_below": S5.SMALL_TOTAL, "test_side_below": S5.SMALL_TEST}, "common_families": S5.COMMON,
               "totals_by_family_dedup": totals, "summary_over_seeds": summary, "per_seed": results}
    print("wrote", C.write_json("06_like_for_like.json", payload), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
