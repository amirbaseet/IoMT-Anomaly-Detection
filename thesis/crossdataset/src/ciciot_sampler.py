"""
Stratified, chunked sampling of the 63 CICIoT2023 merged CSVs (design doc §6).

Two logical passes, but the 8.5 GB feature read happens only ONCE:

  Pass 1  (Label column only, cached): exact per-class counts.
  Pass 2  (features + Label, single read): draw all 3 seed samples at once by
          sampling each class at fraction = min(1, cap / count).

TRAP 2 (strip \r) is applied to the Label column on every chunk.  Rare shared
classes fall below their cap and are kept whole; dominant flooding classes are
down-sampled to the cap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from labels import is_known_ciciot2023, normalize_ciciot2023


def merged_files(merged_dir: Path) -> list[Path]:
    files = sorted(merged_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No merged CSVs in {merged_dir}")
    return files


def count_labels(
    files: list[Path], label_col: str, chunksize: int, cache_path: Path
) -> dict[str, int]:
    """Exact per-class row counts (seed-independent), cached to JSON."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    counts: dict[str, int] = {}
    for path in files:
        for chunk in pd.read_csv(
            path, usecols=[label_col], chunksize=chunksize,
            dtype=str, low_memory=False,
        ):
            norm = chunk[label_col].map(normalize_ciciot2023)
            for lab, c in norm.value_counts().items():
                counts[lab] = counts.get(lab, 0) + int(c)
    cache_path.write_text(json.dumps(counts, indent=2, sort_keys=True))
    return counts


def build_caps(counts: dict[str, int], benign_cap: int, attack_cap: int) -> dict[str, int]:
    """
    Per-class quota: benign gets its own cap, every recognised attack class the
    other, and any unrecognised/malformed label a quota of 0 so those rows are
    never sampled (they would otherwise be mislabelled 'attack').
    """
    caps = {}
    for lab in counts:
        if not is_known_ciciot2023(lab):
            caps[lab] = 0
        elif lab == "BENIGN":
            caps[lab] = benign_cap
        else:
            caps[lab] = attack_cap
    return caps


def sample_all_seeds(
    files: list[Path],
    shared_features: list[str],
    label_col: str,
    counts: dict[str, int],
    caps: dict[str, int],
    seeds: tuple[int, ...],
    chunksize: int,
) -> dict[int, pd.DataFrame]:
    """
    Single feature-read pass producing one sampled frame per seed.

    Each returned frame has the shared features (float32) plus a normalized
    ``raw_label`` column.  Sampling fraction per class = min(1, cap/count),
    so rare classes are kept whole and dominant classes down-sampled.
    """
    fractions = {
        lab: min(1.0, caps[lab] / cnt) for lab, cnt in counts.items() if cnt > 0
    }
    rngs = {s: np.random.default_rng(s) for s in seeds}
    buckets: dict[int, list[pd.DataFrame]] = {s: [] for s in seeds}
    usecols = shared_features + [label_col]

    for path in files:
        for chunk in pd.read_csv(
            path, usecols=usecols, chunksize=chunksize, low_memory=False,
        ):
            labels = chunk[label_col].map(normalize_ciciot2023)
            feats = chunk[shared_features].apply(pd.to_numeric, errors="coerce")
            feats = feats.replace([np.inf, -np.inf], np.nan).astype("float32")
            feats = feats.assign(raw_label=labels.to_numpy())

            frac = labels.map(fractions).to_numpy(dtype=float)
            for seed in seeds:
                keep = rngs[seed].random(len(chunk)) < frac
                if keep.any():
                    buckets[seed].append(feats.loc[keep])

    return {
        seed: (pd.concat(parts, ignore_index=True) if parts else
               pd.DataFrame(columns=usecols))
        for seed, parts in buckets.items()
    }
