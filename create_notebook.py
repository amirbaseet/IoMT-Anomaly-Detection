#!/usr/bin/env python3
"""Generate IoMT_Anomaly_Detection.ipynb from the Phase 2-6 Python source files."""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _id(n: list) -> str:
    idx = n[0]
    n[0] += 1
    return f"cell{idx:04d}"


def md_cell(source: str, cid: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source}


def code_cell(source: str, cid: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_docstring(code: str) -> tuple[str, str]:
    """Remove and return the leading module docstring (triple-quoted)."""
    m = re.match(r'\s*"""(.*?)"""\s*\n', code, re.DOTALL)
    if m:
        return m.group(1).strip(), code[m.end():]
    m = re.match(r"\s*'''(.*?)'''\s*\n", code, re.DOTALL)
    if m:
        return m.group(1).strip(), code[m.end():]
    return "", code


# Section-splitter recognises the three header styles used across the 5 files:
#   # %% SECTION X — ...         (EDA)
#   # %% ===...                   (preprocessing / supervised)
#   # %% [Section X] ...          (fusion)
#   # %% ============...          (unsupervised)
_SECTION_RE = re.compile(
    r"^# %%[ \t]*(={5,}|SECTION|\[|---)",
    re.MULTILINE,
)


def split_sections(code: str) -> list[str]:
    """Return a list of non-empty code blocks split at # %% markers."""
    positions = [m.start() for m in _SECTION_RE.finditer(code)]
    if not positions:
        return [code.strip()] if code.strip() else []

    blocks: list[str] = []
    # text before the first marker
    preamble = code[: positions[0]].strip()
    if preamble:
        blocks.append(preamble)
    # each marked section
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(code)
        block = code[start:end].rstrip()
        if block.strip():
            blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# per-phase descriptions
# ---------------------------------------------------------------------------

PHASE_INTRO = {
    "ciciomt2024_eda.py": (
        "## Phase 2 — Exploratory Data Analysis",
        """\
Performs a full, publication-quality EDA on the CICIoMT2024 dataset
(≈7.16 M train rows, ≈1.61 M test rows, 18 classes, 45 features).

**Key outputs** (written to `eda_output/`):
- `findings.md` — consolidated preprocessing recommendations
- `figures/` — 15+ charts (distributions, correlation, PCA, t-SNE)
- `quality_*.csv` — per-column missing/inf/unique audit
- `imbalance_table.csv` — class counts and ratios
- `high_correlation_pairs.csv` — |r| > 0.85 feature pairs
- `feature_target_cohens_d.csv` — |Cohen's d| ranking (attack vs benign)
- `benign_profile.csv` — Autoencoder reference statistics
- `train_cleaned.csv`, `test_cleaned.csv` — deduped, NaN-filled data

**Prerequisite**: CSV files in `./data/train/` and `./data/test/`.""",
    ),
    "preprocessing_pipeline.py": (
        "## Phase 3 — Preprocessing & Feature Engineering",
        """\
Transforms cleaned CSVs into scaled, labelled, and resampled NumPy arrays
ready for Phases 4–6.

**Two feature variants**:
| Variant | Features | Dropped |
|---------|----------|---------|
| Full    | 44       | Drate only (always 0) |
| Reduced | 28       | Drate + 11 correlated + 5 near-zero noise |

**Scaling strategy** (ColumnTransformer, fit on train only):
- RobustScaler → heavy-tailed (IAT, Rate, Header_Length, Tot sum, …)
- StandardScaler → flag-ratio features (syn_flag_number, …)
- MinMaxScaler → binary protocol indicators (HTTP, TCP, ARP, …)

**SMOTETomek**: targeted oversampling to 50 000 rows for minority classes.

**Outputs** (`preprocessed/`): X/y arrays, scalers (.pkl), label encoders (.json),
benign-only autoencoder sets, and 5 leave-one-attack-out zero-day scenarios.

**Prerequisite**: Phase 2 must have produced `eda_output/train_cleaned.csv`.""",
    ),
    "supervised_training.py": (
        "## Phase 4 — Supervised Model Training (Layer 1)",
        """\
Trains **8 experiments** (RF & XGBoost × reduced/full features × original/SMOTE)
and evaluates at 3 granularities (binary, 6-class category, 19-class multiclass).

| ID | Model | Features | Data     |
|----|-------|----------|----------|
| E1 | RF    | reduced  | original |
| E2 | RF    | reduced  | SMOTE    |
| E3 | XGB   | reduced  | original |
| E4 | XGB   | reduced  | SMOTE    |
| E5 | RF    | full     | original |
| E6 | RF    | full     | SMOTE    |
| **E7** | **XGB** | **full** | **original** ← recommended |
| E8 | XGB   | full     | SMOTE    |

**RF params**: entropy, 200 trees, max_depth=30, class_weight='balanced'
**XGB params**: max_depth=8, lr=0.1, multi:softprob (19-class)

**Outputs** (`results/supervised/`): models (.pkl), predictions (.npy),
confusion matrices, feature importance, comparison CSVs.

**Prerequisite**: Phase 3 must have written `preprocessed/`.""",
    ),
    "unsupervised_training.py": (
        "## Phase 5 — Unsupervised Model Training (Layer 2)",
        """\
Trains an **Autoencoder** and an **Isolation Forest** on benign-only traffic.

**Autoencoder architecture** (symmetric encoder-decoder):
```
Input (44) → Dense(32)+BN+Drop(0.2) → Dense(16)+BN+Drop(0.1)
           → Bottleneck(8)
           → Dense(16)+BN → Dense(32)+BN → Output(44, linear)
```
- Loss: MSE (reconstruction error); Optimizer: Adam lr=1e-3
- Early stopping (patience=10) + ReduceLROnPlateau

**Isolation Forest**: 200 estimators, contamination=0.05

**Threshold selection** (on validation set): p90/p95/p99/mean±2σ/mean±3σ
Best threshold = highest F1 on binary benign-vs-attack classification.

**Outputs** (`results/unsupervised/`): models (.keras / .pkl), MSE score
arrays (.npy), per-class detection rates, zero-day preview, ROC curves.

**Prerequisite**: Phase 3 must have written `preprocessed/full_features/`.
**TensorFlow required**: `pip install tensorflow` (or `tensorflow-metal` on Apple Silicon).""",
    ),
    "fusion_engine.py": (
        "## Phase 6 — Fusion Engine & Zero-Day Simulation (Layer 3)",
        """\
Combines E7 (supervised) and AE+IF (unsupervised) predictions using a
**4-case decision logic** — no retraining required.

| Case | Supervised | Unsupervised | Decision | Confidence |
|------|-----------|-------------|----------|-----------|
| 1 | Attack | Anomaly | **Confirmed Alert** | HIGH |
| 2 | Benign | Anomaly | **Zero-Day Warning** | MEDIUM_HIGH |
| 3 | Attack | Normal | Low-Confidence Alert | MEDIUM_LOW |
| 4 | Benign | Normal | Clear | HIGH |

**Hypothesis tests**:
- **H1**: Paired bootstrap (200 iters) — does fusion improve multiclass macro-F1?
- **H2**: AE recall on samples E7 misclassifies as benign ≥70% on ≥50% of targets.

**Zero-day targets**: Recon_Ping_Sweep, Recon_VulScan, MQTT_Malformed_Data,
MQTT_DoS_Connect_Flood, ARP_Spoofing.

**Outputs** (`results/fusion/`): case arrays (.npy), bootstrap CIs, threshold
sensitivity, H1/H2 verdicts (.json), 5 publication-quality figures.

**Prerequisite**: Phases 4 and 5 must have completed.""",
    ),
}

# ---------------------------------------------------------------------------
# notebook title cell
# ---------------------------------------------------------------------------

TITLE_MD = """\
# IoMT Anomaly Detection — Hybrid Supervised-Unsupervised Framework

**Author**: Amro — M.Sc. AI & ML in Cybersecurity, Sakarya University of Applied Sciences
**Dataset**: CICIoMT2024 (WiFi + MQTT combined, ≈8.7 M network flows, 18 classes, 45 features)

---

## Framework Overview

This notebook implements a full 6-phase pipeline for detecting anomalies and
zero-day attacks in Internet of Medical Things (IoMT) networks.

| Phase | Script | Description |
|-------|--------|-------------|
| 2 | `ciciomt2024_eda.py` | Exploratory Data Analysis |
| 3 | `preprocessing_pipeline.py` | Preprocessing & Feature Engineering |
| 4 | `supervised_training.py` | Supervised Layer (RF + XGBoost) |
| 5 | `unsupervised_training.py` | Unsupervised Layer (Autoencoder + Isolation Forest) |
| 6 | `fusion_engine.py` | Fusion Engine & Zero-Day Simulation |

## Hypotheses

| | Hypothesis | Test |
|-|-----------|------|
| **H1** | Fusion improves 19-class macro-F1 over E7 alone | Paired bootstrap (200 iters), 95% CI |
| **H2** | AE catches ≥70% of attacks E7 misclassifies as benign, on ≥50% of targets | Recall on E7-missed samples |

## Dataset: CICIoMT2024

| Property | Value |
|----------|-------|
| Total rows | ≈8.7 M (train 7.16 M + test 1.61 M) |
| Feature space | 45 network-flow features |
| Attack classes | 17 distinct types + 1 benign |
| Max imbalance ratio | ≈2 211 : 1 (Benign vs Recon_Ping_Sweep) |

Attack families: DDoS (4), DoS (4), Reconnaissance (4), MQTT (5), ARP Spoofing (1)
"""

INSTALL_CODE = """\
# ── Installation ────────────────────────────────────────────────────────────
# Uncomment the lines below if packages are not already installed.

# !pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
# !pip install xgboost joblib pyarrow fastparquet
# !pip install tensorflow               # Autoencoder (Phase 5)
# !pip install tensorflow-metal         # Apple Silicon GPU (Phase 5)
# !pip install tabulate                 # Markdown tables in Phase 6

import sys
print(f"Python : {sys.version}")
try:
    import numpy as np;   print(f"NumPy  : {np.__version__}")
    import pandas as pd;  print(f"Pandas : {pd.__version__}")
    import sklearn;       print(f"sklearn: {sklearn.__version__}")
    import xgboost;       print(f"XGBoost: {xgboost.__version__}")
except ImportError as e:
    print(f"Missing: {e}")
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).parent
    notebooks_dir = root / "notebooks"
    output = root / "IoMT_Anomaly_Detection.ipynb"

    cid: list[int] = [0]
    cells: list[dict] = []

    # ---- title + install ----
    cells.append(md_cell(TITLE_MD, _id(cid)))
    cells.append(code_cell(INSTALL_CODE, _id(cid)))

    # ---- phases ----
    phase_files = [
        "ciciomt2024_eda.py",
        "preprocessing_pipeline.py",
        "supervised_training.py",
        "unsupervised_training.py",
        "fusion_engine.py",
    ]

    for filename in phase_files:
        path = notebooks_dir / filename
        if not path.exists():
            print(f"  !! {path} not found — skipping")
            continue

        print(f"  processing {filename} …")
        raw = read_file(path)
        docstring, code = extract_docstring(raw)

        heading, intro = PHASE_INTRO[filename]

        # phase header markdown
        header_md = f"{heading}\n\n{intro}"
        if docstring:
            trimmed = docstring[:600] + ("…" if len(docstring) > 600 else "")
            header_md += f"\n\n---\n\n*Module docstring:*\n```\n{trimmed}\n```"
        cells.append(md_cell(header_md, _id(cid)))

        # code cells (one per section)
        sections = split_sections(code)
        for block in sections:
            if block.strip():
                cells.append(code_cell(block, _id(cid)))

        print(f"    -> {len(sections)} code cell(s)")

    # ---- write ----
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\nNotebook written -> {output}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
