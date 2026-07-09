"""
Metric computation for Design A (binary) and Design B (5 families), plus
multi-seed aggregation (mean ± σ, TRAP 4).

All functions are pure and return plain dicts / arrays for the reporter.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from labels import FAMILIES
from models import FAMILY_TO_INT


# --------------------------------------------------------------------------- #
# Design A — binary (attack = positive class = 1)
# --------------------------------------------------------------------------- #
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   score_attack: np.ndarray) -> dict:
    """
    y_true / y_pred: 0=benign, 1=attack.  score_attack: higher => more attack-
    like (XGB attack-probability or AE reconstruction error) for ROC-AUC.
    """
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1,
                                           zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1,
                                     zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    # ROC-AUC needs both classes present in y_true.
    out["roc_auc"] = (float(roc_auc_score(y_true, score_attack))
                      if len(np.unique(y_true)) == 2 else float("nan"))
    out["confusion"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out


# --------------------------------------------------------------------------- #
# Design B — 5 families
# --------------------------------------------------------------------------- #
def family_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true / y_pred are family ints (see models.FAMILY_TO_INT)."""
    labels = [FAMILY_TO_INT[f] for f in FAMILIES]
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_family = {
        FAMILIES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                      "f1": float(f[i]), "support": int(support[i])}
        for i in range(len(FAMILIES))
    }
    return {
        "per_family": per_family,
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# --------------------------------------------------------------------------- #
# Multi-seed aggregation
# --------------------------------------------------------------------------- #
def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_scalar(seed_dicts: list[dict], key: str) -> dict:
    """mean±σ for a scalar metric present in each per-seed dict."""
    m, s = mean_std([d[key] for d in seed_dicts])
    return {"mean": m, "std": s, "per_seed": [d[key] for d in seed_dicts]}
