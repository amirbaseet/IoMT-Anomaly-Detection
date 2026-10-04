"""
Shared-feature supervised model training + preprocessing (Layer 1), with the
fit-on-train-only discipline centralised here (TRAP 1).

Exposes:
  fit_imputer            -> SimpleImputer(median) fit on CICIoMT2024 train
  train_xgb_binary       -> XGBClassifier (attack vs benign)
  train_xgb_family       -> XGBClassifier (5 shared families)

The Flow AE (Layer 2) lives in ``ae_worker.py`` and runs in a SEPARATE process
because TensorFlow and XGBoost deadlock when both are live in one process
(duelling OpenMP runtimes).  This module therefore never imports TensorFlow.

Every scaler / imputer is fit on CICIoMT2024 data ONLY.  The CICIoT2023 test
frame is never passed to any ``.fit``; callers use ``.transform`` via the
returned objects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

import config
from labels import ATTACK, BENIGN, FAMILIES

# Fixed label<->int encodings (stable across the whole run / reports).
BINARY_TO_INT = {BENIGN.lower(): 0, ATTACK: 1}
INT_TO_BINARY = {v: k for k, v in BINARY_TO_INT.items()}
FAMILY_TO_INT = {fam: i for i, fam in enumerate(FAMILIES)}
INT_TO_FAMILY = {i: fam for fam, i in FAMILY_TO_INT.items()}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def fit_imputer(x_train: pd.DataFrame) -> SimpleImputer:
    """Median imputer fit on CICIoMT2024 train (handles the coerced NaNs)."""
    imp = SimpleImputer(strategy="median")
    imp.fit(x_train.to_numpy(dtype="float64"))
    return imp


def apply_imputer(imp: SimpleImputer, x: pd.DataFrame) -> np.ndarray:
    """Transform-only; safe for the CICIoT2023 test frame."""
    return imp.transform(x.to_numpy(dtype="float64"))


# --------------------------------------------------------------------------- #
# Supervised — XGBoost (Layer 1)
# --------------------------------------------------------------------------- #
def _xgb(objective: str, eval_metric: str) -> XGBClassifier:
    params = dict(config.XGB_PARAMS_BASE)
    params.update(objective=objective, eval_metric=eval_metric)
    return XGBClassifier(**params)


def train_xgb_binary(x_imputed: np.ndarray, y_binary: np.ndarray) -> XGBClassifier:
    y = np.array([BINARY_TO_INT[v] for v in y_binary], dtype=np.int32)
    model = _xgb("binary:logistic", "logloss")
    model.fit(x_imputed, y)
    return model


def train_xgb_family(x_imputed: np.ndarray, y_family: np.ndarray) -> XGBClassifier:
    y = np.array([FAMILY_TO_INT[v] for v in y_family], dtype=np.int32)
    model = _xgb("multi:softprob", "mlogloss")
    model.fit(x_imputed, y)
    return model
