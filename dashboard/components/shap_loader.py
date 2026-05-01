"""Cached pre-computed SHAP loader for Page 3 (SHAP Explorer).

Phase 7 produced `results/shap/shap_values/shap_values.npy` of shape
(19 classes, 5000 samples, 44 features), computed with `random_state=42`,
`shap_subsample_n=5000`, `shap_background_n=500`. This module wraps the file
behind a `@st.cache_data` loader, exposes per-class / per-category /
top-feature helpers, and includes a SHA256 tripwire that refuses to render
on file corruption.

**Critical: this module is for the PRE-COMPUTED §16 signatures only.**
Page 2's runtime per-flow SHAP uses `shap.TreeExplainer` directly via
`model_loader.get_e7_explainer()`. Don't collide the two code paths.

Aggregation conventions (mirror `notebooks/shap_sensitivity.py:272-284`):
- **Per-class signature** = `|shap_arr[c]|.mean(axis=0)` — one (44,) vector
  per class, computed over ALL 5000 sampled rows in that class's axis slot
  (NOT filtered by y_multiclass).
- **Category signature** = equal-weight mean of the per-class signatures
  for the classes in that category. Equal-weight averaging is what the
  published 0.991 cosine uses; sample-weighted aggregation gives ~0.996.
- **Global signature** = `|shap_arr|.mean(axis=(0,1))` — single (44,)
  vector across the full SHAP cube.

Validation gate: `category_cosine("DDoS", "DoS")` MUST equal 0.990971
(published §16.4 / `comparison.csv:ddos_dos_cosine_test_bg`). Any drift is
an aggregation bug, not a numerical-precision issue.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np
import streamlit as st

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SHAP_VALUES_PATH: Final[Path] = PROJECT_ROOT / "results/shap/shap_values/shap_values.npy"
SHAP_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "results/shap/config.json"
SHAP_SENSITIVITY_PATH: Final[Path] = PROJECT_ROOT / "results/shap/sensitivity/comparison.csv"

# Reproducibility tripwire — SHA256 of shap_values.npy committed at Phase B
# write time. Mismatch indicates file corruption or a different SHAP run was
# substituted; Page 3 refuses to render until investigated.
SHAP_VALUES_SHA256: Final[str] = (
    "c91e6836aec619f17d9aeff93cbac59226d44cd447cc08d011586ff6065df9b0"
)

N_CLASSES: Final[int] = 19
N_FEATURES: Final[int] = 44
N_SAMPLES: Final[int] = 5000


@st.cache_data(ttl=None, show_spinner=False)
def _load_shap_values_raw() -> np.ndarray:
    """Memory-mapped load of the (19, 5000, 44) SHAP cube."""
    return np.load(SHAP_VALUES_PATH, mmap_mode="r")


@st.cache_data(ttl=None, show_spinner=False)
def shap_metadata() -> dict:
    """Class names + feature names + categories from the §16 SHAP config."""
    cfg = json.loads(SHAP_CONFIG_PATH.read_text())
    return {
        "class_names": list(cfg["class_names"]),
        "feature_names": list(cfg["feature_names"]),
        "categories": dict(cfg["categories"]),
    }


@st.cache_data(ttl=None, show_spinner=False)
def verify_sha256() -> tuple[bool, str]:
    """Compute SHA256 of `shap_values.npy` and compare to the committed hash.

    Returns (ok, observed_hex). On mismatch, Page 3 must refuse to render —
    we cannot trust per-class plots if the underlying file is unrecognized.
    """
    h = hashlib.sha256(SHAP_VALUES_PATH.read_bytes()).hexdigest()
    return h == SHAP_VALUES_SHA256, h


@st.cache_data(ttl=None, show_spinner=False)
def per_class_signature() -> np.ndarray:
    """(N_CLASSES, N_FEATURES) — mean |SHAP| per class across all 5000 rows.

    Mirrors `shap_sensitivity.py:category_cosine_matrix` line 277:
        per_class_imp = np.abs(shap_arr).mean(axis=1)
    """
    arr = _load_shap_values_raw()
    return np.asarray(np.abs(arr).mean(axis=1))


@st.cache_data(ttl=None, show_spinner=False)
def global_signature() -> np.ndarray:
    """(N_FEATURES,) — mean |SHAP| across all classes and all samples."""
    arr = _load_shap_values_raw()
    return np.asarray(np.abs(arr).mean(axis=(0, 1)))


def class_idx(name: str) -> int:
    """Class-name → row index in `shap_values.npy` (and class_names list)."""
    return shap_metadata()["class_names"].index(name)


def signature_for_class(name: str) -> np.ndarray:
    """Mean |SHAP| signature for one class, shape (44,)."""
    return per_class_signature()[class_idx(name)]


def signature_for_category(cat: str) -> np.ndarray:
    """Equal-weight mean of per-class signatures across `categories[cat]`.

    This is the canonical aggregation behind the published §16.4 DDoS↔DoS
    cosine of 0.991. Sample-weighted aggregation gives ≈0.996 — close, but
    not bit-exact — so we stick to equal-weight for reproducibility.
    """
    meta = shap_metadata()
    member_idx = [class_idx(m) for m in meta["categories"][cat]]
    return per_class_signature()[member_idx].mean(axis=0)


def top_k_features(name_or_signature: str | np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    """Top-k (feature_name, mean_abs_shap) tuples for a class name or
    pre-computed (44,) signature, sorted by magnitude descending."""
    sig = (
        signature_for_class(name_or_signature)
        if isinstance(name_or_signature, str)
        else np.asarray(name_or_signature)
    )
    if sig.shape != (N_FEATURES,):
        raise ValueError(f"signature must be ({N_FEATURES},), got {sig.shape}")
    feature_names = shap_metadata()["feature_names"]
    order = np.argsort(sig)[::-1][:k]
    return [(feature_names[i], float(sig[i])) for i in order]


def category_cosine(cat_a: str, cat_b: str) -> float:
    """Cosine similarity between two category signatures.

    `category_cosine("DDoS", "DoS")` must reproduce 0.990971 — that's the
    Phase B validation gate.
    """
    a = signature_for_category(cat_a)
    b = signature_for_category(cat_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def closest_class(query: np.ndarray, feature_names_query: list[str]) -> tuple[str, float]:
    """Find the class whose pre-computed signature is closest (by cosine) to
    a runtime SHAP vector for ONE flow.

    Used by Page 2's footer: after the runtime TreeExplainer produces a
    44-dim SHAP vector for a user-input flow, this finds the closest §16
    per-class signature and returns its name + cosine.

    Args:
        query: shape (44,). Per-feature SHAP magnitudes for the predicted
               class. Sign is irrelevant — we abs() it before comparing
               against the pre-computed |SHAP| signatures.
        feature_names_query: feature names in the order `query` is indexed by.
                             If they don't match the canonical SHAP feature
                             order, we reorder first.
    """
    canon = shap_metadata()["feature_names"]
    if list(feature_names_query) != canon:
        idx_map = [feature_names_query.index(name) for name in canon]
        query = np.asarray(query)[idx_map]
    q = np.abs(np.asarray(query))
    if q.shape != (N_FEATURES,):
        raise ValueError(f"query must be ({N_FEATURES},), got {q.shape}")
    sigs = per_class_signature()  # (19, 44)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return shap_metadata()["class_names"][0], 0.0
    cos = (sigs @ q) / (np.linalg.norm(sigs, axis=1) * q_norm)
    best = int(np.argmax(cos))
    return shap_metadata()["class_names"][best], float(cos[best])
