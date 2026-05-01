"""Cached, lazily-imported model loaders for Page 2 (Single Flow Analyzer).

Three models live in this module:
- E7 XGBoost classifier (results/supervised/models/E7_xgb_full_original.pkl)
- StandardScaler (results/unsupervised/models/scaler.pkl)
- AE autoencoder (results/unsupervised/models/autoencoder.keras)

Critical invariants:
1. **Lazy TF import — at the page level, not the module level.** TensorFlow
   loads only when Page 2 (`pages/3_Single_Flow_Analyzer.py`) is opened —
   because Page 2 is the ONLY page that imports this module. Pages 1/3/4/5
   import only `data_loader`, `pareto_chart`, `status_indicators` and remain
   TF-free at cold start. Within this module we deliberately import TF at
   module-load time to claim libomp BEFORE xgboost — see (1a) below.

   1a. **macOS libomp ordering.** xgboost 3.x and TF 2.21 both link to
   libomp on macOS. If xgboost initializes its OMP pool first, the next
   `ae.predict()` call deadlocks waiting on libomp. The fix is order: TF
   must touch libomp first. We achieve this by importing `tensorflow` at
   the TOP of this module, before joblib loads any xgboost Booster. Since
   only Page 2 imports this module, the rest of the dashboard cold-starts
   without TF.

2. **mtime-based cache invalidation.** Streamlit's `@st.cache_resource`
   keys on positional args. Each loader takes an explicit `mtime: float`
   so a freshly-modified model file forces a reload without a full app
   restart.

3. **Reproducibility tripwires.**
   - Tripwire #1 (`verify_e7_tripwire`): re-score X_test[:100], compare to
     committed e7_first100_proba_ref.npy. Tolerance 1e-5 (float32 noise).
   - Tripwire #1b (`verify_ae_tripwire`): recompute mean MSE on
     scaler.transform(X_test[:100]), compare to committed ae_recon_ref.json
     mean_mse. Tolerance 1e-4. Catches scaler-direction bugs.

4. **AE input convention.** The AE consumes `scaler.transform(x)`. Forgetting
   the scaler silently inflates recon error by ~8 orders of magnitude — no
   crash, just every flow flagged anomalous. R4 mitigation.

5. **Graceful AE degradation (Q2 fallback β).** If TF import or AE load
   fails (e.g., environment regression), `load_autoencoder()` returns None
   and Page 2 displays "AE component unavailable" without crashing the
   page. E7 + entropy + SHAP cards stay live.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

# 1a — claim libomp BEFORE xgboost touches it. Module-import-time TF inference
# can deadlock on Python's import lock (TF spawns threads that try to re-enter
# import). So we ONLY import TF + keras here (cheap, ~3 s; both must resolve
# fully now or the lazy `from tensorflow import keras` in `prime_tf()` will
# wedge on a second import while xgboost holds libomp). Defer the actual
# warm-up inference to `prime_tf()`, which Page 2 calls before any xgboost
# predict. Wrapped in try/except so a broken TF env still allows graceful AE
# degradation (Q2 fallback β).
try:
    import tensorflow as _tf  # noqa: F401
    from tensorflow import keras as _keras  # resolves the import once, eagerly
    _TF_AVAILABLE: Final[bool] = True
except Exception:
    _keras = None  # type: ignore[assignment]
    _TF_AVAILABLE = False

import joblib  # noqa: E402 — must come AFTER the TF import on macOS
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

_TF_PRIMED: bool = False  # set True once `prime_tf()` runs ae.predict once

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
COMPONENTS_DIR: Final[Path] = Path(__file__).resolve().parent
PREPROCESSED_DIR: Final[Path] = PROJECT_ROOT / "preprocessed" / "full_features"

E7_PATH: Final[Path] = RESULTS_DIR / "supervised/models/E7_xgb_full_original.pkl"
SCALER_PATH: Final[Path] = RESULTS_DIR / "unsupervised/models/scaler.pkl"
AE_PATH: Final[Path] = RESULTS_DIR / "unsupervised/models/autoencoder.keras"

E7_REF_PATH: Final[Path] = COMPONENTS_DIR / "e7_first100_proba_ref.npy"
AE_REF_PATH: Final[Path] = COMPONENTS_DIR / "ae_recon_ref.json"
X_TEST_PATH: Final[Path] = PREPROCESSED_DIR / "X_test.npy"

E7_TRIPWIRE_TOL: Final[float] = 1e-5
AE_TRIPWIRE_TOL: Final[float] = 1e-4
N_REF_ROWS: Final[int] = 100
N_CLASSES: Final[int] = 19


def _mtime(path: Path) -> float:
    """Filesystem mtime as cache key — modifications force a reload."""
    return path.stat().st_mtime


def prime_tf() -> bool:
    """Force TF to fully initialize before any xgboost predict runs.

    Loads the AE once and runs a dummy inference so TF claims libomp.
    Idempotent (the second call is a no-op). Page 2 calls this BEFORE
    `verify_e7_tripwire` to avoid the macOS xgboost↔TF libomp deadlock
    (R1 / D6 mitigation, see module docstring §1a).

    Returns True if TF is now ready for inference, False if TF is unavailable.
    """
    global _TF_PRIMED
    if _TF_PRIMED:
        return True
    if not _TF_AVAILABLE or _keras is None:
        return False
    try:
        m = _keras.models.load_model(AE_PATH)
        # Use model(x) (eager call) instead of model.predict(x) — predict()
        # spawns a TF data pipeline thread which deadlocks against Streamlit's
        # ScriptRunner thread on macOS. Eager call runs in the current thread.
        m(np.zeros((1, 44), dtype=np.float32), training=False)
        _TF_PRIMED = True
        return True
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_e7(_mtime_key: float) -> Any:
    """Load the E7 XGBoost classifier. Cache invalidates on file mtime change."""
    return joblib.load(E7_PATH)


@st.cache_resource(show_spinner=False)
def load_scaler(_mtime_key: float) -> Any:
    """Load the StandardScaler used by Phase 5 unsupervised models."""
    return joblib.load(SCALER_PATH)


@st.cache_resource(show_spinner=False)
def load_autoencoder(_mtime_key: float) -> Any | None:
    """Load the Phase 5 autoencoder. TF imported here to keep other pages clean.

    Returns None on any load failure (Q2 fallback β: graceful degradation).
    Page 2 must check the return value and show a clear "unavailable" message
    instead of crashing.
    """
    if not _TF_AVAILABLE or _keras is None:
        return None
    try:
        return _keras.models.load_model(AE_PATH)
    except Exception:  # broad: env breakage covers many failure modes
        return None


@st.cache_resource(show_spinner=False)
def load_e7_explainer(_mtime_key: float) -> Any:
    """SHAP TreeExplainer over E7. Used by Page 2 for runtime per-flow SHAP.

    Phase 3 uses pre-computed SHAP from `results/shap/shap_values.npy`; this
    explainer is for ONE user-input flow at request time. Different code path,
    different invariants — do not collapse with the Phase 3 loader.
    """
    import shap  # local: keeps shap out of pages 1/3/4/5 cold-start
    e7 = load_e7(_mtime(E7_PATH))
    return shap.TreeExplainer(e7)


def get_e7() -> Any:
    return load_e7(_mtime(E7_PATH))


def get_scaler() -> Any:
    return load_scaler(_mtime(SCALER_PATH))


def get_autoencoder() -> Any | None:
    return load_autoencoder(_mtime(AE_PATH))


def get_e7_explainer() -> Any:
    return load_e7_explainer(_mtime(E7_PATH))


@st.cache_data(ttl=None, show_spinner=False)
def _load_x_test_first_n(n: int = N_REF_ROWS) -> np.ndarray:
    return np.asarray(np.load(X_TEST_PATH, mmap_mode="r")[:n])


@st.cache_data(ttl=None, show_spinner=False)
def _load_e7_ref() -> np.ndarray:
    return np.load(E7_REF_PATH)


@st.cache_data(ttl=None, show_spinner=False)
def _load_ae_ref() -> dict[str, float]:
    return json.loads(AE_REF_PATH.read_text())


def verify_e7_tripwire() -> tuple[bool, float]:
    """Re-score X_test[:100] and compare to committed reference.

    Returns (ok, max_abs_diff). On mismatch, Page 2 must refuse to score
    user input — it indicates the loaded E7 differs from the one used to
    produce the published §16 results.
    """
    e7 = get_e7()
    ref = _load_e7_ref()
    x = _load_x_test_first_n()
    fresh = e7.predict_proba(x).astype(np.float64)
    if fresh.shape != ref.shape:
        return False, float("inf")
    max_diff = float(np.max(np.abs(fresh - ref)))
    return max_diff <= E7_TRIPWIRE_TOL, max_diff


def verify_ae_tripwire() -> tuple[bool, float, float]:
    """Re-score X_test[:100] through scaler+AE; compare mean MSE.

    Returns (ok, fresh_mean_mse, ref_mean_mse). On mismatch, Page 2 should
    treat the AE card as unavailable — it usually indicates either AE weight
    drift or an accidentally-skipped scaler.transform().
    """
    ae = get_autoencoder()
    if ae is None:
        return False, float("nan"), float("nan")
    scaler = get_scaler()
    x = _load_x_test_first_n()
    xs = scaler.transform(x)
    recon = np.asarray(ae(xs, training=False))
    fresh_mean = float(np.mean((recon - xs) ** 2))
    ref = _load_ae_ref()
    ref_mean = float(ref["mean_mse"])
    return abs(fresh_mean - ref_mean) <= AE_TRIPWIRE_TOL, fresh_mean, ref_mean


def softmax_entropy(proba: np.ndarray) -> float:
    """Shannon entropy of a single 19-class softmax row, base-2 ln (matches §15D).

    The published `e7_entropy.npy` was computed with natural log; preserve that
    convention so percentile comparisons against entropy_thresholds.json are
    meaningful.
    """
    p = np.clip(proba, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def score_flow(x_raw: np.ndarray) -> dict[str, Any]:
    """Run the full pipeline on a single 44-feature flow.

    Args:
        x_raw: shape (44,) or (1, 44), unscaled.

    Returns dict with keys:
        proba          — 19-class softmax (np.ndarray, shape (19,))
        pred_class_idx — argmax index
        entropy        — Shannon entropy in nats
        ae_mse         — None if AE unavailable; else float
        ae_available   — bool
        x_scaled       — scaled 44-feature row (used by SHAP for Page 2)
    """
    x_raw = np.asarray(x_raw, dtype=np.float32).reshape(1, 44)
    e7 = get_e7()
    proba = e7.predict_proba(x_raw)[0].astype(np.float64)
    if proba.shape != (N_CLASSES,):
        raise RuntimeError(f"E7 returned shape {proba.shape}, expected ({N_CLASSES},)")
    pred_idx = int(np.argmax(proba))
    entropy = softmax_entropy(proba)

    scaler = get_scaler()
    x_scaled = scaler.transform(x_raw)

    ae = get_autoencoder()
    if ae is None:
        ae_mse: float | None = None
    else:
        recon = np.asarray(ae(x_scaled, training=False))
        ae_mse = float(np.mean((recon - x_scaled) ** 2))

    return {
        "proba": proba,
        "pred_class_idx": pred_idx,
        "entropy": entropy,
        "ae_mse": ae_mse,
        "ae_available": ae is not None,
        "x_scaled": x_scaled,
    }
