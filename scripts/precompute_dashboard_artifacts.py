"""One-shot pre-computation of dashboard reference artifacts.

Run once (or whenever the upstream models / X_train change). The dashboard
NEVER imports this module — it only consumes the JSON / NPY outputs under
`dashboard/components/`.

Outputs:
    dashboard/components/feature_ranges.json
        Per-feature [p0.1, p1, p50, p99, p99.9] from X_train.npy. Used by
        Page 2's input range validation (yellow flag if user-input falls
        outside [p0.1, p99.9]).

    dashboard/components/e7_first100_proba_ref.npy
        Shape (100, 19) float64. E7 19-class softmax on X_test[:100].
        Used by Page 2's reproducibility tripwire #1: dashboard re-scores
        X_test[:100] at first model load and aborts if max |Δproba| > 1e-6.
        Catches model-version drift between training and serving.

    dashboard/components/ae_recon_ref.json
        Single float: mean MSE of AE on scaler.transform(X_test[:100]).
        Used by Page 2's tripwire #1b: dashboard recomputes the same
        scalar at first AE load and aborts if |Δ| > 1e-4. Catches scaler-
        direction bugs (forgotten transform → wrong by 8 orders of magnitude)
        and AE weight drift.

Verification (run after this script):
    python -c "import json; r=json.load(open('dashboard/components/feature_ranges.json')); print(len(r), 'features')"
    python -c "import numpy as np; a=np.load('dashboard/components/e7_first100_proba_ref.npy'); print(a.shape, a.dtype, a.sum(axis=1)[:3])"
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS_DIR = PROJECT_ROOT / "dashboard" / "components"

X_TRAIN_PATH = PROJECT_ROOT / "preprocessed" / "full_features" / "X_train.npy"
X_TEST_PATH = PROJECT_ROOT / "preprocessed" / "full_features" / "X_test.npy"
E7_MODEL_PATH = PROJECT_ROOT / "results" / "supervised" / "models" / "E7_xgb_full_original.pkl"
SCALER_PATH = PROJECT_ROOT / "results" / "unsupervised" / "models" / "scaler.pkl"
AE_PATH = PROJECT_ROOT / "results" / "unsupervised" / "models" / "autoencoder.keras"
SHAP_CONFIG_PATH = PROJECT_ROOT / "results" / "shap" / "config.json"

PERCENTILES = [0.1, 1.0, 50.0, 99.0, 99.9]
N_REF_ROWS = 100


def load_feature_names() -> list[str]:
    cfg = json.loads(SHAP_CONFIG_PATH.read_text())
    names = cfg["feature_names"]
    assert len(names) == 44, f"expected 44 features, got {len(names)}"
    return names


def compute_feature_ranges(feature_names: list[str]) -> dict[str, dict[str, float]]:
    print(f"loading {X_TRAIN_PATH} ...")
    x_train = np.load(X_TRAIN_PATH, mmap_mode="r")
    print(f"  shape={x_train.shape}, dtype={x_train.dtype}")
    print(f"  computing percentiles {PERCENTILES} per feature ...")
    pcts = np.percentile(x_train, PERCENTILES, axis=0)
    return {
        feature_names[i]: {
            f"p{p}": float(pcts[j, i])
            for j, p in enumerate(PERCENTILES)
        }
        for i in range(len(feature_names))
    }


def compute_e7_reference(n: int = N_REF_ROWS) -> np.ndarray:
    print(f"loading {E7_MODEL_PATH} ...")
    e7 = joblib.load(E7_MODEL_PATH)
    print(f"  type={type(e7).__name__}")
    x_test = np.load(X_TEST_PATH, mmap_mode="r")[:n]
    print(f"  scoring X_test[:{n}] shape={x_test.shape} ...")
    proba = e7.predict_proba(np.asarray(x_test))
    proba = proba.astype(np.float64)
    assert proba.shape == (n, 19), f"expected ({n}, 19), got {proba.shape}"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "softmax rows must sum to 1"
    return proba


def compute_ae_reference(n: int = N_REF_ROWS) -> dict[str, float | int]:
    from tensorflow import keras  # local import; precompute is a one-shot
    print(f"loading {SCALER_PATH} and {AE_PATH} ...")
    scaler = joblib.load(SCALER_PATH)
    ae = keras.models.load_model(AE_PATH)
    x_test = np.asarray(np.load(X_TEST_PATH, mmap_mode="r")[:n])
    x_scaled = scaler.transform(x_test)
    recon = ae.predict(x_scaled, verbose=0)
    mse_per_row = np.mean((recon - x_scaled) ** 2, axis=1)
    return {
        "n_rows": int(n),
        "mean_mse": float(mse_per_row.mean()),
        "max_mse": float(mse_per_row.max()),
        "min_mse": float(mse_per_row.min()),
    }


def main() -> None:
    COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = load_feature_names()

    print("\n=== feature_ranges.json ===")
    ranges = compute_feature_ranges(feature_names)
    out_ranges = COMPONENTS_DIR / "feature_ranges.json"
    out_ranges.write_text(json.dumps(ranges, indent=2) + "\n")
    print(f"  wrote {out_ranges} ({out_ranges.stat().st_size:,} B)")

    print("\n=== e7_first100_proba_ref.npy ===")
    e7_ref = compute_e7_reference()
    out_e7 = COMPONENTS_DIR / "e7_first100_proba_ref.npy"
    np.save(out_e7, e7_ref)
    print(f"  wrote {out_e7} ({out_e7.stat().st_size:,} B)")
    print(f"  argmax distribution (first 10): {e7_ref.argmax(axis=1)[:10].tolist()}")

    print("\n=== ae_recon_ref.json ===")
    ae_ref = compute_ae_reference()
    out_ae = COMPONENTS_DIR / "ae_recon_ref.json"
    out_ae.write_text(json.dumps(ae_ref, indent=2) + "\n")
    print(f"  wrote {out_ae}: {ae_ref}")


if __name__ == "__main__":
    main()
