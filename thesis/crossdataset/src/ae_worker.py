"""
Isolated TensorFlow worker for the Flow autoencoder (Layer 2).

Run as a SUBPROCESS from run.py so TensorFlow never shares a process with
XGBoost.  XGBoost and TensorFlow each bundle their own OpenMP runtime and
deadlock when both are live in one process (confirmed on this machine); a
fresh subprocess gives TF a clean threading world.

Usage:
    python ae_worker.py <jobspec.json>

jobspec.json:
    {
      "benign_train_npy": "<path>",   # imputed CICIoMT2024 benign-train features
      "score_npy": {"name": "<path>", ...},  # imputed matrices to score
      "out_dir": "<path>"             # where mse_<name>.npy + ae_meta.json land
    }

Everything here is fit on CICIoMT2024 benign-train only (TRAP 1); the matrices
in ``score_npy`` are only ever transform+scored, never fit on.

NOTE: this worker deliberately imports NO scipy/sklearn.  Importing scipy before
TensorFlow deadlocks TF's training loop on this macOS/arm64 environment
(confirmed by bisection).  Standardization is therefore done with pure NumPy —
mathematically identical to StandardScaler (``(x - mean) / std`` fit on
benign-train).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import config


def build_ae(input_dim: int):
    """Deterministic dense AE — architecture verbatim from the thesis pipeline."""
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    tf.keras.utils.set_random_seed(config.TRAIN_SEED)
    inp = layers.Input(shape=(input_dim,), name="input")
    x = layers.Dense(32, activation="relu", name="enc_dense_32")(inp)
    x = layers.BatchNormalization(name="enc_bn_32")(x)
    x = layers.Dropout(0.2, name="enc_drop_32")(x)
    x = layers.Dense(16, activation="relu", name="enc_dense_16")(x)
    x = layers.BatchNormalization(name="enc_bn_16")(x)
    x = layers.Dropout(0.1, name="enc_drop_16")(x)
    bottleneck = layers.Dense(8, activation="relu", name="bottleneck")(x)
    x = layers.Dense(16, activation="relu", name="dec_dense_16")(bottleneck)
    x = layers.BatchNormalization(name="dec_bn_16")(x)
    x = layers.Dense(32, activation="relu", name="dec_dense_32")(x)
    x = layers.BatchNormalization(name="dec_bn_32")(x)
    out = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    ae = Model(inputs=inp, outputs=out, name="autoencoder")
    ae.compile(optimizer=tf.keras.optimizers.Adam(config.AE_LEARNING_RATE),
               loss="mse")
    return ae


def reconstruction_mse(model, x_scaled: np.ndarray) -> np.ndarray:
    x_hat = model.predict(x_scaled, batch_size=config.AE_PREDICT_BATCH, verbose=0)
    return np.mean((x_scaled - x_hat) ** 2, axis=1)


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy StandardScaler.fit — mean/std on benign-train (TRAP 1)."""
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0.0] = 1.0                 # guard zero-variance features
    return mu.astype("float32"), sd.astype("float32")


def standardize(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Transform-only; safe for the CICIoT2023 matrices."""
    return ((x - mu) / sd).astype("float32")


def main() -> None:
    from tensorflow.keras import callbacks

    spec = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(spec["out_dir"])
    epochs = int(spec.get("epochs", config.AE_EPOCHS))   # smoke can shorten this
    x_benign = np.load(spec["benign_train_npy"]).astype("float32")

    rng = np.random.default_rng(config.TRAIN_SEED)
    idx = rng.permutation(len(x_benign))
    cut = int(len(idx) * 0.8)
    tr, va = idx[:cut], idx[cut:]

    mu, sd = fit_standardizer(x_benign[tr])              # benign-train only
    x_tr, x_va = standardize(x_benign[tr], mu, sd), standardize(x_benign[va], mu, sd)

    model = build_ae(x_tr.shape[1])
    model.fit(
        x_tr, x_tr, validation_data=(x_va, x_va),
        epochs=epochs, batch_size=config.AE_BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss",
                                    patience=config.AE_PATIENCE,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=5, min_lr=1e-5),
        ],
        verbose=2,
    )

    mse_val = reconstruction_mse(model, x_va)
    thresholds = {f"p{p}": float(np.percentile(mse_val, p))
                  for p in config.AE_THRESHOLD_PERCENTILES}

    for name, path in spec["score_npy"].items():
        x = np.load(path).astype("float32")
        mse = reconstruction_mse(model, standardize(x, mu, sd))
        np.save(out_dir / f"mse_{name}.npy", mse)

    (out_dir / "ae_meta.json").write_text(json.dumps(
        {"thresholds": thresholds,
         "benign_val_mse_stats": {
             "mean": float(mse_val.mean()), "p90": thresholds.get("p90"),
             "p99": thresholds.get("p99")}},
        indent=2))
    print("[ae_worker] done", flush=True)


if __name__ == "__main__":
    main()
