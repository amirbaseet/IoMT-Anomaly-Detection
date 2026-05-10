"""
Phase 6E (proposed) — LSTM-Autoencoder Training
================================================
Path B Tier 2 extension: third Layer-2 substitution alternative alongside the
Phase 5 deterministic AE and the §15E β-VAE family. Implements the bounded,
defensible LSTM-AE work for the README §15E.7 Layer-2 substitution
robustness check.

The Layer-2 component is the only thing that changes; E7, the LOO-XGBoost
ensemble, the deterministic AE, the four β-VAEs, and the entropy-fusion
logic are reused verbatim from saved artifacts. Reuses Phase 5 benign-train
data, scaler, and 80/20 split convention exactly.

Pre-commitment for scope control:
    "If LSTM-AE training does not converge cleanly within the first week,
     document the negative result and stop. Do not escalate into temporal
     smoothing variants or other architectural changes on this CSV."

Usage
-----
    SMOKE=1 python notebooks/lstm_ae_train.py    # 5 epochs on C1 only, 7 sanity checks
    python notebooks/lstm_ae_train.py            # full sweep over C1..C6, 100 epochs each

Sweep size convention (per Phase B brief §5.2):
    "5 distinct architectures + 1 monitoring-only run (C6 = C1 + grad-norm logging)"

Outputs
-------
    results/unsupervised/lstm_ae/_smoke/                 (smoke run)
        model.keras           full saved Keras model (C1 baseline, 5 epochs)
        history.json          {'loss', 'val_loss'} progression
        manifest.json         hyperparameters + diagnostic stats
        smoke_report.json     7 sanity checks × pass/fail
        FAILED.json           only on smoke failure; lists failed checks
    results/unsupervised/lstm_ae/c{1..6}/                (full sweep, per config)
        model.keras           full saved Keras model
        history.json          {'loss', 'val_loss'} progression
        manifest.json         hyperparameters + diagnostic stats
        val_recon_err.npy     per-sample MSE on benign-val   [gitignored]
        test_recon_err.npy    per-sample MSE on full test    [gitignored]
        recon_test_first100.npy    for tripwire build        [gitignored]
        grad_norms.json       only for c6: per-epoch max(grad_norm)
    results/unsupervised/lstm_ae/all_configs_summary.csv     (after full sweep)
    results/unsupervised/lstm_ae/_arch_summary.txt           (one-time)

Loss convention: mean-MSE per sample (Keras `loss="mse"`), matches Phase 5 AE
and yields val_loss directly comparable to AE's best_val_loss = 0.1987786442041397
in results/unsupervised/config.json:66. Gate 1 G1.1 threshold (descriptive only):
val_loss <= 1.5 × 0.1987786442041397 = 0.298167966306. Gate 1 verdict file
gate1_report.json is intentionally NOT written here — that is Phase C, owned
by the human reviewer.
"""

# ============================================================================
# 0 . CONFIG & MODE
# ============================================================================
import os
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SMOKE = bool(int(os.environ.get("SMOKE", "0")))
# SMOKE_OVERRIDE = explicit, transparent bypass of the prior-smoke-FAIL guard.
# Added 2026-05-08 to permit launching the full sweep after a senior-engineer
# review override of the v4 smoke FAIL on (e)+(f) sample-of-one checks while
# (h) population AUC PASS confirmed model health. See _phase_b_pause.json v1.6.
SMOKE_OVERRIDE = bool(int(os.environ.get("SMOKE_OVERRIDE", "0")))
RANDOM_STATE = 42

# ---- paths (mirror Phase 5 unsupervised_training.py and vae_train.py) ----
PREPROCESSED_DIR = Path("./preprocessed/")
PHASE5_DIR       = Path("./results/unsupervised/")
OUTPUT_DIR       = PHASE5_DIR / "lstm_ae"
SCALER_PATH      = PHASE5_DIR / "models" / "scaler.pkl"
AE_CONFIG_PATH   = PHASE5_DIR / "config.json"

# ---- LSTM-AE structural constants (per plan §1.2 final architecture) ----
LATENT_DIM        = 8        # FIXED; matches AE bottleneck and VAE latent
INPUT_DIM         = 44       # number of features per flow
TIMESTEPS         = 44       # within-flow sequence length (per plan §2)
CHANNELS          = 1        # one scalar per timestep

# ---- shared training hyperparameters (mirror Phase 5 AE / VAE) ----
EPOCHS_FULL       = 100
EPOCHS_SMOKE      = 5
BATCH_SIZE        = 512
PATIENCE          = 10
PER_CONFIG_TIME_CAP_S = 3600   # 60 min/config soft cap — raised 2026-05-08 from
                               # 600 to give LSTM-AE the same compute budget as
                               # the AE reference (36 epochs × 86 s/epoch ≈
                               # 3100 s) plus 16% margin. Plan addendum 2026-05-08
                               # documents the rationale; this is a fair-comparison
                               # correction, not scope expansion.
PREDICT_BATCH     = 8192

# ---- reproducibility tolerances ----
ROUNDTRIP_MAX_ABS_DIFF_TOL = 1e-4   # mirrors vae_train.py:77

# ---- AE reference for descriptive Gate 1 fields (per plan §6.1) ----
AE_BEST_VAL_LOSS         = 0.1987786442041397   # config.json:66
AE_REF_THRESHOLD_1_5X    = 1.5 * AE_BEST_VAL_LOSS

# ---- Smoke gate criteria (per plan §8.1, recalibrated 2026-05-07 post Phase C) ----
# (e) threshold recalibrated 0.5 → 5.0 based on population review of v3 smoke output
# (h) added: population AUC ≥ 0.85 across the full 892K-row test set
SMOKE_BENIGN_RECON_MSE_MAX = 5.0    # check (e) — was 0.5, raised above AE benign p99=1.20
SMOKE_VAL_OVER_TRAIN_MAX   = 10.0   # check (d) val_loss <= 10× train_loss
SMOKE_POPULATION_AUC_MIN   = 0.85   # check (h) — well below AE 0.9892 and v3 observed 0.9769

# ---- Gate 1 G1.3 sub-thresholds (per plan §6.3, descriptive only) ----
G1_3_STD_OVER_MEAN_MIN   = 0.1
G1_3_STD_OVER_MEAN_MAX   = 5.0
G1_3_MAX_OVER_MEAN_MAX   = 100.0
G1_3_HIST_NBINS          = 100
G1_3_HIST_CLIP_PCTL      = 99.5
G1_2_GRAD_NORM_MAX       = 1e3      # plan §6.2 (only meaningful for C6)


# ---- Search grid: 5 distinct architectures + 1 monitoring-only (C6) ----
# Per plan §5.2 + Phase B brief reporting convention. All 6 configs run the
# same pure-numpy GradientTape loop (Phase B fix (3) consolidation); the C6
# duplicate of C1 is retained because the plan §5.2 calls for it and because
# C6 is the canonical row whose grad-norm trace is saved as a separate JSON
# artifact. Every config now records max(grad_norm) per epoch in its
# manifest; only C6 additionally writes c6/grad_norms.json (per plan §7).
CONFIGS = [
    {"id": "c1", "enc_units_64": 64,  "enc_units_32": 32, "lr": 1e-3, "rec_dropout": 0.0},
    {"id": "c2", "enc_units_64": 64,  "enc_units_32": 32, "lr": 5e-4, "rec_dropout": 0.0},
    {"id": "c3", "enc_units_64": 32,  "enc_units_32": 16, "lr": 1e-3, "rec_dropout": 0.0},
    {"id": "c4", "enc_units_64": 128, "enc_units_32": 64, "lr": 1e-3, "rec_dropout": 0.0},
    {"id": "c5", "enc_units_64": 64,  "enc_units_32": 32, "lr": 1e-3, "rec_dropout": 0.1},
    {"id": "c6", "enc_units_64": 64,  "enc_units_32": 32, "lr": 1e-3, "rec_dropout": 0.0},
]
GRAD_NORMS_JSON_CONFIGS = {"c6"}     # configs that additionally save grad_norms.json
SMOKE_CONFIG = CONFIGS[0]            # smoke uses C1 baseline

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1 . REPRODUCIBILITY (mirror vae_train.py:81-105)
# ============================================================================
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
random.seed(RANDOM_STATE)

import numpy as np
np.random.seed(RANDOM_STATE)

import pandas as pd
import joblib
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score   # smoke check (h), added 2026-05-07

import tensorflow as tf
# Phase B fix (3) per _phase_b_pause.json v1.2: training and scoring run
# through a pure-numpy + manual GradientTape loop, bypassing tf.data and
# Keras 3 data adapters entirely. This is the actual fix for the macOS
# TF 2.21 + Keras 3.14 + LSTM hang observed in 3 prior runs (logs in
# results/unsupervised/lstm_ae/_phase_b_logs/). Fix (1) eager mode was
# tried first and proved insufficient — the hang is in the data path,
# not XLA — and has been reverted.
from tensorflow.keras import layers, Model, callbacks  # noqa: F401 (callbacks kept for parity)
import keras

tf.random.set_seed(RANDOM_STATE)
try:
    tf.keras.utils.set_random_seed(RANDOM_STATE)
except Exception:
    pass

_T0 = time.time()
def log(msg: str = "") -> None:
    elapsed = time.time() - _T0
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] [+{elapsed:6.1f}s] {msg}", flush=True)

log(f"LSTM-AE training script | SMOKE={SMOKE} | random_state={RANDOM_STATE}")
log(f"TensorFlow {tf.__version__} | tf.keras {tf.keras.__version__} | NumPy {np.__version__}")
assert tf.keras.__version__.startswith("3."), (
    f"Keras 3 required; got {tf.keras.__version__}"
)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    log(f"GPU device(s): {[g.name for g in gpus]}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
else:
    log("No GPU detected; CPU training (LSTM is slower than dense AE here).")


# ============================================================================
# 2 . DATA LOADING (mirror vae_train.py:136-184 verbatim)
# ============================================================================
log("\n" + "=" * 70)
log("SECTION 2 . DATA LOADING")
log("=" * 70)

X_train = np.load(PREPROCESSED_DIR / "full_features" / "X_train.npy").astype(np.float32)
y_train = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_train.csv")

LBL_COL = "label" if "label" in y_train.columns else y_train.columns[0]
y_train_lbl = y_train[LBL_COL].astype(str).values
assert X_train.shape[1] == INPUT_DIM, f"expected {INPUT_DIM} features, got {X_train.shape[1]}"

benign_mask = (y_train_lbl == "Benign")
X_benign_full = X_train[benign_mask]
log(f"Benign rows in train split: {X_benign_full.shape[0]:,}")

# 80/20 split — identical RNG idiom to vae_train.py:151-156 and
# unsupervised_training.py:189-196. Same seed + same array length =>
# identical (X_benign_train, X_benign_val) shape and contents as Phase 5.
rng = np.random.default_rng(RANDOM_STATE)
perm = rng.permutation(X_benign_full.shape[0])
split_idx = int(0.8 * X_benign_full.shape[0])
X_benign_train = X_benign_full[perm[:split_idx]]
X_benign_val   = X_benign_full[perm[split_idx:]]
log(f"AE-train (benign): {X_benign_train.shape}")
log(f"AE-val   (benign): {X_benign_val.shape}")
del X_train, X_benign_full

# ---- Reuse Phase 5 scaler (NEVER refit; per plan §3.2) ----
if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Phase 5 scaler not found at {SCALER_PATH}. Run unsupervised_training.py first."
    )
scaler = joblib.load(SCALER_PATH)
assert scaler.n_features_in_ == INPUT_DIM, (
    f"scaler n_features_in_={scaler.n_features_in_} but expected {INPUT_DIM}"
)
log(f"Loaded Phase 5 scaler: n_samples_seen={scaler.n_samples_seen_:,}, "
    f"means range=[{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")

# Apply scaler in 2D BEFORE the model's internal Reshape (per plan §3.2).
X_benign_train = scaler.transform(X_benign_train).astype(np.float32)
X_benign_val   = scaler.transform(X_benign_val).astype(np.float32)
log(f"After scaling: benign-train mean={X_benign_train.mean():.6f}, "
    f"std={X_benign_train.std():.6f}")

X_test = scaler.transform(
    np.load(PREPROCESSED_DIR / "full_features" / "X_test.npy").astype(np.float32)
).astype(np.float32)
y_test = pd.read_csv(PREPROCESSED_DIR / "full_features" / "y_test.csv")
y_test_lbl = y_test[LBL_COL].astype(str).values if LBL_COL in y_test.columns \
             else y_test[y_test.columns[0]].astype(str).values
benign_test_mask = (y_test_lbl == "Benign")
attack_test_mask = ~benign_test_mask
log(f"Test set (scaled): {X_test.shape}, "
    f"benign={benign_test_mask.sum():,}, attack={attack_test_mask.sum():,}")


# ============================================================================
# 3 . LSTM-AE BUILDER (per plan §1.2 final architecture, §2 sequence formulation)
# ============================================================================
def build_lstm_ae(enc_units_64: int, enc_units_32: int, lr: float,
                  rec_dropout: float = 0.0) -> Model:
    """LSTM-AE seq2seq matching plan §1.2.

    Caller passes 2D X (n, 44); the internal Reshape converts to (n, 44, 1)
    so the LSTM gates operate over 44 timesteps with 1 channel. Output is
    Reshape'd back to (n, 44) so MSE compares directly against the input.
    The decoder uses the Cho/Sutskever RepeatVector + TimeDistributed
    pattern (plan §1.3 deviation 1) — the decision report's bare
    Dense(32) → LSTM(64) → Dense(44) collapses to a non-recurrent decoder
    over a 1-step "sequence" and is rejected.
    """
    inp = layers.Input(shape=(INPUT_DIM,), name="input")
    x = layers.Reshape((TIMESTEPS, CHANNELS), name="enc_reshape")(inp)

    # ---- encoder ----
    x = layers.LSTM(
        enc_units_64, return_sequences=True,
        recurrent_dropout=rec_dropout,
        name="enc_lstm_64",
    )(x)
    x = layers.LSTM(
        enc_units_32, return_sequences=False,
        recurrent_dropout=rec_dropout,
        name="enc_lstm_32",
    )(x)
    z = layers.Dense(LATENT_DIM, activation="linear", name="latent")(x)

    # ---- decoder (symmetric seq2seq, per plan §1.2/§1.3) ----
    h = layers.RepeatVector(TIMESTEPS, name="dec_repeat")(z)
    h = layers.LSTM(
        enc_units_32, return_sequences=True,
        recurrent_dropout=rec_dropout,
        name="dec_lstm_32",
    )(h)
    h = layers.LSTM(
        enc_units_64, return_sequences=True,
        recurrent_dropout=rec_dropout,
        name="dec_lstm_64",
    )(h)
    h = layers.TimeDistributed(
        layers.Dense(CHANNELS, activation="linear"),
        name="dec_td",
    )(h)
    out = layers.Reshape((INPUT_DIM,), name="reconstruction")(h)

    model = Model(inputs=inp, outputs=out,
                  name=f"lstm_ae_{enc_units_64}_{enc_units_32}_lr{lr:g}_rd{rec_dropout:g}")
    # Loss = mean-MSE per sample (plan §1.4); matches Phase 5 AE convention
    # so val_loss is directly comparable to AE_BEST_VAL_LOSS = 0.1987786...
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
    )
    return model


# ============================================================================
# 4 . SCORING + DIAGNOSTICS
# ============================================================================
def model_predict_pure(model: Model, X: np.ndarray,
                       batch_size: int = PREDICT_BATCH) -> np.ndarray:
    """Pure-numpy batched inference; bypasses Keras data adapters.

    Phase B fix (3): the same hang that killed model.fit() likely lives in
    Keras 3's prediction data adapter as well. We avoid model.predict()
    entirely and call the model directly on tf.constant'ed batches, then
    np.concatenate the results.
    """
    n = X.shape[0]
    out_chunks = []
    for start in range(0, n, batch_size):
        x_batch = tf.constant(X[start:start + batch_size])
        y_batch = model(x_batch, training=False).numpy()
        out_chunks.append(y_batch)
    return np.concatenate(out_chunks, axis=0).astype(np.float32)


def compute_recon_mse_batched(model: Model, X: np.ndarray,
                              batch_size: int = PREDICT_BATCH) -> np.ndarray:
    """Per-sample mean-MSE reconstruction error over 44 features.

    X is 2D (n, 44); the model handles the internal reshape. Output is
    a 1D float32 array of length n. Uses model_predict_pure to bypass
    Keras data adapters (Phase B fix (3)).
    """
    x_hat = model_predict_pure(model, X, batch_size=batch_size)
    return np.mean((X - x_hat) ** 2, axis=1).astype(np.float32)


def histogram_unimodal(arr: np.ndarray, nbins: int = G1_3_HIST_NBINS,
                       clip_pctl: float = G1_3_HIST_CLIP_PCTL) -> bool:
    """G1.3 unimodality sub-check (plan §6.3).

    Clip at p99.5 (mirrors unsupervised_training.py:672), build a 100-bin
    histogram, run scipy.find_peaks with height = 0.5×max-bin-height; pass
    if there are 0 or 1 peaks above that threshold.
    """
    if not np.isfinite(arr).any():
        return False
    cap = np.percentile(arr, clip_pctl)
    clipped = np.clip(arr, a_min=None, a_max=cap)
    counts, _ = np.histogram(clipped, bins=nbins)
    if counts.max() == 0:
        return False
    peaks, _ = find_peaks(counts, height=0.5 * counts.max())
    return bool(len(peaks) <= 1)


# ============================================================================
# 5 . TRAINING — pure-numpy GradientTape loop (Phase B fix (3))
# ============================================================================
# All 6 configs use this single training implementation. tf.data.Dataset and
# model.fit() / model.predict() are deliberately avoided — they share a
# Keras 3 data-adapter code path that hangs on macOS TF 2.21 + LSTM (3 prior
# runs all hung at "Epoch 1/5" with the process sleeping; see
# results/unsupervised/lstm_ae/_phase_b_pause.json v1.2).
def train_with_tape(model: Model, lr: float, epochs: int, time_cap_s: float,
                    log_prefix: str):
    """Pure-numpy GradientTape training loop (Phase B fix (3)).

    Behaviour preserved from the prior fit-based path:
      - mean-MSE per sample loss (plan §1.4)
      - Adam(learning_rate=lr)
      - EarlyStopping(monitor=val_loss, patience=PATIENCE, restore_best_weights=True)
      - ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=5, min_lr=1e-6)
      - PER_CONFIG_TIME_CAP_S soft cap

    Differences from the prior path:
      - shuffle is a single np.random.default_rng(RANDOM_STATE) permutation
        per epoch (no tf.data.Dataset, no AUTOTUNE prefetch)
      - validation MSE computed via model_predict_pure (no model.predict)
      - records per-epoch max(grad_norm); every config returns this trace,
        but only configs in GRAD_NORMS_JSON_CONFIGS (currently {"c6"})
        write the separate grad_norms.json artifact
    """
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    history_dict = {"loss": [], "val_loss": []}
    grad_norms_per_epoch = []

    best_val_loss = float("inf")
    patience_counter = 0
    plateau_counter  = 0
    best_weights = None

    n_train = X_benign_train.shape[0]
    rng_train = np.random.default_rng(RANDOM_STATE)

    t_train = time.time()
    over_budget = False
    epochs_actual = 0

    for epoch in range(epochs):
        # explicit start-of-epoch heartbeat — distinguishes "running but slow"
        # from "hung" within the 5-minute hang-detection window
        log(f"  {log_prefix} starting epoch {epoch+1}/{epochs} "
            f"({n_train // BATCH_SIZE + (1 if n_train % BATCH_SIZE else 0)} batches)")
        perm = rng_train.permutation(n_train)
        epoch_losses = []
        epoch_grad_norms = []

        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            x_batch = tf.constant(X_benign_train[batch_idx])

            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                # mean-MSE per sample, then mean over batch (matches loss="mse")
                loss = tf.reduce_mean(
                    tf.reduce_mean(tf.square(x_batch - y_pred), axis=1)
                )
            grads = tape.gradient(loss, model.trainable_weights)
            grad_norm = tf.linalg.global_norm(grads)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            epoch_losses.append(float(loss))
            epoch_grad_norms.append(float(grad_norm))

        train_loss = float(np.mean(epoch_losses))
        max_grad_norm_epoch = float(np.max(epoch_grad_norms))

        # validation pass — pure-tensor batched inference (no Keras data adapters)
        val_pred = model_predict_pure(model, X_benign_val, batch_size=BATCH_SIZE)
        val_loss = float(np.mean(np.mean((X_benign_val - val_pred) ** 2, axis=1)))

        history_dict["loss"].append(train_loss)
        history_dict["val_loss"].append(val_loss)
        grad_norms_per_epoch.append(max_grad_norm_epoch)
        epochs_actual = epoch + 1

        log(f"  {log_prefix} epoch {epoch+1:3d}/{epochs} done: "
            f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"max_grad_norm={max_grad_norm_epoch:.4f}  "
            f"lr={float(optimizer.learning_rate.numpy()):.2e}  "
            f"elapsed={time.time() - t_train:.1f}s")

        # EarlyStopping(restore_best_weights=True) + ReduceLROnPlateau(0.5, p=5)
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            patience_counter = 0
            plateau_counter = 0
            best_weights = [w.numpy() for w in model.weights]
        else:
            patience_counter += 1
            plateau_counter  += 1
            if plateau_counter >= 5:
                new_lr = max(float(optimizer.learning_rate.numpy()) * 0.5, 1e-6)
                optimizer.learning_rate.assign(new_lr)
                plateau_counter = 0
                log(f"  {log_prefix} ReduceLROnPlateau -> lr={new_lr:.2e}")
            if patience_counter >= PATIENCE:
                log(f"  {log_prefix} EarlyStopping at epoch {epoch+1}")
                break

        if (time.time() - t_train) > time_cap_s:
            over_budget = True
            log(f"  {log_prefix} PER_CONFIG_TIME_CAP_S reached")
            break

    if best_weights is not None:
        for w, bw in zip(model.weights, best_weights):
            w.assign(bw)

    train_time = time.time() - t_train
    return history_dict, epochs_actual, train_time, over_budget, grad_norms_per_epoch


# ============================================================================
# 7 . TRAIN ONE CONFIG (dispatch + scoring + manifest emission)
# ============================================================================
def train_one_config(cfg: dict, out_dir: Path, epochs: int,
                     time_cap_s: float):
    """Train a single config; save artifacts; return (manifest, model).

    All configs use the pure-numpy GradientTape loop (Phase B fix (3)).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"\n--- Training {cfg['id']} -> {out_dir} (epochs<={epochs}, "
        f"cap {time_cap_s}s, units={cfg['enc_units_64']}/{cfg['enc_units_32']}, "
        f"lr={cfg['lr']:g}, rec_dropout={cfg['rec_dropout']}) ---")

    model = build_lstm_ae(
        enc_units_64=cfg["enc_units_64"],
        enc_units_32=cfg["enc_units_32"],
        lr=cfg["lr"],
        rec_dropout=cfg["rec_dropout"],
    )

    arch_path = out_dir.parent / "_arch_summary.txt"
    if not arch_path.exists():
        lines = []
        model.summary(print_fn=lambda s: lines.append(s))
        arch_path.write_text("\n".join(lines))
        log("Architecture summary written to lstm_ae/_arch_summary.txt")

    history_dict, epochs_actual, train_time, over_budget, grad_norms = \
        train_with_tape(
            model, lr=cfg["lr"], epochs=epochs, time_cap_s=time_cap_s,
            log_prefix=f"[{cfg['id']}]",
        )

    log(f"{cfg['id']}: {epochs_actual} epochs in {train_time:.1f}s "
        f"({train_time/60:.2f} min){' [OVER BUDGET]' if over_budget else ''}")

    # ---- score val + test ----
    log(f"Scoring benign-val (n={len(X_benign_val):,})...")
    val_recon_err = compute_recon_mse_batched(model, X_benign_val)
    log(f"Scoring full test  (n={len(X_test):,})...")
    test_recon_err = compute_recon_mse_batched(model, X_test)

    # ---- NaN/Inf hard check (mirrors vae_train.py:443-454) ----
    for nm, arr in [("val_recon_err", val_recon_err),
                    ("test_recon_err", test_recon_err)]:
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise RuntimeError(
                f"{cfg['id']}: NaN/Inf in {nm} (n_nan={n_nan}, n_inf={n_inf}). "
                "Likely gradient explosion through unrolled LSTM timesteps."
            )

    # ---- save artifacts ----
    model.save(out_dir / "model.keras")
    np.save(out_dir / "val_recon_err.npy", val_recon_err)
    np.save(out_dir / "test_recon_err.npy", test_recon_err)

    # First-100 reconstruction (for tripwire build later in Phase C).
    # Uses model_predict_pure to avoid Keras data adapters (Phase B fix (3)).
    recon_test_first100 = model_predict_pure(model, X_test[:100], batch_size=100)
    np.save(out_dir / "recon_test_first100.npy", recon_test_first100.astype(np.float32))

    with open(out_dir / "history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    # Every config produces a grad-norm trace under fix (3); only configs in
    # GRAD_NORMS_JSON_CONFIGS additionally write the separate JSON artifact
    # (preserves plan §7 layout where only c6 has grad_norms.json).
    if cfg["id"] in GRAD_NORMS_JSON_CONFIGS:
        with open(out_dir / "grad_norms.json", "w") as f:
            json.dump({
                "max_grad_norm_per_epoch": [float(g) for g in grad_norms],
                "max_grad_norm_overall":    float(np.max(grad_norms)),
                "g1_2_threshold":           G1_2_GRAD_NORM_MAX,
                "g1_2_descriptive_pass":    bool(np.max(grad_norms) <= G1_2_GRAD_NORM_MAX),
            }, f, indent=2)

    # ---- diagnostic stats ----
    best_val_loss = float(min(history_dict["val_loss"]))
    val_recon_mean = float(val_recon_err.mean())
    val_recon_p90  = float(np.percentile(val_recon_err, 90))
    val_recon_p95  = float(np.percentile(val_recon_err, 95))
    val_recon_p99  = float(np.percentile(val_recon_err, 99))
    test_recon_mean = float(test_recon_err.mean())
    test_recon_std  = float(test_recon_err.std())
    test_recon_max  = float(test_recon_err.max())
    test_recon_finite = bool(np.isfinite(test_recon_err).all())
    test_std_over_mean = test_recon_std / max(test_recon_mean, 1e-12)
    test_max_over_mean = test_recon_max / max(test_recon_mean, 1e-12)
    test_recon_unimodal = histogram_unimodal(test_recon_err)

    # ---- descriptive Gate 1 fields (verdict file is Phase C, not here) ----
    # Under fix (3), every config produces a grad-norm trace, so G1.2 is
    # computable for all configs (no None case anymore).
    max_grad_norm_overall = float(np.max(grad_norms))
    g1_1_descriptive_pass = bool(best_val_loss <= AE_REF_THRESHOLD_1_5X)
    g1_2_descriptive_pass = bool(max_grad_norm_overall <= G1_2_GRAD_NORM_MAX)
    g1_3_descriptive_pass = bool(
        test_recon_unimodal
        and (G1_3_STD_OVER_MEAN_MIN <= test_std_over_mean <= G1_3_STD_OVER_MEAN_MAX)
        and (test_max_over_mean <= G1_3_MAX_OVER_MEAN_MAX)
        and test_recon_finite
    )

    manifest = {
        "config_id":             cfg["id"],
        "enc_units_64":          int(cfg["enc_units_64"]),
        "enc_units_32":          int(cfg["enc_units_32"]),
        "learning_rate":         float(cfg["lr"]),
        "recurrent_dropout":     float(cfg["rec_dropout"]),
        "training_path":         "gradient_tape_pure_numpy",
        "input_dim":             int(INPUT_DIM),
        "timesteps":             int(TIMESTEPS),
        "channels":              int(CHANNELS),
        "latent_dim":            int(LATENT_DIM),
        "epochs_max":            int(epochs),
        "epochs_actual":         int(epochs_actual),
        "batch_size":            int(BATCH_SIZE),
        "patience":              int(PATIENCE),
        "best_val_loss":         best_val_loss,
        "training_time_s":       float(train_time),
        "early_stopped_by_budget": bool(over_budget),
        "loss_convention":       "mean-MSE per sample (matches Phase 5 AE: loss=\"mse\")",
        "ae_reference_val_loss": AE_BEST_VAL_LOSS,
        "ae_reference_threshold_1_5x": AE_REF_THRESHOLD_1_5X,
        # diagnostic stats (val)
        "val_recon_mean":        val_recon_mean,
        "val_recon_p90":         val_recon_p90,
        "val_recon_p95":         val_recon_p95,
        "val_recon_p99":         val_recon_p99,
        # diagnostic stats (test)
        "test_recon_mean":       test_recon_mean,
        "test_recon_std":        test_recon_std,
        "test_recon_max":        test_recon_max,
        "test_recon_std_over_mean": float(test_std_over_mean),
        "test_recon_max_over_mean": float(test_max_over_mean),
        "test_recon_unimodal":   test_recon_unimodal,
        "test_recon_finite":     test_recon_finite,
        # descriptive Gate 1 (binary verdict file gate1_report.json is Phase C)
        "gate1_descriptive": {
            "g1_1_val_loss_le_1_5x_ae":  g1_1_descriptive_pass,
            "g1_2_grad_norm_le_1e3":     g1_2_descriptive_pass,
            "g1_3_recon_histogram_ok":   g1_3_descriptive_pass,
        },
        "max_grad_norm_overall":  max_grad_norm_overall,
        "max_grad_norm_per_epoch": [float(g) for g in grad_norms],
        "tf_version":            tf.__version__,
        "keras_version":         tf.keras.__version__,
        "random_state":          int(RANDOM_STATE),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest, model


# ============================================================================
# 8 . ROUND-TRIP SAVE/LOAD VERIFICATION (smoke check g)
# ============================================================================
def verify_roundtrip(model_path: Path, sample_X: np.ndarray,
                     original_pred: np.ndarray):
    reloaded = tf.keras.models.load_model(model_path, compile=False)
    reload_pred = reloaded(sample_X, training=False).numpy()
    abs_diff = np.abs(reload_pred - original_pred)
    return float(abs_diff.max()), float(abs_diff.mean()), reloaded


# ============================================================================
# 9 . SMOKE GATE (5 epochs on C1, 7 sanity checks per plan §8.1)
# ============================================================================
def run_smoke() -> bool:
    log("\n" + "=" * 70)
    log(f"PHASE 1 . SMOKE GATE ({SMOKE_CONFIG['id']}, 5 epochs, _smoke/ dir)")
    log("=" * 70)
    smoke_dir = OUTPUT_DIR / "_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    checks = {}
    failure_reason = None

    # ---- check (a): model builds without error ----
    try:
        probe_model = build_lstm_ae(
            enc_units_64=SMOKE_CONFIG["enc_units_64"],
            enc_units_32=SMOKE_CONFIG["enc_units_32"],
            lr=SMOKE_CONFIG["lr"],
            rec_dropout=SMOKE_CONFIG["rec_dropout"],
        )
        summary_lines = []
        probe_model.summary(print_fn=lambda s: summary_lines.append(s))
        checks["a_model_builds"] = bool(len(summary_lines) > 0)
    except Exception as e:
        log(f"[CHECK a EXC] {type(e).__name__}: {e}")
        checks["a_model_builds"] = False
        failure_reason = f"check (a) build raised {type(e).__name__}: {e}"
        probe_model = None

    # ---- check (b): forward pass shape correct ((4,44) -> (4,44)) ----
    if probe_model is not None:
        try:
            probe_in = np.zeros((4, INPUT_DIM), dtype=np.float32)
            probe_out = probe_model(probe_in, training=False).numpy()
            checks["b_forward_pass_shape"] = bool(
                probe_out.shape == (4, INPUT_DIM)
            )
        except Exception as e:
            log(f"[CHECK b EXC] {type(e).__name__}: {e}")
            checks["b_forward_pass_shape"] = False
            failure_reason = failure_reason or f"check (b) forward raised {type(e).__name__}: {e}"
    else:
        checks["b_forward_pass_shape"] = False

    # If (a) or (b) already failed, write FAILED.json and bail out without
    # spending 5 epochs of training.
    if not (checks.get("a_model_builds") and checks.get("b_forward_pass_shape")):
        smoke_report = {
            "overall_pass":  False,
            "checks":        checks,
            "failure_reason": failure_reason or "build/shape check failed",
        }
        with open(smoke_dir / "smoke_report.json", "w") as f:
            json.dump(smoke_report, f, indent=2)
        with open(smoke_dir / "FAILED.json", "w") as f:
            json.dump({
                "failed_checks": [k for k, v in checks.items() if not v],
                "checks": checks,
                "failure_reason": failure_reason,
            }, f, indent=2)
        log(f"\nSMOKE GATE: FAIL (build/shape) - DO NOT launch full sweep")
        return False

    # ---- run 5-epoch training on C1 baseline ----
    manifest, model = train_one_config(
        cfg=SMOKE_CONFIG, out_dir=smoke_dir,
        epochs=EPOCHS_SMOKE, time_cap_s=PER_CONFIG_TIME_CAP_S,
    )

    with open(smoke_dir / "history.json") as f:
        h = json.load(f)
    train_loss = h["loss"]
    val_loss   = h["val_loss"]

    # ---- check (c): train_loss decreases over 5 epochs ----
    checks["c_train_loss_decreases"] = bool(
        all(np.isfinite(v) for v in train_loss)
        and train_loss[-1] < train_loss[0]
    )

    # ---- check (d): val_loss within 10× train_loss at final epoch ----
    final_train = train_loss[-1]
    final_val   = val_loss[-1]
    if final_train > 0:
        ratio = final_val / final_train
    else:
        ratio = float("inf")
    checks["d_val_within_10x_train"] = bool(
        np.isfinite(final_val)
        and np.isfinite(final_train)
        and ratio <= SMOKE_VAL_OVER_TRAIN_MAX
    )

    # ---- check (e): known benign sample reconstructs with MSE < 0.5 ----
    # X_test is already scaled. Pick the first benign test sample.
    benign_idx = int(np.argmax(benign_test_mask))
    if not benign_test_mask[benign_idx]:
        checks["e_benign_recon_close"] = False
        log("[CHECK e WARN] no benign sample found in test set — unexpected")
        benign_mse = float("nan")
    else:
        x_b = X_test[benign_idx:benign_idx+1]
        x_b_hat = model(tf.constant(x_b), training=False).numpy()
        benign_mse = float(np.mean((x_b - x_b_hat) ** 2))
        checks["e_benign_recon_close"] = bool(benign_mse < SMOKE_BENIGN_RECON_MSE_MAX)

    # ---- check (f): known attack sample has higher MSE than benign sample ----
    attack_idx_arr = np.where(attack_test_mask)[0]
    if len(attack_idx_arr) == 0:
        checks["f_attack_recon_higher"] = False
        log("[CHECK f WARN] no attack sample found in test set — unexpected")
        attack_mse = float("nan")
    else:
        attack_idx = int(attack_idx_arr[0])
        x_a = X_test[attack_idx:attack_idx+1]
        x_a_hat = model(tf.constant(x_a), training=False).numpy()
        attack_mse = float(np.mean((x_a - x_a_hat) ** 2))
        checks["f_attack_recon_higher"] = bool(
            np.isfinite(attack_mse) and np.isfinite(benign_mse)
            and attack_mse > benign_mse
        )

    # ---- check (g): save / reload bit-stable to within 1e-4 ----
    sample_X = X_benign_val[:100]
    original_pred = model(sample_X, training=False).numpy()
    try:
        max_diff, mean_diff, _ = verify_roundtrip(
            smoke_dir / "model.keras", sample_X, original_pred,
        )
        checks["g_roundtrip_bit_stable"] = bool(max_diff < ROUNDTRIP_MAX_ABS_DIFF_TOL)
    except Exception as e:
        log(f"[CHECK g EXC] {type(e).__name__}: {e}")
        checks["g_roundtrip_bit_stable"] = False
        max_diff = float("nan")
        mean_diff = float("nan")

    # ---- check (h): population AUC >= 0.85 across full 892K test set ----
    # Added 2026-05-07 post Phase C population review. (e)+(f) ask the
    # sample-of-one version of this question; (h) asks the population
    # version using test_recon_err.npy that the smoke just produced.
    try:
        test_recon_err_arr = np.load(smoke_dir / "test_recon_err.npy")
        y_true_binary = (~benign_test_mask).astype(np.int8)
        if len(test_recon_err_arr) != len(y_true_binary):
            raise ValueError(
                f"length mismatch: test_recon_err={len(test_recon_err_arr)}, "
                f"y_true={len(y_true_binary)}"
            )
        population_auc = float(roc_auc_score(y_true_binary, test_recon_err_arr))
        checks["h_population_auc_above_floor"] = bool(
            population_auc >= SMOKE_POPULATION_AUC_MIN
        )
    except Exception as e:
        log(f"[CHECK h EXC] {type(e).__name__}: {e}")
        checks["h_population_auc_above_floor"] = False
        population_auc = float("nan")

    # ---- report ----
    log("")
    log("Smoke pass criteria (per plan §8.1):")
    for k, v in checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    log("")
    log("Diagnostics:")
    log(f"  train_loss epochs:      {[f'{v:.4f}' for v in train_loss]}")
    log(f"  val_loss   epochs:      {[f'{v:.4f}' for v in val_loss]}")
    log(f"  benign-sample MSE:      {benign_mse:.6f}  (threshold < {SMOKE_BENIGN_RECON_MSE_MAX})")
    log(f"  attack-sample MSE:      {attack_mse:.6f}")
    log(f"  reload max abs diff:    {max_diff:.6e}  (tolerance < {ROUNDTRIP_MAX_ABS_DIFF_TOL})")
    log(f"  population AUC:         {population_auc:.6f}  (floor >= {SMOKE_POPULATION_AUC_MIN})")

    overall = all(checks.values())

    smoke_report = {
        "overall_pass":  overall,
        "checks":        checks,
        "manifest":      manifest,
        "train_loss_progression": train_loss,
        "val_loss_progression":   val_loss,
        "benign_sample_mse":      benign_mse,
        "attack_sample_mse":      attack_mse,
        "reload_max_abs_diff":    max_diff,
        "reload_mean_abs_diff":   mean_diff,
        "population_auc":         population_auc,
        "smoke_benign_recon_mse_max": SMOKE_BENIGN_RECON_MSE_MAX,
        "smoke_val_over_train_max":   SMOKE_VAL_OVER_TRAIN_MAX,
        "smoke_population_auc_min":   SMOKE_POPULATION_AUC_MIN,
        "roundtrip_max_abs_diff_tol": ROUNDTRIP_MAX_ABS_DIFF_TOL,
    }
    with open(smoke_dir / "smoke_report.json", "w") as f:
        json.dump(smoke_report, f, indent=2)

    if overall:
        log("\n" + "=" * 70)
        log("SMOKE GATE: PASS - safe to launch full sweep")
        log("=" * 70)
    else:
        failed = [k for k, v in checks.items() if not v]
        with open(smoke_dir / "FAILED.json", "w") as f:
            json.dump({
                "failed_checks": failed,
                "checks": checks,
                "failure_reason": failure_reason,
            }, f, indent=2)
        log("\n" + "=" * 70)
        log(f"SMOKE GATE: FAIL ({failed}) - DO NOT launch full sweep")
        log("Per pre-commitment: documenting negative finding and stopping.")
        log("NO architectural escalation.")
        log("=" * 70)
    return overall


# ============================================================================
# 10 . FULL SWEEP (Phase 2)
# ============================================================================
def run_full_sweep() -> None:
    log("\n" + "=" * 70)
    log(f"PHASE 2 . FULL SWEEP — 5 distinct architectures + 1 monitoring-only "
        f"(C6 = C1 + grad-norm logging), max {EPOCHS_FULL} epochs each")
    log("=" * 70)

    rows = []
    for cfg in CONFIGS:
        out_dir = OUTPUT_DIR / cfg["id"]
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists() and (out_dir / "model.keras").exists():
            with open(manifest_path) as f:
                m = json.load(f)
            if m.get("epochs_actual", 0) >= 20:
                # Resume threshold lowered 2026-05-08 from 30 → 20: paired with
                # the old 600 s cap, 30 was the typical "fully trained" mark;
                # under the new 3600 s cap with EarlyStopping(patience=10), a
                # config that legitimately plateaus around epoch 20-30 should
                # be skippable on resume.
                log(f"[+resume] {cfg['id']} already trained "
                    f"({m['epochs_actual']} epochs); skipping")
                rows.append(m)
                continue

        m, _ = train_one_config(
            cfg=cfg, out_dir=out_dir,
            epochs=EPOCHS_FULL, time_cap_s=PER_CONFIG_TIME_CAP_S,
        )
        rows.append(m)

    summary_df = pd.DataFrame([{
        "config_id":              m["config_id"],
        "enc_units_64":           m["enc_units_64"],
        "enc_units_32":           m["enc_units_32"],
        "learning_rate":          m["learning_rate"],
        "recurrent_dropout":      m["recurrent_dropout"],
        "training_path":          m["training_path"],
        "epochs_actual":          m["epochs_actual"],
        "best_val_loss":          round(m["best_val_loss"], 6),
        "ae_reference_val_loss":  round(m["ae_reference_val_loss"], 6),
        "ae_reference_threshold_1_5x": round(m["ae_reference_threshold_1_5x"], 6),
        "training_time_s":        round(m["training_time_s"], 1),
        "val_recon_mean":         round(m["val_recon_mean"], 6),
        "val_recon_p90":          round(m["val_recon_p90"], 6),
        "val_recon_p95":          round(m["val_recon_p95"], 6),
        "val_recon_p99":          round(m["val_recon_p99"], 6),
        "test_recon_mean":        round(m["test_recon_mean"], 6),
        "test_recon_std_over_mean": round(m["test_recon_std_over_mean"], 4),
        "test_recon_max_over_mean": round(m["test_recon_max_over_mean"], 4),
        "test_recon_unimodal":    m["test_recon_unimodal"],
        "test_recon_finite":      m["test_recon_finite"],
        "gate1_g1_1_descriptive_pass": m["gate1_descriptive"]["g1_1_val_loss_le_1_5x_ae"],
        "gate1_g1_2_descriptive_pass": m["gate1_descriptive"]["g1_2_grad_norm_le_1e3"],
        "gate1_g1_3_descriptive_pass": m["gate1_descriptive"]["g1_3_recon_histogram_ok"],
        "early_stopped_by_budget": m["early_stopped_by_budget"],
    } for m in rows])
    summary_df.to_csv(OUTPUT_DIR / "all_configs_summary.csv", index=False)
    log("\nFull-sweep summary (descriptive Gate 1 fields shown for review;")
    log("the binary gate1_report.json is Phase C, written by the human reviewer):")
    log("\n" + summary_df.to_string(index=False))


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    if SMOKE:
        ok = run_smoke()
        raise SystemExit(0 if ok else 1)

    smoke_dir = OUTPUT_DIR / "_smoke"
    if not (smoke_dir / "smoke_report.json").exists():
        log("[WARN] No prior smoke report - running smoke first as a safety net")
        if not run_smoke():
            log("[ABORT] Smoke gate failed; not launching full sweep")
            raise SystemExit(1)
    else:
        with open(smoke_dir / "smoke_report.json") as f:
            sr = json.load(f)
        if not sr.get("overall_pass", False):
            if SMOKE_OVERRIDE:
                failed = [k for k, v in sr.get("checks", {}).items() if not v]
                log("=" * 70)
                log("[OVERRIDE] SMOKE_OVERRIDE=1 set — proceeding despite smoke FAIL")
                log(f"[OVERRIDE] failed checks: {failed}")
                log(f"[OVERRIDE] population AUC: {sr.get('population_auc', 'n/a')}")
                log("[OVERRIDE] override authorized per senior-engineer review;")
                log("[OVERRIDE] see results/unsupervised/lstm_ae/_phase_b_pause.json v1.6")
                log("=" * 70)
            else:
                log("[ABORT] Prior smoke report says FAIL; re-run with SMOKE=1 after fixing,")
                log("[ABORT] or set SMOKE_OVERRIDE=1 to bypass with documented authorization")
                raise SystemExit(1)
        else:
            log(f"[OK] Prior smoke gate PASS at {smoke_dir / 'smoke_report.json'}")
    run_full_sweep()
