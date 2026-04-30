"""
Phase 6D — beta-Variational Autoencoder Training
==================================================
Path B Tier 2 architectural change: replace deterministic Layer-2 AE with a
beta-VAE using ELBO-loss scoring. Reuses Phase 5 benign-train data, scaler,
and 80/20 split convention exactly. Inherits AE structural constants from the
saved Phase 5 model so the architectures cannot drift.

Usage
-----
    SMOKE=1 python notebooks/vae_train.py    # Phase 1: beta=1.0 only, 5 epochs
    python notebooks/vae_train.py            # Phase 2: beta in {0.1, 0.5, 1.0, 4.0}, 100 epochs each

Outputs
-------
    results/unsupervised/vae/_smoke/                     (smoke run)
    results/unsupervised/vae/beta_<beta>/                (full sweep, per beta)
        model.keras           full Functional VAE (Sampling layer registered)
        history.json          {'loss', 'val_loss'} progression
        manifest.json         hyperparameters + diagnostic stats
        val_loglik.npy        per-sample VAE score on benign-val (loss convention)
        test_loglik.npy       per-sample VAE score on full test set
        latent_z_test.npy     z_mean per test sample (n_test, latent_dim)
        components.npz        {val,test}_{recon,kl} terms for diagnostics

Score formula (per sample; higher => more anomalous, matches AE MSE direction):
    score(x) = sum_d (x_d - x_hat_d)^2  +  beta * 0.5 * sum_k [exp(logvar) + mu^2 - 1 - logvar]
    where x_hat is the deterministic decoder output of z_mean (no sampling at
    score-time). This matches the Phase 5 AE convention of no stochasticity at
    scoring while keeping ELBO semantics intact during training.
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
RANDOM_STATE = 42

# ---- paths (mirror Phase 5 unsupervised_training.py) ----
PREPROCESSED_DIR = Path("./preprocessed/")
PHASE5_DIR       = Path("./results/unsupervised/")
OUTPUT_DIR       = PHASE5_DIR / "vae"
SCALER_PATH      = PHASE5_DIR / "models" / "scaler.pkl"
AE_MODEL_PATH    = PHASE5_DIR / "models" / "autoencoder.keras"

# ---- VAE hyperparameters ----
LATENT_DIM       = 8
INPUT_DIM        = 44
BETA_GRID        = [0.1, 0.5, 1.0, 4.0]      # Higgins 2017 standard sweep
BETA_GRID_FALLBACK = [0.1, 0.5, 1.0, 2.0]    # if beta=4.0 NaN-collapses
SMOKE_BETA       = 1.0
EPOCHS_FULL      = 100
EPOCHS_SMOKE     = 5
BATCH_SIZE       = 512                         # match Phase 5 AE
LEARNING_RATE    = 1e-3                        # match Phase 5 AE
PATIENCE         = 10                          # match Phase 5 AE
PER_BETA_TIME_CAP_S = 600                      # 10 min/beta soft cap
PREDICT_BATCH    = 8192                        # match Phase 5 AE
LOGVAR_CLIP_LOW  = -10.0
LOGVAR_CLIP_HIGH = 10.0

# Smoke pass criteria (beta=1.0, 5 epochs, LOSS convention not nats)
# Loose bounds for an under-converged 5-epoch model — these only catch
# CATASTROPHIC failure (NaN, all-zero collapse, blow-up). Strict latent
# geometry checks belong to the full-sweep 100-epoch manifest review.
SMOKE_SCORE_MEDIAN_RANGE   = (3.0, 80.0)
SMOKE_LATENT_MEAN_ABS_MAX  = 5.0   # at 5 ep: KL hasn't pulled encoder to N(0,I) yet
SMOKE_LATENT_STD_RANGE     = (0.1, 10.0)  # catches collapse-to-zero or runaway variance
ROUNDTRIP_MAX_ABS_DIFF_TOL = 1e-4   # tighter now that Sampling honors training=False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1 . REPRODUCIBILITY (mirror Phase 5 unsupervised_training.py:65-104)
# ============================================================================
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
random.seed(RANDOM_STATE)

import numpy as np
np.random.seed(RANDOM_STATE)

import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import keras
from keras import ops as kops

tf.random.set_seed(RANDOM_STATE)
try:
    tf.keras.utils.set_random_seed(RANDOM_STATE)   # TF >= 2.7
except Exception:
    pass

# ---- logging idiom (mirror threshold_sweep.py) ----
_T0 = time.time()
def log(msg: str = "") -> None:
    elapsed = time.time() - _T0
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] [+{elapsed:6.1f}s] {msg}", flush=True)

log(f"VAE training script | SMOKE={SMOKE} | random_state={RANDOM_STATE}")
log(f"TensorFlow {tf.__version__} | tf.keras {tf.keras.__version__} | NumPy {np.__version__}")
assert tf.keras.__version__.startswith("3."), (
    f"Keras 3 required (saved AE was Keras 3.14.0); got {tf.keras.__version__}"
)

# ---- GPU detection ----
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    log(f"GPU device(s): {[g.name for g in gpus]}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
else:
    log("No GPU detected; CPU training (fine for this size).")


# ============================================================================
# 2 . DATA LOADING (mirror Phase 5 SECTION 2 exactly so scaler is reusable)
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

# 80/20 split using identical RNG idiom to Phase 5
rng = np.random.default_rng(RANDOM_STATE)
perm = rng.permutation(X_benign_full.shape[0])
split_idx = int(0.8 * X_benign_full.shape[0])
X_benign_train = X_benign_full[perm[:split_idx]]
X_benign_val   = X_benign_full[perm[split_idx:]]
log(f"AE-train (benign): {X_benign_train.shape}")
log(f"AE-val   (benign): {X_benign_val.shape}")

# Free the full train array (mirror Phase 5)
del X_train, X_benign_full

# ---- Reuse Phase 5 scaler (do NOT refit) ----
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

X_benign_train = scaler.transform(X_benign_train).astype(np.float32)
X_benign_val   = scaler.transform(X_benign_val).astype(np.float32)
log(f"After scaling: benign-train mean={X_benign_train.mean():.6f}, "
    f"std={X_benign_train.std():.6f}")

# Test set for post-training scoring
X_test = scaler.transform(
    np.load(PREPROCESSED_DIR / "full_features" / "X_test.npy").astype(np.float32)
).astype(np.float32)
log(f"Test set (scaled): {X_test.shape}")


# ============================================================================
# 3 . INHERIT AE STRUCTURE FROM SAVED MODEL (per user Q-A3)
# ============================================================================
log("\n" + "=" * 70)
log("SECTION 3 . INHERIT AE STRUCTURAL CONSTANTS FROM SAVED MODEL")
log("=" * 70)

ae_loaded = tf.keras.models.load_model(AE_MODEL_PATH, compile=False)
ae_layers = {layer.name: layer for layer in ae_loaded.layers}

ENC_DENSE_32_UNITS = ae_layers["enc_dense_32"].units
ENC_DENSE_16_UNITS = ae_layers["enc_dense_16"].units
BOTTLENECK_UNITS   = ae_layers["bottleneck"].units
DEC_DENSE_16_UNITS = ae_layers["dec_dense_16"].units
DEC_DENSE_32_UNITS = ae_layers["dec_dense_32"].units
ENC_DROP_32_RATE   = ae_layers["enc_drop_32"].rate
ENC_DROP_16_RATE   = ae_layers["enc_drop_16"].rate

assert BOTTLENECK_UNITS == LATENT_DIM, (
    f"AE bottleneck has {BOTTLENECK_UNITS} units; VAE LATENT_DIM is {LATENT_DIM}. "
    "User constraint Q6: latent_dim must match the AE bottleneck for fair comparison."
)
log(f"Inherited from AE: enc 44 -> {ENC_DENSE_32_UNITS} -> {ENC_DENSE_16_UNITS} -> "
    f"{BOTTLENECK_UNITS} (latent), dec {BOTTLENECK_UNITS} -> {DEC_DENSE_16_UNITS} -> "
    f"{DEC_DENSE_32_UNITS} -> 44; dropout=({ENC_DROP_32_RATE}, {ENC_DROP_16_RATE})")
del ae_loaded


# ============================================================================
# 4 . SAMPLING LAYER (custom subclass; save-friendly via register_keras_serializable)
# ============================================================================
@keras.utils.register_keras_serializable(package="vae", name="Sampling")
class Sampling(layers.Layer):
    """Reparameterization layer: z = z_mean + exp(0.5*clip(logvar)) * eps."""

    def __init__(self, seed: int = RANDOM_STATE,
                 logvar_clip_low: float = LOGVAR_CLIP_LOW,
                 logvar_clip_high: float = LOGVAR_CLIP_HIGH,
                 **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.logvar_clip_low = logvar_clip_low
        self.logvar_clip_high = logvar_clip_high
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        if not training:
            # Deterministic at inference: posterior mode (z = z_mean).
            # Matches Phase 5 AE convention of no stochasticity at scoring time
            # and makes save/reload forward passes bit-stable.
            return z_mean
        z_log_var = kops.clip(z_log_var, self.logvar_clip_low, self.logvar_clip_high)
        eps = keras.random.normal(shape=kops.shape(z_mean), seed=self.seed_generator)
        return z_mean + kops.exp(0.5 * z_log_var) * eps

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "seed": self.seed,
            "logvar_clip_low":  self.logvar_clip_low,
            "logvar_clip_high": self.logvar_clip_high,
        })
        return cfg


@keras.utils.register_keras_serializable(package="vae", name="KLLossLayer")
class KLLossLayer(layers.Layer):
    """Computes KL(q(z|x) || N(0, I)) and adds it as a side-effect loss.

    Keras 3 dropped Functional.add_loss; the supported alternative is layer-
    level add_loss inside call(). This layer is a passthrough on z_mean so
    the loss attaches to the forward graph and shows up in train/val loss.
    """

    def __init__(self, beta: float = 1.0,
                 logvar_clip_low: float = LOGVAR_CLIP_LOW,
                 logvar_clip_high: float = LOGVAR_CLIP_HIGH,
                 **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.logvar_clip_low = logvar_clip_low
        self.logvar_clip_high = logvar_clip_high

    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var_c = kops.clip(z_log_var, self.logvar_clip_low, self.logvar_clip_high)
        kl_per_sample = -0.5 * kops.sum(
            1.0 + z_log_var_c - kops.square(z_mean) - kops.exp(z_log_var_c),
            axis=1,
        )
        kl_loss = kops.mean(kl_per_sample) * self.beta
        self.add_loss(kl_loss)
        return z_mean  # passthrough so this layer is in the forward graph

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "beta": self.beta,
            "logvar_clip_low":  self.logvar_clip_low,
            "logvar_clip_high": self.logvar_clip_high,
        })
        return cfg


@keras.utils.register_keras_serializable(package="vae", name="ReconSumMSE")
class ReconSumMSE(keras.losses.Loss):
    """Per-sample reconstruction loss = sum_d (x - x_hat)^2 (Keras means over batch)."""

    def __init__(self, name: str = "recon_sum_mse", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        return kops.sum(kops.square(y_true - y_pred), axis=1)


# ============================================================================
# 5 . VAE BUILDER (Functional API, layer naming mirrors Phase 5 AE)
# ============================================================================
def build_vae(beta: float) -> Model:
    """Construct beta-VAE matching Phase 5 AE structure with two-head encoder.

    Encoder body: 44 -> 32(relu) -> BN -> Drop(0.2) -> 16(relu) -> BN -> Drop(0.1)
    Latent heads: -> z_mean(8), z_log_var(8) -> Sampling -> z(8)
    Decoder:      8 -> 16(relu) -> BN -> 32(relu) -> BN -> 44(linear)
    Loss (added via add_loss):
        recon = mean_batch[ sum_d (x - x_hat)^2 ]
        kl    = mean_batch[ -0.5 * sum_k (1 + logvar - mu^2 - exp(logvar)) ]
        total = recon + beta * kl
    """
    inp = layers.Input(shape=(INPUT_DIM,), name="input")

    # ---- encoder body (mirrors Phase 5 AE layer naming) ----
    x = layers.Dense(ENC_DENSE_32_UNITS, activation="relu", name="enc_dense_32")(inp)
    x = layers.BatchNormalization(name="enc_bn_32")(x)
    x = layers.Dropout(ENC_DROP_32_RATE, name="enc_drop_32")(x)
    x = layers.Dense(ENC_DENSE_16_UNITS, activation="relu", name="enc_dense_16")(x)
    x = layers.BatchNormalization(name="enc_bn_16")(x)
    x = layers.Dropout(ENC_DROP_16_RATE, name="enc_drop_16")(x)

    # ---- two parallel latent heads ----
    z_mean    = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)

    # KL loss attached as side effect; layer is a passthrough on z_mean so it
    # remains in the forward graph (Keras 3 requires this — Functional.add_loss
    # was removed; layer-level add_loss is the supported alternative).
    z_mean_kl = KLLossLayer(beta=beta, name="kl_loss")([z_mean, z_log_var])
    z = Sampling(name="sampling")([z_mean_kl, z_log_var])

    # ---- decoder ----
    h = layers.Dense(DEC_DENSE_16_UNITS, activation="relu", name="dec_dense_16")(z)
    h = layers.BatchNormalization(name="dec_bn_16")(h)
    h = layers.Dense(DEC_DENSE_32_UNITS, activation="relu", name="dec_dense_32")(h)
    h = layers.BatchNormalization(name="dec_bn_32")(h)
    out = layers.Dense(INPUT_DIM, activation="linear", name="reconstruction")(h)

    vae = Model(inputs=inp, outputs=out, name=f"vae_beta_{beta:g}")

    # ---- compile with custom recon loss (sum over 44 features); KL added inside layer ----
    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=ReconSumMSE(),
    )
    return vae


# ============================================================================
# 6 . SCORING (deterministic; uses z_mean, no sampling at score time)
# ============================================================================
def _build_scoring_submodels(vae: Model):
    """Return (encoder_zmean, encoder_zlogvar, decoder) sub-models for scoring.

    The scoring path is deterministic: z = z_mean (no Sampling). The decoder
    layers are reused via vae.get_layer(...) so we don't re-instantiate weights.
    """
    encoder_zmean = Model(
        inputs=vae.input, outputs=vae.get_layer("z_mean").output, name="enc_zmean",
    )
    encoder_zlogvar = Model(
        inputs=vae.input, outputs=vae.get_layer("z_log_var").output, name="enc_zlogvar",
    )
    z_in = layers.Input(shape=(LATENT_DIM,), name="z_in_score")
    h = vae.get_layer("dec_dense_16")(z_in)
    h = vae.get_layer("dec_bn_16")(h)
    h = vae.get_layer("dec_dense_32")(h)
    h = vae.get_layer("dec_bn_32")(h)
    out = vae.get_layer("reconstruction")(h)
    decoder = Model(inputs=z_in, outputs=out, name="decoder")
    return encoder_zmean, encoder_zlogvar, decoder


def score_vae_per_sample(vae: Model, X: np.ndarray, beta: float,
                         batch_size: int = PREDICT_BATCH):
    """Per-sample (score, recon, kl, z_mean) computed deterministically on z_mean."""
    enc_mu, enc_lv, dec = _build_scoring_submodels(vae)
    z_mean    = enc_mu.predict(X, batch_size=batch_size, verbose=0)
    z_log_var = enc_lv.predict(X, batch_size=batch_size, verbose=0)
    z_log_var = np.clip(z_log_var, LOGVAR_CLIP_LOW, LOGVAR_CLIP_HIGH)
    x_hat = dec.predict(z_mean, batch_size=batch_size, verbose=0)

    recon = np.sum((X - x_hat) ** 2, axis=1).astype(np.float32)
    kl = (-0.5 * np.sum(
        1.0 + z_log_var - z_mean ** 2 - np.exp(z_log_var), axis=1
    )).astype(np.float32)
    score = (recon + beta * kl).astype(np.float32)
    return score, recon, kl, z_mean.astype(np.float32)


# ============================================================================
# 7 . TRAIN ONE BETA
# ============================================================================
def train_one_beta(beta: float, out_dir: Path, epochs: int, time_cap_s: float):
    """Train a single beta-VAE; save artifacts; return (manifest, vae)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"\n--- Training beta={beta} -> {out_dir} "
        f"(epochs<={epochs}, soft cap {time_cap_s}s) ---")

    vae = build_vae(beta)
    if not (out_dir.parent / "_arch_summary.txt").exists():
        # one-time architecture summary, written next to the parent vae/ dir
        lines = []
        vae.summary(print_fn=lambda s: lines.append(s))
        (out_dir.parent / "_arch_summary.txt").write_text("\n".join(lines))
        log("Architecture summary written to vae/_arch_summary.txt")

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
    ]

    t_train = time.time()
    history = vae.fit(
        X_benign_train, X_benign_train,                       # y ignored (add_loss)
        validation_data=(X_benign_val, X_benign_val),
        epochs=epochs, batch_size=BATCH_SIZE,
        callbacks=cbs, verbose=2, shuffle=True,
    )
    train_time = time.time() - t_train
    epochs_actual = len(history.history["loss"])
    over_budget = train_time > time_cap_s
    log(f"beta={beta}: {epochs_actual} epochs in {train_time:.1f}s "
        f"({train_time/60:.2f} min){' [OVER BUDGET]' if over_budget else ''}")

    # ---- score val + test ----
    log(f"Scoring benign-val (n={len(X_benign_val):,})...")
    val_score, val_recon, val_kl, _      = score_vae_per_sample(vae, X_benign_val, beta)
    log(f"Scoring full test  (n={len(X_test):,})...")
    test_score, test_recon, test_kl, test_z = score_vae_per_sample(vae, X_test, beta)

    # ---- NaN/Inf hard check ----
    for nm, arr in [("val_score", val_score), ("test_score", test_score),
                    ("val_recon", val_recon), ("test_recon", test_recon),
                    ("val_kl", val_kl), ("test_kl", test_kl),
                    ("test_z", test_z)]:
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise RuntimeError(
                f"beta={beta}: NaN/Inf in {nm} (n_nan={n_nan}, n_inf={n_inf}). "
                "Likely posterior collapse or exp(z_log_var) overflow despite clip."
            )

    # ---- save artifacts ----
    vae.save(out_dir / "model.keras")
    np.save(out_dir / "val_loglik.npy",    val_score)
    np.save(out_dir / "test_loglik.npy",   test_score)
    np.save(out_dir / "latent_z_test.npy", test_z)
    np.savez(out_dir / "components.npz",
             val_recon=val_recon, val_kl=val_kl,
             test_recon=test_recon, test_kl=test_kl)

    hist_dict = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with open(out_dir / "history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    z_means_per_dim = test_z.mean(axis=0)
    z_stds_per_dim  = test_z.std(axis=0)

    # ---- recon-vs-KL balance (posterior-collapse diagnostic) ----
    val_recon_mean = float(val_recon.mean())
    val_kl_mean    = float(val_kl.mean())
    recon_to_kl_ratio = val_recon_mean / max(val_kl_mean, 1e-9)

    # ---- strict latent-geometry diagnostic (not fatal) ----
    # Convergence target: |z_mean_i| < 0.5 AND |std_i - 1.0| < 0.5 for all 8 dims.
    # KL pressure should pull a converged VAE here; failure indicates the encoder
    # is still producing wider/biased latents at this β. Per-dim flags help see
    # whether the failure is one rogue dim or systemic.
    per_dim_mean_strict_pass = (np.abs(z_means_per_dim) < 0.5).tolist()
    per_dim_std_strict_pass  = (np.abs(z_stds_per_dim - 1.0) < 0.5).tolist()
    latent_geom_strict_pass  = bool(all(per_dim_mean_strict_pass) and all(per_dim_std_strict_pass))

    manifest = {
        "beta":                  float(beta),
        "latent_dim":            int(LATENT_DIM),
        "input_dim":             int(INPUT_DIM),
        "epochs_max":            int(epochs),
        "epochs_actual":         int(epochs_actual),
        "batch_size":            int(BATCH_SIZE),
        "learning_rate":         float(LEARNING_RATE),
        "patience":              int(PATIENCE),
        "best_val_loss":         float(min(history.history["val_loss"])),
        "training_time_s":       float(train_time),
        "early_stopped_by_budget": bool(over_budget),
        "score_formula":         (
            "score = sum_d(x - x_hat)^2 + beta * 0.5 * "
            "sum_k(exp(logvar) + mu^2 - 1 - logvar); higher => more anomalous"
        ),
        "z_mean_used_for_scoring": True,
        "logvar_clip":           [LOGVAR_CLIP_LOW, LOGVAR_CLIP_HIGH],
        "tf_version":            tf.__version__,
        "keras_version":         tf.keras.__version__,
        "random_state":          int(RANDOM_STATE),
        # diagnostic stats (val)
        "val_score_median":      float(np.median(val_score)),
        "val_score_p90":         float(np.percentile(val_score, 90)),
        "val_score_p95":         float(np.percentile(val_score, 95)),
        "val_score_p99":         float(np.percentile(val_score, 99)),
        "val_recon_mean":        val_recon_mean,
        "val_recon_p95":         float(np.percentile(val_recon, 95)),
        "val_kl_mean":           val_kl_mean,
        "val_kl_p95":            float(np.percentile(val_kl, 95)),
        "recon_to_kl_ratio":     float(recon_to_kl_ratio),
        # diagnostic stats (test)
        "test_score_median":     float(np.median(test_score)),
        "test_recon_mean":       float(test_recon.mean()),
        "test_kl_mean":          float(test_kl.mean()),
        # latent geometry — descriptive
        "latent_z_mean_per_dim": z_means_per_dim.astype(float).tolist(),
        "latent_z_std_per_dim":  z_stds_per_dim.astype(float).tolist(),
        "latent_z_mean_overall": float(z_means_per_dim.mean()),
        "latent_z_std_overall":  float(z_stds_per_dim.mean()),
        # latent geometry — strict diagnostic (per user constraint, not fatal)
        "latent_geom_strict_pass":            latent_geom_strict_pass,
        "latent_geom_strict_threshold_mean":  0.5,
        "latent_geom_strict_threshold_std":   0.5,
        "latent_geom_per_dim_mean_pass":      per_dim_mean_strict_pass,
        "latent_geom_per_dim_std_pass":       per_dim_std_strict_pass,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest, vae


# ============================================================================
# 8 . ROUND-TRIP SAVE/LOAD VERIFICATION
# ============================================================================
def verify_roundtrip(model_path: Path, sample_X: np.ndarray, original_pred: np.ndarray):
    """Reload saved model; assert forward pass matches within tolerance."""
    custom_objects = {"Sampling": Sampling}
    reloaded = tf.keras.models.load_model(
        model_path, custom_objects=custom_objects, compile=False,
    )
    reload_pred = reloaded(sample_X, training=False).numpy()
    abs_diff = np.abs(reload_pred - original_pred)
    return float(abs_diff.max()), float(abs_diff.mean()), reloaded


# ============================================================================
# 9 . SMOKE GATE (Phase 1)
# ============================================================================
def run_smoke() -> bool:
    log("\n" + "=" * 70)
    log("PHASE 1 . SMOKE GATE (beta=1.0, 5 epochs, _smoke/ dir)")
    log("=" * 70)
    smoke_dir = OUTPUT_DIR / "_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    manifest, vae = train_one_beta(
        beta=SMOKE_BETA, out_dir=smoke_dir,
        epochs=EPOCHS_SMOKE, time_cap_s=PER_BETA_TIME_CAP_S,
    )

    # ---- pass criteria ----
    checks = {}
    with open(smoke_dir / "history.json") as f:
        h = json.load(f)
    val_loss = h["val_loss"]

    checks["val_loss_finite"]      = bool(all(np.isfinite(v) for v in val_loss))
    checks["val_loss_decreased"]   = bool(val_loss[-1] < val_loss[0])

    val_med = manifest["val_score_median"]
    lo, hi = SMOKE_SCORE_MEDIAN_RANGE
    checks["score_in_sanity_range"] = bool(lo <= val_med <= hi)

    z_means = np.array(manifest["latent_z_mean_per_dim"])
    z_stds  = np.array(manifest["latent_z_std_per_dim"])
    checks["latent_means_centered"]  = bool(np.all(np.abs(z_means) < SMOKE_LATENT_MEAN_ABS_MAX))
    checks["latent_stds_reasonable"] = bool(
        np.all((z_stds > SMOKE_LATENT_STD_RANGE[0]) &
               (z_stds < SMOKE_LATENT_STD_RANGE[1]))
    )

    val_score_loaded = np.load(smoke_dir / "val_loglik.npy")
    checks["no_nan_in_scores"] = bool(np.isfinite(val_score_loaded).all())

    # Round-trip save/load
    sample_X = X_benign_val[:100]
    original_pred = vae(sample_X, training=False).numpy()
    try:
        max_diff, mean_diff, _ = verify_roundtrip(
            smoke_dir / "model.keras", sample_X, original_pred,
        )
        checks["roundtrip_ok"] = bool(max_diff < ROUNDTRIP_MAX_ABS_DIFF_TOL)
    except Exception as e:
        log(f"[ROUNDTRIP EXC] {type(e).__name__}: {e}")
        checks["roundtrip_ok"] = False
        max_diff = float("nan")
        mean_diff = float("nan")

    # ---- report ----
    log("")
    log("Smoke pass criteria:")
    for k, v in checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    log("")
    log("Diagnostics:")
    log(f"  val_loss epochs:        {[f'{v:.4f}' for v in val_loss]}")
    log(f"  val_score median:       {val_med:.4f}    (sanity range {SMOKE_SCORE_MEDIAN_RANGE})")
    log(f"  val_score p90/p95/p99:  "
        f"{manifest['val_score_p90']:.3f} / "
        f"{manifest['val_score_p95']:.3f} / "
        f"{manifest['val_score_p99']:.3f}")
    log(f"  val_recon mean:         {manifest['val_recon_mean']:.4f}")
    log(f"  val_kl mean:            {manifest['val_kl_mean']:.4f}  "
        f"(beta=1.0; sample-balance = {manifest['val_recon_mean']:.2f}+1*{manifest['val_kl_mean']:.2f})")
    log(f"  latent z_mean per dim:  [{z_means.min():.4f}, {z_means.max():.4f}]")
    log(f"  latent z_std  per dim:  [{z_stds.min():.4f}, {z_stds.max():.4f}]")
    log(f"  reload max abs diff:    {max_diff:.6e}")

    overall = all(checks.values())

    smoke_report = {
        "overall_pass":          overall,
        "checks":                checks,
        "manifest":              manifest,
        "val_loss_progression":  val_loss,
        "reload_max_abs_diff":   max_diff,
        "reload_mean_abs_diff":  mean_diff,
        "smoke_score_median_range": list(SMOKE_SCORE_MEDIAN_RANGE),
        "smoke_latent_mean_abs_max": SMOKE_LATENT_MEAN_ABS_MAX,
        "smoke_latent_std_range":   list(SMOKE_LATENT_STD_RANGE),
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
            json.dump({"failed_checks": failed, "checks": checks}, f, indent=2)
        log("\n" + "=" * 70)
        log(f"SMOKE GATE: FAIL ({failed}) - DO NOT launch full sweep")
        log("=" * 70)
    return overall


# ============================================================================
# 10 . FULL SWEEP (Phase 2; not run in SMOKE mode)
# ============================================================================
def run_full_sweep() -> None:
    log("\n" + "=" * 70)
    log(f"PHASE 2 . FULL SWEEP (beta in {BETA_GRID}, max {EPOCHS_FULL} epochs each)")
    log("=" * 70)
    rows = []
    for beta in BETA_GRID:
        out_dir = OUTPUT_DIR / f"beta_{beta:.1f}"
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists() and (out_dir / "model.keras").exists():
            with open(manifest_path) as f:
                m = json.load(f)
            if m.get("epochs_actual", 0) >= 30:
                log(f"[+resume] beta={beta} already trained "
                    f"({m['epochs_actual']} epochs); skipping")
                rows.append(m)
                continue
        m, _ = train_one_beta(
            beta=beta, out_dir=out_dir,
            epochs=EPOCHS_FULL, time_cap_s=PER_BETA_TIME_CAP_S,
        )
        rows.append(m)

    summary_df = pd.DataFrame([{
        "beta":                  m["beta"],
        "epochs_actual":         m["epochs_actual"],
        "best_val_loss":         round(m["best_val_loss"], 4),
        "training_time_s":       round(m["training_time_s"], 1),
        "val_score_median":      round(m["val_score_median"], 4),
        "val_score_p90":         round(m["val_score_p90"], 4),
        "val_score_p95":         round(m["val_score_p95"], 4),
        "val_score_p99":         round(m["val_score_p99"], 4),
        "val_recon_mean":        round(m["val_recon_mean"], 4),
        "val_kl_mean":           round(m["val_kl_mean"], 4),
        "recon_to_kl_ratio":     round(m["recon_to_kl_ratio"], 3),
        "latent_geom_strict_pass": m["latent_geom_strict_pass"],
        "early_stopped_by_budget": m["early_stopped_by_budget"],
    } for m in rows])
    summary_df.to_csv(OUTPUT_DIR / "all_betas_summary.csv", index=False)
    log("\nFull-sweep summary:")
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
            log("[ABORT] Prior smoke report says FAIL; re-run with SMOKE=1 after fixing")
            raise SystemExit(1)
        log(f"[OK] Prior smoke gate PASS at {smoke_dir / 'smoke_report.json'}")
    run_full_sweep()
