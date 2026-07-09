# CLAUDE.md — notebooks/ (training/eval modules — `.py`, NOT ipynb)

Despite the directory name, these are runnable `.py` modules (each has a `__main__` guard), one
per experiment step: `supervised_training`, `unsupervised_training`, `lstm_ae_train`, `vae_train`,
`fusion_engine`, `enhanced_fusion`, `shap_analysis`, `threshold_sweep`, `loo_zero_day`,
`multi_seed_fusion`, `multi_seed_loo`. Read root `../CLAUDE.md` first.

## Run
```
venv/bin/python notebooks/<name>.py                              # constants + __main__; no CLI flags…
venv/bin/python notebooks/multi_seed_loo.py --seeds 1 7 42 100 1729   # …except this one (argparse)
```

## Invariants
- **Multi-seed set is `[1, 7, 42, 100, 1729]`** (`multi_seed_fusion.py:80`, `multi_seed_loo.py:77`).
  Report mean ± σ; never a single seed. (root DN-05/INV-04)
- **AE/IF/VAE get the dedicated StandardScaler on benign-train** — NOT the tree ColumnTransformer.
  (root DN-02/DN-03)
- **Feature/label order comes from `config.json`**; compare models in the same label space.
  (root INV-01)
- The polished `.ipynb` live elsewhere: `IoMT_Anomaly_Detection.ipynb` (root),
  `deliverables/thesis_walkthrough.ipynb`.
