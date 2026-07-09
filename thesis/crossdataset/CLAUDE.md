# CLAUDE.md — thesis/crossdataset (generalization experiment)

Train on CICIoMT2024, evaluate session-disjoint on CICIoT2023 (Design A binary, Design B
5 families) + AE-inversion addendum. Self-contained package under `src/` (`run.py` orchestrates).
Read root `../../CLAUDE.md` first.

## THE FOUR TRAPS (verbatim from the code — never regress these)
- **TRAP 1 — fit-on-train-only.** Scalers/imputers fit on CICIoMT2024 benign-train only.
  `models.py:3`, `ae_worker.py:19,70`, `run.py:14`. (= root DN-03)
- **TRAP 2 — strip `\r`.** Trailing CRLF is stripped from the Label column on every chunk;
  CICIoT2023 CSVs carry it and label-match fails silently otherwise. `labels.py:41`,
  `ciciot_sampler.py:10`. (= root INV-03)
- **TRAP 3 — feature order.** Align a CICIoT2023 frame to the shared-feature model by NAME then
  reorder — never assume positions. `features.py:4,41`. (= root INV-01)
- **TRAP 4 — multi-seed.** Aggregate mean ± σ over `SAMPLING_SEEDS = (42, 43, 44)` — 3
  independent CICIoT2023 samples. `metrics.py:3`, `config.py:28`. (= root DN-05/INV-04)

## Run
```
venv/bin/python -m thesis.crossdataset.src.run [flags]   # real argparse (1 of only 2 in the repo)
```

## Gotchas
- **Stale absolute paths** — `src/config.py:17-21` hardcodes `/Users/amoorabaseet/IoMT-Project/...`
  (the pre-`/code/`-move path). Fix before running on this machine.
- **Honesty caveat** — a drop below the in-dataset baseline is the EXPECTED, honest
  generalization result, not a defect (`results_summary.md:9`). Report the full operating
  characteristic; do not cherry-pick τ.
- Outputs are tagged **Measured**; `results_summary.md` pairs with
  `../../Thesis_Generalization_Analysis_v0.1.md`.
