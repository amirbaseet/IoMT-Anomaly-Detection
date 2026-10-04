# CLAUDE.md — dashboard/ (Streamlit, local-only)

Local Streamlit app over the frozen results (Single Flow Analyzer + SHAP Explorer). Separate
pinned deps in `dashboard/requirements.txt` (`streamlit==1.40.0`, `plotly==5.24.1`). Read root
`../CLAUDE.md` first.

## Run
```
venv/bin/python -m streamlit run dashboard/Home.py       # local; not deployed
```

## Invariants
- **Canonical feature order** is sourced from `results/shap/config.json`'s `feature_names` list —
  exactly the order the upstream models expect (`components/flow_input.py:15-16,33`). Never
  hand-rebuild it. (root INV-01)
- **Page-2 case classifier** must stay aligned to the canonical 5-case `entropy_fusion`; it has
  drifted before and was re-aligned (6aacb8b). (root INV-01)
- The app reads **frozen** precomputed outputs; it does not retrain. Regenerate via
  `scripts/precompute_dashboard_artifacts.py`.
