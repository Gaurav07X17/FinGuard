# Phase 3 — Ingestion Module

This directory contains ingestion utilities and helpers that turn raw uploaded files (CSV/XLSX) into cleaned pandas DataFrames and returns series ready for the risk calculation engine (Phase 2).

Key files
- phase3/ingest.py — parsing helpers, size/row guards, encoding/delimiter detection, NaN policies (reject|drop|impute)
- phase3/mapping.py — auto_guess_mapping(columns) + fuzzy matching for common column names
- phase3/convert.py — convert_prices_to_returns and df_price_column_to_returns
- phase3/mapping_confirm_cli.py — small CLI to preview and accept mappings
- phase3/mapping_confirm_streamlit.py — small Streamlit placeholder for eventual UI
- phase3/tests/test_ingest.py — pytest tests (in this branch)
- phase3/docs/ingestion.md — documentation and examples

Quick start (branch-local)

1. Install dev deps (from repository root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

2. Run Phase 3 tests only:

```bash
pytest -q phase3/tests/test_ingest.py
```

Notes & policies
- This module enforces a strict NaN policy by default (`nan_policy='reject'`). Callers may opt-in to `drop` or `impute` if they explicitly want that behavior.
- Files are processed in-memory and are not persisted. Max file size defaults to 10 MB and max rows to 100k; these are configurable constants in phase3/ingest.py.
- Convert price series to returns with phase3/convert.py::convert_prices_to_returns(prices) before passing to risk_metrics.
