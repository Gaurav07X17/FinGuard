# Phase 2: Risk Calculation Engine

This branch contains the Phase 2 deliverables for the FinTech AI Tool (FinGuard): the standalone calculation engine for portfolio risk metrics, unit tests, and CI.

Quick start
-----------

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

3. Run tests:

```bash
pytest -q
```

Notes
-----
- Input shape: All functions in risk_metrics.py accept periodic returns (not price series). Convert price series to returns using:

```python
returns = prices.pct_change().dropna()
```

- Strict NaN policy: the calculation functions will raise ValueError if the returns series contains NaN values. Clean or impute data in the ingestion layer before calling these tools.

- Sharpe ratio: the risk_free_rate parameter is interpreted as an annualized rate. When calling with `annualize=False`, the function converts the annual rate to a per-period equivalent by dividing by `periods_per_year` (defaults to 252). This makes the non-annualized Sharpe calculation consistent with user expectations.
