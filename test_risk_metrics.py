"""
Sanity & pytest-ready tests for risk_metrics.py.

Covers:
 - volatility, sharpe, var, max_drawdown basic runs
 - explicit check that when annualize=False, Sharpe converts annual RF to per-period RF
 - edge cases: empty series, single obs, NaNs, zero-vol
"""
import numpy as np
import pandas as pd
from risk_metrics import (
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_var,
    calculate_max_drawdown,
)

np.random.seed(42)
n_days = 504
daily_returns = pd.Series(np.random.normal(loc=0.0005, scale=0.012, size=n_days))

def test_volatility_basic():
    res = calculate_volatility(daily_returns)
    assert set(res.keys()) == {"value", "interpretation"}
    assert isinstance(res["value"], float)

def test_sharpe_annualized_and_non_annualized_consistency():
    annual_rf = 0.02
    res_ann = calculate_sharpe_ratio(daily_returns, risk_free_rate=annual_rf, annualize=True)
    res_non = calculate_sharpe_ratio(daily_returns, risk_free_rate=annual_rf, annualize=False, periods_per_year=252)
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    rf_per_period = annual_rf / 252
    expected_sharpe = (mean_return - rf_per_period) / std_return
    assert abs(res_non["value"] - round(float(expected_sharpe), 4)) < 1e-8

def test_var_and_drawdown_basic():
    v = calculate_var(daily_returns)
    assert v["value"] >= 0
    try:
        vp = calculate_var(daily_returns, confidence=0.99, method="parametric")
        assert vp["value"] >= 0
    except ImportError:
        # pytest environment may not have scipy installed
        pass
    d = calculate_max_drawdown(daily_returns)
    assert d["value"] >= 0

def test_edge_cases_raise():
    import pytest
    with pytest.raises(ValueError):
        calculate_volatility(pd.Series([], dtype=float))
    with pytest.raises(ValueError):
        calculate_volatility(pd.Series([0.01]))
    with pytest.raises(ValueError):
        calculate_sharpe_ratio(pd.Series([0.0] * 10), annualize=False)
    with pytest.raises(ValueError):
        calculate_volatility(pd.Series([0.01, np.nan, 0.02]))
    with pytest.raises(ValueError):
        calculate_var(daily_returns, confidence=1.5)
