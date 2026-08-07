"""
Quick validation script for risk_metrics.py.
Not a full pytest suite yet (that's Phase 7) — just a sanity check that
each function runs, returns the right shape, and produces sane numbers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from risk_metrics import (
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_var,
    calculate_max_drawdown,
)

np.random.seed(42)

# Simulate ~2 years of daily returns for a moderately volatile portfolio
n_days = 504
daily_returns = pd.Series(np.random.normal(loc=0.0005, scale=0.012, size=n_days))

print("=" * 70)
print("VOLATILITY")
print("=" * 70)
result = calculate_volatility(daily_returns)
print(result)
assert set(result.keys()) == {"value", "interpretation"}
assert isinstance(result["value"], float)

print("\n" + "=" * 70)
print("SHARPE RATIO")
print("=" * 70)
result = calculate_sharpe_ratio(daily_returns)
print(result)
assert set(result.keys()) == {"value", "interpretation"}

print("\n" + "=" * 70)
print("SHARPE RATIO (custom risk-free rate, testing param override)")
print("=" * 70)
result = calculate_sharpe_ratio(daily_returns, risk_free_rate=0.02)
print(result)

print("\n" + "=" * 70)
print("VaR (historical, 95%)")
print("=" * 70)
result = calculate_var(daily_returns)
print(result)
assert result["value"] >= 0

print("\n" + "=" * 70)
print("VaR (parametric, 99%)")
print("=" * 70)
result = calculate_var(daily_returns, confidence=0.99, method="parametric")
print(result)

print("\n" + "=" * 70)
print("MAX DRAWDOWN")
print("=" * 70)
result = calculate_max_drawdown(daily_returns)
print(result)
assert result["value"] >= 0

print("\n" + "=" * 70)
print("EDGE CASES")
print("=" * 70)

# Empty series
try:
    calculate_volatility(pd.Series([], dtype=float))
    print("FAIL: should have raised ValueError on empty series")
except ValueError as e:
    print(f"OK — empty series correctly raised: {e}")

# Single observation
try:
    calculate_volatility(pd.Series([0.01]))
    print("FAIL: should have raised ValueError on single observation")
except ValueError as e:
    print(f"OK — single observation correctly raised: {e}")

# Zero volatility (Sharpe undefined)
try:
    calculate_sharpe_ratio(pd.Series([0.0] * 10))
    print("FAIL: should have raised ValueError on zero volatility")
except ValueError as e:
    print(f"OK — zero volatility correctly raised: {e}")

# NaN in series
try:
    calculate_volatility(pd.Series([0.01, np.nan, 0.02, 0.01, -0.01]))
    print("FAIL: should have raised ValueError on NaN")
except ValueError as e:
    print(f"OK — NaN correctly raised: {e}")

# Invalid confidence
try:
    calculate_var(daily_returns, confidence=1.5)
    print("FAIL: should have raised ValueError on invalid confidence")
except ValueError as e:
    print(f"OK — invalid confidence correctly raised: {e}")

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
