"""
tool_wrappers.py

Thin, validated wrappers around numerical calculations exposed to the LLM layer.
Each wrapper accepts primitive Python types (lists/numbers) and returns a JSON-serializable
object with keys: 'value' (numeric) and 'interpretation' (string).

These wrappers perform input validation and basic computation locally so tests do not
depend on external code. They are deterministic and mock-friendly.
"""
from typing import List, Dict, Any
import math
import statistics
import numpy as np


def _ensure_returns(returns: List[float]) -> List[float]:
    if not isinstance(returns, (list, tuple)):
        raise TypeError("returns must be a list of numbers")
    cleaned = []
    for r in returns:
        if r is None:
            continue
        try:
            cleaned.append(float(r))
        except Exception:
            raise TypeError("returns must be numeric values")
    if len(cleaned) == 0:
        raise ValueError("returns list is empty after cleaning")
    return cleaned


def calculate_sharpe(returns: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict[str, Any]:
    r = _ensure_returns(returns)
    ex_returns = [x - (risk_free_rate / periods_per_year) for x in r]
    mean = statistics.mean(ex_returns)
    stdev = statistics.pstdev(ex_returns)
    if stdev == 0:
        raise ValueError("zero volatility in returns; Sharpe undefined")
    sharpe = math.sqrt(periods_per_year) * (mean / stdev)
    interpretation = f"Sharpe ratio (annualized) = {sharpe:.4f}"
    return {"value": sharpe, "interpretation": interpretation}


def calculate_volatility(returns: List[float], periods_per_year: int = 252) -> Dict[str, Any]:
    r = _ensure_returns(returns)
    stdev = statistics.pstdev(r)
    ann = stdev * math.sqrt(periods_per_year)
    interpretation = f"Annualized volatility = {ann:.4%}"
    return {"value": ann, "interpretation": interpretation}


def calculate_var(returns: List[float], confidence_level: float = 0.95, method: str = "historical") -> Dict[str, Any]:
    r = _ensure_returns(returns)
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1")
    if method == "historical":
        sorted_r = sorted(r)
        idx = int((1 - confidence_level) * len(sorted_r))
        idx = max(0, min(len(sorted_r) - 1, idx))
        var = -sorted_r[idx]
        interpretation = f"Historical VaR at {confidence_level:.2%} = {var:.4%}"
        return {"value": var, "interpretation": interpretation}
    elif method == "parametric":
        mu = statistics.mean(r)
        sigma = statistics.pstdev(r)
        # z score for one-tailed
        z = float(np.abs(np.percentile(np.random.normal(size=100000), (1 - confidence_level) * 100)))
        # For standard normal the percentile approach above is approximate; use inverse CDF via numpy
        try:
            from scipy import stats
            z = abs(stats.norm.ppf(1 - confidence_level))
        except Exception:
            # fallback approximate z for common levels
            z_map = {0.9: 1.2816, 0.95: 1.645, 0.99: 2.3263}
            z = z_map.get(round(confidence_level, 2), 1.645)
        var = -(mu - z * sigma)
        interpretation = f"Parametric VaR at {confidence_level:.2%} = {var:.4%}"
        return {"value": var, "interpretation": interpretation}
    else:
        raise ValueError("method must be 'historical' or 'parametric'")
