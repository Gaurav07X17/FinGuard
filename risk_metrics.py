"""
risk_metrics.py

Standalone calculation engine for portfolio risk metrics.
No file I/O, no LLM calls, no UI — pure functions only.

Design contract (locked in Phase 1):
- Parameters (risk_free_rate, confidence, etc.) are function arguments
  with sensible defaults, NOT config constants. This lets the NLP layer
  (Phase 4) pass user-specified values straight through via LLM
  function-calling.
- Every function returns a dict of shape {"value": float, "interpretation": str},
  never a bare float. This gives the LLM layer a ready-made explanation
  hook and keeps all four functions consistent for tool-calling schemas.

Input convention: all functions accept an array-like or pandas.Series of PERIODIC
returns (e.g. daily returns). By default we assume 252 trading days per year,
but callers can override periods_per_year for non-daily data.

Important: strict NaN policy — functions will raise ValueError if the input
series contains NaN values. Clean or impute upstream in the ingestion layer.
"""
import numpy as np
import pandas as pd
from typing import Optional

TRADING_DAYS_PER_YEAR = 252


def calculate_volatility(
    returns,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ddof: int = 1,
) -> dict:
    """
    Calculate the standard deviation of periodic returns (volatility).

    Raises ValueError if returns is None, length < 2, or contains NaN.
    """
    returns = _coerce_and_validate_returns(returns)
    per_period_vol = returns.std(ddof=ddof)
    vol = per_period_vol * np.sqrt(periods_per_year) if annualize else per_period_vol
    period_label = "annualized" if annualize else "per-period"

    if vol < 0.10:
        risk_band = "relatively low"
    elif vol < 0.20:
        risk_band = "moderate"
    elif vol < 0.35:
        risk_band = "elevated"
    else:
        risk_band = "high"

    interpretation = (
        f"The portfolio's {period_label} volatility is {vol:.2%}, which is "
        f"{risk_band} risk. This measures how much returns fluctuate around "
        f"their average — higher volatility means larger swings, up or down."
    )

    return {"value": round(float(vol), 6), "interpretation": interpretation}


def calculate_sharpe_ratio(
    returns,
    risk_free_rate: float = 0.04,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ddof: int = 1,
) -> dict:
    """
    Calculate the Sharpe ratio.

    Notes:
    - risk_free_rate is interpreted as an annualized rate. When annualize=False,
      the function converts it to the per-period equivalent by dividing by
      periods_per_year (this matches the test expectations).
    - Raises ValueError for empty/too-short series or zero volatility.
    """
    returns = _coerce_and_validate_returns(returns)
    mean_return = returns.mean()
    vol = returns.std(ddof=ddof)

    if vol == 0:
        raise ValueError("Cannot calculate Sharpe ratio: return series has zero volatility.")

    if annualize:
        annualized_return = mean_return * periods_per_year
        annualized_vol = vol * np.sqrt(periods_per_year)
        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        rf_label = f"{risk_free_rate:.2%} (annual)"
    else:
        rf_per_period = risk_free_rate / periods_per_year
        sharpe = (mean_return - rf_per_period) / vol
        rf_label = f"{rf_per_period:.4%} (per-period equivalent of {risk_free_rate:.2%} annual)"

    if sharpe < 0:
        quality = "poor — the portfolio underperformed the risk-free rate on a risk-adjusted basis"
    elif sharpe < 1:
        quality = "sub-optimal"
    elif sharpe < 2:
        quality = "good"
    else:
        quality = "excellent"

    interpretation = (
        f"The Sharpe ratio is {sharpe:.2f}, which is {quality}. This measures "
        f"return earned per unit of risk taken, above a {rf_label} risk-free rate. "
        f"Higher is better; above 1 is generally considered good."
    )

    return {"value": round(float(sharpe), 4), "interpretation": interpretation}


def calculate_var(
    returns,
    confidence: float = 0.95,
    method: str = "historical",
) -> dict:
    """
    Calculate Value at Risk (VaR).

    Raises ValueError for invalid confidence or method. Raises ImportError if
    parametric method requested but scipy not installed.
    """
    returns = _coerce_and_validate_returns(returns)

    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    if method == "historical":
        percentile = (1 - confidence) * 100
        var = -np.percentile(returns, percentile)
    elif method == "parametric":
        try:
            from scipy.stats import norm  # type: ignore
        except Exception as e:
            raise ImportError("Parametric VaR requires scipy; please install scipy") from e
        z_score = norm.ppf(1 - confidence)
        var = -(returns.mean() + z_score * returns.std())
    else:
        raise ValueError(f"method must be 'historical' or 'parametric', got '{method}'")

    var = max(var, 0.0)

    interpretation = (
        f"At a {confidence:.0%} confidence level, the portfolio's 1-period VaR is "
        f"{var:.2%}. This means there is a {(1-confidence):.0%} chance of "
        f"losing more than {var:.2%} of portfolio value in a single period, "
        f"based on {method} estimation."
    )

    return {"value": round(float(var), 6), "interpretation": interpretation}


def calculate_max_drawdown(returns) -> dict:
    """
    Calculate maximum drawdown (largest peak-to-trough decline).

    Raises ValueError for empty/too-short series or NaNs.
    """
    returns = _coerce_and_validate_returns(returns)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown_series = (cumulative - running_max) / running_max
    max_dd = -drawdown_series.min()

    if max_dd < 0.10:
        severity = "mild"
    elif max_dd < 0.20:
        severity = "moderate"
    elif max_dd < 0.35:
        severity = "significant"
    else:
        severity = "severe"

    interpretation = (
        f"The maximum drawdown was {max_dd:.2%}, a {severity} decline. This "
        f"represents the largest drop from a peak to a subsequent low over "
        f"the observed period — a key measure of downside risk and how long "
        f"the portfolio might take to recover from its worst stretch."
    )

    return {"value": round(float(max_dd), 6), "interpretation": interpretation}


def _coerce_and_validate_returns(returns: Optional[pd.Series]) -> pd.Series:
    """Coerce array-like to pandas.Series and validate. Strict on NaN."""
    if returns is None:
        raise ValueError("returns series is empty.")
    returns = pd.Series(returns)
    if returns.isna().any():
        raise ValueError("returns series contains NaN values — clean data before calculating metrics.")
    if len(returns) == 0:
        raise ValueError("returns series is empty.")
    if len(returns) < 2:
        raise ValueError("returns series must have at least 2 observations.")
    return returns
