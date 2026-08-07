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

Input convention: all functions take a pandas Series of PERIODIC (daily)
returns, e.g. returns = prices.pct_change().dropna()
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> dict:
    """
    Calculate the standard deviation of portfolio returns (volatility).

    Args:
        returns: Series of periodic (daily) returns.
        annualize: If True, scales daily volatility to an annualized figure
            using sqrt(252). Defaults to True since annualized volatility
            is the standard basis for comparison across portfolios.

    Returns:
        dict with:
            value (float): the volatility, as a decimal (e.g. 0.18 = 18%)
            interpretation (str): plain-English explanation

    Raises:
        ValueError: if returns is empty or has fewer than 2 observations.
    """
    _validate_returns(returns)

    daily_vol = returns.std()
    vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else daily_vol
    period_label = "annualized" if annualize else "daily"

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
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    annualize: bool = True,
) -> dict:
    """
    Calculate the Sharpe ratio: risk-adjusted return per unit of volatility.

    Args:
        returns: Series of periodic (daily) returns.
        risk_free_rate: Annualized risk-free rate as a decimal (e.g. 0.04 = 4%).
            Defaults to 4%, a reasonable approximation of recent short-term
            T-bill yields. Exposed as an argument so users can ask
            "what's my Sharpe at a 2% risk-free rate?" via the NLP layer.
        annualize: If True, annualizes both return and volatility before
            computing the ratio. Defaults to True (standard convention).

    Returns:
        dict with:
            value (float): the Sharpe ratio (unitless)
            interpretation (str): plain-English explanation

    Raises:
        ValueError: if returns is empty, has fewer than 2 observations,
            or volatility is zero (undefined ratio).
    """
    _validate_returns(returns)

    mean_return = returns.mean()
    vol = returns.std()

    if vol == 0:
        raise ValueError("Cannot calculate Sharpe ratio: return series has zero volatility.")

    if annualize:
        annualized_return = mean_return * TRADING_DAYS_PER_YEAR
        annualized_vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (annualized_return - risk_free_rate) / annualized_vol
    else:
        sharpe = (mean_return - risk_free_rate) / vol

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
        f"return earned per unit of risk taken, above a {risk_free_rate:.2%} "
        f"risk-free rate. Higher is better; above 1 is generally considered good."
    )

    return {"value": round(float(sharpe), 4), "interpretation": interpretation}


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> dict:
    """
    Calculate Value at Risk (VaR): the maximum expected loss over one period
    at a given confidence level.

    Args:
        returns: Series of periodic (daily) returns.
        confidence: Confidence level as a decimal (e.g. 0.95 = 95%).
            Defaults to 95%, the most common convention. Exposed as an
            argument so users can ask for VaR at custom confidence levels.
        method: "historical" (empirical percentile, default) or
            "parametric" (assumes normally distributed returns).

    Returns:
        dict with:
            value (float): VaR as a positive decimal representing potential
                loss (e.g. 0.032 = a 3.2% expected loss)
            interpretation (str): plain-English explanation

    Raises:
        ValueError: if returns is empty, confidence is not in (0, 1),
            or method is not recognized.
    """
    _validate_returns(returns)

    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    if method == "historical":
        percentile = (1 - confidence) * 100
        var = -np.percentile(returns, percentile)
    elif method == "parametric":
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence)
        var = -(returns.mean() + z_score * returns.std())
    else:
        raise ValueError(f"method must be 'historical' or 'parametric', got '{method}'")

    var = max(var, 0.0)  # VaR shouldn't be negative in this convention

    interpretation = (
        f"At a {confidence:.0%} confidence level, the portfolio's 1-day VaR is "
        f"{var:.2%}. This means there is a {(1-confidence):.0%} chance of "
        f"losing more than {var:.2%} of portfolio value in a single day, "
        f"based on {method} estimation."
    )

    return {"value": round(float(var), 6), "interpretation": interpretation}


def calculate_max_drawdown(returns: pd.Series) -> dict:
    """
    Calculate maximum drawdown: the largest peak-to-trough decline in
    cumulative portfolio value over the observed period.

    Args:
        returns: Series of periodic (daily) returns.

    Returns:
        dict with:
            value (float): max drawdown as a positive decimal (e.g. 0.24 = 24% decline)
            interpretation (str): plain-English explanation

    Raises:
        ValueError: if returns is empty or has fewer than 2 observations.
    """
    _validate_returns(returns)

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


def _validate_returns(returns: pd.Series) -> None:
    """Shared input validation for all metric functions."""
    if returns is None or len(returns) == 0:
        raise ValueError("returns series is empty.")
    if len(returns) < 2:
        raise ValueError("returns series must have at least 2 observations.")
    if returns.isna().any():
        raise ValueError("returns series contains NaN values — clean data before calculating metrics.")
