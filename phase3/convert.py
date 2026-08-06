"""
convert.py

Helpers to convert price series (pandas Series) to returns and a wrapper
that extracts a price column from a DataFrame using a mapping then returns
a cleaned returns Series ready for use by risk_metrics.
"""
import pandas as pd
from typing import Dict


def convert_prices_to_returns(prices: pd.Series) -> pd.Series:
    """Convert a price series (index-preserving) to simple periodic returns.

    Notes:
    - Caller must ensure series is numeric and sorted by date if applicable.
    - This function does not drop NaNs — upstream ingestion must apply the
      NaN policy (reject/drop/impute) before calling the risk engine.
    """
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    if prices.empty:
        raise ValueError("prices series is empty")
    if not pd.api.types.is_numeric_dtype(prices):
        # try to coerce
        prices = pd.to_numeric(prices, errors="coerce")
    returns = prices.pct_change()
    returns = returns.dropna()
    return returns


def df_price_column_to_returns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.Series:
    """Given a DataFrame and a mapping (e.g., from mapping.auto_guess_mapping),
    extract the price column, convert to returns and return the Series.

    mapping should contain key 'price' pointing to the column name.
    """
    price_col = mapping.get("price")
    if price_col is None:
        raise KeyError("mapping must contain a 'price' column")
    if price_col not in df.columns:
        raise KeyError(f"price column '{price_col}' not found in DataFrame")
    prices = df[price_col]
    return convert_prices_to_returns(prices)
