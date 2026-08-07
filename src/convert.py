"""
convert.py

Converts a raw DataFrame + confirmed column mapping into a pandas Series
of daily returns — the exact input shape risk_metrics.py expects.

Handles two portfolio shapes:
    1. Single-series: one row per date, one price/value column
       (e.g. a pre-aggregated portfolio NAV export).
    2. Multi-asset (long format): one row per date PER symbol, with
       price and quantity columns, so portfolio value is computed as
       sum(price * quantity) across symbols for each date.

Design choice: NaN policy is REJECT by default. This module never
silently drops or imputes missing data — it raises with a specific,
actionable error so the user can fix their source file. This matters
for a finance tool: silently interpolating a missing price could
distort every downstream risk number without anyone noticing.
"""

import pandas as pd


class ConversionError(ValueError):
    """Raised when mapped data cannot be safely converted into a returns series."""
    pass


def build_portfolio_value(df: pd.DataFrame, mapping: dict) -> pd.Series:
    """
    Build a single portfolio value time series from raw data + column mapping.

    Args:
        df: Raw DataFrame (as loaded by ingest.load_portfolio_file()).
        mapping: Confirmed column mapping, e.g. {"date": "Date", "price": "NAV"}
            for single-series, or {"date": "Date", "symbol": "Ticker",
            "price": "Price", "quantity": "Shares"} for multi-asset.
            Must already be validated via mapping.validate_mapping().

    Returns:
        pd.Series: portfolio value indexed by date (ascending, no duplicates),
            ready for convert_prices_to_returns().

    Raises:
        ConversionError: if dates are duplicated (single-series case),
            if price/quantity columns contain non-numeric values, if any
            required value is NaN, or if fewer than 2 dates remain.
    """
    date_col = mapping["date"]
    price_col = mapping["price"]
    symbol_col = mapping.get("symbol")
    quantity_col = mapping.get("quantity")

    work = df.copy()

    try:
        work[date_col] = pd.to_datetime(work[date_col])
    except Exception as e:
        raise ConversionError(
            f"Column '{date_col}' could not be parsed as dates. "
            f"Check for inconsistent formats or non-date values. Details: {e}"
        ) from e

    for col, label in [(price_col, "price"), (quantity_col, "quantity")]:
        if col is None:
            continue
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if symbol_col and quantity_col:
        # Multi-asset long format: value = sum(price * quantity) per date
        missing_price = work[price_col].isna()
        missing_qty = work[quantity_col].isna()
        if missing_price.any() or missing_qty.any():
            bad_rows = work.loc[missing_price | missing_qty, [date_col, symbol_col]]
            raise ConversionError(
                f"Found {len(bad_rows)} row(s) with missing or non-numeric price/quantity. "
                f"First affected row: date={bad_rows.iloc[0][date_col]}, "
                f"symbol={bad_rows.iloc[0][symbol_col]}. Fix or remove these rows and re-upload."
            )

        work["_holding_value"] = work[price_col] * work[quantity_col]
        portfolio_value = work.groupby(date_col)["_holding_value"].sum().sort_index()

    else:
        # Single-series: one row per date
        if work[date_col].duplicated().any():
            dupe_dates = work.loc[work[date_col].duplicated(), date_col].dt.date.unique()
            raise ConversionError(
                f"Duplicate dates found ({', '.join(str(d) for d in dupe_dates[:5])}"
                f"{'...' if len(dupe_dates) > 5 else ''}). For a single-series portfolio, "
                f"each date must appear once. If this is a multi-asset portfolio, "
                f"map 'symbol' and 'quantity' columns too."
            )

        if work[price_col].isna().any():
            bad_dates = work.loc[work[price_col].isna(), date_col].dt.date.tolist()
            raise ConversionError(
                f"Missing or non-numeric values in '{price_col}' for date(s): "
                f"{', '.join(str(d) for d in bad_dates[:5])}"
                f"{'...' if len(bad_dates) > 5 else ''}. Fix these rows and re-upload."
            )

        portfolio_value = work.set_index(date_col)[price_col].sort_index()

    if len(portfolio_value) < 2:
        raise ConversionError(
            f"Only {len(portfolio_value)} date(s) of portfolio value could be built. "
            f"At least 2 are needed to calculate returns."
        )

    portfolio_value.name = "portfolio_value"
    return portfolio_value


def convert_prices_to_returns(price_series: pd.Series) -> pd.Series:
    """
    Convert a price/value series into a series of periodic (daily) returns.

    Args:
        price_series: pd.Series of portfolio values indexed by date,
            ascending order, e.g. the output of build_portfolio_value().

    Returns:
        pd.Series: percentage returns between consecutive periods, with
            the first (NaN) observation dropped. This is the exact input
            shape risk_metrics.py functions expect.

    Raises:
        ConversionError: if price_series has fewer than 2 observations,
            or if any value is zero or negative (undefined/invalid for
            return calculation).
    """
    if price_series is None or len(price_series) < 2:
        raise ConversionError("Need at least 2 data points to calculate returns.")

    if (price_series <= 0).any():
        raise ConversionError(
            "Portfolio value series contains zero or negative values, which makes "
            "returns undefined. Check the source data for errors."
        )

    returns = price_series.pct_change().dropna()
    returns.name = "returns"
    return returns


def dataframe_to_returns(df: pd.DataFrame, mapping: dict) -> pd.Series:
    """
    Convenience wrapper: raw DataFrame + confirmed mapping -> returns Series,
    ready to pass directly into risk_metrics.py functions.

    Args:
        df: Raw DataFrame (as loaded by ingest.load_portfolio_file()).
        mapping: Confirmed column mapping (see build_portfolio_value() docstring).

    Returns:
        pd.Series: daily returns, ready for calculate_volatility(),
            calculate_sharpe_ratio(), calculate_var(), calculate_max_drawdown().

    Raises:
        ConversionError: propagated from build_portfolio_value() or
            convert_prices_to_returns().
    """
    portfolio_value = build_portfolio_value(df, mapping)
    return convert_prices_to_returns(portfolio_value)
