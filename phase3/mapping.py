"""
mapping.py

Simple column-mapping heuristics and helpers.
Provides auto_guess_mapping(df.columns) -> mapping dict for keys:
  - date
  - price (or close)
  - symbol (optional)
  - quantity (optional)

Uses difflib for fuzzy matching and common name lists.
"""
from typing import List, Dict, Optional
import difflib

COMMON_DATE_NAMES = ["date", "trade_date", "timestamp", "time"]
COMMON_PRICE_NAMES = ["close", "adj_close", "close_price", "price", "last"]
COMMON_SYMBOL_NAMES = ["symbol", "ticker", "asset", "name"]
COMMON_QTY_NAMES = ["quantity", "qty", "shares"]


def _best_match(col_list: List[str], candidates: List[str]) -> Optional[str]:
    if not col_list:
        return None
    lowered = [c.lower() for c in col_list]
    for cand in candidates:
        if cand in lowered:
            return col_list[lowered.index(cand)]
    # fuzzy match
    matches = difflib.get_close_matches(candidates[0], lowered, n=1, cutoff=0.8)
    if matches:
        return col_list[lowered.index(matches[0])]
    # try any candidate
    for cand in candidates:
        matches = difflib.get_close_matches(cand, lowered, n=1, cutoff=0.7)
        if matches:
            return col_list[lowered.index(matches[0])]
    return None


def auto_guess_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    """Return mapping with keys: date, price, symbol, quantity
    Values are column names or None if not found.
    """
    if isinstance(columns, dict):
        columns = list(columns)
    cols = list(columns)
    date = _best_match(cols, COMMON_DATE_NAMES)
    price = _best_match(cols, COMMON_PRICE_NAMES)
    symbol = _best_match(cols, COMMON_SYMBOL_NAMES)
    qty = _best_match(cols, COMMON_QTY_NAMES)
    return {"date": date, "price": price, "symbol": symbol, "quantity": qty}


def normalize_column_name(name: str) -> str:
    """Return a normalized column name (lower, stripped) suitable for matching."""
    return name.strip().lower()


if __name__ == "__main__":
    # quick demo
    cols = ["TradeDate", "Close", "Ticker", "Shares"]
    print(auto_guess_mapping(cols))
