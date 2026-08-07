"""
mapping.py

Auto-guesses which columns in an uploaded portfolio file correspond to
which semantic roles (date, price, symbol, quantity), so the user isn't
forced to manually rename their Excel columns.

Design choice: this module GUESSES and reports confidence — it never
silently commits to a mapping the caller hasn't confirmed. The Streamlit
UI (Phase 5) is expected to show the guessed mapping to the user for
confirmation before it's used in convert.py.
"""

import difflib
import pandas as pd

# Role -> known aliases (lowercase, no punctuation) used for matching.
# Ordered roughly by how common the alias is in real-world exports.
ROLE_ALIASES = {
    "date": ["date", "trade date", "asof date", "as of date", "timestamp", "period", "day"],
    "price": ["price", "close", "closing price", "nav", "value", "portfolio value",
              "market value", "adj close", "adjusted close", "close price"],
    "symbol": ["symbol", "ticker", "security", "asset", "name", "instrument"],
    "quantity": ["quantity", "qty", "shares", "units", "holding", "position size"],
}

REQUIRED_ROLES = {"date", "price"}
OPTIONAL_ROLES = {"symbol", "quantity"}


def _normalize(col_name: str) -> str:
    """Lowercase and strip common punctuation/whitespace for comparison."""
    return str(col_name).strip().lower().replace("_", " ").replace("-", " ")


def guess_column_mapping(df: pd.DataFrame, min_confidence: float = 0.6) -> dict:
    """
    Guess which DataFrame columns correspond to date/price/symbol/quantity roles.

    Args:
        df: The raw DataFrame loaded by ingest.load_portfolio_file().
        min_confidence: Minimum fuzzy-match score (0-1) to accept a guess.
            Defaults to 0.6. Lower values guess more aggressively but risk
            false matches; higher values are stricter and more likely to
            leave a role unmapped for manual confirmation.

    Returns:
        dict: {
            "mapping": {role: column_name_or_None, ...},
            "confidence": {role: score_or_None, ...},
            "unmapped_columns": [columns not matched to any role],
        }
        Only roles with a confident match appear with a value in "mapping";
        unmatched roles map to None and must be resolved by the caller
        (typically via user confirmation in the UI) before conversion.

    Raises:
        ValueError: if df has no columns at all.
    """
    if df is None or len(df.columns) == 0:
        raise ValueError("DataFrame has no columns to map.")

    columns = list(df.columns)
    normalized_cols = {col: _normalize(col) for col in columns}

    mapping = {}
    confidence = {}
    used_columns = set()

    for role, aliases in ROLE_ALIASES.items():
        best_col = None
        best_score = 0.0

        for col in columns:
            if col in used_columns:
                continue
            norm_col = normalized_cols[col]

            # Exact alias match first (fast path, high confidence)
            if norm_col in aliases:
                best_col, best_score = col, 1.0
                break

            # Fuzzy match against each alias, take the best score for this column
            for alias in aliases:
                score = difflib.SequenceMatcher(None, norm_col, alias).ratio()
                if score > best_score:
                    best_col, best_score = col, score

        if best_col is not None and best_score >= min_confidence:
            mapping[role] = best_col
            confidence[role] = round(best_score, 3)
            used_columns.add(best_col)
        else:
            mapping[role] = None
            confidence[role] = None

    unmapped_columns = [c for c in columns if c not in used_columns]

    return {
        "mapping": mapping,
        "confidence": confidence,
        "unmapped_columns": unmapped_columns,
    }


def validate_mapping(df: pd.DataFrame, mapping: dict) -> None:
    """
    Validate that a (user-confirmed) column mapping is usable before conversion.

    Args:
        df: The DataFrame the mapping refers to.
        mapping: dict of {role: column_name}, e.g. the "mapping" key from
            guess_column_mapping()'s return value, after user confirmation.

    Returns:
        None. Raises on any problem.

    Raises:
        ValueError: if a required role (date, price) is missing or None,
            if a mapped column name doesn't exist in df, or if 'symbol'
            is mapped without 'quantity' (or vice versa) — a partial
            multi-asset mapping that convert.py can't safely use.
    """
    missing_required = [role for role in REQUIRED_ROLES if not mapping.get(role)]
    if missing_required:
        raise ValueError(
            f"Missing required column mapping(s): {', '.join(missing_required)}. "
            f"These must be mapped to proceed."
        )

    for role, col in mapping.items():
        if col is not None and col not in df.columns:
            raise ValueError(
                f"Mapped column '{col}' for role '{role}' does not exist in the uploaded file."
            )

    has_symbol = bool(mapping.get("symbol"))
    has_quantity = bool(mapping.get("quantity"))
    if has_symbol != has_quantity:
        present = "symbol" if has_symbol else "quantity"
        missing = "quantity" if has_symbol else "symbol"
        raise ValueError(
            f"'{present}' is mapped but '{missing}' is not. For multi-asset portfolios, "
            f"both must be mapped together so holdings can be valued correctly."
        )
