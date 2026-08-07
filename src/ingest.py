"""
ingest.py

Handles loading portfolio files (CSV/XLSX) into a pandas DataFrame with
validation. No column interpretation here — that's mapping.py's job.
This module only answers: "is this a readable, safe-sized file?"

Defaults (overridable via function args, same convention as risk_metrics.py):
    max_size_mb = 10
    max_rows = 100_000
    allowed extensions = .csv, .xlsx

Design choice: all processing is in-memory. No temp files are written to
disk, so there's nothing to clean up and nothing that could persist
sensitive portfolio data beyond the request lifecycle.
"""

import io
import os
import pandas as pd

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


class IngestionError(ValueError):
    """Raised when a file fails validation or cannot be parsed.
    Subclasses ValueError so callers can catch broadly if desired."""
    pass


def load_portfolio_file(
    file,
    filename: str,
    max_size_mb: float = 10,
    max_rows: int = 100_000,
    sheet_name=0,
) -> pd.DataFrame:
    """
    Load a portfolio file (CSV or XLSX) into a DataFrame, with validation.

    Args:
        file: A file-like object (e.g. Streamlit UploadedFile, open() handle,
            or io.BytesIO) positioned at the start of the file.
        filename: Original filename, used to determine file type. Required
            because file-like objects don't always expose a reliable name.
        max_size_mb: Maximum allowed file size in megabytes. Defaults to 10.
        max_rows: Maximum allowed number of data rows (excluding header).
            Defaults to 100,000.
        sheet_name: For XLSX files, which sheet to load. Defaults to 0
            (first sheet). Pass a sheet name string to load a specific
            sheet — use list_excel_sheets() first to see available options.

    Returns:
        pd.DataFrame: the raw, unmodified tabular data from the file.

    Raises:
        IngestionError: if the file type is unsupported, the file exceeds
            size or row limits, or the file cannot be parsed.
    """
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{ext}'. Allowed types: "
            f"{', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    raw_bytes = file.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise IngestionError(
            f"File is {size_mb:.1f} MB, which exceeds the {max_size_mb} MB limit. "
            f"Try uploading a smaller file or trimming the date range."
        )
    if size_mb == 0:
        raise IngestionError("File is empty.")

    buffer = io.BytesIO(raw_bytes)

    try:
        if ext == ".csv":
            df = pd.read_csv(buffer, sep=None, engine="python")
        else:  # .xlsx
            df = pd.read_excel(buffer, sheet_name=sheet_name, engine="openpyxl")
    except Exception as e:
        raise IngestionError(
            f"Could not parse '{filename}' as a valid {ext} file. "
            f"It may be corrupted or in an unexpected format. Details: {e}"
        ) from e

    if isinstance(df, dict):
        # sheet_name=None or a list was passed and pandas returned multiple sheets
        raise IngestionError(
            "Multiple sheets were returned — pass a single sheet_name. "
            "Use list_excel_sheets() to see available sheets."
        )

    if df.empty:
        raise IngestionError(f"'{filename}' was parsed but contains no data rows.")

    if len(df) > max_rows:
        raise IngestionError(
            f"File has {len(df):,} rows, which exceeds the {max_rows:,} row limit. "
            f"Try trimming the date range or splitting the file."
        )

    # Drop fully-empty columns/rows that sometimes appear from Excel formatting artifacts
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)

    if df.empty:
        raise IngestionError(f"'{filename}' contains no usable data after removing blank rows/columns.")

    return df


def list_excel_sheets(file, filename: str) -> list:
    """
    List available sheet names in an XLSX file, without fully loading any of them.
    Used to let the user pick a sheet before calling load_portfolio_file().

    Args:
        file: A file-like object positioned at the start of the file.
        filename: Original filename (used to validate it's an .xlsx file).

    Returns:
        list[str]: sheet names in the workbook, in order.

    Raises:
        IngestionError: if the file is not an .xlsx or cannot be opened.
    """
    _, ext = os.path.splitext(filename.lower())
    if ext != ".xlsx":
        raise IngestionError(f"list_excel_sheets only supports .xlsx files, got '{ext}'.")

    raw_bytes = file.read()
    buffer = io.BytesIO(raw_bytes)

    try:
        excel_file = pd.ExcelFile(buffer, engine="openpyxl")
    except Exception as e:
        raise IngestionError(f"Could not open '{filename}' to list sheets. Details: {e}") from e

    return excel_file.sheet_names
