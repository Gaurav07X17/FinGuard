"""
ingest.py

Utilities for parsing uploads (CSV/XLSX) into cleaned pandas DataFrames.
Strict NaN policy: by default required columns must not contain NaN; an
`nan_policy` argument allows 'reject' (default), 'drop' or 'impute'.

This module is ingestion-only: no persistent storage, files processed in-memory.
"""
from io import BytesIO
import pandas as pd
import chardet
import csv

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 100_000
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class IngestError(Exception):
    pass


class UnsupportedFileType(IngestError):
    pass


class FileTooLarge(IngestError):
    pass


class TooManyRows(IngestError):
    pass


class MissingColumns(IngestError):
    pass


class NaNError(IngestError):
    pass


def _detect_delimiter(sample_bytes: bytes) -> str:
    """Try to detect CSV delimiter using csv.Sniffer on a sample string."""
    try:
        sample = sample_bytes.decode("utf-8", errors="replace")[:32768]
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","


def _read_csv_bytes(content: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    sample = content[:65536]
    delimiter = _detect_delimiter(sample)
    df = pd.read_csv(BytesIO(content), delimiter=delimiter, encoding=encoding)
    return df


def _read_excel_bytes(content: bytes, sheet_name=0) -> pd.DataFrame:
    return pd.read_excel(BytesIO(content), sheet_name=sheet_name)


def _ensure_allowed(filename: str, size: int):
    if size > MAX_FILE_SIZE_BYTES:
        raise FileTooLarge(f"File exceeds max size of {MAX_FILE_SIZE_BYTES} bytes")
    low = filename.lower()
    if not any(low.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise UnsupportedFileType("Allowed file types: .csv, .xlsx, .xls")


def ingest_file(content: bytes, filename: str, required_columns: list = None, nan_policy: str = "reject") -> dict:
    """
    Ingest an uploaded file (bytes) and return a dict with DataFrame and metadata.

    Args:
        content: file bytes
        filename: original filename (to infer extension)
        required_columns: list of required column names (after mapping)
        nan_policy: 'reject' (default), 'drop', or 'impute'

    Returns:
        {"df": pd.DataFrame, "meta": {"sheet_names": [...], "rows": N}}
    """
    _ensure_allowed(filename, len(content))
    low = filename.lower()
    # read
    if low.endswith(".csv"):
        # detect encoding
        try:
            enc = chardet.detect(content[:65536])['encoding'] or 'utf-8'
        except Exception:
            enc = 'utf-8'
        df = _read_csv_bytes(content, encoding=enc)
    else:
        # xls/xlsx
        df = _read_excel_bytes(content, sheet_name=0)

    if len(df) > MAX_ROWS:
        raise TooManyRows(f"File has {len(df)} rows, above max of {MAX_ROWS}")

    # required columns
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise MissingColumns(f"Missing required columns: {missing}")

    # NaN policy
    if nan_policy == "reject":
        if df[required_columns].isna().any().any() if required_columns else df.isna().any().any():
            raise NaNError("Input contains NaN values in required columns")
    elif nan_policy == "drop":
        if required_columns:
            df = df.dropna(subset=required_columns)
        else:
            df = df.dropna()
    elif nan_policy == "impute":
        # forward then back
        df = df.fillna(method="ffill").fillna(method="bfill")
    else:
        raise ValueError("nan_policy must be 'reject', 'drop', or 'impute'")

    return {"df": df, "meta": {"rows": len(df), "columns": list(df.columns)}}
