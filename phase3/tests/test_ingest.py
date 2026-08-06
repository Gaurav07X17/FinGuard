import io
import os
import sys
import pandas as pd
import pytest

# ensure repo root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from phase3.ingest import ingest_file, NaNError, MissingColumns
from phase3.mapping import auto_guess_mapping
from phase3.convert import df_price_column_to_returns


def make_csv_bytes(text: str) -> bytes:
    return text.encode('utf-8')


def test_ingest_basic_and_convert():
    csv = "date,close\n2020-01-01,100\n2020-01-02,101\n2020-01-03,102\n"
    b = make_csv_bytes(csv)
    res = ingest_file(b, filename="test.csv", required_columns=["date", "close"], nan_policy="reject")
    df = res["df"]
    assert list(df.columns) == ["date", "close"]
    mapping = auto_guess_mapping(list(df.columns))
    assert mapping["price"] in ("close",)
    returns = df_price_column_to_returns(df, mapping)
    assert len(returns) == 2


def test_ingest_missing_column_raises():
    csv = "date,price\n2020-01-01,100\n"
    b = make_csv_bytes(csv)
    with pytest.raises(MissingColumns):
        ingest_file(b, filename="test.csv", required_columns=["date", "close"], nan_policy="reject")


def test_ingest_rejects_nans():
    csv = "date,close\n2020-01-01,100\n2020-01-02,\n"
    b = make_csv_bytes(csv)
    with pytest.raises(NaNError):
        ingest_file(b, filename="test.csv", required_columns=["date", "close"], nan_policy="reject")


def test_ingest_drop_policy():
    csv = "date,close\n2020-01-01,100\n2020-01-02,\n2020-01-03,102\n"
    b = make_csv_bytes(csv)
    res = ingest_file(b, filename="test.csv", required_columns=["date", "close"], nan_policy="drop")
    df = res["df"]
    assert df.shape[0] == 2
