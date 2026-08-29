import pandas as pd

from scripts.download_finsaber_prices import normalize_finsaber_prices
from src.data.io import read_table, write_table
from src.data.universe import normalize_ticker


def test_ticker_normalization_matches_share_class_separators():
    assert normalize_ticker("BRK-B") == "BRK.B"
    assert normalize_ticker("BRK/B") == "BRK.B"
    assert normalize_ticker("brk.b") == "BRK.B"


def test_finsaber_normalization_uses_adjusted_ohlc_and_preserves_raw_close():
    raw = pd.DataFrame({
        "date": ["2024-01-02"],
        "symbol": ["BRK-B"],
        "cik": [1067983],
        "open": [98.0],
        "high": [110.0],
        "low": [90.0],
        "close": [100.0],
        "adjusted_close": [50.0],
        "volume": [1234],
    })
    out = normalize_finsaber_prices(raw, 2024)
    row = out.iloc[0]
    assert row["ticker"] == "BRK.B"
    assert row["raw_close"] == 100.0
    assert row["close"] == 50.0
    assert row["open"] == 49.0
    assert row["high"] == 55.0
    assert row["low"] == 45.0


def test_finsaber_normalization_excludes_impossible_ohlc_and_counts_it():
    raw = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "symbol": ["AAA", "AAA"],
        "cik": [1, 1],
        "open": [100.0, 100.0],
        "high": [105.0, 95.0],
        "low": [95.0, 90.0],
        "close": [102.0, 100.0],
        "adjusted_close": [51.0, 50.0],
        "volume": [1000, 1000],
    })
    out = normalize_finsaber_prices(raw, 2024)
    assert len(out) == 1
    assert out.attrs["normalization_stats"]["invalid_ohlc_rows"] == 1


def test_table_io_round_trips_parquet(tmp_path):
    path = tmp_path / "prices.parquet"
    frame = pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2025-01-01"]), "close": [10.0]})
    write_table(frame, path)
    loaded = read_table(path)
    assert loaded.iloc[0]["ticker"] == "AAA"
    assert float(loaded.iloc[0]["close"]) == 10.0
