import pandas as pd

from src.data.validation import validate_fundamentals, validate_prices


def test_price_validation_does_not_bridge_reused_ticker_across_ciks():
    prices = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA", "AAA"],
        "cik": [1, 1, 2, 2],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2025-01-01", "2025-01-02"]),
        "open": [10.0, 11.0, 100.0, 101.0],
        "high": [10.5, 11.5, 100.5, 101.5],
        "low": [9.5, 10.5, 99.5, 100.5],
        "close": [10.0, 11.0, 100.0, 101.0],
        "volume": [1000, 1000, 1000, 1000],
    })
    report = validate_prices(prices, min_rows_per_ticker=2)
    assert report.ok
    assert report.stats["instrument_key"] == "ticker+cik"
    assert report.stats["instruments"] == 2
    assert report.stats["close_moves_over_300pct"] == 0


def test_fundamental_validation_enforces_cik_key_and_next_day_guard():
    fundamentals = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "cik": [1, 2],
        "source_filed_date": ["2025-02-15", "2025-02-15"],
        "available_date": ["2025-02-16", "2025-02-16"],
        "roe": [0.2, 0.3],
        "fcf_margin": [0.1, 0.2],
        "debt_to_equity": [0.2, 0.3],
        "current_ratio": [2.0, 1.5],
        "market_cap": [10_000_000_000, 12_000_000_000],
    })
    report = validate_fundamentals(fundamentals)
    assert report.ok
    assert report.stats["instrument_key"] == "ticker+cik"
    assert report.stats["instruments"] == 2
    assert report.stats["availability_lag_days_min"] == 1

    leaked = fundamentals.copy()
    leaked.loc[0, "available_date"] = "2025-02-15"
    report = validate_fundamentals(leaked)
    assert not report.ok
    assert any("available on/before" in error for error in report.errors)
