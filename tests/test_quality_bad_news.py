import pandas as pd

from src.strategies.quality_bad_news import QualityDipConfig, generate_trades, prepare_features


def test_point_in_time_quality_filter_and_trade_generation():
    prices = pd.DataFrame({
        "ticker": ["AAA"] * 6,
        "date": pd.date_range("2025-01-01", periods=6, freq="D"),
        "open": [100, 100, 89, 90, 92, 94],
        "close": [100, 100, 88, 91, 93, 95],
    })
    fundamentals = pd.DataFrame({
        "ticker": ["AAA"],
        "available_date": ["2024-12-20"],
        "roe": [0.20],
        "fcf_margin": [0.10],
        "debt_to_equity": [0.4],
        "current_ratio": [1.8],
        "market_cap": [20_000_000_000],
    })
    cfg = QualityDipConfig(drop_threshold=-0.10, wait_days=0, hold_days=2, round_trip_cost_bps=0)
    trades = generate_trades(prices, fundamentals, cfg)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_price"] == 90
    assert trades.iloc[0]["exit_price"] == 95


def test_future_fundamentals_are_not_used():
    prices = pd.DataFrame({
        "ticker": ["AAA"] * 4,
        "date": pd.date_range("2025-01-01", periods=4, freq="D"),
        "open": [100, 89, 90, 92],
        "close": [100, 88, 91, 93],
    })
    fundamentals = pd.DataFrame({
        "ticker": ["AAA"],
        "available_date": ["2025-02-01"],
        "roe": [0.30],
        "fcf_margin": [0.20],
        "debt_to_equity": [0.1],
        "current_ratio": [2.0],
        "market_cap": [20_000_000_000],
    })
    trades = generate_trades(prices, fundamentals, QualityDipConfig(drop_threshold=-0.10, wait_days=0, hold_days=1))
    assert trades.empty


def test_prepare_features_normalizes_mixed_datetime_resolution():
    prices = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": pd.Series(["2025-01-02", "2025-01-03"], dtype="datetime64[ms]"),
        "open": [100.0, 101.0],
        "close": [100.0, 101.0],
    })
    fundamentals = pd.DataFrame({
        "ticker": ["AAA"],
        "available_date": pd.Series(["2025-01-01"], dtype="datetime64[us]"),
        "roe": [0.20],
        "fcf_margin": [0.10],
        "debt_to_equity": [0.4],
        "current_ratio": [1.8],
        "market_cap": [20_000_000_000],
    })
    features = prepare_features(prices, fundamentals)
    assert str(features["date"].dtype) == "datetime64[ns]"
    assert str(features["available_date"].dtype) == "datetime64[ns]"
    assert features["roe"].tolist() == [0.20, 0.20]
