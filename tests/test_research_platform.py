import pandas as pd

from src.backtest.portfolio import PortfolioConfig, PortfolioSimulator
from src.data.universe import attach_membership, historical_components_to_intervals
from src.data.validation import validate_prices
from src.research.experiment import stable_config_hash
from src.strategies.quality_bad_news import QualityDipConfig, generate_trades


def good_fundamentals(ticker="AAA"):
    return pd.DataFrame({
        "ticker": [ticker],
        "available_date": ["2024-12-20"],
        "roe": [0.25],
        "fcf_margin": [0.12],
        "debt_to_equity": [0.3],
        "current_ratio": [1.8],
        "market_cap": [20_000_000_000],
    })


def test_historical_snapshots_become_non_overlapping_intervals():
    history = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-03"],
        "tickers": ["AAA,BBB", "AAA,CCC"],
    })
    intervals = historical_components_to_intervals(history)
    bbb = intervals[intervals["ticker"] == "BBB"].iloc[0]
    ccc = intervals[intervals["ticker"] == "CCC"].iloc[0]
    assert bbb["end_date"] == pd.Timestamp("2020-01-03")
    assert ccc["start_date"] == pd.Timestamp("2020-01-03")

    frame = pd.DataFrame({"ticker": ["BBB", "BBB"], "date": ["2020-01-02", "2020-01-03"]})
    flagged = attach_membership(frame, intervals)
    assert flagged["in_universe"].tolist() == [True, False]


def test_price_validation_rejects_tiny_history():
    prices = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": ["2025-01-01", "2025-01-02"],
        "open": [10, 10], "high": [11, 11], "low": [9, 9],
        "close": [10, 10], "volume": [100, 100],
    })
    report = validate_prices(prices, min_rows_per_ticker=100)
    assert not report.ok
    assert any("below 100 rows" in error for error in report.errors)


def test_stabilization_requires_a_completed_confirmation_session():
    prices = pd.DataFrame({
        "ticker": ["AAA"] * 6,
        "date": pd.date_range("2025-01-01", periods=6, freq="D"),
        "open": [100, 82, 81, 83, 84, 85],
        "high": [101, 83, 83, 84, 85, 86],
        "low": [99, 79, 80, 82, 83, 84],
        "close": [100, 80, 82, 83, 84, 85],
        "volume": [1000] * 6,
    })
    cfg = QualityDipConfig(
        drop_threshold=-0.10,
        wait_days=0,
        hold_days=1,
        require_stabilization=True,
        round_trip_cost_bps=0,
    )
    trades = generate_trades(prices, good_fundamentals(), cfg)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_date"] == pd.Timestamp("2025-01-04")
    assert trades.iloc[0]["entry_price"] == 83


def test_portfolio_enforces_max_concurrent_positions():
    prices = pd.DataFrame({
        "ticker": ["AAA", "BBB", "AAA", "BBB"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]),
        "open": [10, 20, 11, 20],
        "close": [10, 20, 11, 20],
    })
    trades = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "entry_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "entry_price": [10.0, 20.0],
        "exit_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
        "net_return": [0.10, 0.00],
    })
    simulator = PortfolioSimulator(prices, PortfolioConfig(initial_cash=10_000, max_positions=1, max_position_fraction=0.5))
    metrics, _, accepted = simulator.run(trades)
    assert metrics.accepted_trades == 1
    assert metrics.skipped_trades == 1
    assert len(accepted) == 1


def test_config_hash_is_order_independent():
    assert stable_config_hash({"a": 1, "b": 2}) == stable_config_hash({"b": 2, "a": 1})
