import pandas as pd

from src.backtest.portfolio import PortfolioConfig, PortfolioSimulator
from src.strategies.quality_bad_news import QualityDipConfig, generate_trades


def test_quality_dip_trade_preserves_cik():
    prices = pd.DataFrame({
        "ticker": ["AAA"] * 5,
        "cik": [1] * 5,
        "date": pd.date_range("2025-01-01", periods=5, freq="D"),
        "open": [100, 89, 90, 92, 94],
        "close": [100, 88, 91, 93, 95],
    })
    fundamentals = pd.DataFrame({
        "ticker": ["AAA"], "cik": [1], "available_date": ["2024-12-20"],
        "roe": [0.20], "fcf_margin": [0.10], "debt_to_equity": [0.4],
        "current_ratio": [1.8], "market_cap": [20_000_000_000],
    })
    trades = generate_trades(
        prices, fundamentals,
        QualityDipConfig(drop_threshold=-0.10, wait_days=0, hold_days=1, round_trip_cost_bps=0),
    )
    assert len(trades) == 1
    assert trades.iloc[0]["cik"] == "0000000001"


def test_portfolio_marks_recycled_ticker_with_matching_cik():
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    prices = pd.DataFrame({
        "ticker": ["AAA"] * 6,
        "cik": [1, 1, 1, 2, 2, 2],
        "date": list(dates) + list(dates),
        "open": [10, 12, 11, 100, 50, 60],
        "close": [10, 12, 11, 100, 50, 60],
    })
    trades = pd.DataFrame({
        "ticker": ["AAA"], "cik": [1],
        "entry_date": [dates[0]], "entry_price": [10.0],
        "exit_date": [dates[2]], "net_return": [0.10],
    })
    simulator = PortfolioSimulator(
        prices,
        PortfolioConfig(initial_cash=10_000, max_positions=1, max_position_fraction=1.0),
    )
    metrics, curve, accepted = simulator.run(trades)
    middle = curve[curve["date"] == dates[1]].iloc[0]
    assert middle["equity"] == 12_000.0
    assert metrics.final_equity == 11_000.0
    assert accepted.iloc[0]["cik"] == 1
