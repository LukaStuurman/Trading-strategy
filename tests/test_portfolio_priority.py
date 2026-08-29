import pandas as pd

from src.backtest.portfolio import PortfolioConfig, PortfolioSimulator


def test_same_day_capacity_prefers_quality_then_larger_dip_not_ticker_order():
    prices = pd.DataFrame({
        "ticker": ["AAA", "ZZZ", "AAA", "ZZZ"],
        "date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]),
        "open": [10.0, 20.0, 10.5, 21.0],
        "close": [10.0, 20.0, 10.5, 21.0],
    })
    trades = pd.DataFrame({
        "ticker": ["AAA", "ZZZ"],
        "entry_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
        "entry_price": [10.0, 20.0],
        "exit_date": pd.to_datetime(["2025-01-03", "2025-01-03"]),
        "net_return": [0.05, 0.05],
        "quality_percentile": [0.70, 0.90],
        "signal_return": [-0.20, -0.10],
    })
    simulator = PortfolioSimulator(
        prices,
        PortfolioConfig(initial_cash=10_000, max_positions=1, max_position_fraction=0.5),
    )
    metrics, _, accepted = simulator.run(trades)
    assert metrics.accepted_trades == 1
    assert accepted.iloc[0]["ticker"] == "ZZZ"


def test_equal_quality_prefers_larger_dip():
    prices = pd.DataFrame({
        "ticker": ["AAA", "ZZZ", "AAA", "ZZZ"],
        "date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]),
        "open": [10.0, 20.0, 10.5, 21.0],
        "close": [10.0, 20.0, 10.5, 21.0],
    })
    trades = pd.DataFrame({
        "ticker": ["AAA", "ZZZ"],
        "entry_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
        "entry_price": [10.0, 20.0],
        "exit_date": pd.to_datetime(["2025-01-03", "2025-01-03"]),
        "net_return": [0.05, 0.05],
        "quality_percentile": [0.90, 0.90],
        "signal_return": [-0.20, -0.10],
    })
    simulator = PortfolioSimulator(
        prices,
        PortfolioConfig(initial_cash=10_000, max_positions=1, max_position_fraction=0.5),
    )
    _, _, accepted = simulator.run(trades)
    assert accepted.iloc[0]["ticker"] == "AAA"
