import pandas as pd

from scripts.run_outlier_sensitivity import mark_crossing_trades, price_outliers


def test_outlier_sensitivity_marks_only_matching_instrument_window():
    prices = pd.DataFrame({
        "ticker": ["AAA"] * 6,
        "cik": [1, 1, 1, 2, 2, 2],
        "date": pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03",
            "2025-01-01", "2025-01-02", "2025-01-03",
        ]),
        "close": [10.0, 50.0, 51.0, 10.0, 10.5, 11.0],
    })
    outliers, lookup = price_outliers(prices, threshold=3.0)
    assert len(outliers) == 1
    assert outliers.iloc[0]["_instrument_id"] == "AAA|0000000001"

    trades = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "cik": [1, 2, 1],
        "signal_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03"]),
        "exit_date": pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-03"]),
    })
    marked = mark_crossing_trades(trades, lookup)
    assert marked["crosses_extreme_move"].tolist() == [True, False, False]
