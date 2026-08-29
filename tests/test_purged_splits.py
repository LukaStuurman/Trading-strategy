import pandas as pd

from src.research.robustness import split_trades


def test_split_trades_purges_boundary_crossing_outcomes():
    trades = pd.DataFrame({
        "ticker": ["TRAIN", "CROSS_TRAIN", "VALID", "CROSS_VALID", "OOS"],
        "entry_date": pd.to_datetime([
            "2020-10-01", "2020-10-10", "2021-01-05", "2023-05-10", "2023-05-22",
        ]),
        "exit_date": pd.to_datetime([
            "2020-10-09", "2020-10-20", "2021-02-01", "2023-06-15", "2023-06-20",
        ]),
        "net_return": [0.01, 0.02, 0.03, 0.04, 0.05],
    })
    splits = split_trades(
        trades,
        train_end=pd.Timestamp("2020-10-12"),
        validation_end=pd.Timestamp("2023-05-19"),
    )
    assert splits["train"]["ticker"].tolist() == ["TRAIN"]
    assert splits["validation"]["ticker"].tolist() == ["VALID"]
    assert splits["oos"]["ticker"].tolist() == ["OOS"]
    assert "CROSS_TRAIN" not in pd.concat(splits.values())["ticker"].tolist()
    assert "CROSS_VALID" not in pd.concat(splits.values())["ticker"].tolist()
