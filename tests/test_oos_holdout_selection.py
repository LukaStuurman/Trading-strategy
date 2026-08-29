import pandas as pd

from src.research.robustness import add_neighbor_robustness


def _summary(oos_a: float, oos_b: float) -> pd.DataFrame:
    return pd.DataFrame({
        "drop_threshold": [-0.05, -0.10],
        "wait_days": [1, 1],
        "hold_days": [20, 20],
        "min_quality_percentile": [0.8, 0.8],
        "require_stabilization": [False, False],
        "train_avg_return": [0.01, 0.01],
        "validation_avg_return": [0.03, 0.01],
        "validation_sharpe": [1.0, 0.2],
        "validation_trades": [40, 40],
        "validation_ci_low": [0.005, -0.002],
        "oos_avg_return": [oos_a, oos_b],
        "oos_sharpe": [-99.0, 99.0],
        "oos_trades": [1, 1000],
        "oos_ci_low": [-10.0, 10.0],
    })


def test_selection_score_is_invariant_to_oos_results():
    first = add_neighbor_robustness(_summary(-0.90, 0.90))["selection_score"].tolist()
    second = add_neighbor_robustness(_summary(0.90, -0.90))["selection_score"].tolist()
    assert first == second
    assert first[0] > first[1]
