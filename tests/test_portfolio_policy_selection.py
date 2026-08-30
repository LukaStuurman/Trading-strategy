import pandas as pd

from scripts.run_portfolio_policy_sweep import add_policy_selection_score


def test_policy_selection_score_cannot_use_oos_columns():
    frame = pd.DataFrame({
        "policy": ["A", "B", "C"],
        "train_portfolio_sharpe": [0.8, 0.5, 0.2],
        "validation_portfolio_sharpe": [0.7, 0.6, 0.1],
        "validation_portfolio_cagr": [0.12, 0.10, 0.03],
        "validation_portfolio_max_drawdown": [-0.18, -0.12, -0.05],
        "oos_portfolio_sharpe": [-5.0, 50.0, 500.0],
        "oos_portfolio_total_return": [-0.9, 2.0, 20.0],
    })
    ranked_a = add_policy_selection_score(frame).sort_values(
        "policy_selection_score", ascending=False
    )

    changed = frame.copy()
    changed["oos_portfolio_sharpe"] = [999.0, -999.0, 0.0]
    changed["oos_portfolio_total_return"] = [99.0, -0.99, 0.0]
    ranked_b = add_policy_selection_score(changed).sort_values(
        "policy_selection_score", ascending=False
    )

    assert ranked_a["policy"].tolist() == ranked_b["policy"].tolist()
    assert ranked_a["policy_selection_score"].tolist() == ranked_b["policy_selection_score"].tolist()
