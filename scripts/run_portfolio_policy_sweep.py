#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.portfolio import PortfolioConfig, PortfolioSimulator
from src.data.io import read_table
from src.research.robustness import split_trades


MAX_POSITIONS_GRID = [5, 7, 10, 15, 20]
POSITION_FRACTION_GRID = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
POLICY_SCORE_WEIGHTS = {
    "train_portfolio_sharpe": 0.25,
    "validation_portfolio_sharpe": 0.35,
    "validation_portfolio_cagr": 0.20,
    "validation_portfolio_max_drawdown": 0.10,
    "train_validation_sharpe_stability": 0.10,
}


def add_policy_selection_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank portfolio policies using train/validation columns only.

    OOS columns are deliberately ignored so changing OOS results cannot alter
    the selected sizing/capacity policy.
    """
    out = frame.copy()
    stability = -(
        pd.to_numeric(out["train_portfolio_sharpe"], errors="coerce")
        - pd.to_numeric(out["validation_portfolio_sharpe"], errors="coerce")
    ).abs()
    out["train_validation_sharpe_stability"] = stability

    rank_inputs = {
        "train_portfolio_sharpe": pd.to_numeric(out["train_portfolio_sharpe"], errors="coerce"),
        "validation_portfolio_sharpe": pd.to_numeric(out["validation_portfolio_sharpe"], errors="coerce"),
        "validation_portfolio_cagr": pd.to_numeric(out["validation_portfolio_cagr"], errors="coerce"),
        # Drawdowns are negative, so a less-negative value should rank higher.
        "validation_portfolio_max_drawdown": pd.to_numeric(
            out["validation_portfolio_max_drawdown"], errors="coerce"
        ),
        "train_validation_sharpe_stability": stability,
    }

    score = pd.Series(0.0, index=out.index, dtype=float)
    for column, weight in POLICY_SCORE_WEIGHTS.items():
        ranks = rank_inputs[column].rank(pct=True, method="average").fillna(0.0)
        score = score + weight * ranks
    out["policy_selection_score"] = score
    return out


def _prefix(values: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _run_policy(
    simulator: PortfolioSimulator,
    trades: pd.DataFrame,
    *,
    initial_capital: float,
) -> tuple[dict, pd.DataFrame]:
    metrics, curve, accepted = simulator.run(trades)
    out = metrics.to_dict()

    if curve.empty:
        out.update({
            "average_invested_fraction": 0.0,
            "median_invested_fraction": 0.0,
            "days_ge_90pct_invested": 0.0,
            "days_le_50pct_invested": 1.0,
        })
    else:
        equity = pd.to_numeric(curve["equity"], errors="coerce")
        cash = pd.to_numeric(curve["cash"], errors="coerce")
        valid = equity > 0
        invested = pd.Series(np.nan, index=curve.index, dtype=float)
        invested.loc[valid] = 1.0 - (cash.loc[valid] / equity.loc[valid])
        invested = invested.clip(lower=0.0)
        out.update({
            "average_invested_fraction": float(invested.mean()) if invested.notna().any() else 0.0,
            "median_invested_fraction": float(invested.median()) if invested.notna().any() else 0.0,
            "days_ge_90pct_invested": float((invested >= 0.90).mean()) if invested.notna().any() else 0.0,
            "days_le_50pct_invested": float((invested <= 0.50).mean()) if invested.notna().any() else 1.0,
        })

    total_candidates = metrics.accepted_trades + metrics.skipped_trades
    out["skip_rate"] = (
        float(metrics.skipped_trades / total_candidates) if total_candidates else 0.0
    )
    if accepted.empty:
        out["allocated_turnover_multiple"] = 0.0
        out["average_allocation"] = 0.0
    else:
        allocated = pd.to_numeric(accepted["allocated"], errors="coerce").fillna(0.0)
        out["allocated_turnover_multiple"] = float(allocated.sum() / initial_capital)
        out["average_allocation"] = float(allocated.mean())
    return out, accepted


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)
    args = parser.parse_args()

    prices = read_table(args.prices)
    all_trades = read_table(args.trades)
    leaderboard = read_table(args.leaderboard)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if leaderboard.empty:
        raise ValueError("leaderboard is empty")
    if "variant_id" not in all_trades.columns:
        raise ValueError("top-variant trade file must contain variant_id")

    selected_variant = str(leaderboard.iloc[0]["variant_id"])
    trades = all_trades[all_trades["variant_id"].astype(str) == selected_variant].copy()
    if trades.empty:
        raise ValueError(f"no trades found for validation-selected variant {selected_variant}")

    train_end = pd.Timestamp(args.train_end).normalize()
    validation_end = pd.Timestamp(args.validation_end).normalize()
    splits = split_trades(trades, train_end, validation_end)

    # Build market lookups only once; policy changes only the simulator config.
    simulator = PortfolioSimulator(
        prices,
        PortfolioConfig(
            initial_cash=args.initial_capital,
            max_positions=10,
            max_position_fraction=0.10,
        ),
    )

    rows: list[dict] = []
    accepted_by_policy: dict[tuple[int, float, str], pd.DataFrame] = {}
    for max_positions in MAX_POSITIONS_GRID:
        for fraction in POSITION_FRACTION_GRID:
            simulator.config = PortfolioConfig(
                initial_cash=args.initial_capital,
                max_positions=max_positions,
                max_position_fraction=fraction,
                allow_overlapping_ticker=False,
            )
            row = {
                "max_positions": max_positions,
                "max_position_fraction": fraction,
                "notional_capacity": max_positions * fraction,
            }
            for split_name in ("train", "validation", "oos"):
                split_metrics, accepted = _run_policy(
                    simulator,
                    splits[split_name],
                    initial_capital=args.initial_capital,
                )
                row.update(_prefix(split_metrics, split_name))
                accepted_by_policy[(max_positions, fraction, split_name)] = accepted
            rows.append(row)

    summary = add_policy_selection_score(pd.DataFrame(rows)).sort_values(
        [
            "policy_selection_score",
            "validation_portfolio_sharpe",
            "validation_portfolio_cagr",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary.insert(0, "policy_rank", np.arange(1, len(summary) + 1))
    summary.to_csv(output / "portfolio_policy_sweep.csv", index=False)

    summary.pivot_table(
        index="max_positions",
        columns="max_position_fraction",
        values="validation_portfolio_sharpe",
        aggfunc="first",
    ).to_csv(output / "heatmap_policy_validation_sharpe.csv")
    summary.pivot_table(
        index="max_positions",
        columns="max_position_fraction",
        values="oos_portfolio_sharpe",
        aggfunc="first",
    ).to_csv(output / "heatmap_policy_oos_sharpe.csv")

    chosen = summary.iloc[0]
    baseline_rows = summary[
        (summary["max_positions"] == 10)
        & np.isclose(summary["max_position_fraction"], 0.10)
    ]
    if baseline_rows.empty:
        raise AssertionError("baseline 10 positions x 10% policy missing from grid")
    baseline = baseline_rows.iloc[0]

    accepted_frames = []
    for split_name in ("validation", "oos"):
        accepted = accepted_by_policy[
            (int(chosen["max_positions"]), float(chosen["max_position_fraction"]), split_name)
        ].copy()
        if not accepted.empty:
            accepted.insert(0, "split", split_name)
            accepted_frames.append(accepted)
    if accepted_frames:
        pd.concat(accepted_frames, ignore_index=True).to_csv(
            output / "top_policy_accepted_trades.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output / "top_policy_accepted_trades.csv", index=False)

    manifest = {
        "selected_signal_variant": selected_variant,
        "train_end": train_end.date().isoformat(),
        "validation_end": validation_end.date().isoformat(),
        "test_start": (validation_end + pd.Timedelta(days=1)).date().isoformat(),
        "policy_grid": {
            "max_positions": MAX_POSITIONS_GRID,
            "max_position_fraction": POSITION_FRACTION_GRID,
            "policies": len(summary),
        },
        "selection_rule": (
            "policy_selection_score uses train portfolio Sharpe, validation portfolio Sharpe, "
            "validation CAGR, validation max drawdown and train/validation Sharpe stability only; "
            "OOS columns are diagnostic and never participate in ranking"
        ),
        "selection_weights": POLICY_SCORE_WEIGHTS,
        "selected_policy": {
            "max_positions": int(chosen["max_positions"]),
            "max_position_fraction": float(chosen["max_position_fraction"]),
            "policy_selection_score": float(chosen["policy_selection_score"]),
        },
        "baseline_policy": {
            "max_positions": 10,
            "max_position_fraction": 0.10,
        },
        "oos_status": (
            "diagnostic_not_pristine: this period was already disclosed by the earlier signal "
            "research, so policy OOS results must not be treated as a new untouched confirmation"
        ),
    }
    (output / "portfolio_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    cols = [
        "policy_rank",
        "max_positions",
        "max_position_fraction",
        "policy_selection_score",
        "validation_portfolio_total_return",
        "validation_portfolio_sharpe",
        "validation_portfolio_max_drawdown",
        "validation_average_invested_fraction",
        "validation_skip_rate",
        "oos_portfolio_total_return",
        "oos_portfolio_sharpe",
        "oos_portfolio_max_drawdown",
        "oos_average_invested_fraction",
        "oos_skip_rate",
    ]
    comparison = pd.DataFrame([baseline, chosen]).drop_duplicates(
        ["max_positions", "max_position_fraction"]
    )
    report = [
        "# Portfolio sizing/capacity research",
        "",
        f"Signal variant fixed before this stage: `{selected_variant}`.",
        f"Train through: **{train_end.date()}**; validation through: **{validation_end.date()}**.",
        "",
        "The sizing/capacity policy is selected strictly from train + validation metrics. "
        "OOS columns are never used by `policy_selection_score`.",
        "",
        "**Important:** the 2023-05-20+ period was already exposed in the previous signal study. "
        "It is therefore a diagnostic check here, not a fresh untouched confirmation.",
        "",
        "## Baseline versus selected policy",
        "",
        _markdown_table(comparison, cols),
        "",
        "## Top train/validation-selected policies",
        "",
        _markdown_table(summary.head(10), cols),
        "",
        "## What this stage tests",
        "",
        "- Whether the original 10 positions × 10% sizing rule leaves too much capital idle.",
        "- Whether more/fewer concurrent positions improve risk-adjusted validation performance.",
        "- Whether smaller/larger fixed position fractions improve capital efficiency without using OOS for selection.",
        "- `average_invested_fraction` is daily marked-to-market exposure as a fraction of equity.",
        "- `skip_rate` is the share of otherwise valid trades rejected by cash, capacity, or overlap constraints.",
    ]
    (output / "portfolio_policy_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(summary.head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
