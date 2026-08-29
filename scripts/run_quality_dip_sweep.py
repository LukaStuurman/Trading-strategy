#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.metrics import summarize_trades
from src.backtest.portfolio import PortfolioConfig, PortfolioSimulator
from src.data.io import read_table
from src.data.universe import load_universe_intervals
from src.research.experiment import build_manifest, stable_config_hash, write_manifest
from src.research.robustness import add_neighbor_robustness, bootstrap_mean_ci, chronological_boundaries, split_trades
from src.strategies.quality_bad_news import QualityDipConfig, generate_trades, prepare_features


def _prefix(metrics, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in metrics.to_dict().items()}


def _cfg_from_row(row: pd.Series) -> QualityDipConfig:
    return QualityDipConfig(
        drop_threshold=float(row["drop_threshold"]), wait_days=int(row["wait_days"]),
        hold_days=int(row["hold_days"]), min_quality_percentile=float(row["min_quality_percentile"]),
        require_stabilization=bool(row["require_stabilization"]),
    )


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    x = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in x.iterrows():
        vals = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def _boundaries(prices: pd.DataFrame, train_arg: str | None, validation_arg: str | None):
    if bool(train_arg) != bool(validation_arg):
        raise ValueError("--train-end and --validation-end must be supplied together")
    if not train_arg:
        train_end, validation_end = chronological_boundaries(prices)
        return train_end, validation_end, "chronological_quantiles"
    train_end = pd.Timestamp(train_arg).normalize()
    validation_end = pd.Timestamp(validation_arg).normalize()
    if train_end >= validation_end:
        raise ValueError("train_end must be before validation_end")
    return train_end, validation_end, "fixed_cli_dates"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--fundamentals", required=True)
    parser.add_argument("--universe")
    parser.add_argument("--source-manifest")
    parser.add_argument("--coverage")
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-position-fraction", type=float, default=0.10)
    parser.add_argument("--train-end")
    parser.add_argument("--validation-end")
    args = parser.parse_args()

    prices = read_table(args.prices)
    fundamentals = read_table(args.fundamentals)
    universe = load_universe_intervals(args.universe) if args.universe else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    drops = [-0.05, -0.10, -0.15, -0.20]
    waits = [0, 1, 2]
    holds = [5, 10, 20, 60]
    quality_percentiles = [0.0, 0.50, 0.70, 0.80]
    stabilization = [False, True]

    print(f"Preparing causal features for {len(prices):,} price rows / {prices['ticker'].nunique()} tickers")
    prepared = prepare_features(prices, fundamentals, universe)
    train_end, validation_end, split_source = _boundaries(prices, args.train_end, args.validation_end)
    portfolio_cfg = PortfolioConfig(initial_cash=args.initial_capital, max_positions=args.max_positions, max_position_fraction=args.max_position_fraction)
    simulator = PortfolioSimulator(prices, portfolio_cfg)

    rows = []
    for i, (drop, wait, hold, q_pct, stabilize) in enumerate(product(drops, waits, holds, quality_percentiles, stabilization)):
        cfg = QualityDipConfig(drop_threshold=drop, wait_days=wait, hold_days=hold, min_quality_percentile=q_pct, require_stabilization=stabilize)
        trades = generate_trades(prices, fundamentals, cfg, universe=universe, prepared=prepared)
        splits = split_trades(trades, train_end, validation_end)
        row = summarize_trades(trades).to_dict()
        row.update(_prefix(summarize_trades(splits["train"]), "train"))
        row.update(_prefix(summarize_trades(splits["validation"]), "validation"))
        row.update(_prefix(summarize_trades(splits["oos"]), "oos"))
        ci_low, ci_high = bootstrap_mean_ci(splits["oos"]["net_return"] if not splits["oos"].empty else [], seed=1000 + i)
        row["oos_ci_low"], row["oos_ci_high"] = ci_low, ci_high
        p_metrics, _, _ = simulator.run(trades)
        oos_p_metrics, _, _ = simulator.run(splits["oos"])
        row.update({f"portfolio_{k}": v for k, v in p_metrics.to_dict().items()})
        row.update({f"oos_portfolio_{k}": v for k, v in oos_p_metrics.to_dict().items()})
        row.update({
            "variant_id": stable_config_hash(asdict(cfg)), "drop_threshold": drop, "wait_days": wait,
            "hold_days": hold, "min_quality_percentile": q_pct, "require_stabilization": stabilize,
        })
        rows.append(row)
        if (i + 1) % 48 == 0:
            print(f"Completed {i + 1}/384 variants")

    summary = add_neighbor_robustness(pd.DataFrame(rows)).sort_values(
        ["robustness_score", "oos_avg_return", "oos_trades"], ascending=[False, False, False]
    ).reset_index(drop=True)
    summary.to_csv(output / "parameter_sweep.csv", index=False)
    summary.head(25).to_csv(output / "leaderboard.csv", index=False)
    summary.pivot_table(index="drop_threshold", columns="hold_days", values="oos_avg_return", aggfunc="median").to_csv(output / "heatmap_oos_return.csv")
    summary.pivot_table(index="drop_threshold", columns="hold_days", values="oos_portfolio_sharpe", aggfunc="median").to_csv(output / "heatmap_oos_portfolio_sharpe.csv")

    top_trades = []
    for _, row in summary.head(20).iterrows():
        trades = generate_trades(prices, fundamentals, _cfg_from_row(row), universe=universe, prepared=prepared)
        if not trades.empty:
            trades.insert(0, "variant_id", row["variant_id"])
            top_trades.append(trades)
    if top_trades:
        pd.concat(top_trades, ignore_index=True).to_csv(output / "top_variant_trades.csv", index=False)

    grid = {
        "drop_thresholds": drops, "wait_days": waits, "hold_days": holds,
        "quality_percentiles": quality_percentiles, "stabilization": stabilization,
        "variants": len(rows), "portfolio": asdict(portfolio_cfg),
    }
    data_files = {"prices": args.prices, "fundamentals": args.fundamentals}
    if args.universe:
        data_files["universe"] = args.universe
    if args.source_manifest:
        data_files["finsaber_source_manifest"] = args.source_manifest
    if args.coverage:
        data_files["research_coverage"] = args.coverage
    manifest = build_manifest(
        name="quality_dip_research", parameters=grid, data_files=data_files,
        metadata={
            "train_end": train_end.date().isoformat(), "validation_end": validation_end.date().isoformat(),
            "test_start": (validation_end + pd.Timedelta(days=1)).date().isoformat(), "split_source": split_source,
            "selection_rule": "robustness score emphasizes OOS return, OOS Sharpe, neighboring parameters, validation sign, trade count and bootstrap CI",
            "bulk_price_source": "FINSABER-2 when supplied by the automated pipeline",
        },
    )
    write_manifest(manifest, output / "experiment_manifest.json")

    cols = [
        "variant_id", "drop_threshold", "wait_days", "hold_days", "min_quality_percentile",
        "require_stabilization", "oos_trades", "oos_avg_return", "oos_sharpe", "oos_ci_low", "oos_ci_high",
        "oos_portfolio_total_return", "oos_portfolio_sharpe", "oos_portfolio_max_drawdown",
        "neighbor_positive_fraction", "robustness_score",
    ]
    report = [
        "# Quality-dip robustness report", "", f"Experiment: `{manifest['experiment_id']}`",
        f"Train through: **{train_end.date()}**; validation through: **{validation_end.date()}**; later trades are OOS.",
        f"Split source: **{split_source}**.",
        f"Grid: **{len(rows)} variants** = 4 drops × 3 waits × 4 holds × 4 quality percentiles × stabilization on/off.",
        "", "The ranking is deliberately not the best in-sample Sharpe. It rewards OOS performance and parameter neighborhoods.",
        "", "## Top robust variants", "", _markdown_table(summary.head(10), cols), "", "## Interpretation guardrails", "",
        "- `oos_ci_low > 0` is stronger evidence than a positive point estimate alone.",
        "- Neighbor stability matters: an isolated winning cell is treated as fragile.",
        "- Portfolio metrics enforce capital limits and overlapping-position constraints.",
        "- FINSABER supplies broad historical/delisted prices, but quality results include only tickers with causal fundamental coverage; see `research_coverage.json`.",
        "- Historical membership is still enforced on the signal date; a price record alone never implies S&P 500 membership.",
    ]
    (output / "robustness_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.head(25)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
