from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.metrics import summarize_trades
from src.strategies.quality_bad_news import QualityDipConfig, generate_trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--fundamentals", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prices = pd.read_csv(args.prices)
    fundamentals = pd.read_csv(args.fundamentals)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    all_trades = []
    for drop, wait, hold in product([-0.05, -0.10, -0.15, -0.20], [0, 1, 2], [5, 10, 20, 60]):
        cfg = QualityDipConfig(drop_threshold=drop, wait_days=wait, hold_days=hold)
        trades = generate_trades(prices, fundamentals, cfg)
        metrics = summarize_trades(trades).to_dict()
        metrics.update({"drop_threshold": drop, "wait_days": wait, "hold_days": hold})
        rows.append(metrics)
        if not trades.empty:
            all_trades.append(trades)

    summary = pd.DataFrame(rows).sort_values(["sharpe", "avg_return"], ascending=False)
    summary.to_csv(output / "parameter_sweep.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(output / "all_trades.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
