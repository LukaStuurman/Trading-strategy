from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.metrics import summarize_trades
from src.strategies.orb import ORBConfig, generate_trades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--range-minutes", type=int, default=15)
    parser.add_argument("--target-r", type=float, default=1.5)
    args = parser.parse_args()

    bars = pd.read_csv(args.prices)
    cfg = ORBConfig(opening_range_minutes=args.range_minutes, target_r=args.target_r)
    trades = generate_trades(bars, cfg)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "trades.csv", index=False)
    metrics = pd.DataFrame([summarize_trades(trades).to_dict()])
    metrics.to_csv(output / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
