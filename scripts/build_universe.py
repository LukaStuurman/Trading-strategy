#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data.universe import historical_components_to_intervals

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--history", default=str(ROOT / "data/real/raw/sp500_historical_components.csv"))
    p.add_argument("--output", default=str(ROOT / "data/real/universe_intervals.csv"))
    args = p.parse_args()
    history = pd.read_csv(args.history)
    intervals = historical_components_to_intervals(history)
    if intervals.empty:
        raise RuntimeError("historical S&P data produced no membership intervals")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(args.output, index=False)
    print(f"Wrote {len(intervals):,} membership intervals for {intervals['ticker'].nunique():,} tickers")


if __name__ == "__main__":
    main()
