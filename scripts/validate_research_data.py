#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data.universe import load_universe_intervals
from src.data.validation import validate_all, write_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--fundamentals")
    p.add_argument("--universe")
    p.add_argument("--report", required=True)
    args = p.parse_args()

    prices = pd.read_csv(args.prices)
    fundamentals = pd.read_csv(args.fundamentals) if args.fundamentals else None
    universe = load_universe_intervals(args.universe) if args.universe else None
    report = validate_all(prices, fundamentals, universe)
    write_report(report, args.report)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    report.raise_if_invalid()
    print(f"Validation passed; report written to {Path(args.report)}")


if __name__ == "__main__":
    main()
