#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.io import read_table
from src.data.universe import load_universe_intervals
from src.data.validation import validate_all, write_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--fundamentals")
    p.add_argument("--universe")
    p.add_argument("--report", required=True)
    p.add_argument("--min-total-price-rows", type=int, default=0)
    p.add_argument("--min-price-tickers", type=int, default=0)
    args = p.parse_args()

    prices = read_table(args.prices)
    fundamentals = read_table(args.fundamentals) if args.fundamentals else None
    universe = load_universe_intervals(args.universe) if args.universe else None
    report = validate_all(prices, fundamentals, universe)

    price_stats = report.stats.get("prices", {})
    if args.min_total_price_rows and int(price_stats.get("rows", 0)) < args.min_total_price_rows:
        report.errors.append(
            f"prices: total rows {int(price_stats.get('rows', 0)):,} below completeness gate {args.min_total_price_rows:,}"
        )
    if args.min_price_tickers and int(price_stats.get("tickers", 0)) < args.min_price_tickers:
        report.errors.append(
            f"prices: ticker count {int(price_stats.get('tickers', 0))} below completeness gate {args.min_price_tickers}"
        )

    write_report(report, args.report)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    report.raise_if_invalid()
    print(f"Validation passed; report written to {Path(args.report)}")


if __name__ == "__main__":
    main()
