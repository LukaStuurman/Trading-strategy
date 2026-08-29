#!/usr/bin/env python3
"""Build the quality-research panel from broad FINSABER prices.

FINSABER supplies the broad historical/delisted price universe. Quality-dip
signals can only be evaluated where causal fundamentals exist, so this step
creates a smaller execution panel and writes an explicit coverage report rather
than silently pretending that missing fundamentals do not exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.io import read_table, write_table
from src.data.universe import load_universe_intervals, normalize_ticker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--fundamentals", required=True)
    p.add_argument("--universe", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--coverage", required=True)
    p.add_argument("--warmup-days", type=int, default=10)
    p.add_argument("--min-post-fundamental-rows", type=int, default=100)
    args = p.parse_args()

    prices = read_table(args.prices)
    fundamentals = read_table(args.fundamentals)
    universe = load_universe_intervals(args.universe)

    prices["ticker"] = prices["ticker"].map(normalize_ticker)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    fundamentals["ticker"] = fundamentals["ticker"].map(normalize_ticker)
    fundamentals["available_date"] = pd.to_datetime(fundamentals["available_date"], errors="coerce")

    price_tickers = set(prices["ticker"].dropna().unique())
    universe_tickers = set(universe["ticker"].dropna().unique())
    fundamental_tickers = set(fundamentals["ticker"].dropna().unique())
    historical_price_tickers = price_tickers & universe_tickers
    eligible_before_history_gate = historical_price_tickers & fundamental_tickers

    first_available = (
        fundamentals[fundamentals["ticker"].isin(eligible_before_history_gate)]
        .dropna(subset=["available_date"])
        .groupby("ticker")["available_date"]
        .min()
    )

    candidate_prices = prices[prices["ticker"].isin(eligible_before_history_gate)].copy()
    candidate_prices["first_fundamental_date"] = candidate_prices["ticker"].map(first_available)
    post_available = candidate_prices[
        candidate_prices["first_fundamental_date"].notna()
        & (candidate_prices["date"] >= candidate_prices["first_fundamental_date"])
    ]
    post_counts = post_available.groupby("ticker").size().reindex(sorted(eligible_before_history_gate), fill_value=0)
    usable_tickers = set(post_counts[post_counts >= args.min_post_fundamental_rows].index)
    insufficient = {
        str(ticker): int(count)
        for ticker, count in post_counts[post_counts < args.min_post_fundamental_rows].items()
    }

    panel = candidate_prices[candidate_prices["ticker"].isin(usable_tickers)].copy()
    cutoff = panel["first_fundamental_date"] - pd.to_timedelta(args.warmup_days, unit="D")
    panel = panel[panel["date"] >= cutoff].drop(columns=["first_fundamental_date"])
    panel = panel.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

    if panel.empty:
        raise RuntimeError("Research panel is empty after causal fundamental and history-coverage gates")

    coverage = {
        "source_prices": str(args.prices),
        "research_prices": str(args.output),
        "price_rows_full": int(len(prices)),
        "price_rows_research_panel": int(len(panel)),
        "price_tickers": len(price_tickers),
        "historical_universe_tickers": len(universe_tickers),
        "historical_price_tickers": len(historical_price_tickers),
        "fundamental_tickers": len(fundamental_tickers),
        "eligible_quality_tickers_before_history_gate": len(eligible_before_history_gate),
        "eligible_quality_tickers": len(usable_tickers),
        "fundamental_ticker_coverage_of_historical_prices": (
            len(eligible_before_history_gate) / len(historical_price_tickers) if historical_price_tickers else 0.0
        ),
        "research_ticker_coverage_of_historical_prices": (
            len(usable_tickers) / len(historical_price_tickers) if historical_price_tickers else 0.0
        ),
        "minimum_post_fundamental_price_rows": args.min_post_fundamental_rows,
        "insufficient_post_fundamental_history": insufficient,
        "missing_fundamental_tickers": sorted(historical_price_tickers - fundamental_tickers),
        "price_tickers_outside_membership_source": sorted(price_tickers - universe_tickers),
        "panel_start": panel["date"].min().date().isoformat(),
        "panel_end": panel["date"].max().date().isoformat(),
        "warmup_days_before_first_fundamental": args.warmup_days,
        "fundamental_source_counts": (
            fundamentals.get("fundamental_source", pd.Series(dtype=str)).fillna("unknown").value_counts().to_dict()
        ),
        "interpretation": (
            "FINSABER removes the price-history survivorship bottleneck, but the quality strategy remains limited "
            "to tickers with causal fundamentals and at least the configured amount of post-fundamental price history. "
            "Missing/insufficient tickers are reported explicitly and are not treated as failed quality screens."
        ),
    }

    write_table(panel, args.output)
    target = Path(args.coverage)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(coverage, indent=2, default=str), encoding="utf-8")
    print(
        f"Research panel: {len(panel):,} rows, {len(usable_tickers)} usable tickers "
        f"({len(eligible_before_history_gate)} had fundamentals before history gate); "
        f"research coverage={coverage['research_ticker_coverage_of_historical_prices']:.1%}"
    )
    if insufficient:
        sample = ", ".join(f"{k}={v}" for k, v in list(insufficient.items())[:12])
        print(f"Excluded for <{args.min_post_fundamental_rows} post-fundamental rows: {sample}")


if __name__ == "__main__":
    main()
