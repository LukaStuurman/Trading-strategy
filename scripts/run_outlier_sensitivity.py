#!/usr/bin/env python3
"""Stress-test selected quality-dip variants against extreme adjusted-price moves.

The primary backtest is left untouched. This diagnostic identifies instrument
close-to-close moves above a configurable absolute threshold and recomputes OOS
trade/portfolio metrics after excluding only trades whose signal-to-exit window
crosses one of those dates. OOS membership is determined by entry date, exactly
matching the primary purged split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.metrics import summarize_trades
from src.backtest.portfolio import PortfolioConfig, PortfolioSimulator
from src.data.io import read_table


def _normalize_cik(value) -> str:
    if pd.isna(value):
        return "NO-CIK"
    text = str(value).strip()
    if not text:
        return "NO-CIK"
    try:
        return str(int(float(text))).zfill(10)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())
        return str(int(digits)).zfill(10) if digits else "NO-CIK"


def _instrument_ids(frame: pd.DataFrame) -> pd.Series:
    ticker = frame["ticker"].astype(str).str.upper().str.strip()
    if "cik" not in frame.columns:
        return ticker + "|NO-CIK"
    cik = frame["cik"].map(_normalize_cik)
    return ticker + "|" + cik


def price_outliers(prices: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    p["_instrument_id"] = _instrument_ids(p)
    p = p.dropna(subset=["date", "close"]).sort_values(["_instrument_id", "date"]).reset_index(drop=True)
    p["prev_close"] = p.groupby("_instrument_id", sort=False)["close"].shift(1)
    p["daily_return"] = p["close"] / p["prev_close"] - 1.0
    outliers = p[p["daily_return"].abs() > threshold].copy()
    cols = [c for c in ["ticker", "cik", "_instrument_id", "date", "prev_close", "close", "daily_return"] if c in outliers.columns]
    outliers = outliers[cols].sort_values(["date", "_instrument_id"]).reset_index(drop=True)
    lookup = {
        instrument: np.sort(group["date"].to_numpy(dtype="datetime64[ns]"))
        for instrument, group in outliers.groupby("_instrument_id", sort=False)
    }
    return outliers, lookup


def mark_crossing_trades(trades: pd.DataFrame, lookup: dict[str, np.ndarray]) -> pd.DataFrame:
    t = trades.copy()
    if t.empty:
        t["crosses_extreme_move"] = pd.Series(dtype=bool)
        return t
    t["signal_date"] = pd.to_datetime(t["signal_date"], errors="coerce").dt.normalize()
    t["exit_date"] = pd.to_datetime(t["exit_date"], errors="coerce").dt.normalize()
    t["_instrument_id"] = _instrument_ids(t)

    flags: list[bool] = []
    for instrument, signal_date, exit_date in zip(t["_instrument_id"], t["signal_date"], t["exit_date"]):
        dates = lookup.get(instrument)
        if dates is None or len(dates) == 0 or pd.isna(signal_date) or pd.isna(exit_date):
            flags.append(False)
            continue
        start = np.datetime64(signal_date.to_datetime64())
        end = np.datetime64(exit_date.to_datetime64())
        pos = int(np.searchsorted(dates, start, side="left"))
        flags.append(pos < len(dates) and dates[pos] <= end)
    t["crosses_extreme_move"] = flags
    return t


def _portfolio_fields(prefix: str, metrics) -> dict:
    return {
        f"{prefix}_total_return": metrics.total_return,
        f"{prefix}_sharpe": metrics.sharpe,
        f"{prefix}_max_drawdown": metrics.max_drawdown,
        f"{prefix}_accepted_trades": metrics.accepted_trades,
        f"{prefix}_skipped_trades": metrics.skipped_trades,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
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
    parser.add_argument("--oos-start", required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-position-fraction", type=float, default=0.10)
    args = parser.parse_args()

    prices = read_table(args.prices)
    trades = pd.read_csv(args.trades)
    leaderboard = pd.read_csv(args.leaderboard)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    outliers, lookup = price_outliers(prices, args.threshold)
    outliers.to_csv(output / "price_outliers.csv", index=False)
    marked = mark_crossing_trades(trades, lookup)
    oos_start = pd.Timestamp(args.oos_start).normalize()
    marked["entry_date"] = pd.to_datetime(marked["entry_date"], errors="coerce").dt.normalize()
    simulator = PortfolioSimulator(
        prices,
        PortfolioConfig(
            initial_cash=args.initial_capital,
            max_positions=args.max_positions,
            max_position_fraction=args.max_position_fraction,
        ),
    )

    rank = {str(v): i + 1 for i, v in enumerate(leaderboard["variant_id"].astype(str).tolist())}
    rows = []
    for variant_id, group in marked.groupby("variant_id", sort=False):
        oos = group[group["entry_date"] >= oos_start].copy()
        clean = oos[~oos["crosses_extreme_move"]].copy()
        base_stats = summarize_trades(oos)
        clean_stats = summarize_trades(clean)
        base_port, _, _ = simulator.run(oos)
        clean_port, _, _ = simulator.run(clean)
        rows.append({
            "leaderboard_rank": rank.get(str(variant_id)),
            "variant_id": variant_id,
            "extreme_move_threshold": args.threshold,
            "oos_trades": int(len(oos)),
            "oos_crossing_extreme_move_trades": int(oos["crosses_extreme_move"].sum()),
            "oos_crossing_share": float(oos["crosses_extreme_move"].mean()) if len(oos) else 0.0,
            "clean_oos_trades": int(len(clean)),
            "oos_avg_return": base_stats.avg_return,
            "clean_oos_avg_return": clean_stats.avg_return,
            "avg_return_delta": clean_stats.avg_return - base_stats.avg_return,
            "oos_win_rate": base_stats.win_rate,
            "clean_oos_win_rate": clean_stats.win_rate,
            "oos_profit_factor": base_stats.profit_factor,
            "clean_oos_profit_factor": clean_stats.profit_factor,
            **_portfolio_fields("oos_portfolio", base_port),
            **_portfolio_fields("clean_oos_portfolio", clean_port),
        })

    sensitivity = pd.DataFrame(rows).sort_values(["leaderboard_rank", "variant_id"], na_position="last").reset_index(drop=True)
    sensitivity.to_csv(output / "outlier_sensitivity.csv", index=False)

    report = [
        "# Extreme-price-move sensitivity", "",
        f"Threshold: absolute close-to-close adjusted return > **{args.threshold:.0%}**.",
        f"Detected price outlier rows: **{len(outliers)}**.",
        f"OOS entries start: **{oos_start.date()}**.", "",
        "The primary backtest is not altered. OOS membership follows entry date, matching the primary split. The stress case removes a trade only when an extreme-move date falls between its signal date and exit date, inclusive.",
        "", "## Validation-selected variants", "",
    ]
    show_cols = [
        "leaderboard_rank", "variant_id", "oos_trades", "oos_crossing_extreme_move_trades",
        "oos_avg_return", "clean_oos_avg_return", "oos_portfolio_total_return",
        "clean_oos_portfolio_total_return", "oos_portfolio_max_drawdown", "clean_oos_portfolio_max_drawdown",
    ]
    if sensitivity.empty:
        report.append("No selected-variant OOS trades were available for sensitivity analysis.")
    else:
        report.append(_markdown_table(sensitivity.head(20), show_cols))
    report.extend([
        "", "## Reading this stress test", "",
        "- Similar baseline and clean metrics indicate the selected signal is not being driven by extreme source moves.",
        "- A large deterioration after exclusion is a warning to inspect the named rows in `price_outliers.csv` before trusting the strategy.",
        "- This is a data-quality stress test, not a claim that every >300% move is erroneous.",
    ])
    (output / "outlier_sensitivity.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(sensitivity.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
