from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class QualityDipConfig:
    drop_threshold: float = -0.10
    wait_days: int = 1
    hold_days: int = 20
    min_roe: float = 0.12
    min_fcf_margin: float = 0.05
    max_debt_to_equity: float = 1.5
    min_current_ratio: float = 1.0
    min_market_cap: float = 5_000_000_000
    round_trip_cost_bps: float = 10.0


def _quality_mask(f: pd.DataFrame, cfg: QualityDipConfig) -> pd.Series:
    return (
        (f["roe"] >= cfg.min_roe)
        & (f["fcf_margin"] >= cfg.min_fcf_margin)
        & (f["debt_to_equity"] <= cfg.max_debt_to_equity)
        & (f["current_ratio"] >= cfg.min_current_ratio)
        & (f["market_cap"] >= cfg.min_market_cap)
    )


def generate_trades(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: QualityDipConfig,
) -> pd.DataFrame:
    """Generate long-only trades after a sharp one-day decline in quality companies.

    Point-in-time rule: a fundamental row may only be used if `available_date <= signal_date`.
    For each signal we use the most recently available fundamental observation.

    Required prices columns: ticker,date,open,close
    Required fundamentals columns: ticker,available_date,roe,fcf_margin,debt_to_equity,
        current_ratio,market_cap
    """
    p = prices.copy()
    f = fundamentals.copy()
    p["date"] = pd.to_datetime(p["date"])
    f["available_date"] = pd.to_datetime(f["available_date"])
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    f = f.sort_values(["ticker", "available_date"]).reset_index(drop=True)

    p["daily_return"] = p.groupby("ticker")["close"].pct_change()
    signals = p[p["daily_return"] <= cfg.drop_threshold].copy()
    rows: list[dict] = []

    for sig in signals.itertuples(index=False):
        hist_f = f[(f["ticker"] == sig.ticker) & (f["available_date"] <= sig.date)]
        if hist_f.empty:
            continue
        latest = hist_f.iloc[-1:]
        if not bool(_quality_mask(latest, cfg).iloc[0]):
            continue

        ticker_prices = p[p["ticker"] == sig.ticker].reset_index(drop=True)
        positions = ticker_prices.index[ticker_prices["date"] == sig.date]
        if len(positions) != 1:
            continue
        signal_i = int(positions[0])
        entry_i = signal_i + cfg.wait_days + 1
        exit_i = entry_i + cfg.hold_days
        if exit_i >= len(ticker_prices):
            continue

        entry = ticker_prices.iloc[entry_i]
        exit_ = ticker_prices.iloc[exit_i]
        gross = float(exit_["close"] / entry["open"] - 1.0)
        cost = cfg.round_trip_cost_bps / 10_000.0
        rows.append(
            {
                "ticker": sig.ticker,
                "signal_date": sig.date,
                "signal_return": float(sig.daily_return),
                "entry_date": entry["date"],
                "entry_price": float(entry["open"]),
                "exit_date": exit_["date"],
                "exit_price": float(exit_["close"]),
                "gross_return": gross,
                "net_return": gross - cost,
                "drop_threshold": cfg.drop_threshold,
                "wait_days": cfg.wait_days,
                "hold_days": cfg.hold_days,
            }
        )

    return pd.DataFrame(rows)
