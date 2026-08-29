from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class EarningsMomentumConfig:
    min_eps_surprise: float = 0.10
    min_gap: float = 0.03
    max_gap: float = 0.12
    min_relative_volume: float = 1.5
    hold_days: int = 10
    round_trip_cost_bps: float = 10.0


def generate_trades(prices: pd.DataFrame, events: pd.DataFrame, cfg: EarningsMomentumConfig) -> pd.DataFrame:
    """Daily-bar proxy for after-hours positive earnings momentum.

    Required prices: ticker,date,open,close,volume
    Required events: ticker,event_date,eps_surprise,revenue_beat,after_hours

    The next regular-session open is used as the executable entry proxy. Intraday VWAP
    confirmation requires minute data and is therefore deliberately not faked here.
    """
    p = prices.copy()
    e = events.copy()
    p["date"] = pd.to_datetime(p["date"])
    e["event_date"] = pd.to_datetime(e["event_date"])
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    p["avg_volume_20"] = p.groupby("ticker")["volume"].transform(lambda s: s.shift(1).rolling(20).mean())

    rows: list[dict] = []
    eligible = e[
        (e["eps_surprise"] >= cfg.min_eps_surprise)
        & (e["revenue_beat"].astype(bool))
        & (e["after_hours"].astype(bool))
    ]

    for ev in eligible.itertuples(index=False):
        tp = p[(p["ticker"] == ev.ticker) & (p["date"] > ev.event_date)].reset_index(drop=True)
        prev = p[(p["ticker"] == ev.ticker) & (p["date"] <= ev.event_date)]
        if tp.empty or prev.empty or len(tp) <= cfg.hold_days:
            continue
        prior_close = float(prev.iloc[-1]["close"])
        entry = tp.iloc[0]
        if pd.isna(entry["avg_volume_20"]) or entry["avg_volume_20"] <= 0:
            continue
        gap = float(entry["open"] / prior_close - 1.0)
        rel_vol = float(entry["volume"] / entry["avg_volume_20"])
        if not (cfg.min_gap <= gap <= cfg.max_gap and rel_vol >= cfg.min_relative_volume):
            continue
        exit_ = tp.iloc[cfg.hold_days]
        gross = float(exit_["close"] / entry["open"] - 1.0)
        rows.append({
            "ticker": ev.ticker,
            "event_date": ev.event_date,
            "eps_surprise": float(ev.eps_surprise),
            "gap": gap,
            "relative_volume": rel_vol,
            "entry_date": entry["date"],
            "exit_date": exit_["date"],
            "gross_return": gross,
            "net_return": gross - cfg.round_trip_cost_bps / 10_000.0,
        })
    return pd.DataFrame(rows)
