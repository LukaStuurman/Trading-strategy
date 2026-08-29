from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import pandas as pd


@dataclass(frozen=True)
class ORBConfig:
    opening_range_minutes: int = 15
    target_r: float = 1.5
    entry_cutoff: time = time(12, 0)
    session_close: time = time(16, 0)
    point_value: float = 5.0
    commission_per_side: float = 0.39
    extra_fees_per_side: float = 0.60
    slippage_points_per_side: float = 0.25
    max_trades_per_day: int = 1


def generate_trades(minute_bars: pd.DataFrame, cfg: ORBConfig) -> pd.DataFrame:
    """Simple long/short ORB for ES/MES-like minute data.

    Required columns: datetime,open,high,low,close. Datetimes must already be in
    America/New_York local exchange time.
    """
    df = minute_bars.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["clock"] = df["datetime"].dt.time

    rows: list[dict] = []
    for date_, day in df.groupby("date", sort=True):
        day = day.reset_index(drop=True)
        regular = day[(day["clock"] >= time(9, 30)) & (day["clock"] <= cfg.session_close)]
        if regular.empty:
            continue
        range_end_ts = pd.Timestamp.combine(pd.Timestamp(date_).date(), time(9, 30)) + pd.Timedelta(minutes=cfg.opening_range_minutes)
        orb = regular[(regular["datetime"] >= pd.Timestamp.combine(pd.Timestamp(date_).date(), time(9, 30))) & (regular["datetime"] < range_end_ts)]
        if len(orb) < max(1, cfg.opening_range_minutes - 1):
            continue
        high = float(orb["high"].max())
        low = float(orb["low"].min())
        risk_points = high - low
        if risk_points <= 0:
            continue

        after = regular[(regular["datetime"] >= range_end_ts) & (regular["clock"] <= cfg.entry_cutoff)]
        direction = None
        entry_i = None
        for idx, bar in after.iterrows():
            if float(bar["close"]) > high:
                direction, entry_i = "long", idx
                break
            if float(bar["close"]) < low:
                direction, entry_i = "short", idx
                break
        if entry_i is None:
            continue

        # Use next minute open to avoid entering at a close we only know after the bar completes.
        if entry_i + 1 >= len(day):
            continue
        entry = day.loc[entry_i + 1]
        entry_price = float(entry["open"])
        if direction == "long":
            stop = low
            risk = entry_price - stop
            if risk <= 0:
                continue
            target = entry_price + cfg.target_r * risk
        else:
            stop = high
            risk = stop - entry_price
            if risk <= 0:
                continue
            target = entry_price - cfg.target_r * risk

        exit_price = None
        exit_time = None
        reason = "session_close"
        remainder = day.loc[entry_i + 1:]
        for _, bar in remainder.iterrows():
            if bar["clock"] > cfg.session_close:
                break
            if direction == "long":
                stop_hit = float(bar["low"]) <= stop
                target_hit = float(bar["high"]) >= target
            else:
                stop_hit = float(bar["high"]) >= stop
                target_hit = float(bar["low"]) <= target
            # Conservative ordering when both occur inside one minute.
            if stop_hit:
                exit_price, exit_time, reason = stop, bar["datetime"], "stop"
                break
            if target_hit:
                exit_price, exit_time, reason = target, bar["datetime"], "target"
                break
        if exit_price is None:
            close_rows = regular[regular["clock"] <= cfg.session_close]
            if close_rows.empty:
                continue
            last = close_rows.iloc[-1]
            exit_price, exit_time = float(last["close"]), last["datetime"]

        gross_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        gross_dollars = gross_points * cfg.point_value
        slippage = 2 * cfg.slippage_points_per_side * cfg.point_value
        fees = 2 * (cfg.commission_per_side + cfg.extra_fees_per_side)
        net_dollars = gross_dollars - slippage - fees
        risk_dollars = risk * cfg.point_value
        rows.append({
            "date": date_, "direction": direction, "entry_time": entry["datetime"],
            "entry_price": entry_price, "exit_time": exit_time, "exit_price": exit_price,
            "reason": reason, "opening_range_points": risk_points, "risk_points": risk,
            "gross_pnl": gross_dollars, "net_pnl": net_dollars,
            "net_return": net_dollars / risk_dollars if risk_dollars else 0.0,
            "target_r": cfg.target_r, "opening_range_minutes": cfg.opening_range_minutes,
        })
    return pd.DataFrame(rows)
