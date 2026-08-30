from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.universe import attach_membership, normalize_ticker


def _normalize_cik(value) -> str:
    if value is None or pd.isna(value):
        return "NO-CIK"
    text = str(value).strip()
    if not text:
        return "NO-CIK"
    try:
        text = str(int(float(text)))
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return "NO-CIK"
        text = str(int(digits))
    return text.zfill(10)


def prepare_intraday_features(
    prices: pd.DataFrame,
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build open-time features without using same-session high/low/close as signals.

    Today's open is allowed because every simulated trade enters at that open.
    All momentum, volatility and liquidity features are lagged through the prior
    close/session. Historical index membership is attached on the trade date.
    """
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p["ticker"] = p["ticker"].map(normalize_ticker)
    if "cik" not in p.columns:
        p["cik"] = None
    p["_instrument_id"] = [
        f"{ticker}|{_normalize_cik(cik)}" for ticker, cik in zip(p["ticker"], p["cik"])
    ]
    for col in ("open", "high", "low", "close", "volume"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p = p.dropna(subset=["date", "ticker", "open", "high", "low", "close"]).copy()
    p = p[(p["open"] > 0) & (p["high"] > 0) & (p["low"] > 0) & (p["close"] > 0)].copy()
    p = p.sort_values(["_instrument_id", "date"]).reset_index(drop=True)

    g = p.groupby("_instrument_id", sort=False)
    p["prev_close"] = g["close"].shift(1)
    p["prev_open"] = g["open"].shift(1)
    p["prev_high"] = g["high"].shift(1)
    p["prev_low"] = g["low"].shift(1)
    p["prev2_close"] = g["close"].shift(2)
    p["prev4_close"] = g["close"].shift(4)
    p["prev6_close"] = g["close"].shift(6)

    p["gap_return"] = p["open"] / p["prev_close"] - 1.0
    p["intraday_return"] = p["close"] / p["open"] - 1.0
    p["prev_return"] = p["prev_close"] / p["prev2_close"] - 1.0
    p["prev_intraday_return"] = p["prev_close"] / p["prev_open"] - 1.0
    p["mom3"] = p["prev_close"] / p["prev4_close"] - 1.0
    p["mom5"] = p["prev_close"] / p["prev6_close"] - 1.0
    p["prev_range"] = (p["prev_high"] - p["prev_low"]) / p["prev_close"]

    p["_dollar_volume"] = p["close"] * p["volume"].fillna(0.0)
    p["_cc_return"] = g["close"].pct_change()
    p["adv20"] = g["_dollar_volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).mean()
    )
    p["vol20"] = g["_cc_return"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).std(ddof=1)
    )

    p = attach_membership(p, universe)
    eligible_rank = p["in_universe"].fillna(False) & p["prev_return"].notna()
    p["prev_return_percentile"] = np.nan
    if eligible_rank.any():
        p.loc[eligible_rank, "prev_return_percentile"] = (
            p.loc[eligible_rank]
            .groupby("date")["prev_return"]
            .rank(pct=True, method="average")
            .to_numpy()
        )

    # Cross-sectional prior-session market condition; every input is lagged.
    market = (
        p.loc[eligible_rank, ["date", "prev_return"]]
        .groupby("date", as_index=True)["prev_return"]
        .median()
    )
    p["market_prev_return"] = p["date"].map(market)

    return p.drop(columns=["_dollar_volume", "_cc_return"], errors="ignore")


def intraday_exit_returns(
    frame: pd.DataFrame,
    *,
    stop_loss: float | None,
    take_profit: float | None,
    round_trip_cost_bps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return net open-to-exit returns and exit labels.

    If both stop and target are touched in the same daily bar, the stop is
    assumed to occur first. This is intentionally conservative because daily
    OHLC data cannot reveal the intraday ordering of high and low.
    """
    open_px = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    high_px = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low_px = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    close_px = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)

    gross = close_px / open_px - 1.0
    labels = np.full(len(frame), "close", dtype=object)

    stop_hit = np.zeros(len(frame), dtype=bool)
    if stop_loss is not None:
        stop_hit = low_px <= open_px * (1.0 - float(stop_loss))
        gross[stop_hit] = -float(stop_loss)
        labels[stop_hit] = "stop"

    if take_profit is not None:
        target_hit = high_px >= open_px * (1.0 + float(take_profit))
        target_only = target_hit & ~stop_hit
        gross[target_only] = float(take_profit)
        labels[target_only] = "target"

    net = gross - float(round_trip_cost_bps) / 10_000.0
    return net, labels


def daily_portfolio_metrics(
    trades: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    leverage: float,
    max_positions: int,
    initial_capital: float = 10_000.0,
) -> dict:
    """Equal-slot same-day portfolio with cash for unused slots.

    Each accepted name receives 1/max_positions of equity times the requested
    gross leverage. There is no overnight exposure. A <= -100% portfolio day is
    treated as a blow-up and equity remains at zero thereafter.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    dates = calendar[(calendar >= start) & (calendar <= end)]
    if len(dates) == 0:
        return {
            "final_equity": initial_capital,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "traded_days": 0,
            "average_positions": 0.0,
            "max_positions_used": 0,
            "blown_up": False,
            "worst_day": 0.0,
        }

    t = trades[(trades["date"] >= start) & (trades["date"] <= end)].copy()
    if not t.empty:
        t = t.sort_values(
            ["date", "signal_score", "ticker", "_instrument_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        accepted = t.groupby("date", sort=False).head(int(max_positions))
        grouped = accepted.groupby("date")["net_return"].agg(["sum", "count"])
        daily = (float(leverage) * grouped["sum"] / float(max_positions)).reindex(dates, fill_value=0.0)
        positions = grouped["count"].reindex(dates, fill_value=0)
        accepted_trades = int(len(accepted))
    else:
        daily = pd.Series(0.0, index=dates)
        positions = pd.Series(0, index=dates)
        accepted_trades = 0

    raw = daily.to_numpy(dtype=float)
    blown_up = bool(np.any(raw <= -1.0))
    safe = np.maximum(raw, -1.0)
    equity_curve = float(initial_capital) * np.cumprod(1.0 + safe)
    if blown_up:
        first = int(np.flatnonzero(raw <= -1.0)[0])
        equity_curve[first:] = 0.0

    final_equity = float(equity_curve[-1]) if len(equity_curve) else float(initial_capital)
    total_return = final_equity / float(initial_capital) - 1.0
    years = max(len(dates) / 252.0, 1.0 / 252.0)
    cagr = (final_equity / float(initial_capital)) ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0

    std = float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0
    sharpe = float(np.mean(raw) / std * np.sqrt(252.0)) if std > 0 else 0.0
    peak = np.maximum.accumulate(equity_curve) if len(equity_curve) else np.array([initial_capital])
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, equity_curve / peak - 1.0, -1.0)
    max_drawdown = float(np.nanmin(dd)) if len(dd) else 0.0

    return {
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trades": accepted_trades,
        "traded_days": int((positions > 0).sum()),
        "average_positions": float(positions.mean()),
        "max_positions_used": int(positions.max()) if len(positions) else 0,
        "blown_up": blown_up,
        "worst_day": float(np.min(raw)) if len(raw) else 0.0,
    }
