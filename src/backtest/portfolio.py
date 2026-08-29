from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 10_000.0
    max_positions: int = 10
    max_position_fraction: float = 0.10
    allow_overlapping_ticker: bool = False


@dataclass
class PortfolioMetrics:
    final_equity: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    average_positions: float
    max_positions_used: int
    accepted_trades: int
    skipped_trades: int

    def to_dict(self) -> dict:
        return asdict(self)


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


def _instrument_id(ticker: str, cik) -> str:
    return f"{ticker}|{_normalize_cik(cik)}"


class PortfolioSimulator:
    """Daily mark-to-market simulator for pre-generated entry/exit trades.

    Entries use recorded open prices. Exit proceeds use each trade's `net_return`,
    preserving the strategy's round-trip cost assumption. Active positions are
    marked at the daily close between entry and exit. When CIK is available it
    forms part of the market-data key so recycled ticker symbols never mark to a
    different issuer.
    """

    def __init__(self, prices: pd.DataFrame, config: PortfolioConfig = PortfolioConfig()):
        self.config = config
        p = prices.copy()
        p["date"] = pd.to_datetime(p["date"]).dt.normalize()
        p["ticker"] = p["ticker"].astype(str).str.upper()
        if "cik" not in p.columns:
            p["cik"] = None
        p["_instrument_id"] = [
            _instrument_id(ticker, cik) for ticker, cik in zip(p["ticker"], p["cik"])
        ]
        p = p.sort_values(["date", "_instrument_id"])
        duplicates = p.duplicated(["date", "_instrument_id"])
        if duplicates.any():
            raise ValueError(f"duplicate date/instrument rows in portfolio prices: {int(duplicates.sum())}")
        self.dates = pd.DatetimeIndex(sorted(p["date"].unique()))
        self.open_lookup = p.set_index(["date", "_instrument_id"])["open"].to_dict()
        self.close_lookup = p.set_index(["date", "_instrument_id"])["close"].to_dict()

    def run(self, trades: pd.DataFrame) -> tuple[PortfolioMetrics, pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        if trades.empty:
            m = PortfolioMetrics(cfg.initial_cash, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)
            return m, pd.DataFrame(), pd.DataFrame()

        t = trades.copy().reset_index(drop=True)
        t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
        t["exit_date"] = pd.to_datetime(t["exit_date"]).dt.normalize()
        t["ticker"] = t["ticker"].astype(str).str.upper()
        if "cik" not in t.columns:
            t["cik"] = None
        t["_instrument_id"] = [
            _instrument_id(ticker, cik) for ticker, cik in zip(t["ticker"], t["cik"])
        ]
        t["_trade_id"] = np.arange(len(t))
        entries = {d: g.to_dict("records") for d, g in t.groupby("entry_date")}

        cash = float(cfg.initial_cash)
        active: dict[int, dict] = {}
        accepted, curve_rows = [], []
        skipped = 0
        min_date, max_date = t["entry_date"].min(), t["exit_date"].max()
        dates = self.dates[(self.dates >= min_date) & (self.dates <= max_date)]

        def mark_value(date: pd.Timestamp, use_open: bool = False) -> float:
            lookup = self.open_lookup if use_open else self.close_lookup
            total = 0.0
            for pos in active.values():
                px = lookup.get((date, pos["instrument_id"]))
                if px is None or not np.isfinite(px):
                    px = pos.get("last_price", pos["entry_price"])
                else:
                    pos["last_price"] = float(px)
                total += pos["shares"] * float(px)
            return total

        for date in dates:
            for trade in entries.get(date, []):
                ticker = trade["ticker"]
                if len(active) >= cfg.max_positions:
                    skipped += 1
                    continue
                if not cfg.allow_overlapping_ticker and any(p["ticker"] == ticker for p in active.values()):
                    skipped += 1
                    continue
                equity_open = cash + mark_value(date, use_open=True)
                allocation = min(cash, equity_open * cfg.max_position_fraction)
                entry_price = float(trade["entry_price"])
                if allocation <= 0 or entry_price <= 0:
                    skipped += 1
                    continue
                trade_id = int(trade["_trade_id"])
                active[trade_id] = {
                    "ticker": ticker,
                    "cik": trade.get("cik"),
                    "instrument_id": trade["_instrument_id"],
                    "shares": allocation / entry_price,
                    "allocated": allocation,
                    "entry_price": entry_price,
                    "last_price": entry_price,
                    "exit_date": trade["exit_date"],
                    "net_return": float(trade["net_return"]),
                }
                cash -= allocation
                accepted.append({
                    "trade_id": trade_id,
                    "ticker": ticker,
                    "cik": trade.get("cik"),
                    "entry_date": date,
                    "exit_date": trade["exit_date"],
                    "allocated": allocation,
                })

            for trade_id, pos in list(active.items()):
                if pos["exit_date"] == date:
                    cash += pos["allocated"] * (1.0 + pos["net_return"])
                    del active[trade_id]

            equity = cash + mark_value(date)
            curve_rows.append({"date": date, "equity": equity, "cash": cash, "positions": len(active)})

        curve = pd.DataFrame(curve_rows)
        if curve.empty:
            final_equity, max_dd, avg_pos, max_pos = cash, 0.0, 0.0, 0
            daily = pd.Series(dtype=float)
        else:
            curve["daily_return"] = curve["equity"].pct_change().fillna(0.0)
            curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
            final_equity = float(curve.iloc[-1]["equity"])
            max_dd = float(curve["drawdown"].min())
            avg_pos = float(curve["positions"].mean())
            max_pos = int(curve["positions"].max())
            daily = curve["daily_return"]
        total_return = final_equity / cfg.initial_cash - 1.0
        years = max(len(curve) / 252.0, 1 / 252.0)
        cagr = (final_equity / cfg.initial_cash) ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0
        std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
        vol = std * np.sqrt(252)
        sharpe = float(daily.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        metrics = PortfolioMetrics(
            final_equity=final_equity,
            total_return=float(total_return),
            cagr=float(cagr),
            annualized_volatility=float(vol),
            sharpe=sharpe,
            max_drawdown=max_dd,
            average_positions=avg_pos,
            max_positions_used=max_pos,
            accepted_trades=len(accepted),
            skipped_trades=skipped,
        )
        return metrics, curve, pd.DataFrame(accepted)
