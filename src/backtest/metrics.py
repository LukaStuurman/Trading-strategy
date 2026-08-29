from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    trades: int
    win_rate: float
    avg_return: float
    median_return: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    worst_losing_streak: int

    def to_dict(self) -> dict:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peaks = equity.cummax()
    dd = equity / peaks - 1.0
    return float(dd.min())


def _worst_losing_streak(returns: Iterable[float]) -> int:
    worst = current = 0
    for r in returns:
        if r < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def summarize_trades(trades: pd.DataFrame, return_col: str = "net_return") -> BacktestMetrics:
    if trades.empty:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    r = pd.to_numeric(trades[return_col], errors="coerce").dropna()
    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf

    equity = (1.0 + r).cumprod()
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / std * np.sqrt(len(r))) if std > 0 else 0.0

    return BacktestMetrics(
        trades=int(len(r)),
        win_rate=float((r > 0).mean()),
        avg_return=float(r.mean()),
        median_return=float(r.median()),
        expectancy=float(r.mean()),
        profit_factor=float(profit_factor),
        max_drawdown=_max_drawdown(equity),
        sharpe=sharpe,
        worst_losing_streak=_worst_losing_streak(r),
    )
