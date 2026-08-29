from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_boundaries(prices: pd.DataFrame, train_fraction: float = 0.60, validation_fraction: float = 0.20) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(prices["date"]).dropna().unique()).sort_values().reset_index(drop=True)
    if len(dates) < 5:
        raise ValueError("not enough unique dates for train/validation/test split")
    train_i = min(len(dates) - 3, max(0, int(len(dates) * train_fraction) - 1))
    validation_i = min(len(dates) - 2, max(train_i + 1, int(len(dates) * (train_fraction + validation_fraction)) - 1))
    return pd.Timestamp(dates.iloc[train_i]), pd.Timestamp(dates.iloc[validation_i])


def split_trades(trades: pd.DataFrame, train_end: pd.Timestamp, validation_end: pd.Timestamp, date_col: str = "entry_date") -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {"train": trades.copy(), "validation": trades.copy(), "oos": trades.copy()}
    t = trades.copy()
    t[date_col] = pd.to_datetime(t[date_col])
    return {
        "train": t[t[date_col] <= train_end],
        "validation": t[(t[date_col] > train_end) & (t[date_col] <= validation_end)],
        "oos": t[t[date_col] > validation_end],
    }


def bootstrap_mean_ci(values, confidence: float = 0.95, samples: int = 400, seed: int = 7) -> tuple[float, float]:
    x = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(samples, len(x)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def add_neighbor_robustness(summary: pd.DataFrame) -> pd.DataFrame:
    """Reward stable parameter plateaus instead of an isolated best cell."""
    if summary.empty:
        return summary.copy()
    out = summary.copy()
    param_cols = ["drop_threshold", "wait_days", "hold_days", "min_quality_percentile", "require_stabilization"]
    grids = {c: sorted(out[c].dropna().unique().tolist()) for c in param_cols}
    counts, positive, medians = [], [], []
    for _, row in out.iterrows():
        neighbors = []
        for col in param_cols:
            values = grids[col]
            pos = values.index(row[col])
            adjacent = ([values[pos - 1]] if pos > 0 else []) + ([values[pos + 1]] if pos + 1 < len(values) else [])
            for value in adjacent:
                mask = pd.Series(True, index=out.index)
                for other in param_cols:
                    mask &= out[other].eq(value if other == col else row[other])
                neighbors.extend(out.loc[mask, "oos_avg_return"].dropna().tolist())
        counts.append(len(neighbors))
        positive.append(float(np.mean(np.asarray(neighbors) > 0)) if neighbors else np.nan)
        medians.append(float(np.median(neighbors)) if neighbors else np.nan)
    out["neighbor_count"] = counts
    out["neighbor_positive_fraction"] = positive
    out["neighbor_oos_return_median"] = medians

    def pct_rank(col: str) -> pd.Series:
        return pd.to_numeric(out[col], errors="coerce").rank(pct=True).fillna(0.0)

    trade_score = np.minimum(pd.to_numeric(out["oos_trades"], errors="coerce").fillna(0) / 30.0, 1.0)
    out["robustness_score"] = (
        0.30 * pct_rank("oos_avg_return")
        + 0.20 * pct_rank("oos_sharpe")
        + 0.20 * out["neighbor_positive_fraction"].fillna(0.0)
        + 0.15 * (pd.to_numeric(out["validation_avg_return"], errors="coerce").fillna(0) > 0).astype(float)
        + 0.10 * trade_score
        + 0.05 * (pd.to_numeric(out["oos_ci_low"], errors="coerce").fillna(-1) > 0).astype(float)
    )
    return out
