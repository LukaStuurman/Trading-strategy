from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import pandas as pd


class DataValidationError(RuntimeError):
    pass


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: "ValidationReport", prefix: str) -> None:
        self.errors.extend(f"{prefix}: {x}" for x in other.errors)
        self.warnings.extend(f"{prefix}: {x}" for x in other.warnings)
        self.stats[prefix] = other.stats

    def to_dict(self) -> dict:
        out = asdict(self)
        out["ok"] = self.ok
        return out

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise DataValidationError("; ".join(self.errors))


def validate_prices(prices: pd.DataFrame, min_rows_per_ticker: int = 100) -> ValidationReport:
    r = ValidationReport()
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        r.errors.append(f"missing required columns: {sorted(missing)}")
        return r
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    r.stats.update({
        "rows": int(len(p)),
        "tickers": int(p["ticker"].nunique()),
        "start": None if p["date"].dropna().empty else p["date"].min().date().isoformat(),
        "end": None if p["date"].dropna().empty else p["date"].max().date().isoformat(),
        "min_rows_per_ticker": int(min_rows_per_ticker),
    })
    if p["date"].isna().any():
        r.errors.append(f"{int(p['date'].isna().sum())} invalid dates")
    dupes = int(p.duplicated(["ticker", "date"]).sum())
    if dupes:
        r.errors.append(f"{dupes} duplicate ticker/date rows")
    short = p.groupby("ticker").size()
    short = short[short < min_rows_per_ticker]
    if not short.empty:
        sample = ", ".join(f"{k}={int(v)}" for k, v in short.head(10).items())
        r.errors.append(f"price histories below {min_rows_per_ticker} rows: {sample}")
    if p[["open", "high", "low", "close"]].isna().any(axis=1).any():
        r.errors.append("missing/non-numeric OHLC values found")
    if (p[["open", "high", "low", "close"]] <= 0).any(axis=1).any():
        r.errors.append("non-positive OHLC values found")

    # Relative tolerance avoids rejecting otherwise-consistent adjusted bars due
    # to sub-cent floating point multiplication noise at very large prices.
    magnitude = p[["open", "high", "low", "close"]].abs().max(axis=1).clip(lower=1.0)
    tol = magnitude * 1e-10
    bad_high = p["high"] + tol < p[["open", "close", "low"]].max(axis=1)
    bad_low = p["low"] - tol > p[["open", "close", "high"]].min(axis=1)
    if bad_high.any() or bad_low.any():
        r.errors.append(f"OHLC bounds invalid on {int(bad_high.sum() + bad_low.sum())} rows")
    if (p["volume"].fillna(0) < 0).any():
        r.errors.append("negative volume found")
    ret = p.sort_values(["ticker", "date"]).groupby("ticker")["close"].pct_change()
    extreme = int((ret.abs() > 3.0).sum())
    r.stats["close_moves_over_300pct"] = extreme
    if extreme:
        r.warnings.append(f"{extreme} close-to-close moves exceed 300%; inspect split/source handling")
    return r


def validate_fundamentals(fundamentals: pd.DataFrame) -> ValidationReport:
    r = ValidationReport()
    required = {"ticker", "available_date", "roe", "fcf_margin", "debt_to_equity", "current_ratio", "market_cap"}
    missing = required - set(fundamentals.columns)
    if missing:
        r.errors.append(f"missing required columns: {sorted(missing)}")
        return r
    f = fundamentals.copy()
    f["available_date"] = pd.to_datetime(f["available_date"], errors="coerce")
    r.stats.update({"rows": int(len(f)), "tickers": int(f["ticker"].nunique())})
    if f["available_date"].isna().any():
        r.errors.append("invalid available_date values found")
    dupes = int(f.duplicated(["ticker", "available_date"]).sum())
    if dupes:
        r.errors.append(f"{dupes} duplicate ticker/available_date rows")
    for c in ["roe", "fcf_margin", "debt_to_equity", "current_ratio", "market_cap"]:
        coverage = float(pd.to_numeric(f[c], errors="coerce").notna().mean()) if len(f) else 0.0
        r.stats[f"{c}_coverage"] = coverage
        if coverage < 0.25:
            r.warnings.append(f"{c} coverage is only {coverage:.1%}")
    return r


def validate_universe(intervals: pd.DataFrame) -> ValidationReport:
    r = ValidationReport()
    required = {"ticker", "start_date", "end_date"}
    missing = required - set(intervals.columns)
    if missing:
        r.errors.append(f"missing required columns: {sorted(missing)}")
        return r
    u = intervals.copy()
    u["start_date"] = pd.to_datetime(u["start_date"], errors="coerce")
    u["end_date"] = pd.to_datetime(u["end_date"], errors="coerce")
    r.stats.update({"rows": int(len(u)), "tickers": int(u["ticker"].nunique())})
    if u["start_date"].isna().any():
        r.errors.append("invalid universe start dates found")
    invalid = u["end_date"].notna() & (u["end_date"] <= u["start_date"])
    if invalid.any():
        r.errors.append(f"{int(invalid.sum())} intervals have end_date <= start_date")
    overlaps = 0
    for _, g in u.sort_values(["ticker", "start_date"]).groupby("ticker"):
        prev_end = None
        for row in g.itertuples(index=False):
            if prev_end is not None and pd.notna(prev_end) and row.start_date < prev_end:
                overlaps += 1
            prev_end = row.end_date
    if overlaps:
        r.errors.append(f"{overlaps} overlapping membership intervals")
    return r


def validate_all(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    *,
    min_rows_per_ticker: int = 100,
) -> ValidationReport:
    out = ValidationReport()
    out.merge(validate_prices(prices, min_rows_per_ticker=min_rows_per_ticker), "prices")
    if fundamentals is not None:
        out.merge(validate_fundamentals(fundamentals), "fundamentals")
    if universe is not None:
        out.merge(validate_universe(universe), "universe")
    return out


def write_report(report: ValidationReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
