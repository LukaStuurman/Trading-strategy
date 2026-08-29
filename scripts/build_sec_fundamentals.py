#!/usr/bin/env python3
"""Build point-in-time quality fundamentals from raw SEC Company Facts.

The output matches src/strategies/quality_bad_news.py:
    ticker,available_date,roe,fcf_margin,debt_to_equity,current_ratio,market_cap

Important point-in-time rule
----------------------------
A value becomes usable on its SEC `filed` date, never on the fiscal period end.
For each filing date we use the latest facts filed on or before that date.
Market cap is approximated as shares outstanding * last close available on or
before the filing date. This avoids using today's market cap in old trades.

SEC taxonomy differs by issuer/year. Candidate US-GAAP tags are tried in order
and the selected tag is stored in the audit output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEC_DIR = ROOT / "data" / "real" / "raw" / "sec_companyfacts"
DEFAULT_PRICES = ROOT / "data" / "real" / "prices.csv"
DEFAULT_OUTPUT = ROOT / "data" / "real" / "fundamentals.csv"
DEFAULT_AUDIT = ROOT / "data" / "real" / "fundamentals_audit.csv"

TAGS = {
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "assets_current": ["AssetsCurrent"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "liabilities": ["Liabilities"],
    "debt": [
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtCurrent",
        "ShortTermBorrowings",
    ],
    "debt_long": [
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
    ],
    "shares": [
        "EntityCommonStockSharesOutstanding",
        "CommonStocksIncludingAdditionalPaidInCapitalMember",  # rarely used; filtered by unit
    ],
}

FLOW_KEYS = {"net_income", "revenue", "cfo", "capex"}


def _facts(payload: dict) -> dict:
    return payload.get("facts", {}).get("us-gaap", {}) | payload.get("facts", {}).get("dei", {})


def _unit_rows(fact: dict) -> list[dict]:
    units = fact.get("units", {})
    preferred = ["USD", "shares"]
    rows: list[dict] = []
    for unit in preferred:
        if unit in units:
            rows.extend(units[unit])
    if not rows:
        for values in units.values():
            rows.extend(values)
    return rows


def _select_tag(payload: dict, candidates: Iterable[str]) -> tuple[str | None, list[dict]]:
    facts = _facts(payload)
    for tag in candidates:
        fact = facts.get(tag)
        if fact:
            rows = [r for r in _unit_rows(fact) if r.get("filed") and r.get("val") is not None]
            if rows:
                return tag, rows
    return None, []


def _prepare(rows: list[dict], flow: bool) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["filed", "end", "start", "val", "form", "fp", "fy"])
    df = pd.DataFrame(rows).copy()
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["end"] = pd.to_datetime(df.get("end"), errors="coerce")
    if "start" in df:
        df["start"] = pd.to_datetime(df["start"], errors="coerce")
    else:
        df["start"] = pd.NaT
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df[df["form"].isin(["10-Q", "10-K", "20-F", "40-F"]) if "form" in df else True]
    df = df.dropna(subset=["filed", "val"])
    if flow:
        # Prefer quarterly facts (roughly 60-120 days) when available. Annual/YTD
        # observations are retained as fallback, then converted to a trailing sum.
        df["duration"] = (df["end"] - df["start"]).dt.days
    return df.sort_values(["filed", "end"]).drop_duplicates(["filed", "end"], keep="last")


def _latest_asof(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    x = df[df["filed"] <= date]
    if x.empty:
        return None
    return x.iloc[-1]


def _quarterly_flow_asof(df: pd.DataFrame, date: pd.Timestamp) -> float:
    x = df[df["filed"] <= date].copy()
    if x.empty:
        return np.nan
    q = x[x["duration"].between(60, 120, inclusive="both")]
    if len(q) >= 4:
        q = q.sort_values("end").drop_duplicates("end", keep="last").tail(4)
        if len(q) == 4:
            return float(q["val"].sum())
    # Fallback: latest annual-ish observation. Better to be missing/rough than
    # silently use a future filing.
    annual = x[x["duration"].between(250, 400, inclusive="both")]
    if not annual.empty:
        return float(annual.iloc[-1]["val"])
    return np.nan


def _value_asof(df: pd.DataFrame, date: pd.Timestamp) -> float:
    row = _latest_asof(df, date)
    return float(row["val"]) if row is not None else np.nan


def _price_asof(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float:
    x = prices[(prices["ticker"] == ticker) & (prices["date"] <= date)]
    return float(x.iloc[-1]["close"]) if not x.empty else np.nan


def build_one(ticker: str, payload: dict, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected: dict[str, str | None] = {}
    frames: dict[str, pd.DataFrame] = {}
    for key, candidates in TAGS.items():
        tag, rows = _select_tag(payload, candidates)
        selected[key] = tag
        frames[key] = _prepare(rows, flow=key in FLOW_KEYS)

    filed_dates = sorted({d for df in frames.values() for d in df["filed"].dropna().tolist()})
    out = []
    for available_date in filed_dates:
        revenue = _quarterly_flow_asof(frames["revenue"], available_date)
        net_income = _quarterly_flow_asof(frames["net_income"], available_date)
        cfo = _quarterly_flow_asof(frames["cfo"], available_date)
        capex = _quarterly_flow_asof(frames["capex"], available_date)
        equity = _value_asof(frames["equity"], available_date)
        ca = _value_asof(frames["assets_current"], available_date)
        cl = _value_asof(frames["liabilities_current"], available_date)
        liabilities = _value_asof(frames["liabilities"], available_date)
        debt_short = sum(
            _value_asof(frames[k], available_date) if not frames[k].empty else 0.0
            for k in ["debt"]
        )
        debt_long = _value_asof(frames["debt_long"], available_date)
        shares = _value_asof(frames["shares"], available_date)
        px = _price_asof(prices, ticker, available_date)

        # If debt tags are unavailable, total liabilities is a conservative
        # fallback for leverage screening. Audit records expose which case applied.
        debt_total = np.nansum([debt_short, debt_long])
        if not np.isfinite(debt_total) or debt_total == 0:
            debt_total = liabilities

        roe = net_income / equity if np.isfinite(net_income) and equity > 0 else np.nan
        fcf = cfo - capex if np.isfinite(cfo) and np.isfinite(capex) else np.nan
        fcf_margin = fcf / revenue if np.isfinite(fcf) and revenue > 0 else np.nan
        debt_to_equity = debt_total / equity if np.isfinite(debt_total) and equity > 0 else np.nan
        current_ratio = ca / cl if np.isfinite(ca) and cl > 0 else np.nan
        market_cap = shares * px if np.isfinite(shares) and shares > 0 and np.isfinite(px) else np.nan

        out.append({
            "ticker": ticker,
            "available_date": available_date.date().isoformat(),
            "roe": roe,
            "fcf_margin": fcf_margin,
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
            "market_cap": market_cap,
        })

    audit = pd.DataFrame([
        {"ticker": ticker, "metric": key, "selected_tag": tag or ""}
        for key, tag in selected.items()
    ])
    result = pd.DataFrame(out)
    if not result.empty:
        metric_cols = ["roe", "fcf_margin", "debt_to_equity", "current_ratio", "market_cap"]
        result = result.dropna(subset=metric_cols, how="all").drop_duplicates(["ticker", "available_date"])
    return result, audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sec-dir", default=str(DEFAULT_SEC_DIR))
    p.add_argument("--prices", default=str(DEFAULT_PRICES))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--audit", default=str(DEFAULT_AUDIT))
    args = p.parse_args()

    prices = pd.read_csv(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].str.upper()
    prices = prices.sort_values(["ticker", "date"])

    all_rows, audits = [], []
    for path in sorted(Path(args.sec_dir).glob("*.json")):
        ticker = path.stem.upper()
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows, audit = build_one(ticker, payload, prices)
        if not rows.empty:
            all_rows.append(rows)
        audits.append(audit)
        print(f"[fundamentals] {ticker}: {len(rows):,} point-in-time rows")

    output = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    audit.to_csv(args.audit, index=False)
    print(f"Wrote {len(output):,} rows to {args.output}")


if __name__ == "__main__":
    main()
