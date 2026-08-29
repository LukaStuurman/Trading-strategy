#!/usr/bin/env python3
"""Build point-in-time quality fundamentals from raw SEC Company Facts.

Every value becomes usable on its SEC `filed` date, never the fiscal period end.
The core strategy fields are supplemented with additional profitability, cash-flow
quality and leverage metrics when the issuer taxonomy provides them.
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
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "liabilities": ["Liabilities"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "debt": ["LongTermDebtAndFinanceLeaseObligationsCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "debt_long": ["LongTermDebtAndFinanceLeaseObligationsNoncurrent", "LongTermDebtNoncurrent", "LongTermDebt"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForAdditionsToPropertyPlantAndEquipment"],
    "shares": ["EntityCommonStockSharesOutstanding"],
}
FLOW_KEYS = {"net_income", "revenue", "operating_income", "cfo", "capex"}


def _facts(payload: dict) -> dict:
    return payload.get("facts", {}).get("us-gaap", {}) | payload.get("facts", {}).get("dei", {})


def _unit_rows(fact: dict) -> list[dict]:
    units = fact.get("units", {})
    rows: list[dict] = []
    for unit in ["USD", "shares"]:
        rows.extend(units.get(unit, []))
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
    columns = ["filed", "end", "start", "val", "form", "fp", "fy"]
    if not rows:
        return pd.DataFrame(columns=columns + (["duration"] if flow else []))
    df = pd.DataFrame(rows).copy()
    df["filed"] = pd.to_datetime(df.get("filed"), errors="coerce")
    df["end"] = pd.to_datetime(df.get("end"), errors="coerce")
    df["start"] = pd.to_datetime(df.get("start"), errors="coerce") if "start" in df else pd.NaT
    df["val"] = pd.to_numeric(df.get("val"), errors="coerce")
    if "form" in df:
        df = df[df["form"].isin(["10-Q", "10-K", "20-F", "40-F"])]
    df = df.dropna(subset=["filed", "val"])
    if flow:
        df["duration"] = (df["end"] - df["start"]).dt.days
    return df.sort_values(["filed", "end"]).drop_duplicates(["filed", "end"], keep="last")


def _latest_asof(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    x = df[df["filed"] <= date]
    return None if x.empty else x.iloc[-1]


def _quarterly_flow_asof(df: pd.DataFrame, date: pd.Timestamp) -> float:
    x = df[df["filed"] <= date].copy()
    if x.empty:
        return np.nan
    q = x[x["duration"].between(60, 120, inclusive="both")]
    if len(q) >= 4:
        q = q.sort_values("end").drop_duplicates("end", keep="last").tail(4)
        if len(q) == 4:
            return float(q["val"].sum())
    annual = x[x["duration"].between(250, 400, inclusive="both")]
    return float(annual.iloc[-1]["val"]) if not annual.empty else np.nan


def _value_asof(df: pd.DataFrame, date: pd.Timestamp) -> float:
    row = _latest_asof(df, date)
    return float(row["val"]) if row is not None else np.nan


def _price_asof(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float:
    x = prices[(prices["ticker"] == ticker) & (prices["date"] <= date)]
    return float(x.iloc[-1]["close"]) if not x.empty else np.nan


def build_one(ticker: str, payload: dict, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected, frames = {}, {}
    for key, candidates in TAGS.items():
        tag, rows = _select_tag(payload, candidates)
        selected[key] = tag
        frames[key] = _prepare(rows, flow=key in FLOW_KEYS)

    filed_dates = sorted({d for df in frames.values() for d in df["filed"].dropna().tolist()})
    out = []
    for available_date in filed_dates:
        revenue = _quarterly_flow_asof(frames["revenue"], available_date)
        net_income = _quarterly_flow_asof(frames["net_income"], available_date)
        operating_income = _quarterly_flow_asof(frames["operating_income"], available_date)
        cfo = _quarterly_flow_asof(frames["cfo"], available_date)
        capex = _quarterly_flow_asof(frames["capex"], available_date)
        equity = _value_asof(frames["equity"], available_date)
        assets = _value_asof(frames["assets"], available_date)
        ca = _value_asof(frames["assets_current"], available_date)
        cl = _value_asof(frames["liabilities_current"], available_date)
        liabilities = _value_asof(frames["liabilities"], available_date)
        cash = _value_asof(frames["cash"], available_date)
        debt_short = _value_asof(frames["debt"], available_date)
        debt_long = _value_asof(frames["debt_long"], available_date)
        shares = _value_asof(frames["shares"], available_date)
        px = _price_asof(prices, ticker, available_date)

        debt_total = np.nansum([debt_short, debt_long])
        if not np.isfinite(debt_total) or debt_total == 0:
            debt_total = liabilities
        roe = net_income / equity if np.isfinite(net_income) and equity > 0 else np.nan
        roa = net_income / assets if np.isfinite(net_income) and assets > 0 else np.nan
        operating_margin = operating_income / revenue if np.isfinite(operating_income) and revenue > 0 else np.nan
        fcf = cfo - capex if np.isfinite(cfo) and np.isfinite(capex) else np.nan
        fcf_margin = fcf / revenue if np.isfinite(fcf) and revenue > 0 else np.nan
        fcf_to_net_income = fcf / net_income if np.isfinite(fcf) and np.isfinite(net_income) and abs(net_income) > 1e-12 else np.nan
        asset_turnover = revenue / assets if np.isfinite(revenue) and assets > 0 else np.nan
        debt_to_equity = debt_total / equity if np.isfinite(debt_total) and equity > 0 else np.nan
        net_debt = debt_total - cash if np.isfinite(debt_total) and np.isfinite(cash) else np.nan
        net_debt_to_equity = net_debt / equity if np.isfinite(net_debt) and equity > 0 else np.nan
        current_ratio = ca / cl if np.isfinite(ca) and cl > 0 else np.nan
        market_cap = shares * px if np.isfinite(shares) and shares > 0 and np.isfinite(px) else np.nan

        out.append({
            "ticker": ticker,
            "available_date": available_date.date().isoformat(),
            "roe": roe,
            "roa": roa,
            "operating_margin": operating_margin,
            "fcf_margin": fcf_margin,
            "fcf_to_net_income": fcf_to_net_income,
            "asset_turnover": asset_turnover,
            "debt_to_equity": debt_to_equity,
            "net_debt_to_equity": net_debt_to_equity,
            "current_ratio": current_ratio,
            "market_cap": market_cap,
        })

    audit = pd.DataFrame([{"ticker": ticker, "metric": key, "selected_tag": tag or ""} for key, tag in selected.items()])
    result = pd.DataFrame(out)
    if not result.empty:
        result = result.dropna(subset=["roe", "fcf_margin", "debt_to_equity", "current_ratio", "market_cap"], how="all")
        result = result.drop_duplicates(["ticker", "available_date"])
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
        rows, audit = build_one(ticker, json.loads(path.read_text(encoding="utf-8")), prices)
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
