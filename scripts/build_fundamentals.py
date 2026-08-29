#!/usr/bin/env python3
"""Build research fundamentals from exact SEC Company Facts when available.

GitHub-hosted runners are sometimes blocked by data.sec.gov. In that case this
script falls back to Tenline, a public GitHub dataset derived from SEC filings.
The Tenline fallback is intentionally conservative:

- it uses annual observations only;
- each observation is unavailable until at least period_end + 120 days; and
- when provenance cites a later SEC accession year, availability is delayed to
  December 31 of that accession year.

This prevents a later comparative/restated filing from leaking backwards into
an earlier backtest date. The fallback is therefore causal but deliberately
stale, and is labelled as such in fundamentals_source.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from scripts.build_sec_fundamentals import build_one

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICES = ROOT / "data" / "real" / "prices.csv"
DEFAULT_SEC_DIR = ROOT / "data" / "real" / "raw" / "sec_companyfacts"
DEFAULT_OUTPUT = ROOT / "data" / "real" / "fundamentals.csv"
DEFAULT_AUDIT = ROOT / "data" / "real" / "fundamentals_audit.csv"
DEFAULT_SOURCE = ROOT / "data" / "real" / "fundamentals_source.json"
TENLINE_REPO = "debjitmukherjee1/tenline"
TENLINE_COMMIT_API = f"https://api.github.com/repos/{TENLINE_REPO}/commits/main"
TENLINE_RAW = "https://raw.githubusercontent.com/{repo}/{sha}/site/data/companies/{ticker}.json"
ACC_YEAR = re.compile(r"-(\d{2})-")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Trading-strategy research fallback"})
    return s


def accession_years(record: dict) -> list[int]:
    years: list[int] = []
    for metric in (record.get("prov") or {}).values():
        for item in (metric or {}).get("inputs", []):
            accn = str((item or {}).get("accn", ""))
            match = ACC_YEAR.search(accn)
            if match:
                yy = int(match.group(1))
                years.append(2000 + yy if yy < 70 else 1900 + yy)
    return years


def conservative_available_date(record: dict) -> pd.Timestamp:
    period_end = pd.Timestamp(record["period_end"]).normalize()
    base = period_end + pd.Timedelta(days=120)
    years = accession_years(record)
    if not years:
        return base
    provenance_guard = pd.Timestamp(year=max(years), month=12, day=31)
    return max(base, provenance_guard)


def _price_asof(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float:
    x = prices[(prices["ticker"] == ticker) & (prices["date"] <= date)]
    if x.empty:
        return np.nan
    row = x.iloc[-1]
    col = "raw_close" if "raw_close" in x.columns and pd.notna(row.get("raw_close")) else "close"
    return float(row[col])


def _tenline_commit(session: requests.Session) -> str:
    r = session.get(TENLINE_COMMIT_API, timeout=60)
    r.raise_for_status()
    sha = str(r.json().get("sha", ""))
    if len(sha) < 12:
        raise RuntimeError("Could not resolve Tenline commit SHA")
    return sha


def _download_tenline_record(ticker: str, sha: str, session: requests.Session) -> dict:
    url = TENLINE_RAW.format(repo=TENLINE_REPO, sha=sha, ticker=ticker)
    r = session.get(url, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if str(payload.get("ticker", "")).upper() != ticker:
        raise RuntimeError(f"Tenline ticker mismatch for {ticker}")
    return payload


def build_from_tenline(prices: pd.DataFrame, tickers: Iterable[str], output: Path, audit_path: Path, source_path: Path) -> pd.DataFrame:
    session = _session()
    sha = _tenline_commit(session)
    rows: list[dict] = []
    audits: list[dict] = []
    failed: dict[str, str] = {}

    for ticker in tickers:
        try:
            payload = _download_tenline_record(ticker, sha, session)
        except Exception as exc:
            failed[ticker] = str(exc)
            print(f"[tenline] {ticker}: unavailable ({exc})")
            continue
        count = 0
        for record in payload.get("years", []):
            if not record.get("period_end"):
                continue
            available_date = conservative_available_date(record)
            px = _price_asof(prices, ticker, available_date)
            shares = pd.to_numeric(pd.Series([record.get("diluted_shares")]), errors="coerce").iloc[0]
            market_cap = float(shares * px) if np.isfinite(shares) and shares > 0 and np.isfinite(px) else np.nan
            fcf = pd.to_numeric(pd.Series([record.get("fcf")]), errors="coerce").iloc[0]
            net_income = pd.to_numeric(pd.Series([record.get("net_income")]), errors="coerce").iloc[0]
            fcf_to_ni = float(fcf / net_income) if np.isfinite(fcf) and np.isfinite(net_income) and abs(net_income) > 1e-12 else np.nan
            source_years = accession_years(record)
            rows.append({
                "ticker": ticker,
                "available_date": available_date.date().isoformat(),
                "roe": record.get("roe"),
                "roic": record.get("roic"),
                "roa": np.nan,
                "operating_margin": record.get("operating_margin"),
                "fcf_margin": record.get("fcf_margin"),
                "fcf_to_net_income": fcf_to_ni,
                "asset_turnover": np.nan,
                "debt_to_equity": np.nan,
                "net_debt_to_equity": record.get("net_debt_to_equity"),
                "current_ratio": np.nan,
                "market_cap": market_cap,
                "fundamental_source": "tenline_sec_annual_conservative",
                "source_fiscal_year": record.get("fy"),
                "source_period_end": record.get("period_end"),
                "source_accession_year_max": max(source_years) if source_years else np.nan,
            })
            count += 1
        audits.append({
            "ticker": ticker,
            "metric": "source",
            "selected_tag": "Tenline annual SEC-derived; availability=max(period_end+120d, Dec-31 latest provenance accession year)",
            "rows": count,
        })
        print(f"[tenline] {ticker}: {count} conservative annual rows")

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("Tenline fallback produced no fundamentals")
    result["available_date"] = pd.to_datetime(result["available_date"])
    result = result.sort_values(["ticker", "available_date"]).drop_duplicates(["ticker", "available_date"], keep="last")
    result["available_date"] = result["available_date"].dt.date.astype(str)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    source = {
        "mode": "tenline_sec_annual_conservative",
        "upstream_repo": TENLINE_REPO,
        "upstream_commit": sha,
        "availability_rule": "max(period_end + 120 days, December 31 of latest SEC accession year referenced by provenance)",
        "reason": "Exact data.sec.gov Company Facts unavailable from the GitHub-hosted runner",
        "limitations": [
            "annual rather than quarterly fundamentals",
            "availability deliberately delayed to avoid comparative/restatement look-ahead",
            "current ratio and gross debt/equity are unavailable in Tenline output",
            "market cap uses diluted shares times historical unadjusted/raw close where available",
        ],
        "failed_tickers": failed,
    }
    source_path.write_text(json.dumps(source, indent=2), encoding="utf-8")
    return result


def build_from_sec(prices: pd.DataFrame, sec_dir: Path, tickers: list[str], output: Path, audit_path: Path, source_path: Path) -> pd.DataFrame:
    all_rows, audits = [], []
    for ticker in tickers:
        path = sec_dir / f"{ticker}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows, audit = build_one(ticker, payload, prices)
        if not rows.empty:
            rows["fundamental_source"] = "sec_companyfacts_exact_filing_date"
            all_rows.append(rows)
        audits.append(audit)
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if result.empty:
        raise RuntimeError("SEC Company Facts files existed but produced no fundamentals")
    result.to_csv(output, index=False)
    pd.concat(audits, ignore_index=True).to_csv(audit_path, index=False)
    source_path.write_text(json.dumps({
        "mode": "sec_companyfacts_exact_filing_date",
        "availability_rule": "SEC filed date",
        "tickers": tickers,
    }, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DEFAULT_PRICES))
    p.add_argument("--sec-dir", default=str(DEFAULT_SEC_DIR))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--audit", default=str(DEFAULT_AUDIT))
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    args = p.parse_args()

    prices = pd.read_csv(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices.sort_values(["ticker", "date"])
    tickers = sorted(prices["ticker"].dropna().unique().tolist())
    sec_dir = Path(args.sec_dir)
    output, audit_path, source_path = Path(args.output), Path(args.audit), Path(args.source)
    have_exact_sec = all((sec_dir / f"{ticker}.json").exists() for ticker in tickers)

    if have_exact_sec:
        result = build_from_sec(prices, sec_dir, tickers, output, audit_path, source_path)
    else:
        result = build_from_tenline(prices, tickers, output, audit_path, source_path)
    print(f"Wrote {len(result):,} fundamentals rows to {output}")


if __name__ == "__main__":
    main()
