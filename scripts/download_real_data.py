#!/usr/bin/env python3
"""Download reproducible public historical data.

Daily OHLCV uses Stooq first and a split/dividend-adjusted Yahoo chart fallback.
The Yahoo path also preserves `raw_close` so historical market-cap calculations
can pair as-reported share counts with an unadjusted price.

SEC Company Facts are attempted directly. `--sec-soft-fail` lets a downstream
reproducible fallback take over when data.sec.gov blocks a hosted runner.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
RAW = REAL / "raw"
PRICES_DIR = RAW / "prices"
SEC_DIR = RAW / "sec_companyfacts"
SP500_HISTORY_URL = "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/sp_500_historical_components.csv"
SP500_CURRENT_URL = "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/sp500_constituents.csv"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
STARTER_CIKS = {
    "AAPL": 320193, "MSFT": 789019, "GOOGL": 1652044, "GOOG": 1652044,
    "AMZN": 1018724, "META": 1326801, "NVDA": 1045810, "JPM": 19617,
    "COST": 909832, "HD": 354950, "NKE": 320187,
}


def http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": os.environ.get("SEC_USER_AGENT", "Trading-strategy research 32568576+LukaStuurman@users.noreply.github.com"),
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def ensure_dirs() -> None:
    for p in (REAL, RAW, PRICES_DIR, SEC_DIR):
        p.mkdir(parents=True, exist_ok=True)


def fetch_bytes(url: str, target: Path, session: requests.Session) -> None:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    target.write_bytes(r.content)


def download_sp500_history(session: requests.Session) -> None:
    fetch_bytes(SP500_HISTORY_URL, RAW / "sp500_historical_components.csv", session)
    fetch_bytes(SP500_CURRENT_URL, RAW / "sp500_current_constituents.csv", session)


def stooq_symbol(ticker: str) -> str:
    return ticker.lower().replace(".", "-") + ".us"


def _valid_price_frame(df: pd.DataFrame, start: str, end: str) -> bool:
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return False
    if (pd.Timestamp(end) - pd.Timestamp(start)).days > 365 and len(df) < 100:
        return False
    return len(df) >= 2


def _stooq_prices(ticker: str, start: str, end: str, session: requests.Session) -> pd.DataFrame:
    d1, d2 = start.replace("-", ""), end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol(ticker)}&d1={d1}&d2={d2}&i=d"
    r = session.get(url, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    if not text or text.startswith("No data"):
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(text))
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).lower() for c in df.columns]
    if "close" in df.columns:
        df["raw_close"] = df["close"]
    return df


def _yahoo_prices(ticker: str, start: str, end: str, session: requests.Session) -> pd.DataFrame:
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    params = {
        "period1": period1, "period2": period2, "interval": "1d",
        "events": "div,splits", "includeAdjustedClose": "true",
    }
    r = session.get(YAHOO_CHART.format(ticker=ticker.replace(".", "-")), params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        return pd.DataFrame()
    result = results[0]
    ts = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose", [])
    n = len(ts)
    if not n:
        return pd.DataFrame()
    raw_close = pd.to_numeric(pd.Series(quotes.get("close", [None] * n)), errors="coerce")
    adj_close = pd.to_numeric(pd.Series(adj if len(adj) == n else quotes.get("close", [None] * n)), errors="coerce")
    factor = (adj_close / raw_close).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    data = {
        "date": pd.to_datetime(ts, unit="s", utc=True).date,
        "volume": quotes.get("volume", [None] * n),
        "raw_close": raw_close,
    }
    for col in ["open", "high", "low"]:
        raw = pd.to_numeric(pd.Series(quotes.get(col, [None] * n)), errors="coerce")
        data[col] = raw * factor
    data["close"] = adj_close
    df = pd.DataFrame(data)[["date", "open", "high", "low", "close", "volume", "raw_close"]]
    return df.dropna(subset=["open", "high", "low", "close", "raw_close"]).reset_index(drop=True)


def download_prices(tickers: Iterable[str], start: str, end: str, session: requests.Session) -> pd.DataFrame:
    frames, source_by_ticker = [], {}
    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        df = _stooq_prices(ticker, start, end, session)
        source = "Stooq"
        if not _valid_price_frame(df, start, end):
            print(f"[prices] Stooq invalid for {ticker} ({len(df)} rows); trying adjusted Yahoo")
            df = _yahoo_prices(ticker, start, end, session)
            source = "Yahoo chart API adjusted OHLC + raw_close"
        if not _valid_price_frame(df, start, end):
            raise RuntimeError(f"No valid historical price series for {ticker}; got {len(df)} rows")
        if "raw_close" not in df.columns:
            df["raw_close"] = df["close"]
        df = df[["date", "open", "high", "low", "close", "volume", "raw_close"]].copy()
        df.insert(0, "ticker", ticker)
        df.to_csv(PRICES_DIR / f"{ticker}.csv", index=False)
        frames.append(df)
        source_by_ticker[ticker] = source
        print(f"[prices] {ticker}: {len(df):,} rows via {source}")
        time.sleep(0.2)
    if not frames:
        raise RuntimeError("No price data downloaded")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(REAL / "prices.csv", index=False)
    (REAL / "price_sources.json").write_text(json.dumps(source_by_ticker, indent=2), encoding="utf-8")
    return combined


def sec_ticker_map() -> dict[str, int]:
    mapping = dict(STARTER_CIKS)
    current = RAW / "sp500_current_constituents.csv"
    if current.exists():
        try:
            df = pd.read_csv(current)
            cols = {str(c).lower(): c for c in df.columns}
            symbol_col, cik_col = cols.get("symbol") or cols.get("ticker"), cols.get("cik")
            if symbol_col and cik_col:
                for symbol, cik in df[[symbol_col, cik_col]].dropna().itertuples(index=False):
                    try:
                        mapping[str(symbol).upper()] = int(cik)
                    except (TypeError, ValueError):
                        pass
        except Exception as exc:
            print(f"[sec] warning: could not parse S&P CIK mapping: {exc}")
    return mapping


def download_sec_companyfacts(tickers: Iterable[str], session: requests.Session, *, soft_fail: bool = False) -> dict:
    mapping = sec_ticker_map()
    downloaded: list[str] = []
    failures: dict[str, str] = {}
    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        cik = mapping.get(ticker)
        if cik is None:
            failures[ticker] = "no CIK mapping"
            print(f"[sec] no CIK mapping: {ticker}")
            continue
        url = SEC_FACTS.format(cik=cik)
        last_error = None
        for attempt in range(3):
            try:
                r = session.get(url, timeout=60)
                r.raise_for_status()
                payload = r.json()
                payload["download_metadata"] = {
                    "source_url": url,
                    "downloaded_by": "scripts/download_real_data.py",
                    "point_in_time_note": "Use each fact's filed date, not period end.",
                }
                (SEC_DIR / f"{ticker}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"[sec] {ticker} CIK {cik}")
                downloaded.append(ticker)
                last_error = None
                break
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1.0 + attempt)
        if last_error is not None:
            failures[ticker] = str(last_error)
            if soft_fail:
                print(f"[sec] soft failure for {ticker}: {last_error}; downstream fallback will be used")
                # A 403 from the first well-formed request is normally an egress/IP
                # block. Avoid hammering the endpoint for the remaining tickers.
                if getattr(getattr(last_error, "response", None), "status_code", None) == 403:
                    break
                continue
            raise RuntimeError(f"SEC Company Facts failed for {ticker}: {last_error}")
        time.sleep(0.25)
    return {"downloaded": downloaded, "failures": failures, "complete": not failures and len(downloaded) > 0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,COST,HD,NKE")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=pd.Timestamp.now("UTC").date().isoformat())
    p.add_argument("--skip-prices", action="store_true")
    p.add_argument("--skip-sec", action="store_true")
    p.add_argument("--skip-sp500", action="store_true")
    p.add_argument("--sec-soft-fail", action="store_true", help="Continue when SEC blocks hosted runner; fundamentals builder will use a labelled fallback")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    session = http_session()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not args.skip_sp500:
        download_sp500_history(session)
    if not args.skip_prices:
        download_prices(tickers, args.start, args.end, session)
    sec_status = {"skipped": True}
    if not args.skip_sec:
        sec_status = download_sec_companyfacts(tickers, session, soft_fail=args.sec_soft_fail)
    manifest = {
        "tickers": tickers, "start": args.start, "end": args.end,
        "sources": {
            "prices": "Stooq with adjusted Yahoo chart fallback; raw_close retained for market-cap alignment",
            "sp500_membership": "hanshof/sp500_constituents",
            "fundamentals_primary": "SEC EDGAR Company Facts",
            "fundamentals_fallback": "debjitmukherjee1/tenline pinned GitHub commit, conservative annual availability",
            "earnings_sample": "pingfcc99/Earnings-surprise-on-stock-price Bloomberg-derived 2016 sample",
        },
        "sec_download": sec_status,
    }
    (REAL / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done. Data written under data/real/.")


if __name__ == "__main__":
    main()
