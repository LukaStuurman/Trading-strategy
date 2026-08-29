#!/usr/bin/env python3
"""Download reproducible public historical data used by the backtests.

Sources
-------
Daily OHLCV       : Stooq CSV download endpoint
S&P 500 history   : hanshof/sp500_constituents (GitHub, MIT licensed repo)
SEC fundamentals  : SEC EDGAR company_tickers.json + companyfacts API

The SEC files are deliberately stored raw. They contain filing dates and can be
converted to point-in-time features without pretending that a fiscal period-end
was the date investors knew the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
RAW = REAL / "raw"
PRICES_DIR = RAW / "prices"
SEC_DIR = RAW / "sec_companyfacts"

SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/"
    "sp_500_historical_components.csv"
)
SP500_CURRENT_URL = (
    "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/"
    "sp500_constituents.csv"
)
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": os.environ.get(
            "SEC_USER_AGENT",
            "Trading-strategy research contact: set SEC_USER_AGENT@example.com",
        )
    })
    return s


def ensure_dirs() -> None:
    for p in (REAL, RAW, PRICES_DIR, SEC_DIR):
        p.mkdir(parents=True, exist_ok=True)


def fetch_bytes(url: str, target: Path, session: requests.Session) -> Path:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    target.write_bytes(r.content)
    return target


def download_sp500_history(session: requests.Session) -> None:
    fetch_bytes(SP500_HISTORY_URL, RAW / "sp500_historical_components.csv", session)
    fetch_bytes(SP500_CURRENT_URL, RAW / "sp500_current_constituents.csv", session)


def stooq_symbol(ticker: str) -> str:
    return ticker.lower().replace(".", "-") + ".us"


def download_prices(
    tickers: Iterable[str], start: str, end: str, session: requests.Session
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")

    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol(ticker)}&d1={d1}&d2={d2}&i=d"
        r = session.get(url, timeout=60)
        r.raise_for_status()
        text = r.text.strip()
        if not text or text.startswith("No data"):
            print(f"[prices] no data: {ticker}")
            continue

        target = PRICES_DIR / f"{ticker}.csv"
        target.write_text(text + "\n", encoding="utf-8")
        df = pd.read_csv(target)
        df.columns = [c.lower() for c in df.columns]
        df.insert(0, "ticker", ticker)
        frames.append(df)
        print(f"[prices] {ticker}: {len(df):,} rows")
        time.sleep(0.15)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"date": "date"})
    combined.to_csv(REAL / "prices.csv", index=False)
    return combined


def sec_ticker_map(session: requests.Session) -> dict[str, int]:
    r = session.get(SEC_TICKERS_URL, timeout=60)
    r.raise_for_status()
    payload = r.json()
    return {
        row["ticker"].upper(): int(row["cik_str"])
        for row in payload.values()
    }


def download_sec_companyfacts(tickers: Iterable[str], session: requests.Session) -> None:
    mapping = sec_ticker_map(session)
    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        cik = mapping.get(ticker)
        if cik is None:
            print(f"[sec] no CIK: {ticker}")
            continue
        url = SEC_FACTS.format(cik=cik)
        r = session.get(url, timeout=60)
        r.raise_for_status()
        payload = r.json()
        payload["download_metadata"] = {
            "source_url": url,
            "downloaded_by": "scripts/download_real_data.py",
            "point_in_time_note": "Use each fact's filed date, not merely period end.",
        }
        (SEC_DIR / f"{ticker}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"[sec] {ticker} CIK {cik}")
        # SEC asks automated clients to stay well below its request-rate ceiling.
        time.sleep(0.15)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tickers",
        default="AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,COST,HD,NKE",
        help="Comma separated US tickers",
    )
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=pd.Timestamp.utcnow().date().isoformat())
    p.add_argument("--skip-prices", action="store_true")
    p.add_argument("--skip-sec", action="store_true")
    p.add_argument("--skip-sp500", action="store_true")
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
    if not args.skip_sec:
        download_sec_companyfacts(tickers, session)

    manifest = {
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "sources": {
            "prices": "Stooq",
            "sp500_membership": "hanshof/sp500_constituents",
            "fundamentals": "SEC EDGAR companyfacts",
            "earnings_sample": "pingfcc99/Earnings-surprise-on-stock-price Bloomberg-derived 2016 sample",
        },
    }
    (REAL / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Done. Data written under data/real/.")


if __name__ == "__main__":
    main()
