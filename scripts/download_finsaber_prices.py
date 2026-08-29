#!/usr/bin/env python3
"""Download and normalize the official FINSABER-2 S&P price dataset.

The upstream dataset is partitioned by year in Parquet. We pin the Hugging Face
revision, stream each yearly file through a ParquetWriter, retain raw close for
market-cap calculations, and use split/dividend-adjusted OHLC for return and
execution research.

Rows with impossible OHLC geometry are excluded and counted in the manifest.
A configurable hard ceiling prevents broad source-cleaning from silently hiding
a material upstream data-quality problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from src.data.universe import normalize_ticker

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
DEFAULT_OUTPUT = REAL / "finsaber_prices.parquet"
DEFAULT_MANIFEST = REAL / "finsaber_price_manifest.json"
REPO_ID = "finsaber-team/FINSABER-V2-Data"
PRICE_PATH = "price_daily/year={year}/part-000.parquet"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_finsaber_prices(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    required = {
        "date", "symbol", "cik", "open", "high", "low", "close",
        "adjusted_close", "volume",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"FINSABER price partition {year} missing columns: {sorted(missing)}")

    x = frame[list(required)].copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["ticker"] = x["symbol"].map(normalize_ticker)
    x["cik"] = x["cik"].astype("string").fillna("")
    for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    raw_close = x["close"].copy()
    factor = x["adjusted_close"] / raw_close
    base_valid = (
        x["date"].notna()
        & x["ticker"].ne("")
        & raw_close.gt(0)
        & x["adjusted_close"].gt(0)
        & factor.gt(0)
        & factor.notna()
    )
    invalid_base = int((~base_valid).sum())
    x = x.loc[base_valid].copy()
    factor = factor.loc[base_valid]
    raw_close = raw_close.loc[base_valid]

    # Calculate every adjusted OHLC field with the exact same factor. Using the
    # upstream adjusted_close directly for C while multiplying O/H/L can create
    # tiny floating-point boundary violations even when raw OHLC is consistent.
    out = pd.DataFrame({
        "ticker": x["ticker"].astype("string"),
        "date": x["date"].dt.normalize(),
        "open": x["open"] * factor,
        "high": x["high"] * factor,
        "low": x["low"] * factor,
        "close": raw_close * factor,
        "volume": x["volume"],
        "raw_close": raw_close,
        "cik": x["cik"],
        "source_year": pd.Series(year, index=x.index, dtype="int16"),
    })

    complete = out[["open", "high", "low", "close"]].notna().all(axis=1)
    positive = (out[["open", "high", "low", "close"]] > 0).all(axis=1)
    magnitude = out[["open", "high", "low", "close"]].abs().max(axis=1).clip(lower=1.0)
    tol = magnitude * 1e-10
    high_ok = out["high"] + tol >= out[["open", "close", "low"]].max(axis=1)
    low_ok = out["low"] - tol <= out[["open", "close", "high"]].min(axis=1)
    ohlc_valid = complete & positive & high_ok & low_ok
    invalid_ohlc = int((~ohlc_valid).sum())
    out = out.loc[ohlc_valid].copy()

    before_dupes = len(out)
    out = out.drop_duplicates(["ticker", "date"], keep="last")
    duplicate_rows = int(before_dupes - len(out))
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out.attrs["normalization_stats"] = {
        "input_rows": int(len(frame)),
        "invalid_base_rows": invalid_base,
        "invalid_ohlc_rows": invalid_ohlc,
        "duplicate_ticker_date_rows": duplicate_rows,
        "output_rows": int(len(out)),
    }
    return out


def resolve_revision(repo_id: str, requested: str | None) -> str:
    info = HfApi().dataset_info(repo_id=repo_id, revision=requested or "main")
    if not info.sha:
        raise RuntimeError(f"Could not resolve Hugging Face revision for {repo_id}")
    return str(info.sha)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--revision", help="Optional Hugging Face commit SHA/tag; main is resolved and pinned when omitted")
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--cache-dir")
    p.add_argument("--min-rows", type=int, default=0)
    p.add_argument("--min-tickers", type=int, default=0)
    p.add_argument("--max-invalid-ohlc-rows", type=int, default=1000)
    args = p.parse_args()

    if args.end_year < args.start_year:
        raise ValueError("end-year must be >= start-year")

    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    revision = resolve_revision(args.repo_id, args.revision)
    writer: pq.ParquetWriter | None = None
    row_count = 0
    tickers: set[str] = set()
    yearly_rows: dict[str, int] = {}
    yearly_normalization: dict[str, dict] = {}
    source_files: list[dict] = []
    total_invalid_ohlc = 0

    try:
        for year in range(args.start_year, args.end_year + 1):
            filename = PRICE_PATH.format(year=year)
            local = Path(hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=filename,
                revision=revision,
                cache_dir=args.cache_dir,
            ))
            raw = pd.read_parquet(local)
            normalized = normalize_finsaber_prices(raw, year)
            stats = dict(normalized.attrs.get("normalization_stats", {}))
            yearly_normalization[str(year)] = stats
            total_invalid_ohlc += int(stats.get("invalid_ohlc_rows", 0))
            if normalized.empty:
                raise RuntimeError(f"FINSABER partition {year} produced zero valid price rows")
            table = pa.Table.from_pandas(normalized, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            count = int(len(normalized))
            yearly_rows[str(year)] = count
            row_count += count
            tickers.update(normalized["ticker"].astype(str).unique().tolist())
            source_files.append({"year": year, "filename": filename, "bytes": local.stat().st_size})
            dropped = int(stats.get("input_rows", count) - count)
            print(f"[finsaber] {year}: {count:,} normalized rows ({dropped:,} excluded)")
    finally:
        if writer is not None:
            writer.close()

    if not output.exists():
        raise RuntimeError("FINSABER output parquet was not created")
    if args.min_rows and row_count < args.min_rows:
        raise RuntimeError(f"FINSABER completeness gate failed: {row_count:,} rows < {args.min_rows:,}")
    if args.min_tickers and len(tickers) < args.min_tickers:
        raise RuntimeError(f"FINSABER completeness gate failed: {len(tickers)} tickers < {args.min_tickers}")
    if total_invalid_ohlc > args.max_invalid_ohlc_rows:
        raise RuntimeError(
            f"FINSABER OHLC quality gate failed: {total_invalid_ohlc:,} invalid rows > "
            f"allowed {args.max_invalid_ohlc_rows:,}"
        )

    manifest = {
        "source": "FINSABER-2 official Hugging Face dataset",
        "repo_id": args.repo_id,
        "revision": revision,
        "layout": "price_daily/year=YYYY/part-000.parquet",
        "years": [args.start_year, args.end_year],
        "rows": row_count,
        "tickers": len(tickers),
        "yearly_rows": yearly_rows,
        "yearly_normalization": yearly_normalization,
        "normalization_totals": {
            "invalid_ohlc_rows_excluded": total_invalid_ohlc,
            "max_invalid_ohlc_rows": args.max_invalid_ohlc_rows,
        },
        "source_files": source_files,
        "normalized_output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "schema": ["ticker", "date", "open", "high", "low", "close", "volume", "raw_close", "cik", "source_year"],
        },
        "price_semantics": {
            "open_high_low_close": "split/dividend-adjusted with one adjusted_close/raw_close factor applied consistently to raw O/H/L/C",
            "raw_close": "upstream unadjusted close retained for as-reported share-count market-cap calculations",
            "volume": "upstream raw volume",
            "invalid_ohlc": "excluded from research and counted in yearly_normalization; run fails if exclusion ceiling is exceeded",
        },
        "research_note": "Historical S&P membership is applied separately; presence in FINSABER alone is not treated as index membership.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[finsaber] wrote {row_count:,} rows / {len(tickers)} tickers to {output}")
    print(f"[finsaber] excluded {total_invalid_ohlc:,} invalid OHLC rows")
    print(f"[finsaber] pinned upstream revision {revision}")


if __name__ == "__main__":
    main()
