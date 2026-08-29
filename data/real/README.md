# Real historical data

The automated research pipeline now uses **FINSABER-2** as its primary bulk daily-price source. Large price Parquet files are deliberately not committed to this GitHub repository; only compact provenance, validation, coverage, fundamentals and backtest outputs are committed.

## Bulk prices: FINSABER-2

`scripts/download_finsaber_prices.py` downloads the official Hugging Face dataset `finsaber-team/FINSABER-V2-Data` and pins the resolved upstream revision. The pipeline consumes:

```text
price_daily/year=2000/part-000.parquet
...
price_daily/year=2025/part-000.parquet
```

The normalized local file is `data/real/finsaber_prices.parquet` and contains:

- `ticker`
- `date`
- adjusted `open/high/low/close`
- raw `volume`
- `raw_close` for market-cap alignment
- `cik`
- `source_year`

Adjusted OHLC is computed with `adjusted_close / raw_close`, matching the FINSABER-2 research convention. `finsaber_price_manifest.json` stores the upstream revision, per-year row counts and SHA-256 of the normalized local Parquet.

The full Parquet is ignored by git because it is reproducibly downloadable and too large to belong in normal source control.

## Historical S&P membership

`scripts/download_sp500_membership.py` pins a commit from `hanshof/sp500_constituents` and stores the historical/current source CSVs. `scripts/build_universe.py` converts snapshots to `[start_date, end_date)` membership intervals.

A FINSABER price record does **not** by itself prove that a stock belonged to the S&P 500 on that date. Signal eligibility always uses the separate historical membership intervals.

## Fundamentals

Preferred local source: exact SEC Company Facts with the SEC filing date as `available_date`.

Hosted fallback: `debjitmukherjee1/tenline`, pinned to a Git commit and deliberately delayed with the conservative annual availability rule documented in `fundamentals_source.json`.

FINSABER solves the historical/delisted **price** coverage problem, but Tenline does not cover every former/delisted historical S&P ticker. `research_coverage.json` therefore records the exact intersection of:

1. FINSABER price tickers,
2. historical S&P membership tickers, and
3. tickers with causal fundamentals.

Missing fundamentals are reported as missing data, not treated as failed quality screens.

## Research panel

`data/real/research_prices.parquet` is an ephemeral, smaller Parquet built from the broad FINSABER file. It keeps only tickers with usable causal fundamentals and enough warm-up history around their first available fundamental observation. It is the input to the 384-variant quality-dip sweep.

## Committed provenance and outputs

- `finsaber_price_manifest.json`
- `sp500_membership_manifest.json`
- `sp500_historical_components.csv`
- `sp500_current_constituents.csv`
- `universe_intervals.csv`
- `fundamentals.csv`
- `fundamentals_audit.csv`
- `fundamentals_source.json`
- `research_coverage.json`
- `validation_finsaber_market.json`
- `validation_full.json`
- `earnings_surprise_bloomberg_2016.csv`

## Reproduce locally

```bash
pip install -r requirements.txt

python scripts/download_finsaber_prices.py \
  --start-year 2000 --end-year 2025 \
  --output data/real/finsaber_prices.parquet \
  --manifest data/real/finsaber_price_manifest.json

python scripts/download_sp500_membership.py
python scripts/build_universe.py \
  --history data/real/raw/sp500_historical_components.csv \
  --output data/real/universe_intervals.csv

python scripts/build_fundamentals.py \
  --prices data/real/finsaber_prices.parquet

python scripts/build_research_panel.py \
  --prices data/real/finsaber_prices.parquet \
  --fundamentals data/real/fundamentals.csv \
  --universe data/real/universe_intervals.csv \
  --output data/real/research_prices.parquet \
  --coverage data/real/research_coverage.json
```

## Research constraints

- Price drops remain a **bad-news proxy** until a timestamped event/news classifier is added.
- The broad FINSABER price universe reduces survivorship bias, but final quality conclusions remain bounded by fundamental coverage.
- Daily prices support next-open swing research, not intraday VWAP/10:00 ET execution tests.
- Large source datasets stay outside git; manifests and hashes make them reproducible instead.
