# Trading-strategy

Reproducible Python research framework for systematic trading ideas. The project is intentionally smaller than Lean/Zipline/Nautilus, but borrows the parts that matter for trustworthy research: causal data, explicit historical-universe membership, hard validation gates, portfolio constraints, experiment fingerprints and out-of-sample robustness testing.

## Current strategies

1. **Quality dip / bad-news proxy** — buy financially strong companies after unusually large one-day declines and test recovery over 5/10/20/60 trading days.
2. **Opening Range Breakout (ORB)** — intraday ES/MES opening-range research scaffold with conservative same-bar assumptions.
3. **Positive earnings momentum** — event-driven earnings research scaffold.

> A large price drop is only a **bad-news proxy** unless a timestamped event/news dataset confirms the cause.

## Research architecture

```text
FINSABER-2 historical/delisted prices
              +
point-in-time fundamentals
              +
historical S&P snapshots
              |
        validation gates
              |
   explicit coverage report
              |
    causal feature assembly
              |
 historical membership filter
              |
       signal generation
              |
 portfolio/capital constraints
              |
  train -> validation -> OOS
              |
bootstrap CI + neighbor robustness
              |
 experiment manifest + leaderboard
```

## FINSABER-2 integration

The automated pipeline uses the official `finsaber-team/FINSABER-V2-Data` Hugging Face dataset as its primary bulk daily-price source.

It downloads yearly Parquet partitions from `price_daily/year=2000` through `year=2025`, resolves and pins the upstream dataset revision, and normalizes them into one local research Parquet.

Normalized fields:

- `ticker`
- `date`
- adjusted `open`, `high`, `low`, `close`
- raw `volume`
- `raw_close`
- `cik`
- `source_year`

Adjusted OHLC is derived with the FINSABER convention `adjusted_close / raw_close`. `raw_close` is retained because as-reported historical share counts must not be multiplied by a split-adjusted price when constructing historical market cap.

The large Parquet files are reproducibly downloaded during CI and are **not committed to git**. `finsaber_price_manifest.json` stores the pinned revision, partition counts and SHA-256 of the normalized local data.

## Data correctness rules

- Exact SEC values become usable on the **filing date**, never the fiscal period end.
- If exact local Company Facts are unavailable, the pipeline uses a clearly labelled **Tenline annual SEC-derived fallback** pinned to an upstream Git commit.
- The Tenline fallback availability date is conservatively delayed to `max(period_end + 120 days, Dec 31 of the latest SEC accession year referenced in provenance)`.
- FINSABER price presence never implies S&P membership; historical membership is applied from separately pinned constituent snapshots.
- Current constituents are never silently projected into the past.
- Share-class separators are canonicalized (`BRK-B`, `BRK/B`, `BRK.B` -> `BRK.B`).
- Price files fail validation on duplicates, invalid OHLC, non-positive prices and broad-data completeness gates.
- Missing historical fundamentals are reported in `research_coverage.json`; they are **not** treated as failed quality filters.
- Input files receive SHA-256 fingerprints in experiment manifests.
- Every parameter configuration receives a deterministic variant ID.

## Fundamental coverage caveat

FINSABER fixes the major **price-history survivorship** problem by including a broad historical S&P price universe and delisted names. It does not automatically supply the structured numeric point-in-time fundamentals used by this quality model.

The automated hosted fallback therefore intersects:

1. FINSABER price tickers,
2. historical S&P membership tickers, and
3. tickers with causal fundamentals available from the fundamental source.

`scripts/build_research_panel.py` records the exact intersection and all missing fundamental tickers in `data/real/research_coverage.json`.

This means the pipeline is now broad and survivorship-aware on the **price side**, while final quality conclusions must still be interpreted within measured fundamental coverage. A future structured historical-fundamentals source can be plugged in without changing the FINSABER price layer or backtest engine.

## Quality model

Preferred exact-SEC quality gate:

- ROE >= 12%
- FCF margin >= 5%
- debt/equity <= 1.5
- current ratio >= 1.0
- historical market cap >= $5B

Additional quality inputs when available:

- ROIC
- ROA
- operating margin
- FCF / net income
- asset turnover
- net debt / equity

The annual Tenline fallback does not expose gross debt/equity or current ratio. In that labelled mode the gate requires ROE, FCF margin, historical market cap and **net debt/equity <= 1.0**; missing current ratio is explicitly treated as unavailable rather than invented.

At each trading date the latest available fundamentals are ranked cross-sectionally to create a quality percentile.

## Quality-dip research grid

The default sweep runs **384 variants**:

- drop: -5%, -10%, -15%, -20%
- wait: 0, 1, 2 trading days
- hold: 5, 10, 20, 60 trading days
- minimum quality percentile: 0%, 50%, 70%, 80%
- stabilization filter: off/on

A stabilization-enabled trade requires the next completed session to close above the crash-day close and, when lows are available, not make a lower low. Entry cannot occur before the following session open.

Trade construction is vectorized with per-ticker row offsets so the 384-grid remains practical on the larger FINSABER-derived research panel.

## Robustness instead of cherry-picking

The sweep uses one chronological split shared by all variants:

- first 60%: train
- next 20%: validation
- final 20%: out-of-sample

Each variant receives:

- trade-level train/validation/OOS metrics
- bootstrap 95% CI for OOS mean return
- parameter-neighbor stability
- constrained $10,000 portfolio simulation
- OOS portfolio return, Sharpe and max drawdown

The leaderboard ranks a **robustness score**, not the best in-sample Sharpe.

## Portfolio assumptions

Default research account:

- initial capital: $10,000
- maximum positions: 10
- maximum allocation per position: 10%
- no overlapping positions in the same ticker

## Reproduce the FINSABER pipeline

```bash
pip install -r requirements.txt

python scripts/download_finsaber_prices.py \
  --start-year 2000 --end-year 2025 \
  --output data/real/finsaber_prices.parquet \
  --manifest data/real/finsaber_price_manifest.json

python scripts/download_sp500_membership.py

python scripts/build_universe.py \
  --history data/real/raw/sp_500_historical_components.csv \
  --output data/real/universe_intervals.csv

python scripts/validate_research_data.py \
  --prices data/real/finsaber_prices.parquet \
  --universe data/real/universe_intervals.csv \
  --report data/real/validation_finsaber_market.json

python scripts/build_fundamentals.py \
  --prices data/real/finsaber_prices.parquet \
  --output data/real/fundamentals.csv \
  --audit data/real/fundamentals_audit.csv \
  --source data/real/fundamentals_source.json

python scripts/build_research_panel.py \
  --prices data/real/finsaber_prices.parquet \
  --fundamentals data/real/fundamentals.csv \
  --universe data/real/universe_intervals.csv \
  --output data/real/research_prices.parquet \
  --coverage data/real/research_coverage.json

python scripts/run_quality_dip_sweep.py \
  --prices data/real/research_prices.parquet \
  --fundamentals data/real/fundamentals.csv \
  --universe data/real/universe_intervals.csv \
  --source-manifest data/real/finsaber_price_manifest.json \
  --coverage data/real/research_coverage.json \
  --output results/quality_dip_real
```

## Research outputs

`results/quality_dip_real/`:

- `parameter_sweep.csv`
- `leaderboard.csv`
- `robustness_report.md`
- `experiment_manifest.json`
- `heatmap_oos_return.csv`
- `heatmap_oos_portfolio_sharpe.csv`
- `top_variant_trades.csv`

Compact provenance under `data/real/`:

- `finsaber_price_manifest.json`
- `sp500_membership_manifest.json`
- `fundamentals_source.json`
- `research_coverage.json`
- `validation_finsaber_market.json`
- `validation_full.json`

## Data sources

- Bulk daily prices: official **FINSABER-2** Parquet dataset
- Historical S&P composition: `hanshof/sp500_constituents`, pinned commit
- Preferred fundamentals: SEC EDGAR Company Facts when exact local raw data exists
- Hosted fundamentals fallback: `debjitmukherjee1/tenline`, pinned commit, conservative annual availability
- Earnings sample: Bloomberg-derived 2016 sample from `pingfcc99/Earnings-surprise-on-stock-price`

## Tests

```bash
pytest -q
```

Tests cover point-in-time filing use, FINSABER adjusted-OHLC normalization, share-class ticker canonicalization, Parquet I/O, historical universe intervals, data validation, conservative fallback availability, fallback leverage semantics, stabilization timing, portfolio position limits and deterministic experiment IDs.

## Disclaimer

Research/educational software only. Backtests can be wrong even when code runs successfully; data provenance, survivorship bias, execution assumptions and out-of-sample stability matter more than a single attractive headline return.
