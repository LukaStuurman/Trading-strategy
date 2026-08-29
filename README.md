# Trading-strategy

Reproducible Python research framework for systematic trading ideas. The project is intentionally smaller than Lean/Zipline/Nautilus, but borrows the parts that matter for trustworthy research: point-in-time data, explicit universe membership, data validation gates, portfolio constraints, experiment fingerprints and out-of-sample robustness testing.

## Current strategies

1. **Quality dip / bad-news proxy** — buy financially strong companies after unusually large one-day declines and test the recovery over 5/10/20/60 trading days.
2. **Opening Range Breakout (ORB)** — intraday ES/MES opening-range research with conservative same-bar assumptions.
3. **Positive earnings momentum** — event-driven earnings research scaffold.

> A large price drop is only a **bad-news proxy** unless an event/news dataset confirms the cause.

## Research architecture

```text
raw prices + SEC filings + historical index snapshots
                    |
              validation gate
                    |
       point-in-time feature assembly
                    |
       historical universe membership
                    |
              signal generation
                    |
       portfolio / capital constraints
                    |
        train -> validation -> OOS
                    |
 bootstrap CI + neighbor robustness
                    |
 experiment manifest + leaderboard
```

### Data correctness rules

- SEC values become usable on the **filing date**, never the fiscal period end.
- Historical S&P 500 membership is stored as compact `[start_date, end_date)` intervals.
- Current constituents are never silently projected into the past.
- Price files fail validation when histories are implausibly short, duplicated or internally inconsistent.
- Input files receive SHA-256 fingerprints in each experiment manifest.
- Every parameter configuration gets a deterministic variant ID.

## Quality model

The absolute quality gate currently requires:

- ROE >= 12%
- FCF margin >= 5%
- debt/equity <= 1.5
- current ratio >= 1.0
- historical market cap >= $5B

SEC processing also derives, when available:

- ROA
- operating margin
- FCF / net income
- asset turnover
- net debt / equity

At each trading date the latest available fundamentals for all stocks are ranked cross-sectionally. The strategy can require a minimum quality percentile in addition to the absolute gate.

## Quality-dip research grid

The default sweep now runs **384 variants**:

- drop: -5%, -10%, -15%, -20%
- wait: 0, 1, 2 trading days
- hold: 5, 10, 20, 60 trading days
- minimum quality percentile: 0%, 50%, 70%, 80%
- stabilization filter: off/on

A stabilization-enabled trade requires the next completed session to close above the crash-day close and, when low data exists, not make a lower low. Entry therefore cannot occur before the following session open.

## Robustness instead of cherry-picking

The sweep uses one chronological split shared by all variants:

- first 60%: train
- next 20%: validation
- final 20%: out-of-sample

Each variant receives:

- trade-level metrics for train/validation/OOS
- bootstrap 95% CI for OOS mean return
- parameter-neighbor stability
- a constrained $10,000 portfolio simulation
- OOS portfolio return, Sharpe and max drawdown

The leaderboard ranks a **robustness score**, not the best in-sample Sharpe. Neighboring parameter settings and OOS evidence are deliberately rewarded.

## Portfolio assumptions

Default research account:

- initial capital: $10,000
- maximum positions: 10
- maximum allocation per position: 10%
- no overlapping positions in the same ticker

The simulator marks active positions to daily close and realizes the strategy's configured round-trip cost at exit.

## Real-data pipeline

```bash
python scripts/download_real_data.py \
  --tickers AAPL,MSFT,GOOGL,AMZN,META,NVDA,JPM,COST,HD,NKE \
  --start 2000-01-01

python scripts/build_universe.py \
  --history data/real/raw/sp500_historical_components.csv \
  --output data/real/universe_intervals.csv

python scripts/build_sec_fundamentals.py \
  --prices data/real/prices.csv \
  --output data/real/fundamentals.csv \
  --audit data/real/fundamentals_audit.csv

python scripts/validate_research_data.py \
  --prices data/real/prices.csv \
  --fundamentals data/real/fundamentals.csv \
  --universe data/real/universe_intervals.csv \
  --report data/real/validation_full.json

python scripts/run_quality_dip_sweep.py \
  --prices data/real/prices.csv \
  --fundamentals data/real/fundamentals.csv \
  --universe data/real/universe_intervals.csv \
  --output results/quality_dip_real
```

## Research outputs

`results/quality_dip_real/` contains:

- `parameter_sweep.csv` — every tested configuration
- `leaderboard.csv` — top robust variants
- `robustness_report.md` — human-readable research summary
- `experiment_manifest.json` — git commit, parameters and SHA-256 data fingerprints
- `heatmap_oos_return.csv` — median OOS return parameter surface
- `heatmap_oos_portfolio_sharpe.csv` — median portfolio Sharpe surface
- `top_variant_trades.csv` — trades for the leading configurations

## Data sources

- Daily OHLCV: Stooq with Yahoo chart API fallback
- Historical S&P 500 composition: `hanshof/sp500_constituents`
- Point-in-time fundamentals: SEC EDGAR Company Facts
- Earnings sample: Bloomberg-derived 2016 sample from `pingfcc99/Earnings-surprise-on-stock-price`

The current automated workflow starts with ten large stocks to validate the full pipeline. That is **not** a survivorship-bias-free final research universe. The historical membership layer is already in place for expansion to the full historical S&P 500 set, but delisted-security price availability still needs explicit coverage before broad results should be treated as definitive.

## Tests

```bash
pytest -q
```

Tests cover point-in-time filing use, historical universe intervals, data validation, stabilization timing, portfolio position limits and deterministic experiment IDs.

## Disclaimer

Research/educational software only. Backtests can be wrong even when code runs successfully; data provenance, survivorship bias, execution assumptions and out-of-sample stability matter more than a single attractive headline return.
