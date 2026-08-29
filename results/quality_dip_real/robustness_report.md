# Quality-dip robustness report

Experiment: `2e0b8fd83bd20342`
Train through: **2020-10-12**; validation through: **2023-05-19**; later trades are OOS.
Split source: **fixed_cli_dates**.
Grid: **384 variants** = 4 drops × 3 waits × 4 holds × 4 quality percentiles × stabilization on/off.

The ranking is deliberately not the best in-sample Sharpe. It rewards OOS performance and parameter neighborhoods.

## Top robust variants

| variant_id | drop_threshold | wait_days | hold_days | min_quality_percentile | require_stabilization | oos_trades | oos_avg_return | oos_sharpe | oos_ci_low | oos_ci_high | oos_portfolio_total_return | oos_portfolio_sharpe | oos_portfolio_max_drawdown | neighbor_positive_fraction | robustness_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 753d3c3a53b3 | -0.0500 | 2 | 20 | 0.5000 | False | 60 | 0.0288 | 1.9392 | 0.0040 | 0.0616 | 0.0398 | 0.2668 | -0.0928 | 1.0000 | 0.9803 |
| f06d1b9ccbf5 | -0.0500 | 0 | 60 | 0.5000 | False | 55 | 0.0784 | 2.7109 | 0.0298 | 0.1279 | 0.0536 | 0.2785 | -0.1454 | 0.8333 | 0.9645 |
| 747d3d0318fe | -0.0500 | 1 | 60 | 0.0000 | False | 55 | 0.0641 | 2.2740 | 0.0055 | 0.1222 | 0.0410 | 0.2198 | -0.1377 | 0.8333 | 0.9595 |
| 4e3bc0aec66c | -0.0500 | 0 | 60 | 0.0000 | False | 55 | 0.0784 | 2.7109 | 0.0287 | 0.1325 | 0.0536 | 0.2785 | -0.1454 | 0.8000 | 0.9578 |
| 6480b7b974fa | -0.0500 | 0 | 20 | 0.5000 | False | 60 | 0.0349 | 2.1724 | 0.0028 | 0.0617 | 0.0420 | 0.2859 | -0.0805 | 0.8571 | 0.9544 |
| 8a16e2137ddf | -0.0500 | 2 | 60 | 0.0000 | False | 55 | 0.0607 | 2.2044 | 0.0100 | 0.1118 | 0.0309 | 0.1803 | -0.1565 | 0.8000 | 0.9502 |
| 0a62007297e5 | -0.0500 | 0 | 20 | 0.0000 | False | 60 | 0.0349 | 2.1724 | 0.0066 | 0.0667 | 0.0420 | 0.2859 | -0.0805 | 0.8333 | 0.9496 |
| e4315cbd9379 | -0.0500 | 1 | 60 | 0.5000 | False | 55 | 0.0641 | 2.2740 | 0.0083 | 0.1138 | 0.0410 | 0.2198 | -0.1377 | 0.7143 | 0.9357 |
| 436c1319ceb4 | -0.0500 | 2 | 20 | 0.0000 | False | 60 | 0.0288 | 1.9392 | -0.0010 | 0.0578 | 0.0398 | 0.2668 | -0.0928 | 1.0000 | 0.9303 |
| 6a944dda46dc | -0.0500 | 2 | 60 | 0.5000 | False | 55 | 0.0607 | 2.2044 | 0.0087 | 0.1133 | 0.0309 | 0.1803 | -0.1565 | 0.6667 | 0.9236 |

## Interpretation guardrails

- `oos_ci_low > 0` is stronger evidence than a positive point estimate alone.
- Neighbor stability matters: an isolated winning cell is treated as fragile.
- Portfolio metrics enforce capital limits and overlapping-position constraints.
- FINSABER supplies broad historical/delisted prices, but quality results include only tickers with causal fundamental coverage; see `research_coverage.json`.
- Historical membership is still enforced on the signal date; a price record alone never implies S&P 500 membership.
