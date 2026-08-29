# Quality-dip robustness report

Experiment: `64ba9742a6b4f70a`
Train through: **2020-10-12**; validation through: **2023-05-19**; later entries are OOS.
Split source: **fixed_cli_dates**. Train/validation trades crossing a cutoff are purged until their outcomes are fully observable.
Grid: **384 variants** = 4 drops × 3 waits × 4 holds × 4 quality percentiles × stabilization on/off.

Variant ranking is determined strictly from train + validation. OOS return, Sharpe, CI and portfolio results are reported only after that ranking is fixed.

## Top validation-selected variants and untouched OOS results

| variant_id | drop_threshold | wait_days | hold_days | min_quality_percentile | require_stabilization | validation_trades | validation_avg_return | validation_sharpe | validation_ci_low | validation_ci_high | neighbor_validation_positive_fraction | selection_score | oos_trades | oos_avg_return | oos_sharpe | oos_ci_low | oos_ci_high | oos_portfolio_total_return | oos_portfolio_sharpe | oos_portfolio_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22c44ea4dbb6 | -0.1000 | 0 | 60 | 0.0000 | False | 59 | 0.0889 | 2.5966 | 0.0282 | 0.1582 | 1.0000 | 0.9633 | 92 | 0.0692 | 2.7708 | 0.0258 | 0.1128 | 0.0791 | 0.2561 | -0.2431 |
| 59a060ade294 | -0.1000 | 0 | 60 | 0.5000 | False | 54 | 0.0946 | 2.5422 | 0.0164 | 0.1817 | 1.0000 | 0.9630 | 82 | 0.0734 | 2.6501 | 0.0215 | 0.1306 | 0.0732 | 0.2468 | -0.2431 |
| 425bb9e15b68 | -0.1000 | 0 | 20 | 0.0000 | False | 61 | 0.0694 | 2.8669 | 0.0275 | 0.1209 | 1.0000 | 0.9612 | 100 | 0.0263 | 1.9186 | 0.0004 | 0.0539 | 0.0956 | 0.3349 | -0.1449 |
| c48255f35be3 | -0.1000 | 0 | 20 | 0.7000 | False | 26 | 0.1319 | 2.8411 | 0.0484 | 0.2328 | 1.0000 | 0.9606 | 46 | 0.0086 | 0.4589 | -0.0262 | 0.0425 | 0.0042 | 0.0671 | -0.1005 |
| f2526719447b | -0.1000 | 0 | 20 | 0.5000 | False | 56 | 0.0681 | 2.6584 | 0.0175 | 0.1214 | 1.0000 | 0.9565 | 90 | 0.0258 | 1.7212 | -0.0033 | 0.0594 | 0.0807 | 0.3010 | -0.1449 |
| 7b972fdcd1af | -0.1000 | 1 | 20 | 0.7000 | False | 26 | 0.1104 | 2.7015 | 0.0387 | 0.1805 | 1.0000 | 0.9552 | 46 | 0.0030 | 0.1549 | -0.0303 | 0.0439 | -0.0304 | -0.1006 | -0.1102 |
| 19190144c1bf | -0.1000 | 2 | 20 | 0.7000 | False | 26 | 0.1162 | 2.4050 | 0.0286 | 0.2138 | 1.0000 | 0.9515 | 46 | -0.0080 | -0.4874 | -0.0388 | 0.0204 | -0.0650 | -0.2607 | -0.1636 |
| 387f28f5dfb4 | -0.1000 | 2 | 10 | 0.7000 | False | 28 | 0.0819 | 2.3064 | 0.0266 | 0.1558 | 1.0000 | 0.9501 | 46 | -0.0243 | -2.1353 | -0.0468 | -0.0014 | -0.1220 | -0.6931 | -0.1423 |
| 40fa69c97a44 | -0.1000 | 2 | 60 | 0.0000 | False | 57 | 0.0769 | 2.1700 | 0.0179 | 0.1448 | 1.0000 | 0.9482 | 92 | 0.0518 | 2.3047 | 0.0122 | 0.0935 | 0.0515 | 0.2000 | -0.2569 |
| 5062e7e97823 | -0.1000 | 1 | 10 | 0.7000 | False | 28 | 0.0687 | 2.4116 | 0.0160 | 0.1231 | 1.0000 | 0.9475 | 46 | -0.0177 | -1.4874 | -0.0410 | 0.0070 | -0.0935 | -0.5231 | -0.1097 |

## Interpretation guardrails

- OOS columns do not participate in `selection_score`; changing only OOS values cannot change the leaderboard order.
- `oos_ci_low > 0` is stronger post-selection evidence than a positive OOS point estimate alone.
- Neighbor stability is measured on validation performance, not OOS performance.
- Portfolio metrics enforce capital limits and overlapping-position constraints.
- FINSABER supplies broad historical/delisted prices, but quality results include only tickers with causal fundamental coverage; see `research_coverage.json`.
- Historical membership is enforced on the signal date; a price record alone never implies S&P 500 membership.
