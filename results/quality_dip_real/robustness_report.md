# Quality-dip robustness report

Experiment: `4c32970901a4dbeb`
Train through: **2020-10-12**; validation through: **2023-05-19**; later trades are OOS.
Grid: **384 variants** = 4 drops × 3 waits × 4 holds × 4 quality percentiles × stabilization on/off.

The ranking is deliberately not the best in-sample Sharpe. It rewards OOS performance and parameter neighborhoods.

## Top robust variants

| variant_id | drop_threshold | wait_days | hold_days | min_quality_percentile | require_stabilization | oos_trades | oos_avg_return | oos_sharpe | oos_ci_low | oos_ci_high | oos_portfolio_total_return | oos_portfolio_sharpe | oos_portfolio_max_drawdown | neighbor_positive_fraction | robustness_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4e3bc0aec66c | -0.0500 | 0 | 60 | 0.0000 | False | 762 | 0.1013 | 12.7865 | 0.0853 | 0.1153 | 0.3155 | 0.6355 | -0.3134 | 1.0000 | 0.9914 |
| 8a16e2137ddf | -0.0500 | 2 | 60 | 0.0000 | False | 761 | 0.0944 | 12.4373 | 0.0796 | 0.1096 | 0.2314 | 0.5061 | -0.3136 | 1.0000 | 0.9885 |
| f06d1b9ccbf5 | -0.0500 | 0 | 60 | 0.5000 | False | 629 | 0.0945 | 10.7577 | 0.0768 | 0.1109 | 0.2582 | 0.5628 | -0.3273 | 1.0000 | 0.9878 |
| 747d3d0318fe | -0.0500 | 1 | 60 | 0.0000 | False | 762 | 0.0928 | 12.2612 | 0.0794 | 0.1078 | 0.2030 | 0.4513 | -0.3153 | 1.0000 | 0.9833 |
| de616963c744 | -0.0500 | 0 | 60 | 0.7000 | False | 212 | 0.1026 | 6.1514 | 0.0688 | 0.1364 | 0.2366 | 0.5376 | -0.2931 | 1.0000 | 0.9820 |
| 9a6c66f873e5 | -0.1000 | 1 | 60 | 0.0000 | False | 123 | 0.1058 | 4.4393 | 0.0570 | 0.1563 | 0.1916 | 0.4864 | -0.2155 | 1.0000 | 0.9779 |
| 22c44ea4dbb6 | -0.1000 | 0 | 60 | 0.0000 | False | 123 | 0.1053 | 4.4011 | 0.0611 | 0.1511 | 0.0890 | 0.2721 | -0.2395 | 1.0000 | 0.9755 |
| 6a944dda46dc | -0.0500 | 2 | 60 | 0.5000 | False | 628 | 0.0870 | 10.3681 | 0.0686 | 0.1024 | 0.2271 | 0.5105 | -0.3157 | 1.0000 | 0.9740 |
| 40fa69c97a44 | -0.1000 | 2 | 60 | 0.0000 | False | 123 | 0.0979 | 4.4140 | 0.0501 | 0.1349 | 0.1605 | 0.4131 | -0.2356 | 1.0000 | 0.9719 |
| e4315cbd9379 | -0.0500 | 1 | 60 | 0.5000 | False | 629 | 0.0847 | 10.1206 | 0.0661 | 0.1018 | 0.2397 | 0.5201 | -0.3182 | 1.0000 | 0.9708 |

## Interpretation guardrails

- `oos_ci_low > 0` is stronger evidence than a positive point estimate alone.
- Neighbor stability matters: an isolated winning cell is treated as fragile.
- Portfolio metrics enforce capital limits and overlapping-position constraints.
- FINSABER supplies broad historical/delisted prices, but quality results include only tickers with causal fundamental coverage; see `research_coverage.json`.
- Historical membership is still enforced on the signal date; a price record alone never implies S&P 500 membership.
