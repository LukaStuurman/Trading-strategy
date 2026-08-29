# Quality-dip robustness report

Experiment: `d907921c8b41cd9d`
Train through: **2015-12-24**; validation through: **2021-04-26**; later trades are OOS.
Grid: **384 variants** = 4 drops × 3 waits × 4 holds × 4 quality percentiles × stabilization on/off.

The ranking is deliberately not the best in-sample Sharpe. It rewards OOS performance and parameter neighborhoods.

## Top robust variants

| variant_id | drop_threshold | wait_days | hold_days | min_quality_percentile | require_stabilization | oos_trades | oos_avg_return | oos_sharpe | oos_ci_low | oos_ci_high | oos_portfolio_total_return | oos_portfolio_sharpe | oos_portfolio_max_drawdown | neighbor_positive_fraction | robustness_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 436c1319ceb4 | -0.0500 | 2 | 20 | 0.0000 | False | 68 | 0.0374 | 3.0034 | 0.0177 | 0.0611 | 0.0503 | 0.2469 | -0.0788 | 1.0000 | 0.9643 |
| 753d3c3a53b3 | -0.0500 | 2 | 20 | 0.5000 | False | 63 | 0.0359 | 2.6948 | 0.0107 | 0.0606 | 0.0320 | 0.1704 | -0.0957 | 1.0000 | 0.9625 |
| 6a944dda46dc | -0.0500 | 2 | 60 | 0.5000 | False | 57 | 0.1185 | 4.0405 | 0.0654 | 0.1765 | 0.0998 | 0.3837 | -0.0716 | 0.8333 | 0.9607 |
| e4315cbd9379 | -0.0500 | 1 | 60 | 0.5000 | False | 57 | 0.1091 | 3.5221 | 0.0499 | 0.1637 | 0.0797 | 0.3196 | -0.0821 | 0.8571 | 0.9602 |
| f06d1b9ccbf5 | -0.0500 | 0 | 60 | 0.5000 | False | 58 | 0.1133 | 3.6631 | 0.0557 | 0.1641 | 0.0782 | 0.3182 | -0.0806 | 0.8333 | 0.9576 |
| 747d3d0318fe | -0.0500 | 1 | 60 | 0.0000 | False | 62 | 0.1131 | 3.9158 | 0.0554 | 0.1729 | 0.1395 | 0.4970 | -0.0763 | 0.8333 | 0.9573 |
| 0a62007297e5 | -0.0500 | 0 | 20 | 0.0000 | False | 68 | 0.0336 | 2.4748 | 0.0093 | 0.0572 | 0.0450 | 0.2235 | -0.0899 | 1.0000 | 0.9568 |
| ac386fee1cc2 | -0.0500 | 1 | 20 | 0.0000 | False | 68 | 0.0330 | 2.4696 | 0.0067 | 0.0571 | 0.0493 | 0.2325 | -0.0779 | 1.0000 | 0.9555 |
| 8a16e2137ddf | -0.0500 | 2 | 60 | 0.0000 | False | 62 | 0.1227 | 4.4964 | 0.0680 | 0.1733 | 0.1716 | 0.5845 | -0.0716 | 0.8000 | 0.9553 |
| 4e3bc0aec66c | -0.0500 | 0 | 60 | 0.0000 | False | 63 | 0.1168 | 4.0401 | 0.0638 | 0.1670 | 0.1379 | 0.4943 | -0.0767 | 0.8000 | 0.9527 |

## Interpretation guardrails

- `oos_ci_low > 0` is stronger evidence than a positive point estimate alone.
- Neighbor stability matters: an isolated winning cell is treated as fragile.
- Portfolio metrics enforce capital limits and overlapping-position constraints.
- A historical universe filters signal eligibility; it does not magically restore unavailable delisted price histories.
