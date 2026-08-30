# Leveraged intraday strategy sweep

Tested **6,656 signal/exit variants** from **416 base signals**, then **920 leverage/capacity policies** on the strongest train/validation candidates.
Costs: **20 bps round trip per notional**. All entries are at the open and all exits occur the same trading day.
Signals use only data known by the open. Daily OHLC cannot reveal whether a stop or target happened first, so simultaneous touches are scored as a stop.

Train: **2001-01-01–2018-12-31**; validation: **2019-01-01–2021-12-31**; OOS: **2022-01-01–2024-12-31**; final holdout: **2025-01-01–2025-12-31**.
Neither OOS nor holdout columns participate in selection.

## Best overall policies selected on train + validation

| policy_rank | family | leverage | max_positions | stop_loss | take_profit | policy_selection_score | validation_total_return | validation_sharpe | validation_max_drawdown | oos_total_return | oos_sharpe | oos_max_drawdown | holdout_total_return | holdout_sharpe | holdout_max_drawdown | holdout_trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | relative_weakness_reversal | 1.5000 | 5 | 0.0200 | nan | 0.8965 | 0.5267 | 0.8887 | -0.1065 | 0.0601 | 0.1970 | -0.1685 | 0.1173 | 0.7327 | -0.1062 | 84 |
| 2 | relative_weakness_reversal | 1.0000 | 5 | 0.0200 | nan | 0.8948 | 0.3386 | 0.8887 | -0.0719 | 0.0475 | 0.1970 | -0.1151 | 0.0799 | 0.7327 | -0.0717 | 84 |
| 3 | relative_weakness_reversal | 2.0000 | 5 | 0.0200 | nan | 0.8928 | 0.7255 | 0.8887 | -0.1402 | 0.0655 | 0.1970 | -0.2194 | 0.1526 | 0.7327 | -0.1397 | 84 |
| 4 | relative_weakness_reversal | 1.0000 | 3 | 0.0200 | nan | 0.8846 | 0.3929 | 0.8284 | -0.0956 | -0.0299 | -0.0219 | -0.1833 | 0.0892 | 0.6432 | -0.0895 | 72 |
| 5 | gap_down_reversal | 1.0000 | 10 | nan | 0.0800 | 0.8812 | 0.4370 | 0.8855 | -0.1064 | -0.4113 | -1.1823 | -0.4634 | 0.0065 | 0.1249 | -0.1369 | 531 |
| 6 | relative_weakness_reversal | 3.0000 | 5 | 0.0200 | nan | 0.8810 | 1.1477 | 0.8887 | -0.2048 | 0.0559 | 0.1970 | -0.3135 | 0.2170 | 0.7327 | -0.2040 | 84 |
| 7 | relative_weakness_reversal | 1.5000 | 3 | 0.0200 | nan | 0.8799 | 0.6084 | 0.8284 | -0.1411 | -0.0595 | -0.0219 | -0.2657 | 0.1282 | 0.6432 | -0.1320 | 72 |
| 8 | relative_weakness_reversal | 2.0000 | 3 | 0.0200 | nan | 0.8742 | 0.8321 | 0.8284 | -0.1850 | -0.0973 | -0.0219 | -0.3422 | 0.1631 | 0.6432 | -0.1731 | 72 |
| 9 | gap_down_reversal | 1.5000 | 10 | nan | 0.0800 | 0.8719 | 0.6813 | 0.8855 | -0.1614 | -0.5583 | -1.1823 | -0.6141 | -0.0018 | 0.1249 | -0.1992 | 531 |
| 10 | relative_weakness_reversal | 4.0000 | 5 | 0.0200 | nan | 0.8674 | 1.5871 | 0.8887 | -0.2658 | 0.0215 | 0.1970 | -0.3982 | 0.2722 | 0.7327 | -0.2647 | 84 |

## Best policy per strategy family

| policy_rank | family | leverage | max_positions | stop_loss | take_profit | policy_selection_score | validation_total_return | validation_sharpe | validation_max_drawdown | oos_total_return | oos_sharpe | oos_max_drawdown | holdout_total_return | holdout_sharpe | holdout_max_drawdown | holdout_trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | relative_weakness_reversal | 1.5000 | 5 | 0.0200 | nan | 0.8965 | 0.5267 | 0.8887 | -0.1065 | 0.0601 | 0.1970 | -0.1685 | 0.1173 | 0.7327 | -0.1062 | 84 |
| 5 | gap_down_reversal | 1.0000 | 10 | nan | 0.0800 | 0.8812 | 0.4370 | 0.8855 | -0.1064 | -0.4113 | -1.1823 | -0.4634 | 0.0065 | 0.1249 | -0.1369 | 531 |
| 53 | strength_pullback | 4.0000 | 20 | 0.0200 | 0.0800 | 0.8098 | 0.5009 | 0.7100 | -0.0874 | -0.0117 | -0.0728 | -0.0520 | -0.1835 | -1.7161 | -0.1992 | 66 |
| 353 | volatility_reversal | 1.0000 | 5 | nan | 0.0800 | 0.5887 | 0.2062 | 0.4032 | -0.2829 | -0.0682 | -0.0515 | -0.4091 | -0.0452 | -0.1667 | -0.1723 | 241 |
| 364 | oversold_3d_reversal | 1.0000 | 20 | nan | nan | 0.5692 | 0.1534 | 0.3822 | -0.1434 | -0.1156 | -0.6395 | -0.1433 | 0.0261 | 0.2253 | -0.0796 | 586 |
| 403 | prior_day_crash_rebound | 1.0000 | 20 | nan | nan | 0.5288 | -0.0173 | -0.0370 | -0.0892 | 0.0281 | 0.3412 | -0.0391 | 0.0045 | 0.1078 | -0.0352 | 250 |
| 447 | oversold_5d_reversal | 1.0000 | 20 | nan | nan | 0.4911 | -0.0394 | -0.0741 | -0.1336 | 0.0284 | 0.2303 | -0.0636 | 0.1228 | 0.7147 | -0.0633 | 245 |
| 531 | gap_up_continuation | 1.0000 | 3 | nan | 0.0200 | 0.4189 | -0.0145 | -0.0718 | -0.1109 | 0.0070 | 0.1159 | -0.0269 | -0.0111 | -0.3184 | -0.0389 | 9 |

## Guardrails

- Leverage multiplies both profit/loss and transaction cost exposure; unused position slots remain cash.
- A portfolio day at or below -100% is treated as a blow-up and equity stays at zero.
- Historical S&P membership is enforced on each trade date to reduce survivorship bias.
- Same-day stop/target ordering is unknowable from daily bars; stop-first makes the test conservative but intraday bars are still required before live use.
