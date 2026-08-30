# Recent-year intraday refinement

Focused on the three families that remained useful in the previous 2025 study. Tested **31,500 signal/exit variants** from **1,050 focused base configs**, followed by **1,848 leverage/capacity policies**.

Only recent years matter here: 2021-2023 train, 2024 validation, 2025 diagnostic. 2025 never participates in ranking because the seed families were already chosen with knowledge of their 2025 behavior.
Execution costs remain **20 bps round trip per notional** and all trades open and close the same day.

## Best recent-regime policies

| policy_rank | family | leverage | max_positions | stop_loss | take_profit | policy_selection_score | y2021_return | y2022_return | y2023_return | y2024_return | validation_sharpe | validation_max_drawdown | y2025_return | diagnostic_2025_sharpe | diagnostic_2025_max_drawdown | diagnostic_2025_trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | gap_down_reversal | 3.0000 | 7 | nan | 0.0600 | 0.7677 | 0.1354 | 0.2071 | 0.2432 | 0.1678 | 1.0293 | -0.0695 | 0.6448 | 2.0091 | -0.0498 | 49 |
| 2 | relative_weakness_reversal | 2.5000 | 5 | nan | 0.0800 | 0.7664 | 0.1573 | 0.3321 | 0.2092 | 0.1437 | 0.9814 | -0.0691 | 0.7605 | 2.1509 | -0.0497 | 40 |
| 3 | gap_down_reversal | 2.5000 | 7 | nan | 0.0600 | 0.7658 | 0.1132 | 0.1774 | 0.2012 | 0.1400 | 1.0293 | -0.0580 | 0.5204 | 2.0091 | -0.0416 | 49 |
| 4 | relative_weakness_reversal | 3.0000 | 5 | nan | 0.0800 | 0.7656 | 0.1877 | 0.3944 | 0.2510 | 0.1719 | 0.9814 | -0.0827 | 0.9552 | 2.1509 | -0.0596 | 40 |
| 5 | gap_down_reversal | 3.0000 | 10 | nan | 0.0600 | 0.7632 | 0.1042 | 0.2232 | 0.1189 | 0.1524 | 1.0703 | -0.0488 | 0.3851 | 1.5242 | -0.0595 | 55 |
| 6 | gap_down_reversal | 2.5000 | 10 | nan | 0.0600 | 0.7590 | 0.0868 | 0.1886 | 0.0992 | 0.1269 | 1.0703 | -0.0407 | 0.3162 | 1.5242 | -0.0496 | 55 |
| 7 | gap_down_reversal | 3.0000 | 7 | nan | nan | 0.7588 | 0.1527 | 0.4695 | 0.3441 | 0.1709 | 0.9443 | -0.0791 | 0.9544 | 1.8710 | -0.0498 | 49 |
| 8 | relative_weakness_reversal | 2.0000 | 5 | nan | 0.0800 | 0.7588 | 0.1265 | 0.2679 | 0.1673 | 0.1152 | 0.9814 | -0.0555 | 0.5811 | 2.1509 | -0.0398 | 40 |
| 9 | gap_down_reversal | 2.0000 | 7 | nan | 0.0600 | 0.7585 | 0.0908 | 0.1455 | 0.1596 | 0.1120 | 1.0293 | -0.0465 | 0.4031 | 2.0091 | -0.0334 | 49 |
| 10 | relative_weakness_reversal | 3.0000 | 10 | nan | 0.0800 | 0.7570 | 0.1220 | 0.2902 | 0.0862 | 0.1484 | 1.0802 | -0.0418 | 0.4257 | 1.6147 | -0.0374 | 50 |
| 11 | gap_down_reversal | 2.5000 | 7 | nan | nan | 0.7570 | 0.1276 | 0.3938 | 0.2905 | 0.1429 | 0.9443 | -0.0661 | 0.7630 | 1.8710 | -0.0416 | 49 |
| 12 | relative_weakness_reversal | 3.0000 | 7 | nan | 0.0800 | 0.7567 | 0.1523 | 0.3242 | 0.1367 | 0.1477 | 1.0311 | -0.0594 | 0.5624 | 1.8575 | -0.0427 | 44 |
| 13 | relative_weakness_reversal | 2.5000 | 7 | nan | 0.0800 | 0.7544 | 0.1270 | 0.2730 | 0.1148 | 0.1231 | 1.0311 | -0.0496 | 0.4563 | 1.8575 | -0.0356 | 44 |
| 14 | gap_down_reversal | 3.0000 | 5 | nan | 0.0600 | 0.7540 | 0.1728 | 0.1912 | 0.4173 | 0.1862 | 0.9648 | -0.0969 | 0.7809 | 1.9097 | -0.0693 | 45 |
| 15 | gap_down_reversal | 3.0000 | 7 | nan | 0.0800 | 0.7537 | 0.1320 | 0.3070 | 0.2569 | 0.1545 | 0.9433 | -0.0791 | 0.7171 | 2.0038 | -0.0498 | 49 |

## Best per seed family

| policy_rank | family | leverage | max_positions | stop_loss | take_profit | policy_selection_score | y2021_return | y2022_return | y2023_return | y2024_return | validation_sharpe | validation_max_drawdown | y2025_return | diagnostic_2025_sharpe | diagnostic_2025_max_drawdown | diagnostic_2025_trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | gap_down_reversal | 3.0000 | 7 | nan | 0.0600 | 0.7677 | 0.1354 | 0.2071 | 0.2432 | 0.1678 | 1.0293 | -0.0695 | 0.6448 | 2.0091 | -0.0498 | 49 |
| 2 | relative_weakness_reversal | 2.5000 | 5 | nan | 0.0800 | 0.7664 | 0.1573 | 0.3321 | 0.2092 | 0.1437 | 0.9814 | -0.0691 | 0.7605 | 2.1509 | -0.0497 | 40 |
| 1338 | oversold_5d_reversal | 1.0000 | 20 | nan | 0.0800 | 0.3155 | -0.0406 | 0.0254 | 0.0302 | 0.0122 | 0.7083 | -0.0112 | 0.1054 | 1.1247 | -0.0247 | 163 |

## Interpretation guardrails

- Ranking uses 2021-2024 only; 2025 is diagnostic, not a clean holdout.
- The score rewards Sharpe, 2024 CAGR/drawdown, train-validation stability, positive-year share and the worst pre-2025 year.
- Historical S&P membership, lagged liquidity/momentum and open-time signals remain enforced.
- Daily bars still cannot reveal exact intraday stop/target ordering; simultaneous touches are scored stop-first.
