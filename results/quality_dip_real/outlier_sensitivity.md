# Extreme-price-move sensitivity

Threshold: absolute close-to-close adjusted return > **300%**.
Detected price outlier rows: **69**.
OOS entries start: **2023-05-20**.

The primary backtest is not altered. OOS membership follows entry date, matching the primary split. The stress case removes a trade only when an extreme-move date falls between its signal date and exit date, inclusive.

## Validation-selected variants

| leaderboard_rank | variant_id | oos_trades | oos_crossing_extreme_move_trades | oos_avg_return | clean_oos_avg_return | oos_portfolio_total_return | clean_oos_portfolio_total_return | oos_portfolio_max_drawdown | clean_oos_portfolio_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 22c44ea4dbb6 | 92 | 0 | 0.0692 | 0.0692 | 0.0791 | 0.0791 | -0.2431 | -0.2431 |
| 2 | 59a060ade294 | 82 | 0 | 0.0734 | 0.0734 | 0.0732 | 0.0732 | -0.2431 | -0.2431 |
| 3 | 425bb9e15b68 | 100 | 0 | 0.0263 | 0.0263 | 0.0956 | 0.0956 | -0.1449 | -0.1449 |
| 4 | c48255f35be3 | 46 | 0 | 0.0086 | 0.0086 | 0.0042 | 0.0042 | -0.1005 | -0.1005 |
| 5 | f2526719447b | 90 | 0 | 0.0258 | 0.0258 | 0.0807 | 0.0807 | -0.1449 | -0.1449 |
| 6 | 7b972fdcd1af | 46 | 0 | 0.0030 | 0.0030 | -0.0304 | -0.0304 | -0.1102 | -0.1102 |
| 7 | 19190144c1bf | 46 | 0 | -0.0080 | -0.0080 | -0.0650 | -0.0650 | -0.1636 | -0.1636 |
| 8 | 387f28f5dfb4 | 46 | 0 | -0.0243 | -0.0243 | -0.1220 | -0.1220 | -0.1423 | -0.1423 |
| 9 | 40fa69c97a44 | 92 | 0 | 0.0518 | 0.0518 | 0.0515 | 0.0515 | -0.2569 | -0.2569 |
| 10 | 5062e7e97823 | 46 | 0 | -0.0177 | -0.0177 | -0.0935 | -0.0935 | -0.1097 | -0.1097 |
| 11 | 9a6c66f873e5 | 92 | 0 | 0.0604 | 0.0604 | 0.0301 | 0.0301 | -0.2459 | -0.2459 |
| 12 | 48b69f980f3b | 82 | 0 | 0.0685 | 0.0685 | 0.0662 | 0.0662 | -0.2459 | -0.2459 |
| 13 | cc6b7037cfb9 | 82 | 0 | 0.0593 | 0.0593 | 0.0848 | 0.0848 | -0.2569 | -0.2569 |
| 14 | ed4ac4e65bc9 | 234 | 0 | 0.0565 | 0.0565 | 0.3254 | 0.3254 | -0.2963 | -0.2963 |
| 15 | 4e3bc0aec66c | 626 | 0 | 0.0913 | 0.0913 | -0.1308 | -0.1308 | -0.3759 | -0.3759 |
| 16 | 999457ba2b01 | 100 | 0 | 0.0193 | 0.0193 | 0.0376 | 0.0376 | -0.2070 | -0.2070 |
| 17 | f06d1b9ccbf5 | 567 | 0 | 0.0959 | 0.0959 | -0.1162 | -0.1162 | -0.3933 | -0.3933 |
| 18 | 8a16e2137ddf | 625 | 0 | 0.0805 | 0.0805 | -0.1189 | -0.1189 | -0.3688 | -0.3688 |
| 19 | 6a944dda46dc | 566 | 0 | 0.0847 | 0.0847 | -0.1367 | -0.1367 | -0.3992 | -0.3992 |
| 20 | 8974cd1eaea5 | 100 | 0 | 0.0205 | 0.0205 | 0.0685 | 0.0685 | -0.1472 | -0.1472 |

## Reading this stress test

- Similar baseline and clean metrics indicate the selected signal is not being driven by extreme source moves.
- A large deterioration after exclusion is a warning to inspect the named rows in `price_outliers.csv` before trusting the strategy.
- This is a data-quality stress test, not a claim that every >300% move is erroneous.
