# Frozen winners — independent 2026 holdout

Price source: **Yahoo adjusted daily OHLC**, independent of the FINSABER source used to select the rules.
Scored period: **2026-01-01 through 2026-08-28**. No 2026 parameter is used for tuning.

| Strategy | Return | CAGR | Sharpe | Max DD | Trades | 1x/10 return | 1x/10 Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_relative_weakness_reversal_v2 | -2.82% | -4.27% | -0.325 | -7.04% | 12 | -0.42% | -0.325 |
| frozen_gap_down_reversal_v2 | -20.58% | -29.66% | -1.402 | -23.46% | 42 | -4.36% | -1.537 |

## Interpretation guardrails

- These rules were frozen before this 2026 test.
- 2026 is evaluated once; do not retune these parameters using this result.
- Yahoo is a new price source, which reduces dependence on FINSABER-specific adjustments.
- Daily bars still approximate execution at the open and target touch; real intraday bid/ask and slippage are not modeled.
