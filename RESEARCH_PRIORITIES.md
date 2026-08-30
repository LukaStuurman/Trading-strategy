# Research priorities

Updated: 2026-08-30

This roadmap shifts the project away from trying to out-speed HFTs and toward research that can realistically exploit the footprints of large algorithmic/institutional execution. The core idea is to detect persistent meta-order flow, distinguish temporary liquidity pressure from informed price discovery, and test strategies with execution costs that reflect adverse selection and market impact.

## P0 — Intraday microstructure data and execution realism

**Highest priority. Do this before another large parameter sweep or higher-leverage search.**

Build a recent intraday dataset with at least 1-minute or 5-minute OHLCV and, where obtainable, quotes / NBBO, spread, trade direction, auction information and depth/order-book snapshots.

Required research fields/features:

- intraday volume relative to the normal time-of-day volume curve;
- signed-volume or buy/sell imbalance proxies;
- bid-ask spread and spread percentile;
- realized volatility and short-horizon volatility shock;
- price impact per unit of volume;
- distance from VWAP / session VWAP trajectory;
- market- and sector-relative return/flow;
- opening/closing auction imbalance when available;
- liquidity/depth proxies and recovery after a shock.

Upgrade the backtester so fills are not assumed at frictionless bar prices. Model at minimum:

- half/full spread depending on order type;
- slippage increasing with volatility and participation rate;
- market-impact penalty as trade size rises relative to volume;
- conservative same-bar stop/target ordering;
- delayed or failed fills for limit orders;
- opening-auction execution separately from continuous trading.

**Reason:** latency-arbitrage races occur on microsecond horizons and are concentrated among a few specialized firms, so competing directly on speed is not a realistic edge for this project. The exploitable research space is instead slower institutional execution and the liquidity response around it.

## P1 — Detect large algorithmic/meta-order footprints

Create a reusable `institutional_flow` feature layer that estimates whether a stock is likely experiencing a large parent order split into many child trades.

Prioritize signals such as:

1. persistent same-direction signed volume across consecutive intraday bars;
2. abnormal participation rate versus the stock's normal intraday volume curve;
3. monotonic VWAP drift with repeated shallow pullbacks;
4. elevated price impact with flow persistence;
5. repeated volume bursts without proportional news/market movement;
6. sector-correlated flow versus stock-specific flow;
7. post-shock liquidity recovery speed.

Do **not** treat trade size alone as an institutional identifier. Large algorithms deliberately split orders, and both unusually large and unusually small trades can contain institutional information.

Output a causal `metaorder_score` / `flow_persistence_score` available only from information known at each decision timestamp.

## P2 — Test two competing strategy families: continuation vs liquidity reversion

Large algorithmic orders can generate either persistent information-driven impact or temporary liquidity pressure. The research engine should explicitly test both hypotheses rather than assuming every large move mean-reverts.

### A. Persistent-flow continuation

Enter in the direction of suspected meta-order flow when:

- flow persistence remains high;
- price impact does not decay after short pauses;
- market/sector confirmation is present;
- spreads are not abnormally wide;
- liquidity is sufficient for realistic execution.

Exit when flow persistence breaks, impact decays, VWAP structure fails, or by end of day.

### B. Temporary-impact / forced-flow reversal

Fade a large move only when evidence suggests non-informational liquidity pressure:

- extreme signed-flow/volume shock;
- price moves much more than sector/market peers;
- flow intensity starts falling;
- spread/depth begins to normalize;
- price impact decays rather than propagates;
- no fresh information signal supports continuation.

This family is the microstructure-aware successor to the current gap-down / relative-weakness reversal ideas.

## P3 — Adverse-selection and “do-not-trade” filters

Build filters that identify periods where automated liquidity providers are likely to have an informational advantage.

Potential no-trade / size-down conditions:

- spread in extreme percentile;
- volatility shock plus persistent one-sided flow;
- immediate cross-asset/sector repricing;
- repeated failed reversals while flow remains one-sided;
- unusually high impact per unit volume;
- opening minutes around major public information events;
- low liquidity combined with large intended participation rate.

The goal is not only to find entry signals; avoiding toxic liquidity can improve strategy quality materially.

## P4 — Market-impact-aware position sizing before leverage optimization

Replace leverage-first optimization with liquidity-aware sizing.

For every candidate trade calculate position size from:

- account risk budget;
- recent volatility;
- spread;
- expected intraday volume;
- intended participation rate;
- estimated market impact.

Use a concave impact model as a baseline rather than assuming costs scale linearly with size. Keep 1x–1.5x leverage as the primary research range until execution-aware tests show that extra leverage improves *net* risk-adjusted return. Higher leverage should be a downstream stress test, not a source of apparent alpha.

## P5 — Validation protocol for the new microstructure strategies

Keep strategy discovery and proof separate.

Required protocol:

1. freeze feature definitions before inspecting the final holdout;
2. use rolling / walk-forward train and validation windows;
3. rank on net returns after spread, slippage and impact;
4. report performance by calendar year and market regime;
5. require positive results across parameter neighbors, not one optimum;
6. test sensitivity to 2x and 3x assumed transaction costs;
7. evaluate capacity by increasing participation rate;
8. keep a genuinely untouched future period for final confirmation;
9. never reclassify a previously inspected year as pristine OOS.

The existing 2025 results remain diagnostic because they have already influenced which strategy families are considered promising.

## P6 — Cross-asset and cross-sectional confirmation

After P0–P5 work reliably, add signals that large automated traders can process rapidly across related securities:

- sector ETF versus constituent lead/lag;
- index future / ETF shock versus stock response;
- peer-stock flow confirmation;
- market-wide order-flow state.

Use these primarily as confirmation/regime features, not as a latency-arbitrage strategy.

## Lower priority / pause for now

Until the microstructure layer is built, deprioritize:

- very large brute-force sweeps on daily OHLC-only strategies;
- searching mainly for higher leverage;
- sub-second/HFT latency strategies;
- adding many indicators that do not represent flow, liquidity or execution;
- optimizing on 2025 because it is already diagnostic rather than untouched.

## Implementation order

1. acquire recent intraday data and define a reproducible schema;
2. build execution-cost and fill model;
3. implement institutional-flow/meta-order features;
4. implement continuation, reversal and no-trade families;
5. run small causal sanity tests;
6. run walk-forward parameter sweeps;
7. perform cost/capacity stress tests;
8. freeze winners;
9. test on untouched future data.

## Scientific basis

The priority order is motivated by market-microstructure research rather than by the previous backtest winners alone:

- **Baron, Brogaard, Hagströmer & Kirilenko — High-Frequency Market Making to Large Institutional Trades (Review of Financial Studies, 2019):** HFT behavior changes around large institutional trades; HFTs trade more in the same direction and mean-revert inventory faster, while large informed traders can face higher transaction costs.
- **Budish, Cramton, Shim et al. — Quantifying the High-Frequency Trading Arms Race (Quarterly Journal of Economics, 2022):** latency races are extremely fast and concentrated among a small number of firms, supporting the decision not to compete on speed.
- **Li, Wang & Ye — Who Provides Liquidity, and When? (NBER Working Paper 25972):** institutional execution algorithms and HFTs strategically interact in liquidity provision; order type and tick/liquidity conditions matter.
- **Campbell, Ramadorai & Vuolteenaho — Caught on Tape: Institutional Order Flow and Stock Returns (Journal of Financial Economics, 2009):** inferred institutional flow contains return-predictive information and institutional selling can create liquidity-provision opportunities.
- **Boulatov, Hendershott & Livdan — Informed Trading and Portfolio Returns (Review of Economic Studies, 2013):** informed order flow can positively predict future returns and future informed flow, supporting a continuation branch alongside reversal.
- **Order splitting and interacting with a counterparty (Journal of Financial Markets, 2023/2024):** institutions split parent orders and their interaction with natural counterparties can substantially change execution cost, supporting explicit meta-order persistence and liquidity modeling.
- **Foucault, Pagano & Röell — Market Liquidity: Theory, Evidence, and Policy (Oxford, 2023):** order flow mixes information and noise; large trades can cause both persistent price discovery and temporary deviations, making regime classification central.

## Success criterion

A strategy is ready for serious shadow testing only when its edge survives:

- realistic spread/slippage/impact;
- multiple recent market regimes;
- parameter-neighbor checks;
- higher-cost stress tests;
- liquidity/capacity constraints;
- and an untouched future holdout.

Headline backtest return alone is not a success criterion.
