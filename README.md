# Trading Strategy Backtests

Reproduceerbaar Python-framework om eenvoudige tradingstrategieën te testen zonder de resultaten mooier te maken dan ze zijn.

## Strategieën

1. **Opening Range Breakout (ORB)**
   - opening range van 5/15/30/60 minuten
   - long/short breakout
   - instelbare R-multiple target
   - maximaal aantal trades per dag
   - transactiekosten en slippage

2. **Positive earnings momentum**
   - earnings surprise filter
   - gap filter
   - holding periods van 1/5/10/20/60 handelsdagen
   - bedoeld voor point-in-time earnings event data

3. **Quality bad-news mean reversion**
   - zoekt sterke bedrijven die hard dalen
   - varianten voor daling van 5/10/15/20%
   - wachttijd 0/1/2 dagen
   - holding periods 5/10/20/60 dagen
   - kwaliteit wordt uitsluitend bepaald met fundamentals die vóór de entry bekend waren

## Belangrijkste ontwerpregels

- Geen look-ahead bias.
- Geen gebruik van toekomstige fundamentals.
- Kosten en slippage staan standaard aan.
- Resultaten worden per trade opgeslagen.
- Parameter sweeps tonen alle varianten, niet alleen de winnaar.
- In-sample en out-of-sample perioden kunnen apart worden gerapporteerd.
- De engine maakt expliciet onderscheid tussen prijsdata, event-data en point-in-time fundamentals.

## Installatie

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick start

```bash
python scripts/run_quality_dip_sweep.py \
  --prices data/sample/prices.csv \
  --fundamentals data/sample/fundamentals.csv \
  --output results/quality_dip
```

```bash
python scripts/run_orb.py --prices data/sample/intraday.csv --output results/orb
```

## Vereiste data

Zie [`data/README.md`](data/README.md). Grote commerciële datasets worden niet in Git opgenomen. Kleine fixtures en openbare voorbeelddata wel.

## Metrics

- aantal trades
- winrate
- gemiddelde en mediane trade return
- expectancy
- profit factor
- maximale drawdown
- Sharpe-ratio
- CAGR waar een echte equity curve beschikbaar is
- worst losing streak

## Onderzoeksdoel

Dit project is researchsoftware, geen financieel advies. Een historisch positief resultaat is geen bewijs dat een strategie live winstgevend blijft.
