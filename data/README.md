# Data

Dit project houdt code en kleine testfixtures in Git, maar geen grote commerciële marktdata.

## 1. Dagelijkse aandelenprijzen

Bestand: `prices.csv`

```text
ticker,date,open,high,low,close,volume
AAPL,2025-01-02,....
```

Gebruik bij voorkeur split/dividend-consistente data en documenteer of prijzen adjusted of unadjusted zijn. Voor gap- en executable-open tests zijn echte historische opens nodig; gebruik niet blind adjusted OHLC zonder te controleren hoe corporate actions zijn verwerkt.

## 2. Point-in-time fundamentals

Bestand: `fundamentals.csv`

```text
ticker,available_date,roe,fcf_margin,debt_to_equity,current_ratio,market_cap
```

`available_date` is de eerste datum waarop de cijfers daadwerkelijk publiek beschikbaar waren. Dat voorkomt look-ahead bias. Een fiscal quarter-end is dus niet automatisch de beschikbare datum.

De quality-dip baseline vereist:

- ROE >= 12%
- FCF margin >= 5%
- debt/equity <= 1.5
- current ratio >= 1.0
- market cap >= $5 miljard

Deze thresholds zijn hypotheses, geen geoptimaliseerde waarheid.

## 3. Earnings events

Bestand: `earnings.csv`

```text
ticker,event_date,eps_surprise,revenue_beat,after_hours
```

`eps_surprise` wordt als fractie opgeslagen: `0.12` = 12% beat. `event_date` moet de daadwerkelijke publicatiedatum zijn. Als exacte timestamps beschikbaar zijn, bewaar die in een extra `event_datetime` kolom.

## 4. Intraday ES/MES

Bestand: `intraday.csv`

```text
datetime,open,high,low,close,volume
2026-01-02 09:30:00,...
```

De huidige ORB-engine verwacht timestamps in `America/New_York` exchange local time. Converteer UTC-bronnen vóór de backtest en behandel DST expliciet.

## Openbare datasets die als bron/fixture kunnen dienen

- GitHub `getdata-finance/es-1m-ohlcv-stocks-historical-data` bevat een openbare ES 1-minute CSV die bruikbaar is voor ORB-experimenten. Controleer altijd contractcontinuïteit, timezone, rollmethodiek en licentie voordat resultaten worden gepubliceerd.
- GitHub `pingfcc99/Earnings-surprise-on-stock-price` bevat een kleine historische Bloomberg-afgeleide earnings-surprise dataset voor S&P 500-bedrijven (2016), bruikbaar als sanity-check maar niet voldoende als moderne out-of-sample test.
- SEC EDGAR kan worden gebruikt om point-in-time fundamentals op te bouwen. Voor serieus onderzoek heeft dit de voorkeur boven een huidige snapshot van fundamentals.

## Niet doen

- Huidige S&P 500 constituents gebruiken voor een backtest vanaf 2000 zonder historische membership.
- Huidige fundamentals terugprojecteren naar oude trades.
- Een earningsdatum gebruiken zonder te weten of de release vóór opening of na sluiting kwam.
- De beste parametercombinatie rapporteren zonder alle geteste varianten en out-of-sample resultaten te tonen.
