#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from src.data.universe import load_universe_intervals, normalize_ticker


def fetch(ticker, start, end):
    symbol = normalize_ticker(ticker).replace('.', '-')
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
    params = {'period1': int(start.timestamp()), 'period2': int(end.timestamp()), 'interval': '1d', 'events': 'div,splits', 'includeAdjustedClose': 'true'}
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers={'User-Agent':'Mozilla/5.0'}, timeout=30)
            r.raise_for_status()
            result = (r.json().get('chart',{}).get('result') or [None])[0]
            if not result: return []
            q = result['indicators']['quote'][0]
            adj = (result['indicators'].get('adjclose') or [{'adjclose':[]}])[0]['adjclose']
            out=[]
            for i, ts in enumerate(result.get('timestamp',[])):
                vals=[q[k][i] for k in ['open','high','low','close','volume']]
                if any(v is None for v in vals[:4]): continue
                o,h,l,c,v=map(float,vals)
                a=float(adj[i]) if i < len(adj) and adj[i] is not None else c
                f=a/c if c else 1.0
                if not np.isfinite(f) or f<=0: f=1.0
                out.append({'date':datetime.fromtimestamp(ts,tz=timezone.utc).date(),'ticker':normalize_ticker(ticker),'cik':None,'open':o*f,'high':h*f,'low':l*f,'close':a,'volume':float(v)/f if v is not None else np.nan})
            return out
        except Exception:
            time.sleep(attempt+1)
    return []


def main():
    p=argparse.ArgumentParser(); p.add_argument('--universe',required=True); p.add_argument('--output',required=True); p.add_argument('--manifest',required=True); a=p.parse_args()
    start=pd.Timestamp('2025-10-01',tz='UTC'); end=pd.Timestamp(datetime.now(timezone.utc).date()+pd.Timedelta(days=1),tz='UTC')
    u=load_universe_intervals(a.universe); s=start.tz_localize(None); e=end.tz_localize(None)
    mask=(u.start_date<e)&(u.end_date.isna()|(u.end_date>s)); tickers=sorted(u.loc[mask,'ticker'].unique())
    rows=[]; ok=0; failed=[]
    for n,t in enumerate(tickers,1):
        x=fetch(t,start,end)
        if x: rows+=x; ok+=1
        else: failed.append(t)
        if n%50==0: print(n,len(tickers),ok,len(rows))
    if ok<450: raise RuntimeError(f'Yahoo success only {ok}/{len(tickers)}')
    df=pd.DataFrame(rows); df['date']=pd.to_datetime(df.date); df=df.sort_values(['ticker','date'])
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); df.to_parquet(a.output,index=False)
    y=df[df.date>=pd.Timestamp('2026-01-01')]
    manifest={'source':'Yahoo Finance chart v8','selection_source':'FINSABER','independent_price_source':True,'tickers_requested':len(tickers),'tickers_succeeded':ok,'failed_tickers':failed,'rows':len(df),'first_date':str(df.date.min().date()),'last_date':str(df.date.max().date()),'scored_2026_last_date':str(y.date.max().date()) if len(y) else None,'adjustment':'OHLC adjusted with adjclose/close; volume inverse-adjusted'}
    Path(a.manifest).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
