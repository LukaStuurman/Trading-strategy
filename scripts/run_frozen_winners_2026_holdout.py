#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.io import read_table
from src.data.universe import load_universe_intervals
from src.strategies.intraday_daily import prepare_intraday_features, intraday_exit_returns, daily_portfolio_metrics

COST_BPS=20.0

def candidates(f, s):
    cfg=s['signal']; fam=s['family']
    m=(f.in_universe.fillna(False)&(f.open>=5)&(f.prev_close>=5)&(f.adv20>=float(cfg['min_adv20']))&f.gap_return.notna()&f.prev_return.notna()&(f.gap_return.abs()<=0.50)&(f.prev_return.abs()<=0.60))
    if fam=='relative_weakness_reversal':
        m &= (f.prev_return_percentile<=float(cfg['prev_return_percentile_max']))&(f.gap_return<=float(cfg['gap_max']))&(f.mom3<=float(cfg['mom3_max']))&(f.market_prev_return<=float(cfg['market_prev_return_max']))
        score=(1-f.prev_return_percentile)+(-f.gap_return).clip(lower=0)*2+(-f.mom3).clip(lower=0)*0.30
    elif fam=='gap_down_reversal':
        m &= (f.gap_return<=float(cfg['gap_max']))&(f.prev_return<=float(cfg['prev_return_max']))&(f.mom3<=float(cfg['mom3_max']))
        score=-f.gap_return+(-f.prev_return).clip(lower=0)*0.50+(-f.mom3).clip(lower=0)*0.20
    else: raise ValueError(fam)
    cols=['ticker','_instrument_id','date','open','high','low','close']
    out=f.loc[m,cols].copy(); out['signal_score']=pd.to_numeric(score.loc[m],errors='coerce').to_numpy()
    return out.dropna(subset=['signal_score'])

def metrics_for(trades, calendar, leverage, max_positions, start, end):
    return daily_portfolio_metrics(trades,calendar,start=start,end=end,leverage=leverage,max_positions=max_positions)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prices',required=True); p.add_argument('--universe',required=True); p.add_argument('--strategies',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    prices=read_table(a.prices); universe=load_universe_intervals(a.universe); frozen=json.loads(Path(a.strategies).read_text())
    features=prepare_intraday_features(prices,universe); features=features[features.date>=pd.Timestamp('2025-10-01')].copy()
    calendar=pd.DatetimeIndex(sorted(features.date.dropna().unique())); scored=calendar[calendar>=pd.Timestamp('2026-01-01')]
    if len(scored)==0: raise RuntimeError('No 2026 Yahoo sessions available')
    start=pd.Timestamp('2026-01-01'); end=scored.max(); outdir=Path(a.output); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; trade_frames=[]
    for s in frozen['strategies']:
        c=candidates(features,s).sort_values(['date','signal_score','ticker','_instrument_id'],ascending=[True,False,True,True],kind='stable')
        net,labels=intraday_exit_returns(c,stop_loss=None,take_profit=float(s['exit']['take_profit']),round_trip_cost_bps=COST_BPS)
        c=c.copy(); c['net_return']=net; c['exit_reason']=labels
        exact=metrics_for(c,calendar,float(s['portfolio']['leverage']),int(s['portfolio']['max_positions']),start,end)
        one_x=metrics_for(c,calendar,1.0,10,start,end)
        row={'name':s['name'],'family':s['family'],'source_variant_id':s['source_variant_id'],'scored_start':str(start.date()),'scored_end':str(end.date()),'leverage':s['portfolio']['leverage'],'max_positions':s['portfolio']['max_positions'],'take_profit':s['exit']['take_profit'],'raw_candidates_2026':int((c.date>=start).sum())}
        row.update({f'policy_{k}':v for k,v in exact.items()}); row.update({f'signal_1x10_{k}':v for k,v in one_x.items()}); rows.append(row)
        accepted=c[(c.date>=start)&(c.date<=end)].groupby('date',sort=False).head(int(s['portfolio']['max_positions'])).copy(); accepted.insert(0,'strategy',s['name']); trade_frames.append(accepted)
    df=pd.DataFrame(rows); df.to_csv(outdir/'holdout_2026_results.csv',index=False)
    if trade_frames: pd.concat(trade_frames,ignore_index=True).to_csv(outdir/'holdout_2026_trades.csv',index=False)
    report=['# Frozen winners — independent 2026 holdout','',f'Price source: **Yahoo adjusted daily OHLC**, independent of the FINSABER source used to select the rules.',f'Scored period: **{start.date()} through {end.date()}**. No 2026 parameter is used for tuning.','', '| Strategy | Return | CAGR | Sharpe | Max DD | Trades | 1x/10 return | 1x/10 Sharpe |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in df.iterrows(): report.append(f"| {r['name']} | {r['policy_total_return']:.2%} | {r['policy_cagr']:.2%} | {r['policy_sharpe']:.3f} | {r['policy_max_drawdown']:.2%} | {int(r['policy_trades'])} | {r['signal_1x10_total_return']:.2%} | {r['signal_1x10_sharpe']:.3f} |")
    report += ['', '## Interpretation guardrails', '', '- These rules were frozen before this 2026 test.', '- 2026 is evaluated once; do not retune these parameters using this result.', '- Yahoo is a new price source, which reduces dependence on FINSABER-specific adjustments.', '- Daily bars still approximate execution at the open and target touch; real intraday bid/ask and slippage are not modeled.']
    (outdir/'holdout_2026_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    manifest={'strategy_file':a.strategies,'price_source':'Yahoo Finance chart v8 adjusted OHLC','selection_price_source':'FINSABER','scored_start':str(start.date()),'scored_end':str(end.date()),'no_retuning':True,'round_trip_cost_bps':COST_BPS}
    (outdir/'holdout_2026_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(df.to_string(index=False))

if __name__=='__main__': main()
