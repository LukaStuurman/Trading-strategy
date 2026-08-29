#!/usr/bin/env python3
"""Build point-in-time quality fundamentals from compact SEC mirror facts.

The builder uses raw Company Facts rows and their SEC `filed` timestamps. Because
Company Facts exposes a filing date but not a trustworthy publication time for
our daily backtest, each snapshot becomes usable one calendar day after `filed`.
That conservative guard prevents an after-close filing from leaking into the
same day's signal. The implementation keeps one incremental context state per
metric so the broad historical build remains fast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_sec_fundamentals import TAGS, FLOW_KEYS
from src.data.io import read_table
from src.data.universe import normalize_ticker

MIRROR_TAGS = {
    **TAGS,
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "income_tax": ["IncomeTaxExpenseBenefit"],
}
MIRROR_FLOW_KEYS = set(FLOW_KEYS) | {"pretax_income", "income_tax"}
AVAILABILITY_LAG = pd.Timedelta(days=1)


def normalize_cik(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        text = str(int(float(text)))
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return None
        text = str(int(digits))
    return text.zfill(10)


def _metric_frame(facts: pd.DataFrame, candidates: list[str], flow: bool) -> pd.DataFrame:
    x = facts[facts["tag"].isin(candidates)].copy()
    if x.empty:
        cols = ["filed", "start", "end", "val", "form", "fp", "fy", "tag", "accn", "priority"]
        return pd.DataFrame(columns=cols + (["duration"] if flow else []))
    priority = {tag: i for i, tag in enumerate(candidates)}
    x["priority"] = x["tag"].map(priority).fillna(len(priority)).astype("int16")
    for col in ["filed", "start", "end"]:
        x[col] = pd.to_datetime(x.get(col), errors="coerce")
    x["val"] = pd.to_numeric(x["val"], errors="coerce")
    x = x.dropna(subset=["filed", "end", "val"])
    x = x[x["end"] <= x["filed"] + pd.Timedelta(days=7)]
    if flow:
        x = x.dropna(subset=["start"])
        x["duration"] = (x["end"] - x["start"]).dt.days
        x = x[x["duration"].between(45, 420, inclusive="both")]
        context_cols = ["filed", "start", "end"]
    else:
        context_cols = ["filed", "end"]
    x = x.sort_values(context_cols + ["priority"]).drop_duplicates(context_cols, keep="first")
    return x.sort_values(["filed", "end", "priority"]).reset_index(drop=True)


def _flow_snapshot(state: dict[tuple[pd.Timestamp, pd.Timestamp], tuple[int, int, float]], date: pd.Timestamp) -> float:
    if not state:
        return np.nan
    quarters: dict[pd.Timestamp, tuple[int, float]] = {}
    by_start: dict[pd.Timestamp, list[tuple[pd.Timestamp, int, int, float]]] = {}
    annual: list[tuple[pd.Timestamp, int, float]] = []
    for (start, end), (duration, priority, value) in state.items():
        if 65 <= duration <= 115:
            cur = quarters.get(end)
            if cur is None or priority < cur[0]:
                quarters[end] = (priority, value)
        if 65 <= duration <= 400:
            by_start.setdefault(start, []).append((end, duration, priority, value))
        if 300 <= duration <= 400:
            annual.append((end, priority, value))
    for group in by_start.values():
        group.sort(key=lambda row: (row[1], row[2], row[0]))
        by_duration: dict[int, tuple[pd.Timestamp, int, int, float]] = {}
        for row in group:
            duration = row[1]
            existing = by_duration.get(duration)
            if existing is None or row[2] < existing[2]:
                by_duration[duration] = row
        ordered = [by_duration[d] for d in sorted(by_duration)]
        for prev, curr in zip(ordered, ordered[1:]):
            increment = curr[1] - prev[1]
            if 55 <= increment <= 120:
                end = curr[0]
                derived = curr[3] - prev[3]
                if np.isfinite(derived) and end not in quarters:
                    quarters[end] = (10_000 + curr[2], float(derived))
    quarter_rows = [(end, value[1]) for end, value in sorted(quarters.items()) if np.isfinite(value[1])]
    if len(quarter_rows) >= 4:
        last4 = quarter_rows[-4:]
        if (last4[-1][0] - last4[0][0]).days <= 380:
            return float(sum(v for _, v in last4))
    if not annual:
        return np.nan
    annual.sort(key=lambda row: (row[0], -row[1]))
    end, _priority, value = annual[-1]
    if (date - end).days > 550:
        return np.nan
    return float(value)


def _flow_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    state: dict[tuple[pd.Timestamp, pd.Timestamp], tuple[int, int, float]] = {}
    out: dict[pd.Timestamp, float] = {}
    for filed, group in frame.groupby("filed", sort=True):
        filed = pd.Timestamp(filed)
        for row in group.itertuples(index=False):
            state[(pd.Timestamp(row.start), pd.Timestamp(row.end))] = (
                int(row.duration), int(row.priority), float(row.val)
            )
        out[filed] = _flow_snapshot(state, filed)
    return pd.Series(out, dtype=float).sort_index()


def _instant_series(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if frame.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    state: dict[pd.Timestamp, tuple[int, float]] = {}
    latest_out: dict[pd.Timestamp, float] = {}
    average_out: dict[pd.Timestamp, float] = {}
    for filed, group in frame.groupby("filed", sort=True):
        filed = pd.Timestamp(filed)
        for row in group.itertuples(index=False):
            end = pd.Timestamp(row.end)
            # Preference was resolved within this filing by _metric_frame. A
            # later filing always replaces the older context, even after a tag
            # migration or restatement.
            state[end] = (int(row.priority), float(row.val))
        if not state:
            continue
        latest_end = max(state)
        latest_val = state[latest_end][1]
        latest_out[filed] = latest_val
        prior_ends = [end for end in state if end <= latest_end - pd.Timedelta(days=180)]
        if prior_ends:
            prior_val = state[max(prior_ends)][1]
            average_out[filed] = float(np.nanmean([prior_val, latest_val]))
        else:
            average_out[filed] = latest_val
    return pd.Series(latest_out, dtype=float).sort_index(), pd.Series(average_out, dtype=float).sort_index()


def _asof_series(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    if series.empty:
        return pd.Series(np.nan, index=dates, dtype=float)
    return series.reindex(series.index.union(dates)).sort_index().ffill().reindex(dates).astype(float)


def _safe_div(num: pd.Series, den: pd.Series, *, positive_den: bool = True) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    valid = n.notna() & d.notna()
    valid &= d.gt(0) if positive_den else d.abs().gt(1e-12)
    out = pd.Series(np.nan, index=n.index, dtype=float)
    out.loc[valid] = n.loc[valid] / d.loc[valid]
    return out


def _price_series_asof(pg: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    price_col = "raw_close" if "raw_close" in pg.columns else "close"
    right = pg[["date", price_col]].copy().sort_values("date")
    right[price_col] = pd.to_numeric(right[price_col], errors="coerce")
    left = pd.DataFrame({"filed_date": dates})
    merged = pd.merge_asof(
        left, right, left_on="filed_date", right_on="date",
        direction="backward", allow_exact_matches=True,
    )
    return pd.Series(merged[price_col].to_numpy(), index=dates, dtype=float)


def _join_unique(values: pd.Series, limit: int | None = None) -> str:
    items = sorted({str(v) for v in values.dropna() if str(v)})
    if limit is not None:
        items = items[:limit]
    return ",".join(items)


def build_from_mirror_facts(prices: pd.DataFrame, facts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    p = prices.copy()
    p["ticker"] = p["ticker"].map(normalize_ticker)
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p["cik"] = p["cik"].map(normalize_cik)
    p = p.dropna(subset=["ticker", "date", "cik"]).sort_values(["ticker", "cik", "date"])

    f = facts.copy()
    f["cik"] = f["cik"].map(normalize_cik)
    for col in ["filed", "start", "end"]:
        f[col] = pd.to_datetime(f.get(col), errors="coerce")
    f["val"] = pd.to_numeric(f["val"], errors="coerce")
    f = f.dropna(subset=["cik", "filed", "end", "tag", "val"])

    facts_by_cik = {cik: g for cik, g in f.groupby("cik", sort=False)}
    rows: list[pd.DataFrame] = []
    audits: list[dict] = []
    built_tickers: set[str] = set()
    built_ciks: set[str] = set()
    multi_cik = p[["ticker", "cik"]].drop_duplicates().groupby("ticker")["cik"].nunique()
    multi_cik_tickers = sorted(multi_cik[multi_cik > 1].index.tolist())

    pairs = list(p.groupby(["ticker", "cik"], sort=False))
    for pair_i, ((ticker, cik), pg) in enumerate(pairs, start=1):
        cf = facts_by_cik.get(cik)
        if cf is None or cf.empty:
            continue
        pg = pg.sort_values("date").reset_index(drop=True)
        pair_start = pd.Timestamp(pg["date"].min())
        pair_end = pd.Timestamp(pg["date"].max())
        cf = cf[
            (cf["filed"] >= pair_start - pd.Timedelta(days=550))
            & (cf["filed"] <= pair_end + pd.Timedelta(days=7))
        ].copy()
        if cf.empty:
            continue

        frames = {key: _metric_frame(cf, candidates, key in MIRROR_FLOW_KEYS) for key, candidates in MIRROR_TAGS.items()}
        filed_dates = pd.DatetimeIndex(sorted(pd.to_datetime(cf["filed"].dropna().unique())))
        if filed_dates.empty:
            continue
        state = pd.DataFrame(index=filed_dates)
        for key in MIRROR_FLOW_KEYS:
            state[key] = _asof_series(_flow_series(frames[key]), filed_dates)
        for key in set(MIRROR_TAGS) - MIRROR_FLOW_KEYS:
            latest, average = _instant_series(frames[key])
            state[key] = _asof_series(latest, filed_dates)
            if key in {"equity", "assets"}:
                state[f"avg_{key}"] = _asof_series(average, filed_dates)

        debt = pd.concat([state["debt"], state["debt_long"]], axis=1).sum(axis=1, min_count=1)
        fcf = state["cfo"] - state["capex"]
        state["roe"] = _safe_div(state["net_income"], state["avg_equity"])
        state["roa"] = _safe_div(state["net_income"], state["avg_assets"])
        state["operating_margin"] = _safe_div(state["operating_income"], state["revenue"])
        state["fcf_margin"] = _safe_div(fcf, state["revenue"])
        state["fcf_to_net_income"] = _safe_div(fcf, state["net_income"], positive_den=False)
        state["asset_turnover"] = _safe_div(state["revenue"], state["avg_assets"])
        state["debt_to_equity"] = _safe_div(debt, state["equity"])
        net_debt = debt - state["cash"]
        state["net_debt_to_equity"] = _safe_div(net_debt, state["equity"])
        state["current_ratio"] = _safe_div(state["assets_current"], state["liabilities_current"])

        price = _price_series_asof(pg, filed_dates)
        market_cap = state["shares"] * price
        market_cap[(state["shares"] <= 0) | state["shares"].isna() | price.isna()] = np.nan
        state["market_cap"] = market_cap

        tax_rate = _safe_div(state["income_tax"], state["pretax_income"]).clip(0.0, 0.40)
        invested_capital = state["equity"] + debt - state["cash"]
        nopat = state["operating_income"] * (1.0 - tax_rate)
        state["roic"] = _safe_div(nopat, invested_capital)

        source_forms = pd.Series("", index=filed_dates)
        source_accessions = pd.Series("", index=filed_dates)
        if "form" in cf.columns:
            source_forms = cf.groupby("filed")["form"].apply(_join_unique).reindex(filed_dates).fillna("")
        if "accn" in cf.columns:
            source_accessions = cf.groupby("filed")["accn"].apply(lambda s: _join_unique(s, 5)).reindex(filed_dates).fillna("")

        out = pd.DataFrame({
            "ticker": ticker,
            "cik": cik,
            "source_filed_date": filed_dates,
            "available_date": filed_dates + AVAILABILITY_LAG,
            "roe": state["roe"], "roic": state["roic"], "roa": state["roa"],
            "operating_margin": state["operating_margin"], "fcf_margin": state["fcf_margin"],
            "fcf_to_net_income": state["fcf_to_net_income"], "asset_turnover": state["asset_turnover"],
            "debt_to_equity": state["debt_to_equity"], "net_debt_to_equity": state["net_debt_to_equity"],
            "current_ratio": state["current_ratio"], "market_cap": state["market_cap"],
            "fundamental_source": "sec_companyfacts_mirror_pit",
            "source_forms": source_forms, "source_accessions": source_accessions,
        })
        metric_cols = ["roe", "fcf_margin", "debt_to_equity", "net_debt_to_equity", "market_cap"]
        out = out[out[metric_cols].notna().any(axis=1)].copy()
        if not out.empty:
            rows.append(out)
            built_tickers.add(ticker)
            built_ciks.add(cik)
        for key, frame in frames.items():
            audits.append({
                "ticker": ticker, "cik": cik, "metric": key,
                "selected_tags": ",".join(sorted(frame["tag"].dropna().unique().tolist())) if not frame.empty else "",
                "fact_contexts": int(len(frame)), "pair_output_rows": int(len(out)),
            })
        if pair_i % 100 == 0 or pair_i == len(pairs):
            print(f"[fundamentals] processed {pair_i}/{len(pairs)} ticker-CIK pairs; built={len(built_tickers)} tickers")

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if result.empty:
        raise RuntimeError("SEC mirror facts produced no point-in-time fundamentals")
    for col in ["source_filed_date", "available_date"]:
        result[col] = pd.to_datetime(result[col])
    result = result.sort_values(["ticker", "cik", "available_date"]).drop_duplicates(["ticker", "cik", "available_date"], keep="last")
    result["source_filed_date"] = result["source_filed_date"].dt.date.astype(str)
    result["available_date"] = result["available_date"].dt.date.astype(str)
    audit = pd.DataFrame(audits)
    metadata = {
        "mode": "sec_companyfacts_mirror_pit",
        "availability_rule": "SEC filed date + 1 calendar day; next observed trading day consumes the snapshot",
        "availability_lag_days": 1,
        "built_tickers": len(built_tickers), "built_ciks": len(built_ciks),
        "price_tickers": int(p["ticker"].nunique()), "price_ciks": int(p["cik"].nunique()),
        "ticker_coverage": len(built_tickers) / p["ticker"].nunique() if p["ticker"].nunique() else 0.0,
        "cik_coverage": len(built_ciks) / p["cik"].nunique() if p["cik"].nunique() else 0.0,
        "multi_cik_tickers": multi_cik_tickers,
        "ttm_rule": "derive standalone quarters from direct 3-month and cumulative contexts; fallback to latest annual context",
        "balance_rule": "average latest balance with prior balance at least 180 days earlier when available",
        "engine": "incremental filing-context state with vectorized cross-metric calculations",
    }
    return result, audit, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--facts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--source")
    parser.add_argument("--mirror-manifest")
    args = parser.parse_args()
    prices = read_table(args.prices)
    facts = read_table(args.facts)
    result, audit, metadata = build_from_mirror_facts(prices, facts)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    audit.to_csv(args.audit, index=False)
    if args.source:
        if args.mirror_manifest and Path(args.mirror_manifest).exists():
            upstream = json.loads(Path(args.mirror_manifest).read_text(encoding="utf-8"))
            metadata["upstream"] = {
                "upstream_repo": upstream.get("upstream_repo"), "release_tag": upstream.get("release_tag"),
                "release_id": upstream.get("release_id"), "published_at": upstream.get("published_at"),
                "compact_output_sha256": upstream.get("compact_output_sha256"),
                "input_ciks": upstream.get("input_ciks"), "matched_ciks": upstream.get("matched_ciks"),
            }
        Path(args.source).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(result):,} point-in-time rows for {metadata['built_tickers']:,} tickers "
        f"({metadata['ticker_coverage']:.1%} price-ticker coverage)"
    )


if __name__ == "__main__":
    main()
