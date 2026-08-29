#!/usr/bin/env python3
"""Build point-in-time quality fundamentals from compact SEC mirror facts.

Unlike the older annual fallback, this builder uses raw Company Facts rows and
their SEC `filed` timestamps. Candidate taxonomy tags are selected per context,
so issuers can change tags through time without losing older history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_sec_fundamentals import TAGS, FLOW_KEYS

MIRROR_TAGS = {
    **TAGS,
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "income_tax": ["IncomeTaxExpenseBenefit"],
}
MIRROR_FLOW_KEYS = set(FLOW_KEYS) | {"pretax_income", "income_tax"}
from src.data.io import read_table
from src.data.universe import normalize_ticker


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
    columns = ["filed", "start", "end", "val", "form", "fp", "fy", "tag", "accn"]
    x = facts[facts["tag"].isin(candidates)].copy()
    if x.empty:
        return pd.DataFrame(columns=columns + (["duration"] if flow else []))
    priority = {tag: i for i, tag in enumerate(candidates)}
    x["priority"] = x["tag"].map(priority).fillna(len(priority)).astype(int)
    for col in ["filed", "start", "end"]:
        x[col] = pd.to_datetime(x[col], errors="coerce")
    x["val"] = pd.to_numeric(x["val"], errors="coerce")
    x = x.dropna(subset=["filed", "end", "val"])
    x = x[x["end"] <= x["filed"] + pd.Timedelta(days=7)]
    if flow:
        x = x.dropna(subset=["start"])
        x["duration"] = (x["end"] - x["start"]).dt.days
        x = x[x["duration"].between(45, 420, inclusive="both")]
    context_cols = ["filed", "start", "end"] if flow else ["filed", "end"]
    x = x.sort_values(context_cols + ["priority"]).drop_duplicates(context_cols, keep="first")
    return x.sort_values(["filed", "end", "priority"]).reset_index(drop=True)


def _latest_contexts_asof(frame: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame
    x = frame[frame["filed"] <= date].copy()
    if x.empty:
        return x
    context_cols = ["start", "end"] if "duration" in x.columns else ["end"]
    x = x.sort_values(["filed", "priority"], ascending=[True, False]).drop_duplicates(context_cols, keep="last")
    return x.sort_values("end").reset_index(drop=True)


def _instant_asof(frame: pd.DataFrame, date: pd.Timestamp) -> float:
    x = _latest_contexts_asof(frame, date)
    if x.empty:
        return np.nan
    latest_end = x["end"].max()
    row = x[x["end"] == latest_end].sort_values("priority").iloc[0]
    return float(row["val"])


def _average_balance_asof(frame: pd.DataFrame, date: pd.Timestamp) -> float:
    x = _latest_contexts_asof(frame, date)
    if x.empty:
        return np.nan
    x = x.sort_values("end").drop_duplicates("end", keep="last")
    latest = x.iloc[-1]
    prior = x[x["end"] <= latest["end"] - pd.Timedelta(days=180)]
    if prior.empty:
        return float(latest["val"])
    return float(np.nanmean([float(prior.iloc[-1]["val"]), float(latest["val"])]))


def _quarter_values(frame: pd.DataFrame, date: pd.Timestamp) -> list[tuple[pd.Timestamp, float]]:
    x = _latest_contexts_asof(frame, date)
    if x.empty:
        return []

    quarters: dict[pd.Timestamp, tuple[int, float]] = {}
    direct = x[x["duration"].between(65, 115, inclusive="both")]
    for _, row in direct.iterrows():
        end = pd.Timestamp(row["end"])
        priority = int(row["priority"])
        current = quarters.get(end)
        if current is None or priority < current[0]:
            quarters[end] = (priority, float(row["val"]))

    cumulative = x[x["duration"].between(65, 400, inclusive="both")].copy()
    for _, group in cumulative.groupby("start", sort=False):
        g = group.sort_values(["duration", "priority"]).drop_duplicates("duration", keep="first")
        rows = list(g.itertuples(index=False))
        for prev, curr in zip(rows, rows[1:]):
            prev_duration = float(prev.duration)
            curr_duration = float(curr.duration)
            increment = curr_duration - prev_duration
            if 55 <= increment <= 120 and curr_duration <= 400:
                end = pd.Timestamp(curr.end)
                derived = float(curr.val) - float(prev.val)
                quarters.setdefault(end, (10_000 + int(curr.priority), derived))

    return [(end, val[1]) for end, val in sorted(quarters.items()) if np.isfinite(val[1])]


def _ttm_flow_asof(frame: pd.DataFrame, date: pd.Timestamp) -> float:
    quarters = _quarter_values(frame, date)
    if len(quarters) >= 4:
        last4 = quarters[-4:]
        if (last4[-1][0] - last4[0][0]).days <= 380:
            return float(sum(v for _, v in last4))

    x = _latest_contexts_asof(frame, date)
    annual = x[x["duration"].between(300, 400, inclusive="both")]
    if annual.empty:
        return np.nan
    row = annual.sort_values(["end", "priority"]).iloc[-1]
    if (date - pd.Timestamp(row["end"])).days > 550:
        return np.nan
    return float(row["val"])


def _raw_price_asof(group: pd.DataFrame, date: pd.Timestamp) -> float:
    if group.empty:
        return np.nan
    pos = int(group["date"].searchsorted(date, side="right")) - 1
    if pos < 0:
        return np.nan
    row = group.iloc[pos]
    if "raw_close" in group.columns and pd.notna(row.get("raw_close")):
        return float(row["raw_close"])
    return float(row["close"])


def build_from_mirror_facts(prices: pd.DataFrame, facts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    p = prices.copy()
    p["ticker"] = p["ticker"].map(normalize_ticker)
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p["cik"] = p["cik"].map(normalize_cik)
    p = p.dropna(subset=["ticker", "date", "cik"]).sort_values(["ticker", "cik", "date"])

    f = facts.copy()
    f["cik"] = f["cik"].map(normalize_cik)
    f["filed"] = pd.to_datetime(f["filed"], errors="coerce")
    f["start"] = pd.to_datetime(f.get("start"), errors="coerce")
    f["end"] = pd.to_datetime(f["end"], errors="coerce")
    f["val"] = pd.to_numeric(f["val"], errors="coerce")
    f = f.dropna(subset=["cik", "filed", "end", "tag", "val"])

    facts_by_cik = {cik: g.copy() for cik, g in f.groupby("cik", sort=False)}
    rows: list[dict] = []
    audits: list[dict] = []
    built_tickers: set[str] = set()
    built_ciks: set[str] = set()
    multi_cik = p[["ticker", "cik"]].drop_duplicates().groupby("ticker")["cik"].nunique()
    multi_cik_tickers = sorted(multi_cik[multi_cik > 1].index.tolist())

    for (ticker, cik), pg in p.groupby(["ticker", "cik"], sort=False):
        cf = facts_by_cik.get(cik)
        if cf is None or cf.empty:
            continue
        pg = pg.sort_values("date").reset_index(drop=True)
        pair_start = pg["date"].min()
        pair_end = pg["date"].max()
        cf = cf[
            (cf["filed"] >= pair_start - pd.Timedelta(days=550))
            & (cf["filed"] <= pair_end + pd.Timedelta(days=7))
        ].copy()
        if cf.empty:
            continue

        frames = {
            key: _metric_frame(cf, candidates, flow=key in MIRROR_FLOW_KEYS)
            for key, candidates in MIRROR_TAGS.items()
        }
        filed_dates = sorted(cf["filed"].dropna().unique().tolist())
        pair_count = 0
        for available_raw in filed_dates:
            available_date = pd.Timestamp(available_raw)
            if available_date > pair_end + pd.Timedelta(days=7):
                continue

            revenue = _ttm_flow_asof(frames["revenue"], available_date)
            net_income = _ttm_flow_asof(frames["net_income"], available_date)
            operating_income = _ttm_flow_asof(frames["operating_income"], available_date)
            cfo = _ttm_flow_asof(frames["cfo"], available_date)
            capex = _ttm_flow_asof(frames["capex"], available_date)
            pretax_income = _ttm_flow_asof(frames["pretax_income"], available_date)
            income_tax = _ttm_flow_asof(frames["income_tax"], available_date)

            equity = _instant_asof(frames["equity"], available_date)
            avg_equity = _average_balance_asof(frames["equity"], available_date)
            assets = _instant_asof(frames["assets"], available_date)
            avg_assets = _average_balance_asof(frames["assets"], available_date)
            ca = _instant_asof(frames["assets_current"], available_date)
            cl = _instant_asof(frames["liabilities_current"], available_date)
            cash = _instant_asof(frames["cash"], available_date)
            debt_short = _instant_asof(frames["debt"], available_date)
            debt_long = _instant_asof(frames["debt_long"], available_date)
            shares = _instant_asof(frames["shares"], available_date)

            debt_parts = [v for v in [debt_short, debt_long] if np.isfinite(v)]
            debt_total = float(sum(debt_parts)) if debt_parts else np.nan
            roe = net_income / avg_equity if np.isfinite(net_income) and np.isfinite(avg_equity) and avg_equity > 0 else np.nan
            roa = net_income / avg_assets if np.isfinite(net_income) and np.isfinite(avg_assets) and avg_assets > 0 else np.nan
            operating_margin = operating_income / revenue if np.isfinite(operating_income) and np.isfinite(revenue) and revenue > 0 else np.nan
            fcf = cfo - capex if np.isfinite(cfo) and np.isfinite(capex) else np.nan
            fcf_margin = fcf / revenue if np.isfinite(fcf) and np.isfinite(revenue) and revenue > 0 else np.nan
            fcf_to_ni = fcf / net_income if np.isfinite(fcf) and np.isfinite(net_income) and abs(net_income) > 1e-12 else np.nan
            asset_turnover = revenue / avg_assets if np.isfinite(revenue) and np.isfinite(avg_assets) and avg_assets > 0 else np.nan
            debt_to_equity = debt_total / equity if np.isfinite(debt_total) and np.isfinite(equity) and equity > 0 else np.nan
            net_debt = debt_total - cash if np.isfinite(debt_total) and np.isfinite(cash) else np.nan
            net_debt_to_equity = net_debt / equity if np.isfinite(net_debt) and np.isfinite(equity) and equity > 0 else np.nan
            current_ratio = ca / cl if np.isfinite(ca) and np.isfinite(cl) and cl > 0 else np.nan
            px = _raw_price_asof(pg, available_date)
            market_cap = shares * px if np.isfinite(shares) and shares > 0 and np.isfinite(px) else np.nan

            tax_rate = income_tax / pretax_income if np.isfinite(income_tax) and np.isfinite(pretax_income) and pretax_income > 0 else np.nan
            if np.isfinite(tax_rate):
                tax_rate = float(np.clip(tax_rate, 0.0, 0.40))
            invested_capital = equity + debt_total - cash if np.isfinite(equity) and np.isfinite(debt_total) and np.isfinite(cash) else np.nan
            nopat = operating_income * (1.0 - tax_rate) if np.isfinite(operating_income) and np.isfinite(tax_rate) else np.nan
            roic = nopat / invested_capital if np.isfinite(nopat) and np.isfinite(invested_capital) and invested_capital > 0 else np.nan

            if not any(np.isfinite(v) for v in [roe, fcf_margin, debt_to_equity, net_debt_to_equity, market_cap]):
                continue
            filing_rows = cf[cf["filed"] == available_date]
            accessions = sorted({str(v) for v in filing_rows.get("accn", pd.Series(dtype=str)).dropna() if str(v)})
            forms = sorted({str(v) for v in filing_rows.get("form", pd.Series(dtype=str)).dropna() if str(v)})

            rows.append({
                "ticker": ticker,
                "cik": cik,
                "available_date": available_date.date().isoformat(),
                "roe": roe,
                "roic": roic,
                "roa": roa,
                "operating_margin": operating_margin,
                "fcf_margin": fcf_margin,
                "fcf_to_net_income": fcf_to_ni,
                "asset_turnover": asset_turnover,
                "debt_to_equity": debt_to_equity,
                "net_debt_to_equity": net_debt_to_equity,
                "current_ratio": current_ratio,
                "market_cap": market_cap,
                "fundamental_source": "sec_companyfacts_mirror_pit",
                "source_forms": ",".join(forms),
                "source_accessions": ",".join(accessions[:5]),
            })
            pair_count += 1

        if pair_count:
            built_tickers.add(ticker)
            built_ciks.add(cik)
        for key, frame in frames.items():
            audits.append({
                "ticker": ticker,
                "cik": cik,
                "metric": key,
                "selected_tags": ",".join(sorted(frame["tag"].dropna().unique().tolist())) if not frame.empty else "",
                "fact_contexts": int(len(frame)),
                "pair_output_rows": pair_count,
            })

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("SEC mirror facts produced no point-in-time fundamentals")
    result["available_date"] = pd.to_datetime(result["available_date"])
    result = result.sort_values(["ticker", "cik", "available_date"]).drop_duplicates(["ticker", "cik", "available_date"], keep="last")
    result["available_date"] = result["available_date"].dt.date.astype(str)
    audit = pd.DataFrame(audits)
    metadata = {
        "mode": "sec_companyfacts_mirror_pit",
        "availability_rule": "SEC filed date from raw Company Facts mirror",
        "built_tickers": len(built_tickers),
        "built_ciks": len(built_ciks),
        "price_tickers": int(p["ticker"].nunique()),
        "price_ciks": int(p["cik"].nunique()),
        "ticker_coverage": len(built_tickers) / p["ticker"].nunique() if p["ticker"].nunique() else 0.0,
        "cik_coverage": len(built_ciks) / p["cik"].nunique() if p["cik"].nunique() else 0.0,
        "multi_cik_tickers": multi_cik_tickers,
        "ttm_rule": "derive standalone quarters from direct 3-month and cumulative contexts; fallback to latest annual context",
        "balance_rule": "average latest balance with prior balance at least 180 days earlier when available",
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
                "upstream_repo": upstream.get("upstream_repo"),
                "release_tag": upstream.get("release_tag"),
                "release_id": upstream.get("release_id"),
                "published_at": upstream.get("published_at"),
                "compact_output_sha256": upstream.get("compact_output_sha256"),
                "input_ciks": upstream.get("input_ciks"),
                "matched_ciks": upstream.get("matched_ciks"),
            }
        Path(args.source).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(result):,} point-in-time rows for {metadata['built_tickers']:,} tickers "
        f"({metadata['ticker_coverage']:.1%} price-ticker coverage)"
    )


if __name__ == "__main__":
    main()
