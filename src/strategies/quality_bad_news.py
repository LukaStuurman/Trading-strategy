from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data.universe import attach_membership, normalize_ticker


@dataclass(frozen=True)
class QualityDipConfig:
    drop_threshold: float = -0.10
    wait_days: int = 1
    hold_days: int = 20
    min_roe: float = 0.12
    min_fcf_margin: float = 0.05
    max_debt_to_equity: float = 1.5
    max_net_debt_to_equity: float = 1.0
    min_current_ratio: float = 1.0
    allow_missing_current_ratio: bool = True
    min_market_cap: float = 5_000_000_000
    min_quality_percentile: float = 0.0
    require_stabilization: bool = False
    round_trip_cost_bps: float = 10.0


CORE_FUNDAMENTALS = [
    "roe", "fcf_margin", "debt_to_equity", "net_debt_to_equity",
    "current_ratio", "market_cap",
]
POSITIVE_QUALITY_FACTORS = [
    "roe", "roic", "fcf_margin", "current_ratio", "roa", "operating_margin",
    "fcf_to_net_income", "asset_turnover",
]
NEGATIVE_QUALITY_FACTORS = ["debt_to_equity", "net_debt_to_equity"]


def _numeric_column(f: pd.DataFrame, name: str) -> pd.Series:
    if name not in f.columns:
        return pd.Series(np.nan, index=f.index, dtype=float)
    return pd.to_numeric(f[name], errors="coerce")


def _quality_mask(f: pd.DataFrame, cfg: QualityDipConfig) -> pd.Series:
    roe = _numeric_column(f, "roe")
    fcf_margin = _numeric_column(f, "fcf_margin")
    debt_to_equity = _numeric_column(f, "debt_to_equity")
    net_debt_to_equity = _numeric_column(f, "net_debt_to_equity")
    current_ratio = _numeric_column(f, "current_ratio")
    market_cap = _numeric_column(f, "market_cap")

    leverage_ok = (debt_to_equity.notna() & (debt_to_equity <= cfg.max_debt_to_equity)) | (
        debt_to_equity.isna()
        & net_debt_to_equity.notna()
        & (net_debt_to_equity <= cfg.max_net_debt_to_equity)
    )
    liquidity_ok = current_ratio >= cfg.min_current_ratio
    if cfg.allow_missing_current_ratio:
        liquidity_ok = liquidity_ok | current_ratio.isna()

    return (
        (roe >= cfg.min_roe)
        & (fcf_margin >= cfg.min_fcf_margin)
        & leverage_ok
        & liquidity_ok
        & (market_cap >= cfg.min_market_cap)
    )


def _attach_quality_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ranked = []
    for col in POSITIVE_QUALITY_FACTORS:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            ranked.append(values.groupby(out["date"]).rank(pct=True, ascending=True))
    for col in NEGATIVE_QUALITY_FACTORS:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            ranked.append(values.groupby(out["date"]).rank(pct=True, ascending=False))
    if not ranked:
        out["quality_percentile"] = np.nan
    else:
        out["quality_percentile"] = pd.concat(ranked, axis=1).mean(axis=1, skipna=True)
    return out


def _datetime_ns(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").astype("datetime64[ns]")


def _normalize_cik_value(value) -> str | None:
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


def _fundamental_groups(f: pd.DataFrame, columns: list[str]):
    specific: dict[tuple[str, str], pd.DataFrame] = {}
    fallback: dict[str, pd.DataFrame] = {}
    if "cik" in f.columns:
        with_cik = f[f["cik"].notna()]
        specific = {
            (ticker, cik): g[columns].sort_values("available_date")
            for (ticker, cik), g in with_cik.groupby(["ticker", "cik"], sort=False)
        }
        no_cik = f[f["cik"].isna()]
        fallback = {
            ticker: g[columns].sort_values("available_date")
            for ticker, g in no_cik.groupby("ticker", sort=False)
        }
    else:
        fallback = {
            ticker: g[columns].sort_values("available_date")
            for ticker, g in f.groupby("ticker", sort=False)
        }
    return specific, fallback


def prepare_features(prices: pd.DataFrame, fundamentals: pd.DataFrame, universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build causal features while isolating reused ticker symbols by CIK."""
    p = prices.copy()
    f = fundamentals.copy()
    p["date"] = _datetime_ns(p["date"])
    f["available_date"] = _datetime_ns(f["available_date"])
    p["ticker"] = p["ticker"].map(normalize_ticker)
    f["ticker"] = f["ticker"].map(normalize_ticker)

    if "cik" in p.columns:
        p["cik"] = p["cik"].map(_normalize_cik_value)
    else:
        p["cik"] = None
    if "cik" in f.columns:
        f["cik"] = f["cik"].map(_normalize_cik_value)

    p = p.dropna(subset=["date", "ticker"]).copy()
    p["_instrument_id"] = p["ticker"] + "|" + p["cik"].fillna("NO-CIK")
    p = p.sort_values(["_instrument_id", "date"]).reset_index(drop=True)
    f = f.dropna(subset=["available_date", "ticker"]).sort_values(["ticker", "available_date"]).reset_index(drop=True)
    p["daily_return"] = p.groupby("_instrument_id", sort=False)["close"].pct_change()

    fundamental_cols = [c for c in f.columns if c not in {"ticker", "cik"}]
    specific_groups, fallback_groups = _fundamental_groups(f, fundamental_cols)
    merged = []
    for _, gp in p.groupby("_instrument_id", sort=False):
        gp = gp.sort_values("date")
        ticker = str(gp["ticker"].iloc[0])
        cik = gp["cik"].iloc[0]
        parts = []
        if cik is not None and not pd.isna(cik):
            specific = specific_groups.get((ticker, str(cik)))
            if specific is not None and not specific.empty:
                specific = specific.copy()
                specific["_specific_source"] = 1
                parts.append(specific)
        generic = fallback_groups.get(ticker)
        if generic is not None and not generic.empty:
            generic = generic.copy()
            generic["_specific_source"] = 0
            parts.append(generic)

        if not parts:
            x = gp.copy()
            for col in fundamental_cols:
                x[col] = pd.NaT if col == "available_date" else np.nan
        else:
            gf = pd.concat(parts, ignore_index=True)
            gf = gf.sort_values(["available_date", "_specific_source"]).drop_duplicates("available_date", keep="last").drop(columns="_specific_source")
            x = pd.merge_asof(gp, gf, left_on="date", right_on="available_date", direction="backward", allow_exact_matches=True)
        merged.append(x)

    out = pd.concat(merged, ignore_index=True) if merged else p.copy()
    out = attach_membership(out, universe)
    out = _attach_quality_percentile(out)
    out = out.sort_values(["_instrument_id", "date"]).reset_index(drop=True)
    out["_ticker_row"] = out.groupby("_instrument_id", sort=False).cumcount().astype("int32")
    return out


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "ticker", "cik", "signal_date", "signal_return", "quality_percentile",
        "fundamental_source", "fundamental_available_date", "entry_date",
        "entry_price", "exit_date", "exit_price", "gross_return", "net_return",
        "drop_threshold", "wait_days", "hold_days", "min_quality_percentile",
        "require_stabilization",
    ])


def generate_trades(prices: pd.DataFrame, fundamentals: pd.DataFrame, cfg: QualityDipConfig, *, universe: pd.DataFrame | None = None, prepared: pd.DataFrame | None = None) -> pd.DataFrame:
    features = prepared if prepared is not None else prepare_features(prices, fundamentals, universe)
    if features.empty:
        return _empty_trades()

    quality = _quality_mask(features, cfg)
    q_pct = pd.to_numeric(features.get("quality_percentile"), errors="coerce")
    signal_mask = (
        (pd.to_numeric(features["daily_return"], errors="coerce") <= cfg.drop_threshold)
        & features["in_universe"].fillna(False)
        & quality
        & q_pct.notna()
        & (q_pct >= cfg.min_quality_percentile)
    )
    signal_cols = ["ticker", "cik", "_instrument_id", "date", "daily_return", "quality_percentile", "close", "_ticker_row"]
    if "low" in features.columns:
        signal_cols.append("low")
    if "fundamental_source" in features.columns:
        signal_cols.append("fundamental_source")
    if "available_date" in features.columns:
        signal_cols.append("available_date")
    signals = features.loc[signal_mask, signal_cols].copy()
    if signals.empty:
        return _empty_trades()

    signals = signals.rename(columns={
        "date": "signal_date", "daily_return": "signal_return", "close": "signal_close",
        "low": "signal_low", "available_date": "fundamental_available_date", "_ticker_row": "signal_i",
    })
    if "signal_low" not in signals.columns:
        signals["signal_low"] = np.nan
    if "fundamental_source" not in signals.columns:
        signals["fundamental_source"] = "unknown"
    if "fundamental_available_date" not in signals.columns:
        signals["fundamental_available_date"] = pd.NaT

    join_keys = ["_instrument_id"]
    if cfg.require_stabilization:
        confirm_lookup = features[join_keys + ["_ticker_row", "close"] + (["low"] if "low" in features.columns else [])].copy()
        confirm_lookup = confirm_lookup.rename(columns={"_ticker_row": "confirm_i", "close": "confirm_close", "low": "confirm_low"})
        signals["confirm_i"] = signals["signal_i"] + 1
        signals = signals.merge(confirm_lookup, on=join_keys + ["confirm_i"], how="inner")
        stabilized = signals["confirm_close"] > signals["signal_close"]
        if "confirm_low" in signals.columns:
            both_lows = signals["confirm_low"].notna() & signals["signal_low"].notna()
            stabilized &= (~both_lows) | (signals["confirm_low"] >= signals["signal_low"])
        signals = signals.loc[stabilized].copy()
        if signals.empty:
            return _empty_trades()

    minimum_offset = 2 if cfg.require_stabilization else 1
    entry_offset = max(cfg.wait_days + 1, minimum_offset)
    signals["entry_i"] = signals["signal_i"] + entry_offset
    entry_lookup = features[join_keys + ["_ticker_row", "date", "open"]].rename(columns={"_ticker_row": "entry_i", "date": "entry_date", "open": "entry_price"})
    signals = signals.merge(entry_lookup, on=join_keys + ["entry_i"], how="inner")
    if signals.empty:
        return _empty_trades()

    signals["exit_i"] = signals["entry_i"] + cfg.hold_days
    exit_lookup = features[join_keys + ["_ticker_row", "date", "close"]].rename(columns={"_ticker_row": "exit_i", "date": "exit_date", "close": "exit_price"})
    signals = signals.merge(exit_lookup, on=join_keys + ["exit_i"], how="inner")
    if signals.empty:
        return _empty_trades()

    signals["gross_return"] = signals["exit_price"] / signals["entry_price"] - 1.0
    signals["net_return"] = signals["gross_return"] - cfg.round_trip_cost_bps / 10_000.0
    signals["drop_threshold"] = cfg.drop_threshold
    signals["wait_days"] = cfg.wait_days
    signals["hold_days"] = cfg.hold_days
    signals["min_quality_percentile"] = cfg.min_quality_percentile
    signals["require_stabilization"] = cfg.require_stabilization
    return signals[_empty_trades().columns.tolist()].sort_values(["signal_date", "ticker", "cik"]).reset_index(drop=True)
