from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data.universe import attach_membership


@dataclass(frozen=True)
class QualityDipConfig:
    drop_threshold: float = -0.10
    wait_days: int = 1
    hold_days: int = 20
    min_roe: float = 0.12
    min_fcf_margin: float = 0.05
    max_debt_to_equity: float = 1.5
    min_current_ratio: float = 1.0
    min_market_cap: float = 5_000_000_000
    min_quality_percentile: float = 0.0
    require_stabilization: bool = False
    round_trip_cost_bps: float = 10.0


CORE_FUNDAMENTALS = ["roe", "fcf_margin", "debt_to_equity", "current_ratio", "market_cap"]
POSITIVE_QUALITY_FACTORS = [
    "roe", "fcf_margin", "current_ratio", "roa", "operating_margin",
    "fcf_to_net_income", "asset_turnover",
]
NEGATIVE_QUALITY_FACTORS = ["debt_to_equity", "net_debt_to_equity"]


def _quality_mask(f: pd.DataFrame, cfg: QualityDipConfig) -> pd.Series:
    return (
        (f["roe"] >= cfg.min_roe)
        & (f["fcf_margin"] >= cfg.min_fcf_margin)
        & (f["debt_to_equity"] <= cfg.max_debt_to_equity)
        & (f["current_ratio"] >= cfg.min_current_ratio)
        & (f["market_cap"] >= cfg.min_market_cap)
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


def prepare_features(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a causal daily feature table once for a whole parameter sweep.

    Every price row receives only the most recent fundamental row whose
    `available_date <= date`. Quality percentiles are then calculated
    cross-sectionally using only those as-of values.
    """
    p = prices.copy()
    f = fundamentals.copy()
    p["date"] = pd.to_datetime(p["date"])
    f["available_date"] = pd.to_datetime(f["available_date"])
    p["ticker"] = p["ticker"].astype(str).str.upper()
    f["ticker"] = f["ticker"].astype(str).str.upper()
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    f = f.sort_values(["ticker", "available_date"]).reset_index(drop=True)
    p["daily_return"] = p.groupby("ticker")["close"].pct_change()

    merged = []
    fundamental_cols = [c for c in f.columns if c != "ticker"]
    for ticker, gp in p.groupby("ticker", sort=False):
        gf = f[f["ticker"] == ticker][fundamental_cols].sort_values("available_date")
        gp = gp.sort_values("date")
        if gf.empty:
            x = gp.copy()
            for col in fundamental_cols:
                x[col] = pd.NaT if col == "available_date" else np.nan
        else:
            x = pd.merge_asof(
                gp,
                gf,
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        merged.append(x)
    out = pd.concat(merged, ignore_index=True) if merged else p.copy()
    out = attach_membership(out, universe)
    out = _attach_quality_percentile(out)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def generate_trades(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: QualityDipConfig,
    *,
    universe: pd.DataFrame | None = None,
    prepared: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate long-only quality mean-reversion trades without look-ahead.

    `wait_days=0` means next-session open. If stabilization is required, the
    next full session must close above the crash-day close (and not make a lower
    low when low data exists), so the earliest entry moves to the following open.
    """
    features = prepared.copy() if prepared is not None else prepare_features(prices, fundamentals, universe)
    signals = features[
        (features["daily_return"] <= cfg.drop_threshold)
        & features["in_universe"].fillna(False)
    ].copy()
    rows: list[dict] = []

    for sig in signals.itertuples(index=False):
        latest = pd.DataFrame([{c: getattr(sig, c, np.nan) for c in CORE_FUNDAMENTALS}])
        if not bool(_quality_mask(latest, cfg).iloc[0]):
            continue
        quality_percentile = float(getattr(sig, "quality_percentile", np.nan))
        if not np.isfinite(quality_percentile) or quality_percentile < cfg.min_quality_percentile:
            continue

        tp = features[features["ticker"] == sig.ticker].reset_index(drop=True)
        positions = tp.index[tp["date"] == sig.date]
        if len(positions) != 1:
            continue
        signal_i = int(positions[0])
        earliest_entry = signal_i + 1

        if cfg.require_stabilization:
            confirm_i = signal_i + 1
            if confirm_i >= len(tp):
                continue
            confirm = tp.iloc[confirm_i]
            signal_row = tp.iloc[signal_i]
            stabilized = float(confirm["close"]) > float(signal_row["close"])
            if "low" in tp.columns and pd.notna(confirm.get("low")) and pd.notna(signal_row.get("low")):
                stabilized = stabilized and float(confirm["low"]) >= float(signal_row["low"])
            if not stabilized:
                continue
            earliest_entry = confirm_i + 1

        entry_i = max(signal_i + cfg.wait_days + 1, earliest_entry)
        exit_i = entry_i + cfg.hold_days
        if exit_i >= len(tp):
            continue
        entry = tp.iloc[entry_i]
        exit_ = tp.iloc[exit_i]
        gross = float(exit_["close"] / entry["open"] - 1.0)
        cost = cfg.round_trip_cost_bps / 10_000.0
        rows.append({
            "ticker": sig.ticker,
            "signal_date": sig.date,
            "signal_return": float(sig.daily_return),
            "quality_percentile": quality_percentile,
            "entry_date": entry["date"],
            "entry_price": float(entry["open"]),
            "exit_date": exit_["date"],
            "exit_price": float(exit_["close"]),
            "gross_return": gross,
            "net_return": gross - cost,
            "drop_threshold": cfg.drop_threshold,
            "wait_days": cfg.wait_days,
            "hold_days": cfg.hold_days,
            "min_quality_percentile": cfg.min_quality_percentile,
            "require_stabilization": cfg.require_stabilization,
        })
    return pd.DataFrame(rows)
