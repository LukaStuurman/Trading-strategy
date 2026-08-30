#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io import read_table
from src.data.universe import load_universe_intervals
from src.research.experiment import stable_config_hash
from src.strategies.intraday_daily import (
    daily_portfolio_metrics,
    intraday_exit_returns,
    prepare_intraday_features,
)


LIQUIDITY_GRID = [20_000_000.0, 50_000_000.0]
STOP_GRID = [None, 0.02, 0.04, 0.06]
TARGET_GRID = [None, 0.02, 0.04, 0.08]
LEVERAGE_GRID = [1.0, 1.5, 2.0, 3.0, 4.0]
MAX_POSITIONS_GRID = [3, 5, 10, 20]
ROUND_TRIP_COST_BPS = 20.0


def _base_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for adv in LIQUIDITY_GRID:
        for gap in [-0.01, -0.02, -0.03, -0.05, -0.08]:
            for prev_max in [None, 0.0, -0.03, -0.05]:
                rows.append({
                    "family": "gap_down_reversal",
                    "min_adv20": adv,
                    "gap_max": gap,
                    "prev_return_max": prev_max,
                })

        for gap in [0.01, 0.02, 0.03, 0.05, 0.08]:
            for prev_min in [None, 0.0, 0.03, 0.05]:
                for mom_min in [None, 0.0, 0.05]:
                    rows.append({
                        "family": "gap_up_continuation",
                        "min_adv20": adv,
                        "gap_min": gap,
                        "prev_return_min": prev_min,
                        "mom3_min": mom_min,
                    })

        for prev_max in [-0.03, -0.05, -0.08, -0.10, -0.15]:
            for max_abs_gap in [0.01, 0.03, 0.05]:
                rows.append({
                    "family": "prior_day_crash_rebound",
                    "min_adv20": adv,
                    "prev_return_max": prev_max,
                    "max_abs_gap": max_abs_gap,
                })

        for window in [3, 5]:
            for mom_max in [-0.05, -0.08, -0.12, -0.15, -0.20]:
                for max_abs_gap in [0.01, 0.03, 0.05]:
                    rows.append({
                        "family": f"oversold_{window}d_reversal",
                        "min_adv20": adv,
                        "momentum_column": f"mom{window}",
                        "momentum_max": mom_max,
                        "max_abs_gap": max_abs_gap,
                    })

        for prev_min in [0.03, 0.05, 0.08, 0.10]:
            for gap_max in [-0.01, -0.02, -0.03, -0.05]:
                for mom_min in [None, 0.03]:
                    rows.append({
                        "family": "strength_pullback",
                        "min_adv20": adv,
                        "prev_return_min": prev_min,
                        "gap_max": gap_max,
                        "mom3_min": mom_min,
                    })

        for prev_intraday_max in [-0.02, -0.04, -0.06, -0.08]:
            for prev_range_min in [0.03, 0.05, 0.08]:
                for gap_max in [0.0, -0.02]:
                    rows.append({
                        "family": "volatility_reversal",
                        "min_adv20": adv,
                        "prev_intraday_max": prev_intraday_max,
                        "prev_range_min": prev_range_min,
                        "gap_max": gap_max,
                    })

        for pct_max in [0.05, 0.10, 0.20]:
            for gap_max in [0.0, -0.01, -0.03]:
                for mom_max in [None, 0.0, -0.05]:
                    rows.append({
                        "family": "relative_weakness_reversal",
                        "min_adv20": adv,
                        "prev_return_percentile_max": pct_max,
                        "gap_max": gap_max,
                        "mom3_max": mom_max,
                    })
    return rows


def _base_candidates(features: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    common = (
        features["in_universe"].fillna(False)
        & (features["open"] >= 5.0)
        & (features["prev_close"] >= 5.0)
        & (features["adv20"] >= float(cfg["min_adv20"]))
        & features["gap_return"].notna()
        & features["prev_return"].notna()
        & (features["gap_return"].abs() <= 0.50)
        & (features["prev_return"].abs() <= 0.60)
    )
    family = cfg["family"]
    mask = common.copy()

    if family == "gap_down_reversal":
        mask &= features["gap_return"] <= float(cfg["gap_max"])
        if cfg["prev_return_max"] is not None:
            mask &= features["prev_return"] <= float(cfg["prev_return_max"])
        score = -features["gap_return"] + (-features["prev_return"]).clip(lower=0) * 0.20

    elif family == "gap_up_continuation":
        mask &= features["gap_return"] >= float(cfg["gap_min"])
        if cfg["prev_return_min"] is not None:
            mask &= features["prev_return"] >= float(cfg["prev_return_min"])
        if cfg["mom3_min"] is not None:
            mask &= features["mom3"] >= float(cfg["mom3_min"])
        score = features["gap_return"] + features["prev_return"].clip(lower=0) * 0.20

    elif family == "prior_day_crash_rebound":
        mask &= features["prev_return"] <= float(cfg["prev_return_max"])
        mask &= features["gap_return"].abs() <= float(cfg["max_abs_gap"])
        score = -features["prev_return"] + (-features["gap_return"]).clip(lower=0) * 0.20

    elif family.startswith("oversold_"):
        col = str(cfg["momentum_column"])
        mask &= features[col] <= float(cfg["momentum_max"])
        mask &= features["gap_return"].abs() <= float(cfg["max_abs_gap"])
        score = -features[col] + (-features["gap_return"]).clip(lower=0) * 0.10

    elif family == "strength_pullback":
        mask &= features["prev_return"] >= float(cfg["prev_return_min"])
        mask &= features["gap_return"] <= float(cfg["gap_max"])
        if cfg["mom3_min"] is not None:
            mask &= features["mom3"] >= float(cfg["mom3_min"])
        score = features["prev_return"] - features["gap_return"]

    elif family == "volatility_reversal":
        mask &= features["prev_intraday_return"] <= float(cfg["prev_intraday_max"])
        mask &= features["prev_range"] >= float(cfg["prev_range_min"])
        mask &= features["gap_return"] <= float(cfg["gap_max"])
        score = -features["prev_intraday_return"] + features["prev_range"]

    elif family == "relative_weakness_reversal":
        mask &= features["prev_return_percentile"] <= float(cfg["prev_return_percentile_max"])
        mask &= features["gap_return"] <= float(cfg["gap_max"])
        if cfg["mom3_max"] is not None:
            mask &= features["mom3"] <= float(cfg["mom3_max"])
        score = (1.0 - features["prev_return_percentile"]) + (-features["gap_return"]).clip(lower=0)

    else:
        raise ValueError(f"unknown family {family}")

    columns = ["ticker", "_instrument_id", "date", "open", "high", "low", "close"]
    out = features.loc[mask, columns].copy()
    out["signal_score"] = pd.to_numeric(score.loc[mask], errors="coerce").to_numpy()
    return out.dropna(subset=["signal_score"]).reset_index(drop=True)


def _with_exit(candidates: pd.DataFrame, stop_loss: float | None, take_profit: float | None) -> pd.DataFrame:
    out = candidates.copy()
    net, labels = intraday_exit_returns(
        out,
        stop_loss=stop_loss,
        take_profit=take_profit,
        round_trip_cost_bps=ROUND_TRIP_COST_BPS,
    )
    out["net_return"] = net
    out["exit_reason"] = labels
    return out


def _prefix(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _split_metrics(
    trades: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    leverage: float,
    max_positions: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> dict:
    row: dict[str, Any] = {}
    for name, start, end in [
        ("train", train_start, train_end),
        ("validation", validation_start, validation_end),
        ("oos", oos_start, oos_end),
        ("holdout", holdout_start, holdout_end),
    ]:
        metrics = daily_portfolio_metrics(
            trades,
            calendar,
            start=start,
            end=end,
            leverage=leverage,
            max_positions=max_positions,
        )
        row.update(_prefix(metrics, name))
    return row


def _selection_score(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    train_sharpe = pd.to_numeric(out["train_sharpe"], errors="coerce")
    val_sharpe = pd.to_numeric(out["validation_sharpe"], errors="coerce")
    val_cagr = pd.to_numeric(out["validation_cagr"], errors="coerce")
    val_dd = pd.to_numeric(out["validation_max_drawdown"], errors="coerce")
    stability = -(train_sharpe - val_sharpe).abs()
    out[f"{prefix}_sharpe_stability"] = stability

    components = [
        (train_sharpe, 0.20),
        (val_sharpe, 0.35),
        (val_cagr, 0.20),
        (val_dd, 0.15),
        (stability, 0.10),
    ]
    score = pd.Series(0.0, index=out.index, dtype=float)
    for values, weight in components:
        score += values.rank(pct=True, method="average").fillna(0.0) * weight

    eligible = (
        (pd.to_numeric(out["train_trades"], errors="coerce") >= 100)
        & (pd.to_numeric(out["validation_trades"], errors="coerce") >= 40)
        & (pd.to_numeric(out["validation_traded_days"], errors="coerce") >= 20)
        & ~out["train_blown_up"].astype(bool)
        & ~out["validation_blown_up"].astype(bool)
    )
    out[f"{prefix}_selection_score"] = score.where(eligible, -1.0)
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(_fmt(row[c]) for c in columns) + " |" for _, row in frame[columns].iterrows()]
    return "\n".join([header, sep] + rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--oos-end", default="2024-12-31")
    args = parser.parse_args()

    prices = read_table(args.prices)
    universe = load_universe_intervals(args.universe)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Preparing intraday open-time features for {len(prices):,} rows")
    features = prepare_intraday_features(prices, universe)
    calendar = pd.DatetimeIndex(sorted(features["date"].dropna().unique()))
    if len(calendar) == 0:
        raise ValueError("no trading dates")

    train_start = max(pd.Timestamp("2001-01-01"), calendar.min())
    train_end = pd.Timestamp(args.train_end).normalize()
    validation_start = train_end + pd.Timedelta(days=1)
    validation_end = pd.Timestamp(args.validation_end).normalize()
    oos_start = validation_end + pd.Timedelta(days=1)
    oos_end = pd.Timestamp(args.oos_end).normalize()
    holdout_start = oos_end + pd.Timedelta(days=1)
    holdout_end = calendar.max()

    base_configs = _base_configs()
    exit_policies = [(stop, target) for stop in STOP_GRID for target in TARGET_GRID]
    print(f"Base signals: {len(base_configs)}; exit policies: {len(exit_policies)}; signal variants: {len(base_configs) * len(exit_policies):,}")

    signal_rows: list[dict] = []
    for i, cfg in enumerate(base_configs, start=1):
        candidates = _base_candidates(features, cfg)
        if candidates.empty:
            continue
        ranked = candidates.sort_values(
            ["date", "signal_score", "ticker", "_instrument_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        baseline_candidates = ranked.groupby("date", sort=False).head(10).reset_index(drop=True)

        for stop_loss, take_profit in exit_policies:
            trades = _with_exit(baseline_candidates, stop_loss, take_profit)
            variant_cfg = {
                "signal": cfg,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            }
            row = {
                "variant_id": stable_config_hash(variant_cfg),
                "family": cfg["family"],
                "base_config_json": json.dumps(cfg, sort_keys=True),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
                "raw_candidates": len(candidates),
            }
            row.update(_split_metrics(
                trades,
                calendar,
                leverage=1.0,
                max_positions=10,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                oos_start=oos_start,
                oos_end=oos_end,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
            ))
            signal_rows.append(row)

        if i % 40 == 0:
            print(f"Completed {i}/{len(base_configs)} base signals -> {len(signal_rows):,} variants")

    signal = _selection_score(pd.DataFrame(signal_rows), "signal")
    signal = signal.sort_values(
        ["signal_selection_score", "validation_sharpe", "validation_cagr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    signal.insert(0, "signal_rank", np.arange(1, len(signal) + 1))
    signal.to_csv(output / "signal_variant_sweep.csv", index=False)
    signal.head(100).to_csv(output / "signal_leaderboard.csv", index=False)

    # Preserve family diversity before the leverage/capacity sweep.
    selected_ids: list[str] = signal.head(20)["variant_id"].astype(str).tolist()
    for _, group in signal[signal["signal_selection_score"] >= 0].groupby("family", sort=False):
        selected_ids.extend(group.head(5)["variant_id"].astype(str).tolist())
    selected_ids = list(dict.fromkeys(selected_ids))[:60]
    selected_signal = signal[signal["variant_id"].astype(str).isin(selected_ids)].copy()

    policy_rows: list[dict] = []
    top_trade_frames: list[pd.DataFrame] = []
    for _, sig in selected_signal.iterrows():
        cfg = json.loads(str(sig["base_config_json"]))
        candidates = _base_candidates(features, cfg)
        trades = _with_exit(
            candidates,
            None if pd.isna(sig["stop_loss"]) else float(sig["stop_loss"]),
            None if pd.isna(sig["take_profit"]) else float(sig["take_profit"]),
        )
        for leverage in LEVERAGE_GRID:
            for max_positions in MAX_POSITIONS_GRID:
                row = {
                    "variant_id": str(sig["variant_id"]),
                    "family": str(sig["family"]),
                    "base_config_json": str(sig["base_config_json"]),
                    "stop_loss": sig["stop_loss"],
                    "take_profit": sig["take_profit"],
                    "leverage": leverage,
                    "max_positions": max_positions,
                }
                row.update(_split_metrics(
                    trades,
                    calendar,
                    leverage=leverage,
                    max_positions=max_positions,
                    train_start=train_start,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    holdout_start=holdout_start,
                    holdout_end=holdout_end,
                ))
                policy_rows.append(row)

    policy = _selection_score(pd.DataFrame(policy_rows), "policy")
    policy = policy.sort_values(
        ["policy_selection_score", "validation_sharpe", "validation_cagr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    policy.insert(0, "policy_rank", np.arange(1, len(policy) + 1))
    policy.to_csv(output / "portfolio_policy_sweep.csv", index=False)
    policy.head(100).to_csv(output / "portfolio_leaderboard.csv", index=False)

    champions = (
        policy[policy["policy_selection_score"] >= 0]
        .sort_values("policy_rank")
        .groupby("family", sort=False)
        .head(1)
        .sort_values("policy_rank")
        .reset_index(drop=True)
    )
    champions.to_csv(output / "family_champions.csv", index=False)

    # Save accepted trades for the top overall policy and every family champion.
    export_rows = pd.concat([policy.head(1), champions], ignore_index=True).drop_duplicates(
        ["variant_id", "leverage", "max_positions"]
    )
    for _, row in export_rows.iterrows():
        cfg = json.loads(str(row["base_config_json"]))
        cand = _base_candidates(features, cfg)
        tr = _with_exit(
            cand,
            None if pd.isna(row["stop_loss"]) else float(row["stop_loss"]),
            None if pd.isna(row["take_profit"]) else float(row["take_profit"]),
        )
        tr = tr.sort_values(
            ["date", "signal_score", "ticker", "_instrument_id"],
            ascending=[True, False, True, True],
            kind="stable",
        ).groupby("date", sort=False).head(int(row["max_positions"])).copy()
        tr.insert(0, "strategy_family", row["family"])
        tr.insert(0, "variant_id", row["variant_id"])
        tr.insert(0, "leverage", row["leverage"])
        top_trade_frames.append(tr)
    if top_trade_frames:
        pd.concat(top_trade_frames, ignore_index=True).to_csv(output / "top_strategy_trades.csv", index=False)

    manifest = {
        "name": "leveraged_intraday_daily_bar_research",
        "data_rule": "historical S&P 500 membership; FINSABER adjusted OHLC; signal uses only prior-session data plus current open",
        "execution_rule": "buy at current open and exit same session at conservative stop/target/close; if stop and target both touched, stop is assumed first",
        "round_trip_cost_bps_per_notional": ROUND_TRIP_COST_BPS,
        "base_signal_configs": len(base_configs),
        "exit_policies": len(exit_policies),
        "signal_variants": len(signal),
        "policy_variants": len(policy),
        "leverage_grid": LEVERAGE_GRID,
        "max_positions_grid": MAX_POSITIONS_GRID,
        "splits": {
            "train": [train_start.date().isoformat(), train_end.date().isoformat()],
            "validation": [validation_start.date().isoformat(), validation_end.date().isoformat()],
            "oos": [oos_start.date().isoformat(), oos_end.date().isoformat()],
            "holdout": [holdout_start.date().isoformat(), holdout_end.date().isoformat()],
        },
        "selection_rule": "signal and leverage/capacity rankings use train + validation only; OOS 2022-2024 and 2025 holdout never participate in ranking",
        "families": sorted(signal["family"].unique().tolist()),
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    cols = [
        "policy_rank", "family", "leverage", "max_positions", "stop_loss", "take_profit",
        "policy_selection_score", "validation_total_return", "validation_sharpe", "validation_max_drawdown",
        "oos_total_return", "oos_sharpe", "oos_max_drawdown",
        "holdout_total_return", "holdout_sharpe", "holdout_max_drawdown", "holdout_trades",
    ]
    report = [
        "# Leveraged intraday strategy sweep",
        "",
        f"Tested **{len(signal):,} signal/exit variants** from **{len(base_configs)} base signals**, then **{len(policy):,} leverage/capacity policies** on the strongest train/validation candidates.",
        f"Costs: **{ROUND_TRIP_COST_BPS:.0f} bps round trip per notional**. All entries are at the open and all exits occur the same trading day.",
        "Signals use only data known by the open. Daily OHLC cannot reveal whether a stop or target happened first, so simultaneous touches are scored as a stop.",
        "",
        f"Train: **{train_start.date()}–{train_end.date()}**; validation: **{validation_start.date()}–{validation_end.date()}**; OOS: **{oos_start.date()}–{oos_end.date()}**; final holdout: **{holdout_start.date()}–{holdout_end.date()}**.",
        "Neither OOS nor holdout columns participate in selection.",
        "",
        "## Best overall policies selected on train + validation",
        "",
        _markdown(policy.head(10), cols),
        "",
        "## Best policy per strategy family",
        "",
        _markdown(champions, cols),
        "",
        "## Guardrails",
        "",
        "- Leverage multiplies both profit/loss and transaction cost exposure; unused position slots remain cash.",
        "- A portfolio day at or below -100% is treated as a blow-up and equity stays at zero.",
        "- Historical S&P membership is enforced on each trade date to reduce survivorship bias.",
        "- Same-day stop/target ordering is unknowable from daily bars; stop-first makes the test conservative but intraday bars are still required before live use.",
    ]
    (output / "intraday_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(policy.head(15)[cols].to_string(index=False))
    print("\nFamily champions:\n")
    print(champions[cols].to_string(index=False))


if __name__ == "__main__":
    main()
