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


ROUND_TRIP_COST_BPS = 20.0
LIQUIDITY_GRID = [20_000_000.0, 50_000_000.0, 100_000_000.0]
STOP_GRID = [None, 0.01, 0.015, 0.02, 0.025, 0.03]
TARGET_GRID = [None, 0.02, 0.04, 0.06, 0.08]
LEVERAGE_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
MAX_POSITIONS_GRID = [3, 5, 7, 10, 15, 20]


def focused_configs() -> list[dict[str, Any]]:
    """Refine only the families that remained useful in the 2025 study."""
    rows: list[dict[str, Any]] = []

    # 2025 winner family: relative weakness + another negative opening gap.
    for adv in LIQUIDITY_GRID:
        for pct in [0.10, 0.15, 0.20, 0.25]:
            for gap in [-0.015, -0.02, -0.025, -0.03, -0.04]:
                for mom3 in [None, -0.03, -0.05, -0.08]:
                    for market_max in [None, 0.0]:
                        rows.append({
                            "family": "relative_weakness_reversal",
                            "min_adv20": adv,
                            "prev_return_percentile_max": pct,
                            "gap_max": gap,
                            "mom3_max": mom3,
                            "market_prev_return_max": market_max,
                        })

    # The simpler 2025 gap-down pattern: prior loss + fresh gap down.
    for adv in LIQUIDITY_GRID:
        for gap in [-0.02, -0.025, -0.03, -0.04, -0.05]:
            for prev_max in [0.0, -0.02, -0.03, -0.05, -0.08]:
                for mom3 in [None, -0.03, -0.05]:
                    for market_max in [None, 0.0]:
                        rows.append({
                            "family": "gap_down_reversal",
                            "min_adv20": adv,
                            "gap_max": gap,
                            "prev_return_max": prev_max,
                            "mom3_max": mom3,
                            "market_prev_return_max": market_max,
                        })

    # A third family that was positive in 2025: deep 5-day oversold reversal.
    for adv in LIQUIDITY_GRID:
        for mom5 in [-0.10, -0.12, -0.15, -0.18, -0.20]:
            for max_gap in [0.02, 0.03, 0.05, 0.08]:
                for pct_max in [None, 0.30]:
                    rows.append({
                        "family": "oversold_5d_reversal",
                        "min_adv20": adv,
                        "mom5_max": mom5,
                        "max_abs_gap": max_gap,
                        "prev_return_percentile_max": pct_max,
                    })
    return rows


def candidates(features: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
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
    family = str(cfg["family"])
    mask = common.copy()

    if family == "relative_weakness_reversal":
        mask &= features["prev_return_percentile"] <= float(cfg["prev_return_percentile_max"])
        mask &= features["gap_return"] <= float(cfg["gap_max"])
        if cfg["mom3_max"] is not None:
            mask &= features["mom3"] <= float(cfg["mom3_max"])
        if cfg["market_prev_return_max"] is not None:
            mask &= features["market_prev_return"] <= float(cfg["market_prev_return_max"])
        score = (
            (1.0 - features["prev_return_percentile"])
            + (-features["gap_return"]).clip(lower=0.0) * 2.0
            + (-features["mom3"]).clip(lower=0.0) * 0.30
        )

    elif family == "gap_down_reversal":
        mask &= features["gap_return"] <= float(cfg["gap_max"])
        mask &= features["prev_return"] <= float(cfg["prev_return_max"])
        if cfg["mom3_max"] is not None:
            mask &= features["mom3"] <= float(cfg["mom3_max"])
        if cfg["market_prev_return_max"] is not None:
            mask &= features["market_prev_return"] <= float(cfg["market_prev_return_max"])
        score = (
            -features["gap_return"]
            + (-features["prev_return"]).clip(lower=0.0) * 0.50
            + (-features["mom3"]).clip(lower=0.0) * 0.20
        )

    elif family == "oversold_5d_reversal":
        mask &= features["mom5"] <= float(cfg["mom5_max"])
        mask &= features["gap_return"].abs() <= float(cfg["max_abs_gap"])
        if cfg["prev_return_percentile_max"] is not None:
            mask &= features["prev_return_percentile"] <= float(cfg["prev_return_percentile_max"])
        score = -features["mom5"] + (-features["gap_return"]).clip(lower=0.0) * 0.20

    else:
        raise ValueError(f"unknown family {family}")

    cols = ["ticker", "_instrument_id", "date", "open", "high", "low", "close"]
    out = features.loc[mask, cols].copy()
    out["signal_score"] = pd.to_numeric(score.loc[mask], errors="coerce").to_numpy()
    return out.dropna(subset=["signal_score"]).reset_index(drop=True)


def with_exit(frame: pd.DataFrame, stop: float | None, target: float | None) -> pd.DataFrame:
    out = frame.copy()
    net, labels = intraday_exit_returns(
        out,
        stop_loss=stop,
        take_profit=target,
        round_trip_cost_bps=ROUND_TRIP_COST_BPS,
    )
    out["net_return"] = net
    out["exit_reason"] = labels
    return out


def prefixed(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def period_metrics(
    trades: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    leverage: float,
    max_positions: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    periods = {
        "train": (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")),
        "validation": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        "diagnostic_2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
    }
    for name, (start, end) in periods.items():
        row.update(prefixed(daily_portfolio_metrics(
            trades,
            calendar,
            start=start,
            end=end,
            leverage=leverage,
            max_positions=max_positions,
        ), name))

    for year in range(2021, 2026):
        m = daily_portfolio_metrics(
            trades,
            calendar,
            start=pd.Timestamp(f"{year}-01-01"),
            end=pd.Timestamp(f"{year}-12-31"),
            leverage=leverage,
            max_positions=max_positions,
        )
        row[f"y{year}_return"] = m["total_return"]
        row[f"y{year}_sharpe"] = m["sharpe"]
        row[f"y{year}_max_drawdown"] = m["max_drawdown"]
        row[f"y{year}_trades"] = m["trades"]
    return row


def selection_score(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rank from 2021-2024 only. 2025 is deliberately excluded."""
    out = frame.copy()
    train_sharpe = pd.to_numeric(out["train_sharpe"], errors="coerce")
    val_sharpe = pd.to_numeric(out["validation_sharpe"], errors="coerce")
    val_cagr = pd.to_numeric(out["validation_cagr"], errors="coerce")
    val_dd = pd.to_numeric(out["validation_max_drawdown"], errors="coerce")
    stability = -(train_sharpe - val_sharpe).abs()
    positive_year_share = pd.DataFrame({
        year: pd.to_numeric(out[f"y{year}_return"], errors="coerce") > 0
        for year in range(2021, 2025)
    }).mean(axis=1)
    worst_pre2025_return = pd.concat(
        [pd.to_numeric(out[f"y{year}_return"], errors="coerce") for year in range(2021, 2025)],
        axis=1,
    ).min(axis=1)

    out[f"{prefix}_sharpe_stability"] = stability
    out[f"{prefix}_positive_year_share_2021_2024"] = positive_year_share
    out[f"{prefix}_worst_year_return_2021_2024"] = worst_pre2025_return

    score = pd.Series(0.0, index=out.index, dtype=float)
    components = [
        (train_sharpe, 0.18),
        (val_sharpe, 0.27),
        (val_cagr, 0.15),
        (val_dd, 0.12),
        (stability, 0.08),
        (positive_year_share, 0.12),
        (worst_pre2025_return, 0.08),
    ]
    for values, weight in components:
        score += pd.Series(values).rank(pct=True, method="average").fillna(0.0) * weight

    eligible = (
        (pd.to_numeric(out["train_trades"], errors="coerce") >= 60)
        & (pd.to_numeric(out["validation_trades"], errors="coerce") >= 15)
        & ~out["train_blown_up"].astype(bool)
        & ~out["validation_blown_up"].astype(bool)
    )
    out[f"{prefix}_selection_score"] = score.where(eligible, -1.0)
    return out


def fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prices = read_table(args.prices)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    # Keep enough 2020 history to seed lagged/rolling features, but score only 2021-2025.
    prices = prices[prices["date"] >= pd.Timestamp("2020-10-01")].copy()
    universe = load_universe_intervals(args.universe)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Preparing recent intraday features for {len(prices):,} price rows")
    features = prepare_intraday_features(prices, universe)
    features = features[(features["date"] >= pd.Timestamp("2021-01-01")) & (features["date"] <= pd.Timestamp("2025-12-31"))].copy()
    calendar = pd.DatetimeIndex(sorted(features["date"].dropna().unique()))

    configs = focused_configs()
    exits = [(s, t) for s in STOP_GRID for t in TARGET_GRID]
    print(f"Focused base configs: {len(configs):,}; exit policies: {len(exits)}; signal variants: {len(configs) * len(exits):,}")

    signal_rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs, start=1):
        cand = candidates(features, cfg)
        if cand.empty:
            continue
        cand = cand.sort_values(
            ["date", "signal_score", "ticker", "_instrument_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        baseline = cand.groupby("date", sort=False).head(10).reset_index(drop=True)
        for stop, target in exits:
            trades = with_exit(baseline, stop, target)
            variant_cfg = {
                "signal": cfg,
                "stop_loss": stop,
                "take_profit": target,
                "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            }
            row = {
                "variant_id": stable_config_hash(variant_cfg),
                "family": cfg["family"],
                "base_config_json": json.dumps(cfg, sort_keys=True),
                "stop_loss": stop,
                "take_profit": target,
                "raw_candidates": len(cand),
            }
            row.update(period_metrics(trades, calendar, leverage=1.0, max_positions=10))
            signal_rows.append(row)
        if i % 100 == 0:
            print(f"Completed {i}/{len(configs)} base configs -> {len(signal_rows):,} signal variants")

    signal = selection_score(pd.DataFrame(signal_rows), "signal")
    signal = signal.sort_values(
        ["signal_selection_score", "validation_sharpe", "validation_cagr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    signal.insert(0, "signal_rank", np.arange(1, len(signal) + 1))
    signal.to_csv(output / "signal_variant_sweep.csv", index=False)
    signal.head(150).to_csv(output / "signal_leaderboard.csv", index=False)

    selected_ids: list[str] = signal[signal["signal_selection_score"] >= 0].head(30)["variant_id"].astype(str).tolist()
    for _, group in signal[signal["signal_selection_score"] >= 0].groupby("family", sort=False):
        selected_ids.extend(group.head(12)["variant_id"].astype(str).tolist())
    selected_ids = list(dict.fromkeys(selected_ids))[:60]
    selected_signal = signal[signal["variant_id"].astype(str).isin(selected_ids)].copy()

    policy_rows: list[dict[str, Any]] = []
    for _, sig in selected_signal.iterrows():
        cfg = json.loads(str(sig["base_config_json"]))
        cand = candidates(features, cfg)
        trades = with_exit(
            cand,
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
                row.update(period_metrics(trades, calendar, leverage=leverage, max_positions=max_positions))
                policy_rows.append(row)

    policy = selection_score(pd.DataFrame(policy_rows), "policy")
    policy = policy.sort_values(
        ["policy_selection_score", "validation_sharpe", "validation_cagr"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    policy.insert(0, "policy_rank", np.arange(1, len(policy) + 1))
    policy.to_csv(output / "portfolio_policy_sweep.csv", index=False)
    policy.head(150).to_csv(output / "portfolio_leaderboard.csv", index=False)

    champions = (
        policy[policy["policy_selection_score"] >= 0]
        .sort_values("policy_rank")
        .groupby("family", sort=False)
        .head(1)
        .sort_values("policy_rank")
        .reset_index(drop=True)
    )
    champions.to_csv(output / "family_champions.csv", index=False)

    manifest = {
        "name": "recent_intraday_refinement_2021_2025",
        "seed_reason": "families chosen because they remained positive/useful in the prior 2025 diagnostic study",
        "data_window": ["2021-01-01", "2025-12-31"],
        "feature_warmup_start": "2020-10-01",
        "selection_train": ["2021-01-01", "2023-12-31"],
        "selection_validation": ["2024-01-01", "2024-12-31"],
        "diagnostic_2025": ["2025-01-01", "2025-12-31"],
        "warning_2025": "2025 is not pristine because family choice was informed by the earlier 2025 study; it never participates in parameter or leverage ranking here",
        "families": sorted(signal["family"].unique().tolist()),
        "base_configs": len(configs),
        "exit_policies": len(exits),
        "signal_variants": len(signal),
        "policy_variants": len(policy),
        "cost_bps_round_trip_per_notional": ROUND_TRIP_COST_BPS,
        "leverage_grid": LEVERAGE_GRID,
        "max_positions_grid": MAX_POSITIONS_GRID,
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    cols = [
        "policy_rank", "family", "leverage", "max_positions", "stop_loss", "take_profit",
        "policy_selection_score", "y2021_return", "y2022_return", "y2023_return", "y2024_return",
        "validation_sharpe", "validation_max_drawdown", "y2025_return", "diagnostic_2025_sharpe",
        "diagnostic_2025_max_drawdown", "diagnostic_2025_trades",
    ]
    report = [
        "# Recent-year intraday refinement",
        "",
        f"Focused on the three families that remained useful in the previous 2025 study. Tested **{len(signal):,} signal/exit variants** from **{len(configs):,} focused base configs**, followed by **{len(policy):,} leverage/capacity policies**.",
        "",
        "Only recent years matter here: 2021-2023 train, 2024 validation, 2025 diagnostic. 2025 never participates in ranking because the seed families were already chosen with knowledge of their 2025 behavior.",
        f"Execution costs remain **{ROUND_TRIP_COST_BPS:.0f} bps round trip per notional** and all trades open and close the same day.",
        "",
        "## Best recent-regime policies",
        "",
        markdown(policy.head(15), cols),
        "",
        "## Best per seed family",
        "",
        markdown(champions, cols),
        "",
        "## Interpretation guardrails",
        "",
        "- Ranking uses 2021-2024 only; 2025 is diagnostic, not a clean holdout.",
        "- The score rewards Sharpe, 2024 CAGR/drawdown, train-validation stability, positive-year share and the worst pre-2025 year.",
        "- Historical S&P membership, lagged liquidity/momentum and open-time signals remain enforced.",
        "- Daily bars still cannot reveal exact intraday stop/target ordering; simultaneous touches are scored stop-first.",
    ]
    (output / "recent_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(policy.head(15)[cols].to_string(index=False))
    print("\nFamily champions:\n")
    print(champions[cols].to_string(index=False))


if __name__ == "__main__":
    main()
