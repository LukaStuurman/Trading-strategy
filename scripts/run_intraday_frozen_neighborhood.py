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
from src.strategies.intraday_daily import daily_portfolio_metrics, prepare_intraday_features
from scripts.run_intraday_recent_focus import candidates, with_exit, period_metrics, selection_score

ROUND_TRIP_COST_BPS = 20.0
LEVERAGE_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
MAX_POSITIONS_GRID = [3, 5, 7, 10]
TARGETS_GAP = [None, 0.04, 0.05, 0.06, 0.07, 0.08]
TARGETS_REL = [None, 0.06, 0.07, 0.08, 0.09, 0.10]

FROZEN = {
    "gap_down_reversal": {
        "signal": {
            "family": "gap_down_reversal",
            "min_adv20": 50_000_000.0,
            "gap_max": -0.04,
            "prev_return_max": -0.02,
            "mom3_max": -0.05,
            "market_prev_return_max": None,
        },
        "target": 0.06,
        "leverage": 3.0,
        "max_positions": 7,
    },
    "relative_weakness_reversal": {
        "signal": {
            "family": "relative_weakness_reversal",
            "min_adv20": 50_000_000.0,
            "prev_return_percentile_max": 0.15,
            "gap_max": -0.04,
            "mom3_max": -0.05,
            "market_prev_return_max": 0.0,
        },
        "target": 0.08,
        "leverage": 2.5,
        "max_positions": 5,
    },
}


def local_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Tight neighborhood around the frozen gap-down baseline.
    for adv in [20_000_000.0, 50_000_000.0, 100_000_000.0]:
        for gap in [-0.035, -0.04, -0.045, -0.05]:
            for prev in [-0.015, -0.02, -0.025, -0.03]:
                for mom3 in [-0.04, -0.05, -0.06, -0.08]:
                    for market in [None, 0.0]:
                        rows.append({
                            "family": "gap_down_reversal",
                            "min_adv20": adv,
                            "gap_max": gap,
                            "prev_return_max": prev,
                            "mom3_max": mom3,
                            "market_prev_return_max": market,
                        })

    # Tight neighborhood around the frozen relative-weakness baseline.
    for adv in [20_000_000.0, 50_000_000.0, 100_000_000.0]:
        for pct in [0.10, 0.125, 0.15, 0.175, 0.20]:
            for gap in [-0.035, -0.04, -0.045, -0.05]:
                for mom3 in [-0.04, -0.05, -0.06, -0.08]:
                    for market in [None, 0.0]:
                        rows.append({
                            "family": "relative_weakness_reversal",
                            "min_adv20": adv,
                            "prev_return_percentile_max": pct,
                            "gap_max": gap,
                            "mom3_max": mom3,
                            "market_prev_return_max": market,
                        })
    return rows


def is_frozen_signal(cfg: dict[str, Any], target: float | None) -> bool:
    frozen = FROZEN[str(cfg["family"])]
    return cfg == frozen["signal"] and target == frozen["target"]


def prefixed(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def full_period_metrics(trades: pd.DataFrame, calendar: pd.DatetimeIndex, *, leverage: float, max_positions: int) -> dict[str, Any]:
    row = period_metrics(trades, calendar, leverage=leverage, max_positions=max_positions)
    m = daily_portfolio_metrics(
        trades,
        calendar,
        start=pd.Timestamp("2021-01-01"),
        end=pd.Timestamp("2025-12-31"),
        leverage=leverage,
        max_positions=max_positions,
    )
    row.update(prefixed(m, "all_2021_2025"))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    prices = read_table(args.prices)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices[prices["date"] >= pd.Timestamp("2020-10-01")].copy()
    universe = load_universe_intervals(args.universe)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    features = prepare_intraday_features(prices, universe)
    features = features[(features["date"] >= "2021-01-01") & (features["date"] <= "2025-12-31")].copy()
    calendar = pd.DatetimeIndex(sorted(features["date"].dropna().unique()))

    configs = local_configs()
    signal_rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs, 1):
        cand = candidates(features, cfg)
        if cand.empty:
            continue
        cand = cand.sort_values(["date", "signal_score", "ticker", "_instrument_id"], ascending=[True, False, True, True], kind="stable")
        baseline = cand.groupby("date", sort=False).head(10).reset_index(drop=True)
        targets = TARGETS_GAP if cfg["family"] == "gap_down_reversal" else TARGETS_REL
        for target in targets:
            trades = with_exit(baseline, None, target)
            variant_cfg = {"signal": cfg, "stop_loss": None, "take_profit": target, "round_trip_cost_bps": ROUND_TRIP_COST_BPS}
            row = {
                "variant_id": stable_config_hash(variant_cfg),
                "family": cfg["family"],
                "base_config_json": json.dumps(cfg, sort_keys=True),
                "stop_loss": None,
                "take_profit": target,
                "raw_candidates": len(cand),
                "is_frozen_signal": is_frozen_signal(cfg, target),
            }
            row.update(full_period_metrics(trades, calendar, leverage=1.0, max_positions=10))
            signal_rows.append(row)
        if i % 100 == 0:
            print(f"Completed {i}/{len(configs)} local configs -> {len(signal_rows)} signal variants")

    signal = selection_score(pd.DataFrame(signal_rows), "signal")
    signal = signal.sort_values(["signal_selection_score", "validation_sharpe", "validation_cagr"], ascending=[False, False, False]).reset_index(drop=True)
    signal.insert(0, "signal_rank", np.arange(1, len(signal) + 1))
    signal.to_csv(outdir / "signal_neighborhood_sweep.csv", index=False)
    signal.head(200).to_csv(outdir / "signal_leaderboard.csv", index=False)

    # Keep broad local diversity: top 30 each family plus the frozen baselines.
    selected = []
    for _, g in signal[signal["signal_selection_score"] >= 0].groupby("family", sort=False):
        selected.extend(g.head(30)["variant_id"].astype(str).tolist())
    selected.extend(signal[signal["is_frozen_signal"]]["variant_id"].astype(str).tolist())
    selected = list(dict.fromkeys(selected))
    selected_signal = signal[signal["variant_id"].astype(str).isin(selected)].copy()

    policy_rows: list[dict[str, Any]] = []
    for _, sig in selected_signal.iterrows():
        cfg = json.loads(str(sig["base_config_json"]))
        cand = candidates(features, cfg)
        trades = with_exit(cand, None, None if pd.isna(sig["take_profit"]) else float(sig["take_profit"]))
        for lev in LEVERAGE_GRID:
            for cap in MAX_POSITIONS_GRID:
                frozen = FROZEN[str(sig["family"])]
                row = {
                    "variant_id": str(sig["variant_id"]),
                    "family": str(sig["family"]),
                    "base_config_json": str(sig["base_config_json"]),
                    "stop_loss": None,
                    "take_profit": sig["take_profit"],
                    "leverage": lev,
                    "max_positions": cap,
                    "is_frozen_policy": bool(sig["is_frozen_signal"]) and lev == frozen["leverage"] and cap == frozen["max_positions"],
                }
                row.update(full_period_metrics(trades, calendar, leverage=lev, max_positions=cap))
                policy_rows.append(row)

    policy = selection_score(pd.DataFrame(policy_rows), "policy")
    policy = policy.sort_values(["policy_selection_score", "validation_sharpe", "validation_cagr"], ascending=[False, False, False]).reset_index(drop=True)
    policy.insert(0, "policy_rank", np.arange(1, len(policy) + 1))
    policy.to_csv(outdir / "portfolio_neighborhood_sweep.csv", index=False)
    policy.head(200).to_csv(outdir / "portfolio_leaderboard.csv", index=False)

    baseline_compare = policy[policy["is_frozen_policy"]].copy().sort_values("family")
    baseline_compare.to_csv(outdir / "frozen_baseline_comparison.csv", index=False)

    family_best = policy[policy["policy_selection_score"] >= 0].sort_values("policy_rank").groupby("family", sort=False).head(1)
    family_best.to_csv(outdir / "family_best.csv", index=False)

    # Conservative shortlist: positive in every pre-2025 year and >=15 validation trades.
    robust = policy[
        (policy["policy_positive_year_share_2021_2024"] >= 1.0)
        & (policy["validation_trades"] >= 15)
        & (policy["validation_sharpe"] > 0)
    ].copy()
    robust = robust.sort_values(["policy_selection_score", "validation_max_drawdown"], ascending=[False, False])
    robust.head(100).to_csv(outdir / "robust_shortlist.csv", index=False)

    manifest = {
        "name": "frozen_intraday_neighborhood_robustness",
        "frozen_baselines_file": "strategies/frozen_intraday_baselines_2026-08-30.json",
        "families": sorted(signal["family"].unique().tolist()),
        "base_configs": len(configs),
        "signal_variants": len(signal),
        "policy_variants": len(policy),
        "selection": "2021-2023 train + 2024 validation only; 2025 diagnostic excluded from ranking",
        "stops": [None],
        "gap_targets": TARGETS_GAP,
        "relative_weakness_targets": TARGETS_REL,
        "leverage_grid": LEVERAGE_GRID,
        "max_positions_grid": MAX_POSITIONS_GRID,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
    }
    (outdir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    cols = [
        "policy_rank", "family", "leverage", "max_positions", "take_profit", "is_frozen_policy",
        "policy_selection_score", "y2021_return", "y2022_return", "y2023_return", "y2024_return",
        "validation_sharpe", "validation_max_drawdown", "y2025_return", "diagnostic_2025_sharpe",
        "diagnostic_2025_max_drawdown", "diagnostic_2025_trades"
    ]
    lines = [
        "# Frozen intraday neighborhood test",
        "",
        f"Tested {len(signal):,} local signal/exit variants and {len(policy):,} leverage/capacity policies around two frozen baselines.",
        "2025 is diagnostic only and never participates in ranking.",
        "",
        "## Top local policies",
        "",
        policy.head(20)[cols].to_markdown(index=False),
        "",
        "## Frozen baselines in the new local sweep",
        "",
        baseline_compare[cols].to_markdown(index=False),
        "",
        "## Best per family",
        "",
        family_best[cols].to_markdown(index=False),
        "",
        "## Guardrails",
        "",
        "- No stop-loss variants are used here, matching the frozen winners and avoiding daily-bar stop/target path ambiguity.",
        "- Historical S&P membership and lagged open-time features remain enforced.",
        "- A local variant is not treated as proven merely because its 2025 diagnostic is high.",
    ]
    (outdir / "neighborhood_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(policy.head(20)[cols].to_string(index=False))
    print("\nFrozen baselines:\n", baseline_compare[cols].to_string(index=False))


if __name__ == "__main__":
    main()
