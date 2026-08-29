import math

import pandas as pd

from scripts.build_mirror_fundamentals import build_from_mirror_facts
from scripts.download_sec_facts_mirror import cik_bucket, normalize_cik
from src.strategies.quality_bad_news import prepare_features


def _fact(tag, val, filed="2025-02-15", start=None, end="2024-12-31", unit="USD", accn="0001-25-000001"):
    return {
        "cik": "0000000001", "namespace": "us-gaap", "tag": tag, "unit": unit,
        "start": start, "end": end, "val": val, "accn": accn, "fy": 2024,
        "fp": "FY", "form": "10-K", "filed": filed, "frame": None,
    }


def test_cik_normalization_and_mirror_sharding():
    assert normalize_cik(320193) == "0000320193"
    assert normalize_cik("0000320193") == "0000320193"
    assert cik_bucket("0000320193") == 320193 % 64


def test_mirror_fundamentals_use_next_day_after_filed_date_and_do_not_look_ahead():
    prices = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"], "cik": [1, 1, 1],
        "date": pd.to_datetime(["2024-12-31", "2025-02-15", "2026-02-15"]),
        "open": [49.0, 50.0, 55.0], "high": [51.0, 51.0, 56.0],
        "low": [48.0, 49.0, 54.0], "close": [50.0, 50.0, 55.0],
        "raw_close": [50.0, 50.0, 55.0], "volume": [1_000_000] * 3,
    })
    facts = [
        _fact("Revenues", 1000.0, start="2024-01-01"),
        _fact("NetIncomeLoss", 100.0, start="2024-01-01"),
        _fact("OperatingIncomeLoss", 150.0, start="2024-01-01"),
        _fact("NetCashProvidedByUsedInOperatingActivities", 160.0, start="2024-01-01"),
        _fact("PaymentsToAcquirePropertyPlantAndEquipment", 60.0, start="2024-01-01"),
        _fact("StockholdersEquity", 500.0), _fact("Assets", 1000.0),
        _fact("AssetsCurrent", 300.0), _fact("LiabilitiesCurrent", 150.0),
        _fact("CashAndCashEquivalentsAtCarryingValue", 20.0),
        _fact("ShortTermBorrowings", 20.0), _fact("LongTermDebtNoncurrent", 80.0),
        # Some issuers expose the US-GAAP balance-sheet tag instead of the DEI
        # cover-page tag. It must still produce a causal market-cap estimate.
        _fact("CommonStockSharesOutstanding", 100.0, unit="shares"),
        _fact("Revenues", 100.0, filed="2026-02-15", start="2025-01-01", end="2025-12-31", accn="0001-26-000001"),
        _fact("NetIncomeLoss", -500.0, filed="2026-02-15", start="2025-01-01", end="2025-12-31", accn="0001-26-000001"),
    ]
    result, audit, meta = build_from_mirror_facts(prices, pd.DataFrame(facts))
    first = result[result["available_date"] == "2025-02-16"].iloc[0]
    assert first["source_filed_date"] == "2025-02-15"
    assert not (result["available_date"] == result["source_filed_date"]).any()
    assert math.isclose(first["roe"], 0.20)
    assert math.isclose(first["fcf_margin"], 0.10)
    assert math.isclose(first["debt_to_equity"], 0.20)
    assert math.isclose(first["current_ratio"], 2.0)
    assert math.isclose(first["market_cap"], 5000.0)
    share_audit = audit[audit["metric"] == "shares"].iloc[0]
    assert "CommonStockSharesOutstanding" in share_audit["selected_tags"]
    assert first["fundamental_source"] == "sec_companyfacts_mirror_pit"
    assert meta["availability_lag_days"] == 1
    assert meta["built_tickers"] == 1
    assert not audit.empty


def test_prepare_features_isolates_reused_ticker_by_cik():
    prices = pd.DataFrame({
        "ticker": ["AAA"] * 6, "cik": [1, 1, 1, 2, 2, 2],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2025-01-01", "2025-01-02", "2025-01-03"]),
        "open": [100, 99, 98, 100, 80, 82], "close": [100, 99, 98, 100, 80, 82],
    })
    fundamentals = pd.DataFrame({
        "ticker": ["AAA"], "cik": ["0000000001"], "available_date": ["2019-12-20"],
        "roe": [0.20], "fcf_margin": [0.10], "debt_to_equity": [0.2],
        "current_ratio": [2.0], "market_cap": [20_000_000_000],
    })
    prepared = prepare_features(prices, fundamentals)
    second_issuer = prepared[prepared["cik"] == "0000000002"]
    assert second_issuer["roe"].isna().all()
    assert pd.isna(second_issuer.iloc[0]["daily_return"])
