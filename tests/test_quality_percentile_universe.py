import pandas as pd

from src.strategies.quality_bad_news import _attach_quality_percentile


def test_quality_percentile_ignores_non_members_in_cross_section():
    frame = pd.DataFrame({
        "ticker": ["LOW", "HIGH", "OUT"],
        "date": pd.to_datetime(["2025-01-02"] * 3),
        "in_universe": [True, True, False],
        "roe": [0.10, 0.30, 9.99],
    })
    ranked = _attach_quality_percentile(frame)
    low = ranked.loc[ranked["ticker"] == "LOW", "quality_percentile"].iloc[0]
    high = ranked.loc[ranked["ticker"] == "HIGH", "quality_percentile"].iloc[0]
    outside = ranked.loc[ranked["ticker"] == "OUT", "quality_percentile"].iloc[0]
    assert low == 0.5
    assert high == 1.0
    assert pd.isna(outside)
