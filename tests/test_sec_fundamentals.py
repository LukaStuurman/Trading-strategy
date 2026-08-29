import pandas as pd

from scripts.build_sec_fundamentals import build_one


def fact(rows, unit="USD"):
    return {"units": {unit: rows}}


def test_fundamentals_never_use_future_filing():
    q = lambda start, end, filed, val: {
        "start": start, "end": end, "filed": filed, "val": val, "form": "10-Q"
    }
    instant = lambda end, filed, val: {
        "end": end, "filed": filed, "val": val, "form": "10-Q"
    }

    payload = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": fact([
                    q("2023-01-01", "2023-03-31", "2023-05-01", 100),
                    q("2023-04-01", "2023-06-30", "2023-08-01", 110),
                    q("2023-07-01", "2023-09-30", "2023-11-01", 120),
                    q("2023-10-01", "2023-12-31", "2024-02-01", 130),
                ]),
                "NetIncomeLoss": fact([
                    q("2023-01-01", "2023-03-31", "2023-05-01", 10),
                    q("2023-04-01", "2023-06-30", "2023-08-01", 11),
                    q("2023-07-01", "2023-09-30", "2023-11-01", 12),
                    q("2023-10-01", "2023-12-31", "2024-02-01", 13),
                ]),
                "NetCashProvidedByUsedInOperatingActivities": fact([
                    q("2023-01-01", "2023-03-31", "2023-05-01", 20),
                    q("2023-04-01", "2023-06-30", "2023-08-01", 21),
                    q("2023-07-01", "2023-09-30", "2023-11-01", 22),
                    q("2023-10-01", "2023-12-31", "2024-02-01", 23),
                ]),
                "PaymentsToAcquirePropertyPlantAndEquipment": fact([
                    q("2023-01-01", "2023-03-31", "2023-05-01", 5),
                    q("2023-04-01", "2023-06-30", "2023-08-01", 5),
                    q("2023-07-01", "2023-09-30", "2023-11-01", 5),
                    q("2023-10-01", "2023-12-31", "2024-02-01", 5),
                ]),
                "StockholdersEquity": fact([
                    instant("2023-03-31", "2023-05-01", 200),
                    instant("2023-12-31", "2024-02-01", 250),
                ]),
                "AssetsCurrent": fact([instant("2023-03-31", "2023-05-01", 100)]),
                "LiabilitiesCurrent": fact([instant("2023-03-31", "2023-05-01", 50)]),
                "LongTermDebtNoncurrent": fact([instant("2023-03-31", "2023-05-01", 40)]),
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": fact([
                    {**instant("2023-03-31", "2023-05-01", 10), "form": "10-Q"}
                ], unit="shares")
            },
        }
    }

    prices = pd.DataFrame({
        "ticker": ["TEST", "TEST"],
        "date": pd.to_datetime(["2023-05-01", "2024-02-01"]),
        "close": [20.0, 30.0],
    })
    rows, _ = build_one("TEST", payload, prices)
    may = rows[rows["available_date"] == "2023-05-01"].iloc[0]
    feb = rows[rows["available_date"] == "2024-02-01"].iloc[0]

    # The May filing must use equity 200, never the future 250 value.
    assert may["market_cap"] == 200.0
    assert may["current_ratio"] == 2.0
    assert feb["roe"] > may["roe"]
