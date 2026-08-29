from __future__ import annotations

from pathlib import Path
import pandas as pd


def normalize_ticker(value: object) -> str:
    """Canonicalize common US share-class ticker separators."""
    return str(value).strip().upper().replace("/", ".").replace("-", ".")


def historical_components_to_intervals(history: pd.DataFrame) -> pd.DataFrame:
    """Convert dated constituent snapshots to [start_date, end_date) intervals.

    Supports the hanshof `date,tickers` format and long `date,ticker|symbol` data.
    """
    if history.empty:
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])
    h = history.copy()
    cols = {str(c).strip().lower(): c for c in h.columns}
    if "date" not in cols:
        raise ValueError("historical universe requires a date column")
    date_col = cols["date"]
    h[date_col] = pd.to_datetime(h[date_col], errors="coerce")
    h = h.dropna(subset=[date_col]).sort_values(date_col)

    snapshots: list[tuple[pd.Timestamp, set[str]]] = []
    if "tickers" in cols:
        member_col = cols["tickers"]
        for date, group in h.groupby(date_col, sort=True):
            value = group.iloc[-1][member_col]
            members = {
                normalize_ticker(x)
                for x in str(value).split(",")
                if str(x).strip() and str(x).lower() != "nan"
            }
            snapshots.append((pd.Timestamp(date).normalize(), members))
    else:
        member_col = cols.get("ticker") or cols.get("symbol")
        if member_col is None:
            raise ValueError("historical universe requires tickers, ticker, or symbol column")
        for date, group in h.groupby(date_col, sort=True):
            members = {normalize_ticker(x) for x in group[member_col].dropna() if str(x).strip()}
            snapshots.append((pd.Timestamp(date).normalize(), members))

    active: dict[str, pd.Timestamp] = {}
    previous: set[str] = set()
    rows: list[dict] = []
    for date, members in snapshots:
        for ticker in sorted(previous - members):
            rows.append({"ticker": ticker, "start_date": active.pop(ticker), "end_date": date})
        for ticker in sorted(members - previous):
            active[ticker] = date
        previous = members
    for ticker, start in sorted(active.items()):
        rows.append({"ticker": ticker, "start_date": start, "end_date": pd.NaT})

    out = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])
    return out.sort_values(["ticker", "start_date"]).reset_index(drop=True) if not out.empty else out


def load_universe_intervals(path: str | Path) -> pd.DataFrame:
    u = pd.read_csv(path)
    required = {"ticker", "start_date", "end_date"}
    if not required.issubset(u.columns):
        raise ValueError(f"universe intervals require columns {sorted(required)}")
    u["ticker"] = u["ticker"].map(normalize_ticker)
    u["start_date"] = pd.to_datetime(u["start_date"], errors="coerce")
    u["end_date"] = pd.to_datetime(u["end_date"], errors="coerce")
    return u.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def attach_membership(frame: pd.DataFrame, intervals: pd.DataFrame | None, date_col: str = "date") -> pd.DataFrame:
    """Attach an `in_universe` flag without materializing a daily membership table.

    Group indices are built once, so this stays practical for multi-million-row
    FINSABER panels instead of rescanning the full frame for every ticker.
    """
    out = frame.copy()
    if intervals is None or intervals.empty:
        out["in_universe"] = True
        return out
    out[date_col] = pd.to_datetime(out[date_col])
    out["ticker"] = out["ticker"].map(normalize_ticker)
    out["in_universe"] = False
    u = intervals.copy()
    u["ticker"] = u["ticker"].map(normalize_ticker)
    u["start_date"] = pd.to_datetime(u["start_date"], errors="coerce")
    u["end_date"] = pd.to_datetime(u["end_date"], errors="coerce")

    row_groups = out.groupby("ticker", sort=False).indices
    for ticker, spans in u.groupby("ticker", sort=False):
        idx = row_groups.get(ticker)
        if idx is None or len(idx) == 0:
            continue
        idx = pd.Index(idx)
        dates = out.loc[idx, date_col]
        member = pd.Series(False, index=idx)
        for span in spans.itertuples(index=False):
            mask = dates >= span.start_date
            if pd.notna(span.end_date):
                mask &= dates < span.end_date
            member |= mask
        out.loc[idx, "in_universe"] = member.to_numpy()
    return out
