from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read CSV or Parquet with one interface.

    Bulk research data should prefer Parquet; CSV remains supported for the
    small committed fixtures and derived tables already used by the project.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p, columns=columns)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p, usecols=columns)
    raise ValueError(f"Unsupported table format: {p}")


def write_table(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(p, index=index)
        return
    if suffix in {".csv", ".txt"}:
        frame.to_csv(p, index=index)
        return
    raise ValueError(f"Unsupported table format: {p}")
