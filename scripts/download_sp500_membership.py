#!/usr/bin/env python3
"""Download historical S&P 500 membership from a pinned GitHub revision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
RAW = REAL / "raw"
REPO = "hanshof/sp500_constituents"
COMMIT_API = f"https://api.github.com/repos/{REPO}/commits/main"
FILES = {
    "historical": "sp_500_historical_components.csv",
    "current": "sp500_constituents.csv",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(RAW))
    p.add_argument("--manifest", default=str(REAL / "sp500_membership_manifest.json"))
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Trading-strategy research"})
    commit_response = session.get(COMMIT_API, timeout=60)
    commit_response.raise_for_status()
    revision = str(commit_response.json().get("sha", ""))
    if len(revision) < 12:
        raise RuntimeError("Could not resolve historical S&P membership revision")

    downloaded = {}
    for key, filename in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{revision}/{filename}"
        response = session.get(url, timeout=120)
        response.raise_for_status()
        target = output_dir / filename
        target.write_bytes(response.content)
        downloaded[key] = {"path": str(target), "bytes": target.stat().st_size, "source": url}
        print(f"[membership] {filename}: {target.stat().st_size:,} bytes")

    manifest = {
        "source": REPO,
        "revision": revision,
        "files": downloaded,
        "note": "Historical snapshots are converted to [start_date,end_date) intervals before research.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[membership] pinned revision {revision}")


if __name__ == "__main__":
    main()
