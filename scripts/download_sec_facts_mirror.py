#!/usr/bin/env python3
"""Download a compact point-in-time SEC Company Facts mirror for FINSABER CIKs.

The upstream `deeleeramone/sec-company-facts` release mirrors SEC Company Facts
into sharded Parquet files. Raw facts preserve the SEC `filed` date, which is
the availability timestamp required for causal backtests.

Only CIKs present in the supplied FINSABER price file and only tags needed by
our quality model are retained. Large upstream shards are cached locally but
never committed to this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import requests

from scripts.build_mirror_fundamentals import MIRROR_TAGS
from src.data.io import read_table

UPSTREAM_REPO = "deeleeramone/sec-company-facts"
RELEASE_TAG = "parquet-latest"
RELEASE_API = f"https://api.github.com/repos/{UPSTREAM_REPO}/releases/tags/{RELEASE_TAG}"
ALLOWED_FORMS = {"10-Q", "10-K", "20-F", "40-F"}
FACT_COLUMNS = [
    "cik", "tag_id", "unit", "start", "end", "val", "accn_id",
    "fy", "fp", "form", "filed", "frame",
]


def normalize_cik(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if "." in text:
            text = str(int(float(text)))
        else:
            text = str(int(text))
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return None
        text = str(int(digits))
    return text.zfill(10)


def cik_bucket(cik: str) -> int:
    return int(cik) % 64


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "LukaStuurman/Trading-strategy academic research",
        "Accept": "application/vnd.github+json",
    })
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _release_assets(session: requests.Session) -> tuple[dict, dict[str, dict]]:
    response = session.get(RELEASE_API, timeout=60)
    response.raise_for_status()
    release = response.json()
    assets: dict[str, dict] = {}
    url = release["assets_url"]
    page = 1
    while True:
        r = session.get(url, params={"per_page": 100, "page": page}, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for asset in batch:
            assets[asset["name"]] = asset
        if len(batch) < 100:
            break
        page += 1
    return release, assets


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_digest(asset: dict) -> str | None:
    digest = str(asset.get("digest") or "")
    return digest.split(":", 1)[1] if digest.startswith("sha256:") else None


def _download_asset(asset: dict, cache_dir: Path, session: requests.Session) -> Path:
    target = cache_dir / asset["name"]
    expected = _expected_digest(asset)
    if target.exists() and (not expected or _sha256(target) == expected):
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    url = asset["browser_download_url"]
    with session.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    if expected:
        actual = _sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Digest mismatch for {asset['name']}: {actual} != {expected}")
    tmp.replace(target)
    return target


def _wanted_tags() -> set[str]:
    return {tag for candidates in MIRROR_TAGS.values() for tag in candidates}


def _read_tag_map(path: Path) -> pd.DataFrame:
    tags = pd.read_parquet(path)
    tags["tag"] = tags["tag"].astype(str)
    if "namespace" in tags.columns:
        tags["namespace"] = tags["namespace"].astype(str)
        tags = tags[tags["namespace"].isin(["us-gaap", "dei"])]
    tags = tags[tags["tag"].isin(_wanted_tags())].copy()
    if tags.empty:
        raise RuntimeError("SEC mirror xbrl_tags contains none of the required quality tags")
    return tags[["tag_id", "namespace", "tag"]].drop_duplicates("tag_id")


def _read_accessions(path: Path) -> pd.DataFrame:
    acc = pd.read_parquet(path)
    cols = [c for c in ["accn_id", "accn"] if c in acc.columns]
    if set(cols) != {"accn_id", "accn"}:
        raise RuntimeError(f"Unexpected accessions schema: {list(acc.columns)}")
    return acc[cols].drop_duplicates("accn_id")


def _filter_shard(path: Path, wanted_ciks: set[str], wanted_tag_ids: set[int]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=FACT_COLUMNS)
    frame["cik"] = frame["cik"].map(normalize_cik)
    frame["tag_id"] = pd.to_numeric(frame["tag_id"], errors="coerce")
    frame = frame[
        frame["cik"].isin(wanted_ciks)
        & frame["tag_id"].isin(wanted_tag_ids)
        & frame["form"].isin(ALLOWED_FORMS)
        & frame["filed"].notna()
        & frame["val"].notna()
    ].copy()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default=".sec-mirror-cache")
    parser.add_argument("--min-matched-ciks", type=int, default=600)
    args = parser.parse_args()

    prices = read_table(args.prices)
    if "cik" not in prices.columns:
        raise RuntimeError("FINSABER prices must contain CIK for SEC mirror linkage")
    ciks = sorted({c for c in prices["cik"].map(normalize_cik).dropna().tolist()})
    if not ciks:
        raise RuntimeError("No valid CIKs found in FINSABER prices")
    wanted_ciks = set(ciks)
    buckets = sorted({cik_bucket(cik) for cik in ciks})

    session = _session()
    release, assets = _release_assets(session)
    required_support = ["xbrl_tags.parquet", "accessions.parquet"]
    missing_assets = [name for name in required_support if name not in assets]
    shard_names = [f"facts_enc-b{bucket:05d}.parquet" for bucket in buckets]
    missing_assets.extend(name for name in shard_names if name not in assets)
    if missing_assets:
        raise RuntimeError(f"SEC mirror release is missing required assets: {missing_assets[:10]}")

    cache = Path(args.cache_dir)
    tag_path = _download_asset(assets["xbrl_tags.parquet"], cache, session)
    accession_path = _download_asset(assets["accessions.parquet"], cache, session)
    tag_map = _read_tag_map(tag_path)
    accessions = _read_accessions(accession_path)
    wanted_tag_ids = set(pd.to_numeric(tag_map["tag_id"], errors="coerce").dropna().astype(int))

    chunks: list[pd.DataFrame] = []
    used_assets: list[dict] = []
    for idx, name in enumerate(shard_names, start=1):
        asset = assets[name]
        shard_path = _download_asset(asset, cache, session)
        chunk = _filter_shard(shard_path, wanted_ciks, wanted_tag_ids)
        if not chunk.empty:
            chunks.append(chunk)
        used_assets.append({
            "name": name,
            "size": int(asset.get("size") or 0),
            "digest": asset.get("digest"),
        })
        if idx % 8 == 0 or idx == len(shard_names):
            print(f"[sec-mirror] processed {idx}/{len(shard_names)} fact shards")

    if not chunks:
        raise RuntimeError("SEC Company Facts mirror produced no relevant facts for FINSABER CIKs")
    facts = pd.concat(chunks, ignore_index=True)
    facts = facts.merge(tag_map, on="tag_id", how="left")
    facts = facts.merge(accessions, on="accn_id", how="left")
    facts["filed"] = pd.to_datetime(facts["filed"], errors="coerce")
    facts["start"] = pd.to_datetime(facts["start"], errors="coerce")
    facts["end"] = pd.to_datetime(facts["end"], errors="coerce")
    facts = facts.dropna(subset=["cik", "tag", "filed", "val"])
    facts = facts.sort_values(["cik", "filed", "end", "tag"]).drop_duplicates(
        ["cik", "tag", "unit", "start", "end", "filed", "val", "accn_id"], keep="last"
    )

    matched_ciks = sorted(set(facts["cik"].dropna().astype(str)))
    if len(matched_ciks) < args.min_matched_ciks:
        raise RuntimeError(
            f"SEC mirror coverage too low: {len(matched_ciks)} matched CIKs < {args.min_matched_ciks}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    facts.to_parquet(output, index=False)

    manifest = {
        "source": "deeleeramone/sec-company-facts GitHub release mirror of SEC Company Facts",
        "upstream_repo": UPSTREAM_REPO,
        "release_tag": release.get("tag_name"),
        "release_id": release.get("id"),
        "published_at": release.get("published_at"),
        "availability_field": "filed",
        "sharding_rule": "CIK modulo 64",
        "input_ciks": len(ciks),
        "matched_ciks": len(matched_ciks),
        "cik_coverage": len(matched_ciks) / len(ciks),
        "buckets": buckets,
        "rows": int(len(facts)),
        "tags": sorted(facts["tag"].dropna().unique().tolist()),
        "support_assets": {
            name: {
                "size": int(assets[name].get("size") or 0),
                "digest": assets[name].get("digest"),
            }
            for name in required_support
        },
        "fact_assets": used_assets,
        "compact_output_sha256": _sha256(output),
        "notes": [
            "Only facts filed on or before a simulated date may be used.",
            "Only FINSABER CIKs and quality-model tags are retained.",
            "Upstream shard files remain cache-only and are not committed.",
        ],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[sec-mirror] wrote {len(facts):,} facts for {len(matched_ciks):,}/{len(ciks):,} CIKs "
        f"({manifest['cik_coverage']:.1%})"
    )


if __name__ == "__main__":
    main()
