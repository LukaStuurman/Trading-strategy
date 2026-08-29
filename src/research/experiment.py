from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_config_hash(value: Any, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_commit() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def build_manifest(*, name: str, parameters: dict, data_files: dict[str, str | Path], metadata: dict | None = None) -> dict:
    data = {}
    for key, raw_path in data_files.items():
        path = Path(raw_path)
        data[key] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }
    identity = {"name": name, "parameters": parameters, "data": data}
    return {
        "experiment_id": stable_config_hash(identity, 16),
        "name": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "parameters": parameters,
        "data": data,
        "metadata": metadata or {},
    }


def write_manifest(manifest: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
