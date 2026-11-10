"""Content identities shared by rollout, training, serving and evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path


def artifact_hash(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    if not files or not path.exists():
        raise ValueError(f"missing checkpoint artifact: {path}")
    for item in files:
        name = str(item.relative_to(path)) if path.is_dir() else item.name
        digest.update(name.encode() + b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()

