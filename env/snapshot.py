from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class QuarantineSink(Protocol):
    def quarantine(self, vm_id: str, reason: str) -> None: ...


@dataclass(frozen=True)
class SnapshotVerification:
    path: Path
    expected_sha256: str
    actual_sha256: str

    @property
    def ok(self) -> bool:
        return self.expected_sha256 == self.actual_sha256


class SnapshotDriftError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: Path, expected_sha256: str) -> SnapshotVerification:
    if len(expected_sha256) != 64:
        raise ValueError("expected snapshot SHA-256 must contain 64 hex characters")
    try:
        int(expected_sha256, 16)
    except ValueError as exc:
        raise ValueError("expected snapshot SHA-256 is not hexadecimal") from exc
    result = SnapshotVerification(path, expected_sha256.lower(), sha256_file(path))
    if not result.ok:
        raise SnapshotDriftError(
            f"snapshot hash mismatch for {path}: expected {result.expected_sha256}, "
            f"got {result.actual_sha256}"
        )
    return result
