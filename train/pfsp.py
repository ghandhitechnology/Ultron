from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import atomic_write_json
from .schema_v1 import Role


@dataclass(frozen=True)
class PoolEntry:
    checkpoint_id: str
    path: str
    role: Role
    win_rate_vs_live: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.win_rate_vs_live <= 1.0:
            raise ValueError("win_rate_vs_live must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "path": self.path,
            "role": self.role.value,
            "win_rate_vs_live": self.win_rate_vs_live,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoolEntry":
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            path=str(data["path"]),
            role=Role(data["role"]),
            win_rate_vs_live=float(data["win_rate_vs_live"]),
        )


def pfsp_weight(entry: PoolEntry) -> float:
    return max(entry.win_rate_vs_live * (1.0 - entry.win_rate_vs_live), 1e-6)


def pfsp_sample(
    pool: list[PoolEntry], opponent_role: Role, *, rng: random.Random | None = None
) -> PoolEntry:
    candidates = [entry for entry in pool if entry.role == opponent_role]
    if not candidates:
        raise ValueError(f"no opponent checkpoints for role {opponent_role.value}")
    chooser = rng or random
    return chooser.choices(candidates, weights=[pfsp_weight(entry) for entry in candidates], k=1)[0]


def assign_group_opponent(
    groups: list[str],
    pool: list[PoolEntry],
    role: Role,
    *,
    rng: random.Random | None = None,
) -> dict[str, str]:
    return {
        group_id: pfsp_sample(pool, role, rng=rng).checkpoint_id
        for group_id in dict.fromkeys(groups)
    }


def update_pool(entries: list[PoolEntry], new_entry: PoolEntry, limit: int = 8) -> list[PoolEntry]:
    if limit < 1:
        raise ValueError("pool limit must be positive")
    updated = [entry for entry in entries if entry.checkpoint_id != new_entry.checkpoint_id]
    updated.append(new_entry)
    if len(updated) <= limit:
        return updated
    return sorted(updated, key=pfsp_weight, reverse=True)[:limit]


def load_pool(path: Path) -> list[PoolEntry]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    return [PoolEntry.from_dict(entry) for entry in payload.get("entries", [])]


def save_pool(path: Path, generation: int, entries: list[PoolEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation": generation,
        "entries": [entry.to_dict() for entry in entries],
    }
    atomic_write_json(path, payload)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Update the local PFSP checkpoint manifest.")
    parser.add_argument("--update-pool", action="store_true")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/checkpoints/pfsp_pool.json"))
    args = parser.parse_args()
    if not args.update_pool:
        parser.error("--update-pool is required")
    save_pool(args.manifest, args.generation, load_pool(args.manifest))


if __name__ == "__main__":
    _main()
