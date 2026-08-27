from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

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


def _main() -> None:
    parser = argparse.ArgumentParser(description="Update the local PFSP checkpoint manifest.")
    parser.add_argument("--update-pool", action="store_true")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/checkpoints/pfsp_pool.json"))
    args = parser.parse_args()
    if not args.update_pool:
        parser.error("--update-pool is required")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generation": args.generation, "entries": []}
    if args.manifest.exists():
        payload = json.loads(args.manifest.read_text())
        payload["generation"] = args.generation
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    _main()
