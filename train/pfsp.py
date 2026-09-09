from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import atomic_write_json
from .schema_v1 import Role
from .study import AdaptiveMix, Method


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


def sample_study_opponent(
    method: Method,
    *,
    single: PoolEntry,
    latest: PoolEntry,
    historical: list[PoolEntry],
    reference_pool: list[PoolEntry],
    learner_relative: dict[str, float],
    rng: random.Random,
    mix: AdaptiveMix,
) -> PoolEntry:
    match method:
        case Method.FIXED_SINGLE:
            return single
        case Method.FIXED_DIVERSE:
            if not reference_pool:
                raise ValueError("fixed diverse pool is empty")
            return rng.choice(reference_pool)
        case Method.ADAPTIVE_LATEST:
            return latest
        case Method.ADAPTIVE_HISTORY:
            return sample_adaptive_history(
                latest=latest,
                historical=historical,
                learner_relative=learner_relative,
                rng=rng,
                mix=mix,
            )
        case Method.UNTRAINED:
            raise ValueError("untrained control does not sample training opponents")
        case _:
            raise AssertionError(f"unhandled method {method!r}")


def sample_adaptive_history(
    *,
    latest: PoolEntry,
    historical: list[PoolEntry],
    learner_relative: dict[str, float],
    rng: random.Random,
    mix: AdaptiveMix,
) -> PoolEntry:
    if not historical:
        return latest
    draw = rng.random()
    if draw < mix.latest:
        return latest
    if draw < mix.latest + mix.difficult:
        return _difficult(historical, learner_relative, rng)
    return rng.choice(historical)


def _difficult(
    historical: list[PoolEntry],
    learner_relative: dict[str, float],
    rng: random.Random,
) -> PoolEntry:
    weights = [
        max(learner_relative.get(entry.checkpoint_id, entry.win_rate_vs_live), 1e-6)
        for entry in historical
    ]
    return rng.choices(historical, weights=weights, k=1)[0]


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
