from __future__ import annotations

from dataclasses import dataclass

from ultron.train.episode_runner import (
    EpisodeRunner,
    FinalProbe,
    ProfileLoader,
    RestoreFn,
    TurnExecutor,
)


@dataclass(frozen=True)
class JobSpec:
    episode_count: int
    generation: int
    profile_id: str
    group_id: str
    opponent_checkpoint_id: str
    snapshot_sha256: str
    turns_per_side: int = 8

    def __post_init__(self) -> None:
        if self.episode_count < 1:
            raise ValueError("episode_count must be >= 1")
        if self.turns_per_side < 0:
            raise ValueError("turns_per_side must be >= 0")
        if self.generation < 0:
            raise ValueError("generation must be >= 0")


@dataclass(frozen=True)
class Seams:
    restore: RestoreFn
    run_turn: TurnExecutor
    final_probe: FinalProbe
    load_profile: ProfileLoader

    @classmethod
    def from_runner(cls, runner: EpisodeRunner) -> Seams:
        return cls(
            restore=runner.restore,
            run_turn=runner.run_turn,
            final_probe=runner.final_probe,
            load_profile=runner.load_profile,
        )
