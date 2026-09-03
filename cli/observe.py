from __future__ import annotations

import time
from typing import Any, Iterable

from ultron.train.episode_runner import EpisodeRunner, GuestVm
from ultron.train.schema_v1 import Role, TrajectoryStep

from .model import (
    Clock,
    EpisodeCase,
    ErrorEvent,
    EventSink,
    EpisodeEndedEvent,
    JobEndedEvent,
    ProbeEvent,
    RestoreEvent,
    ToolObservedEvent,
    TurnEndedEvent,
    TurnStartedEvent,
)


def instrument_runner(
    runner: EpisodeRunner,
    *,
    episode_index: int,
    emit: EventSink,
    clock: Clock,
) -> EpisodeRunner:
    """Clone runner with observed callbacks. The original is never mutated."""
    original_restore = runner.restore
    original_run_turn = runner.run_turn
    original_probe = runner.final_probe
    sequence = 0

    def restore(vm: GuestVm, sha: str) -> None:
        start = clock()
        original_restore(vm, sha)
        end = clock()
        emit(
            RestoreEvent(
                episode_index=episode_index,
                guest_id=vm.vm_id,
                image_ref=vm.image_ref,
                duration_s=end - start,
                at_s=end,
            )
        )

    def run_turn(
        vm: GuestVm, role: Role, profile: dict[str, Any], turn: int
    ) -> list[TrajectoryStep]:
        nonlocal sequence
        emit(
            TurnStartedEvent(
                episode_index=episode_index,
                turn_index=turn,
                role=role,
                at_s=clock(),
            )
        )
        start = clock()
        steps = original_run_turn(vm, role, profile, turn)
        end = clock()
        for step in steps:
            for tool in step.tool_events:
                emit(
                    ToolObservedEvent(
                        episode_index=episode_index,
                        turn_index=turn,
                        role=role,
                        sequence=sequence,
                        tool=tool,
                        at_s=clock(),
                    )
                )
                sequence += 1
        emit(
            TurnEndedEvent(
                episode_index=episode_index,
                turn_index=turn,
                role=role,
                duration_s=end - start,
                trajectory_step_count=len(steps),
                at_s=end,
            )
        )
        return steps

    def final_probe(vm: GuestVm, profile: dict[str, Any]) -> Any:
        start = clock()
        result = original_probe(vm, profile)
        end = clock()
        emit(
            ProbeEvent(
                episode_index=episode_index,
                result=result,
                duration_s=end - start,
                at_s=end,
            )
        )
        return result

    return EpisodeRunner(
        snapshot_sha256=runner.snapshot_sha256,
        load_profile=runner.load_profile,
        run_turn=run_turn,
        final_probe=final_probe,
        restore=restore,
        turns_per_side=runner.turns_per_side,
    )


def run_job(
    runner: EpisodeRunner,
    cases: Iterable[EpisodeCase],
    *,
    emit: EventSink,
    clock: Clock | None = None,
) -> None:
    """Blocking loop suitable for a worker thread. Emits per-episode and job ends."""
    tick = clock or time.monotonic
    job_start = tick()
    episode_index = 0
    try:
        for episode_index, case in enumerate(cases):
            instrumented = instrument_runner(
                runner, episode_index=episode_index, emit=emit, clock=tick
            )
            start = tick()
            trajectories = instrumented.run(case.config, case.vm)
            end = tick()
            emit(
                EpisodeEndedEvent(
                    episode_index=episode_index,
                    duration_s=end - start,
                    terminal=trajectories[0].terminal,
                    at_s=end,
                )
            )
    except Exception as exc:
        emit(
            ErrorEvent(
                at_s=tick(),
                operation=f"episode {episode_index}",
                message=str(exc),
                detail=type(exc).__name__,
            )
        )
        return
    end = tick()
    emit(JobEndedEvent(at_s=end, duration_s=end - job_start))
