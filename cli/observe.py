from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

from ultron.cli.model import (
    EpisodeEnded,
    JobEnded,
    JobError,
    JobEvent,
    JobMeta,
    ModelResponseStarted,
    ModelResponseDelta,
    ModelResponseFinished,
    ResponseArchiveOpened,
    ProbeFinished,
    ProbeStarted,
    RestoreFinished,
    RestoreStarted,
    ToolObserved,
    TurnEnded,
    TurnStarted,
)
from ultron.env.backend import IsolationBackend
from ultron.cli.responses import ResponseArchive
from ultron.response_stream import ResponseUpdate, observe_responses
from ultron.train.episode_runner import EpisodeConfig, EpisodeRunner, GuestVm

Emit = Callable[[JobEvent], None]
Clock = Callable[[], float]


@dataclass(frozen=True)
class EpisodeCase:
    config: EpisodeConfig
    vm: GuestVm


def instrument_runner(
    runner: EpisodeRunner,
    *,
    episode_index: int,
    emit: Emit,
    clock: Clock,
    cancel: Event | None = None,
) -> EpisodeRunner:
    orig_restore = runner.restore
    orig_turn = runner.run_turn
    orig_probe = runner.final_probe

    def restore(vm: GuestVm, sha: str) -> None:
        if cancel is not None and cancel.is_set():
            raise CancelledError("Battle view closed")
        isolation = vm.isolation
        if not isinstance(isolation, IsolationBackend):
            isolation = IsolationBackend(isolation)
        emit(
            RestoreStarted(
                episode_index=episode_index,
                guest_id=vm.vm_id,
                image_ref=vm.image_ref,
                isolation=isolation,
                at_s=clock(),
            )
        )
        started = clock()
        orig_restore(vm, sha)
        emit(
            RestoreFinished(
                episode_index=episode_index,
                guest_id=vm.vm_id,
                host_address=vm.host_address,
                duration_s=clock() - started,
                at_s=clock(),
            )
        )

    def run_turn(vm: GuestVm, role, profile: dict[str, Any], turn: int):
        if cancel is not None and cancel.is_set():
            raise CancelledError("Battle view closed")
        emit(
            TurnStarted(
                episode_index=episode_index,
                turn_index=turn,
                role=role,
                at_s=clock(),
            )
        )
        started = clock()

        def response(update: ResponseUpdate) -> None:
            if update.kind == "started" and cancel is not None and cancel.is_set():
                raise CancelledError("Battle view closed")
            common = dict(
                episode_index=episode_index,
                turn_index=turn,
                role=role,
                response_id=update.response_id,
                at_s=clock(),
            )
            if update.kind == "started":
                emit(ModelResponseStarted(**common, model=update.model))
            elif update.kind == "delta":
                emit(ModelResponseDelta(**common, text=update.text))
                if cancel is not None and cancel.is_set():
                    raise CancelledError("Battle view closed")
            else:
                emit(ModelResponseFinished(**common, status=update.status, error=update.error))

        with observe_responses(response):
            steps = orig_turn(vm, role, profile, turn)
        seq = 0
        for step in steps:
            for tool in step.tool_events:
                emit(
                    ToolObserved(
                        episode_index=episode_index,
                        turn_index=turn,
                        role=role,
                        sequence=seq,
                        tool=tool,
                        at_s=clock(),
                    )
                )
                seq += 1
        emit(
            TurnEnded(
                episode_index=episode_index,
                turn_index=turn,
                role=role,
                duration_s=clock() - started,
                step_count=len(steps),
                at_s=clock(),
            )
        )
        return steps

    def final_probe(vm: GuestVm, profile: dict[str, Any]):
        if cancel is not None and cancel.is_set():
            raise CancelledError("Battle view closed")
        emit(ProbeStarted(episode_index=episode_index, at_s=clock()))
        started = clock()
        result = orig_probe(vm, profile)
        emit(
            ProbeFinished(
                episode_index=episode_index,
                result=result,
                duration_s=clock() - started,
                at_s=clock(),
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


def drive_job(
    meta: JobMeta,
    runner: EpisodeRunner,
    cases: Sequence[EpisodeCase] | Iterable[EpisodeCase],
    *,
    emit: Emit,
    clock: Clock,
    responses_dir: Path | None = None,
    cancel: Event | None = None,
    archive: ResponseArchive | None = None,
) -> None:
    job_started = clock()
    original_emit = emit
    archive = archive or ResponseArchive(meta, responses_dir)

    def emit(event: JobEvent) -> None:
        archive.record(event)
        original_emit(event)

    try:
        emit(ResponseArchiveOpened(path=str(archive.path.resolve()), at_s=clock()))
        completed = 0
        for index, case in enumerate(cases):
            if index >= meta.episodes_planned:
                break
            observed = instrument_runner(
                runner, episode_index=index, emit=emit, clock=clock, cancel=cancel
            )
            episode_started = clock()
            trajectories = observed.run(case.config, case.vm)
            if not trajectories:
                raise RuntimeError(f"episode {index} produced no trajectories")
            emit(
                EpisodeEnded(
                    episode_index=index,
                    duration_s=clock() - episode_started,
                    terminal=trajectories[0].terminal,
                    guest_id=case.vm.vm_id,
                    at_s=clock(),
                )
            )
            completed += 1
        if completed != meta.episodes_planned:
            emit(
                JobError(
                    message=f"completed {completed} of {meta.episodes_planned} episodes",
                    operation="drive",
                    at_s=clock(),
                )
            )
            return
        emit(JobEnded(duration_s=clock() - job_started, at_s=clock()))
    except BaseException as exc:
        cancelled = isinstance(exc, (CancelledError, KeyboardInterrupt, SystemExit))
        error = JobError(
            message=str(exc) or type(exc).__name__,
            operation="cancel" if cancelled else "drive",
            at_s=clock(),
        )
        try:
            archive.record(error)
        except Exception:
            # Disk failures must still reach the UI through its independent queue.
            pass
        original_emit(error)
        raise
