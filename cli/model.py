from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, Protocol, TypeAlias

from ultron.env.backend import IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.episode_runner import EpisodeConfig, GuestVm
from ultron.train.schema_v1 import Role, TerminalOutcome, ToolEvent

RECENT_EVENT_LIMIT = 500


class Phase(str, Enum):
    RESTORING = "restoring"
    ATTACKING = "attacking"
    DEFENDING = "defending"
    PROBING = "probing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class JobSpec:
    version: str
    generation: int
    profile_id: str
    isolation_backend: IsolationBackend
    episode_count: int
    turns_per_side: int


@dataclass(frozen=True)
class EpisodeCase:
    config: EpisodeConfig
    vm: GuestVm


@dataclass(frozen=True)
class EpisodeSummary:
    episode_index: int
    duration_s: float
    probe: ProbeResult
    terminal: TerminalOutcome


@dataclass(frozen=True)
class RestoreEvent:
    episode_index: int
    guest_id: str
    image_ref: str
    duration_s: float
    at_s: float
    kind: Literal["restore"] = "restore"


@dataclass(frozen=True)
class TurnStartedEvent:
    episode_index: int
    turn_index: int
    role: Role
    at_s: float
    kind: Literal["turn_started"] = "turn_started"


@dataclass(frozen=True)
class TurnEndedEvent:
    episode_index: int
    turn_index: int
    role: Role
    duration_s: float
    trajectory_step_count: int
    at_s: float
    kind: Literal["turn_ended"] = "turn_ended"


@dataclass(frozen=True)
class ToolObservedEvent:
    episode_index: int
    turn_index: int
    role: Role
    sequence: int
    tool: ToolEvent
    at_s: float
    kind: Literal["tool"] = "tool"


@dataclass(frozen=True)
class ProbeEvent:
    episode_index: int
    result: ProbeResult
    duration_s: float
    at_s: float
    kind: Literal["probe"] = "probe"


@dataclass(frozen=True)
class EpisodeEndedEvent:
    episode_index: int
    duration_s: float
    terminal: TerminalOutcome
    at_s: float
    kind: Literal["episode_ended"] = "episode_ended"


@dataclass(frozen=True)
class JobEndedEvent:
    at_s: float
    duration_s: float
    kind: Literal["job_ended"] = "job_ended"


@dataclass(frozen=True)
class ErrorEvent:
    at_s: float
    operation: str
    message: str
    detail: str | None = None
    kind: Literal["error"] = "error"


JobEvent: TypeAlias = (
    RestoreEvent
    | TurnStartedEvent
    | TurnEndedEvent
    | ToolObservedEvent
    | ProbeEvent
    | EpisodeEndedEvent
    | JobEndedEvent
    | ErrorEvent
)


@dataclass(frozen=True)
class SnapshotData:
    spec: JobSpec
    started_at_s: float
    prior_episodes: tuple[EpisodeSummary, ...]
    recent_events: tuple[JobEvent, ...]


@dataclass(frozen=True)
class RestoringSnapshot:
    data: SnapshotData
    episode_index: int
    phase: Literal[Phase.RESTORING] = Phase.RESTORING


@dataclass(frozen=True)
class AttackingSnapshot:
    data: SnapshotData
    episode_index: int
    turn_index: int
    phase: Literal[Phase.ATTACKING] = Phase.ATTACKING


@dataclass(frozen=True)
class DefendingSnapshot:
    data: SnapshotData
    episode_index: int
    turn_index: int
    phase: Literal[Phase.DEFENDING] = Phase.DEFENDING


@dataclass(frozen=True)
class ProbingSnapshot:
    data: SnapshotData
    episode_index: int
    result: ProbeResult | None
    phase: Literal[Phase.PROBING] = Phase.PROBING


@dataclass(frozen=True)
class CompleteSnapshot:
    data: SnapshotData
    final_episode: EpisodeSummary
    ended_at_s: float | None
    phase: Literal[Phase.COMPLETE] = Phase.COMPLETE


@dataclass(frozen=True)
class FailedSnapshot:
    data: SnapshotData
    error: ErrorEvent
    phase: Literal[Phase.FAILED] = Phase.FAILED


JobSnapshot: TypeAlias = (
    RestoringSnapshot
    | AttackingSnapshot
    | DefendingSnapshot
    | ProbingSnapshot
    | CompleteSnapshot
    | FailedSnapshot
)


@dataclass(frozen=True)
class JobProgress:
    completed_episodes: int
    total_episodes: int
    completed_turns: int
    total_turns: int


class InvalidTransition(ValueError):
    pass


class Clock(Protocol):
    def __call__(self) -> float: ...


class EventSink(Protocol):
    def __call__(self, event: JobEvent) -> None: ...


def active_role(snapshot: JobSnapshot) -> Role | None:
    if isinstance(snapshot, AttackingSnapshot):
        return Role.ATTACKER
    if isinstance(snapshot, DefendingSnapshot):
        return Role.DEFENDER
    return None


def initial_snapshot(spec: JobSpec, *, started_at_s: float) -> RestoringSnapshot:
    data = SnapshotData(
        spec=spec,
        started_at_s=started_at_s,
        prior_episodes=(),
        recent_events=(),
    )
    return RestoringSnapshot(data=data, episode_index=0)


def _record(data: SnapshotData, event: JobEvent) -> SnapshotData:
    tail = (*data.recent_events, event)[-RECENT_EVENT_LIMIT:]
    return replace(data, recent_events=tail)


def _illegal(snapshot: JobSnapshot, event: JobEvent) -> InvalidTransition:
    return InvalidTransition(
        f"{event.kind} is illegal in phase {snapshot.phase.value}"
    )


def apply(snapshot: JobSnapshot, event: JobEvent) -> JobSnapshot:
    """Pure transition. Illegal event order raises InvalidTransition."""
    if isinstance(snapshot, (CompleteSnapshot, FailedSnapshot)):
        raise _illegal(snapshot, event)
    if isinstance(event, ErrorEvent):
        return FailedSnapshot(data=_record(snapshot.data, event), error=event)
    if isinstance(snapshot, RestoringSnapshot):
        return _from_restoring(snapshot, event)
    if isinstance(snapshot, (AttackingSnapshot, DefendingSnapshot)):
        return _from_turn(snapshot, event)
    if isinstance(snapshot, ProbingSnapshot):
        return _from_probing(snapshot, event)
    raise _illegal(snapshot, event)


def _from_restoring(snapshot: RestoringSnapshot, event: JobEvent) -> JobSnapshot:
    data = _record(snapshot.data, event)
    if isinstance(event, RestoreEvent):
        return AttackingSnapshot(
            data=data, episode_index=snapshot.episode_index, turn_index=0
        )
    if isinstance(event, JobEndedEvent):
        if not snapshot.data.prior_episodes:
            raise _illegal(snapshot, event)
        return CompleteSnapshot(
            data=data,
            final_episode=snapshot.data.prior_episodes[-1],
            ended_at_s=event.at_s,
        )
    raise _illegal(snapshot, event)


def _from_turn(
    snapshot: AttackingSnapshot | DefendingSnapshot, event: JobEvent
) -> JobSnapshot:
    data = _record(snapshot.data, event)
    if isinstance(event, TurnStartedEvent):
        if event.role is Role.ATTACKER:
            return AttackingSnapshot(
                data=data,
                episode_index=snapshot.episode_index,
                turn_index=event.turn_index,
            )
        return DefendingSnapshot(
            data=data,
            episode_index=snapshot.episode_index,
            turn_index=event.turn_index,
        )
    if isinstance(event, (ToolObservedEvent, TurnEndedEvent)):
        return replace(snapshot, data=data)
    if isinstance(event, ProbeEvent):
        return ProbingSnapshot(
            data=data, episode_index=snapshot.episode_index, result=event.result
        )
    raise _illegal(snapshot, event)


def _from_probing(snapshot: ProbingSnapshot, event: JobEvent) -> JobSnapshot:
    data = _record(snapshot.data, event)
    if isinstance(event, ProbeEvent):
        return ProbingSnapshot(
            data=data, episode_index=snapshot.episode_index, result=event.result
        )
    if isinstance(event, EpisodeEndedEvent):
        if snapshot.result is None:
            raise _illegal(snapshot, event)
        summary = EpisodeSummary(
            episode_index=snapshot.episode_index,
            duration_s=event.duration_s,
            probe=snapshot.result,
            terminal=event.terminal,
        )
        return RestoringSnapshot(
            data=replace(data, prior_episodes=(*data.prior_episodes, summary)),
            episode_index=snapshot.episode_index + 1,
        )
    raise _illegal(snapshot, event)


def progress(snapshot: JobSnapshot) -> JobProgress:
    data = snapshot.data
    completed_episodes = len(data.prior_episodes)
    total_turns = data.spec.turns_per_side * 2
    if isinstance(snapshot, (AttackingSnapshot, DefendingSnapshot)):
        completed_turns = min(snapshot.turn_index + 1, total_turns)
    elif isinstance(snapshot, ProbingSnapshot):
        completed_turns = total_turns
    elif isinstance(snapshot, (CompleteSnapshot, FailedSnapshot)):
        completed_turns = total_turns
    else:
        completed_turns = 0
    return JobProgress(
        completed_episodes=completed_episodes,
        total_episodes=data.spec.episode_count,
        completed_turns=completed_turns,
        total_turns=total_turns,
    )


def estimate_eta_s(snapshot: JobSnapshot) -> float | None:
    """None until at least one episode completes, then remaining * mean duration."""
    durations = tuple(s.duration_s for s in snapshot.data.prior_episodes)
    if not durations:
        return None
    mean = sum(durations) / len(durations)
    remaining = snapshot.data.spec.episode_count - len(durations)
    if remaining <= 0:
        return 0.0
    return mean * remaining
