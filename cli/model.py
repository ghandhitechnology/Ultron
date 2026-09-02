from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, TypeAlias

from ultron.env.backend import IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.schema_v1 import Role, TerminalOutcome, ToolEvent

LOG_LIMIT = 200
EVENT_LIMIT = 200


class Phase(str, Enum):
    RESTORING = "restoring"
    ATTACKING = "attacking"
    DEFENDING = "defending"
    PROBING = "probing"
    COMPLETE = "complete"
    FAILED = "failed"


class InvalidTransition(ValueError):
    pass


@dataclass(frozen=True)
class JobMeta:
    generation: int
    profile_id: str
    isolation: IsolationBackend
    episodes_planned: int
    turns_per_side: int
    version: str = "0.1.0"
    group_id: str = "demo"
    snapshot_sha256: str = ""


@dataclass(frozen=True)
class EpisodeSummary:
    episode_index: int
    duration_s: float
    terminal: TerminalOutcome
    guest_id: str
    probe: ProbeResult | None = None


@dataclass(frozen=True)
class RestoreStarted:
    episode_index: int
    guest_id: str
    image_ref: str
    isolation: IsolationBackend
    at_s: float
    kind: Literal["restore_started"] = "restore_started"


@dataclass(frozen=True)
class RestoreFinished:
    episode_index: int
    guest_id: str
    host_address: str
    duration_s: float
    at_s: float
    kind: Literal["restore_finished"] = "restore_finished"


@dataclass(frozen=True)
class TurnStarted:
    episode_index: int
    turn_index: int
    role: Role
    at_s: float
    kind: Literal["turn_started"] = "turn_started"


@dataclass(frozen=True)
class TurnEnded:
    episode_index: int
    turn_index: int
    role: Role
    duration_s: float
    step_count: int
    at_s: float
    kind: Literal["turn_ended"] = "turn_ended"


@dataclass(frozen=True)
class ToolObserved:
    episode_index: int
    turn_index: int
    role: Role
    sequence: int
    tool: ToolEvent
    at_s: float
    kind: Literal["tool"] = "tool"


@dataclass(frozen=True)
class ProbeStarted:
    episode_index: int
    at_s: float
    kind: Literal["probe_started"] = "probe_started"


@dataclass(frozen=True)
class ProbeFinished:
    episode_index: int
    result: ProbeResult
    duration_s: float
    at_s: float
    kind: Literal["probe_finished"] = "probe_finished"


@dataclass(frozen=True)
class EpisodeEnded:
    episode_index: int
    duration_s: float
    terminal: TerminalOutcome
    guest_id: str
    at_s: float
    kind: Literal["episode_ended"] = "episode_ended"


@dataclass(frozen=True)
class JobEnded:
    duration_s: float
    at_s: float
    kind: Literal["job_ended"] = "job_ended"


@dataclass(frozen=True)
class JobError:
    message: str
    operation: str
    at_s: float
    kind: Literal["error"] = "error"


JobEvent: TypeAlias = (
    RestoreStarted
    | RestoreFinished
    | TurnStarted
    | TurnEnded
    | ToolObserved
    | ProbeStarted
    | ProbeFinished
    | EpisodeEnded
    | JobEnded
    | JobError
)


@dataclass(frozen=True)
class JobProgress:
    completed_episodes: int
    total_episodes: int
    turn_index: int | None
    total_turns: int


@dataclass(frozen=True)
class JobSnapshot:
    meta: JobMeta
    phase: Phase
    started_at_s: float
    episode_index: int = 0
    turn_index: int | None = None
    guest_id: str = ""
    host_address: str = ""
    image_ref: str = ""
    last_attacker: str = "waiting"
    last_defender: str = "waiting"
    attacker_tools: int = 0
    defender_tools: int = 0
    log: tuple[str, ...] = ()
    recent: tuple[JobEvent, ...] = ()
    completed: tuple[EpisodeSummary, ...] = ()
    probe: ProbeResult | None = None
    last_tool: ToolEvent | None = None
    last_tool_role: Role | None = None
    error: str | None = None
    last_terminal: TerminalOutcome | None = None

    @property
    def active_role(self) -> Role | None:
        if self.phase is Phase.ATTACKING:
            return Role.ATTACKER
        if self.phase is Phase.DEFENDING:
            return Role.DEFENDER
        return None


def initial_snapshot(meta: JobMeta, *, started_at_s: float) -> JobSnapshot:
    if meta.episodes_planned < 1:
        raise ValueError("episodes_planned must be >= 1")
    if meta.turns_per_side < 1:
        raise ValueError("turns_per_side must be >= 1")
    return JobSnapshot(meta=meta, phase=Phase.RESTORING, started_at_s=started_at_s)


def progress(snapshot: JobSnapshot) -> JobProgress:
    return JobProgress(
        completed_episodes=len(snapshot.completed),
        total_episodes=snapshot.meta.episodes_planned,
        turn_index=snapshot.turn_index,
        total_turns=snapshot.meta.turns_per_side * 2,
    )


def estimate_eta_s(snapshot: JobSnapshot) -> float | None:
    if snapshot.phase is Phase.COMPLETE:
        return 0.0
    if not snapshot.completed:
        return None
    mean = sum(item.duration_s for item in snapshot.completed) / len(snapshot.completed)
    remaining = snapshot.meta.episodes_planned - len(snapshot.completed)
    return max(0.0, mean * remaining)


def apply(snapshot: JobSnapshot, event: JobEvent) -> JobSnapshot:
    if snapshot.phase is Phase.COMPLETE:
        raise InvalidTransition("job already complete")
    if snapshot.phase is Phase.FAILED and event.kind != "error":
        raise InvalidTransition("job already failed")
    recent = _bounded(snapshot.recent + (event,), EVENT_LIMIT)
    log = _bounded(snapshot.log + (_log_line(event),), LOG_LIMIT)
    base = replace(snapshot, recent=recent, log=log)
    if isinstance(event, RestoreStarted):
        return _restore_started(base, event)
    if isinstance(event, RestoreFinished):
        return _restore_finished(base, event)
    if isinstance(event, TurnStarted):
        return _turn_started(base, event)
    if isinstance(event, TurnEnded):
        return _turn_ended(base, event)
    if isinstance(event, ToolObserved):
        return _tool(base, event)
    if isinstance(event, ProbeStarted):
        return _probe_started(base, event)
    if isinstance(event, ProbeFinished):
        return _probe_finished(base, event)
    if isinstance(event, EpisodeEnded):
        return _episode_ended(base, event)
    if isinstance(event, JobEnded):
        return _job_ended(base, event)
    if isinstance(event, JobError):
        return replace(base, phase=Phase.FAILED, error=event.message)
    raise InvalidTransition(f"unknown event {type(event)!r}")


def _bounded(items: tuple, limit: int) -> tuple:
    return items[-limit:]


def _require_episode(snapshot: JobSnapshot, episode_index: int) -> None:
    if episode_index != snapshot.episode_index:
        raise InvalidTransition(
            f"episode {episode_index} does not match current {snapshot.episode_index}"
        )


def _restore_started(snapshot: JobSnapshot, event: RestoreStarted) -> JobSnapshot:
    if snapshot.phase not in (Phase.RESTORING, Phase.ATTACKING, Phase.DEFENDING, Phase.PROBING):
        raise InvalidTransition(f"restore cannot start in {snapshot.phase.value}")
    _require_episode(snapshot, event.episode_index)
    return replace(
        snapshot,
        phase=Phase.RESTORING,
        guest_id=event.guest_id,
        image_ref=event.image_ref,
        host_address="",
        probe=None,
        last_terminal=None,
        attacker_tools=0,
        defender_tools=0,
        last_attacker="waiting",
        last_defender="waiting",
        turn_index=None,
    )


def _restore_finished(snapshot: JobSnapshot, event: RestoreFinished) -> JobSnapshot:
    if snapshot.phase is not Phase.RESTORING:
        raise InvalidTransition("restore finished while not restoring")
    _require_episode(snapshot, event.episode_index)
    return replace(
        snapshot,
        guest_id=event.guest_id,
        host_address=event.host_address,
    )


def _turn_started(snapshot: JobSnapshot, event: TurnStarted) -> JobSnapshot:
    if snapshot.phase not in (Phase.RESTORING, Phase.ATTACKING, Phase.DEFENDING):
        raise InvalidTransition(f"turn cannot start in {snapshot.phase.value}")
    _require_episode(snapshot, event.episode_index)
    phase = Phase.ATTACKING if event.role is Role.ATTACKER else Phase.DEFENDING
    attacker_tools = 0 if event.role is Role.ATTACKER else snapshot.attacker_tools
    defender_tools = 0 if event.role is Role.DEFENDER else snapshot.defender_tools
    return replace(
        snapshot,
        phase=phase,
        turn_index=event.turn_index,
        attacker_tools=attacker_tools,
        defender_tools=defender_tools,
    )


def _turn_ended(snapshot: JobSnapshot, event: TurnEnded) -> JobSnapshot:
    expected = Phase.ATTACKING if event.role is Role.ATTACKER else Phase.DEFENDING
    if snapshot.phase is not expected:
        raise InvalidTransition(f"turn end for {event.role.value} in {snapshot.phase.value}")
    _require_episode(snapshot, event.episode_index)
    return snapshot


def _tool(snapshot: JobSnapshot, event: ToolObserved) -> JobSnapshot:
    expected = Phase.ATTACKING if event.role is Role.ATTACKER else Phase.DEFENDING
    if snapshot.phase is not expected:
        raise InvalidTransition(f"tool for {event.role.value} in {snapshot.phase.value}")
    _require_episode(snapshot, event.episode_index)
    label = _tool_label(event.tool)
    if event.role is Role.ATTACKER:
        return replace(
            snapshot,
            last_attacker=label,
            attacker_tools=snapshot.attacker_tools + 1,
            last_tool=event.tool,
            last_tool_role=event.role,
        )
    return replace(
        snapshot,
        last_defender=label,
        defender_tools=snapshot.defender_tools + 1,
        last_tool=event.tool,
        last_tool_role=event.role,
    )


def _probe_started(snapshot: JobSnapshot, event: ProbeStarted) -> JobSnapshot:
    if snapshot.phase not in (Phase.ATTACKING, Phase.DEFENDING, Phase.RESTORING):
        raise InvalidTransition(f"probe cannot start in {snapshot.phase.value}")
    _require_episode(snapshot, event.episode_index)
    return replace(snapshot, phase=Phase.PROBING, turn_index=None)


def _probe_finished(snapshot: JobSnapshot, event: ProbeFinished) -> JobSnapshot:
    if snapshot.phase is not Phase.PROBING:
        raise InvalidTransition("probe finished while not probing")
    _require_episode(snapshot, event.episode_index)
    return replace(snapshot, probe=event.result)


def _episode_ended(snapshot: JobSnapshot, event: EpisodeEnded) -> JobSnapshot:
    if snapshot.phase is not Phase.PROBING:
        raise InvalidTransition("episode ended before probe")
    _require_episode(snapshot, event.episode_index)
    summary = EpisodeSummary(
        episode_index=event.episode_index,
        duration_s=event.duration_s,
        terminal=event.terminal,
        guest_id=event.guest_id,
        probe=snapshot.probe,
    )
    completed = snapshot.completed + (summary,)
    nxt = event.episode_index + 1
    if nxt >= snapshot.meta.episodes_planned:
        return replace(
            snapshot,
            completed=completed,
            last_terminal=event.terminal,
            episode_index=event.episode_index,
        )
    return replace(
        snapshot,
        phase=Phase.RESTORING,
        completed=completed,
        last_terminal=event.terminal,
        episode_index=nxt,
        turn_index=None,
        probe=None,
        attacker_tools=0,
        defender_tools=0,
        last_attacker="waiting",
        last_defender="waiting",
    )


def _job_ended(snapshot: JobSnapshot, event: JobEnded) -> JobSnapshot:
    if len(snapshot.completed) != snapshot.meta.episodes_planned:
        raise InvalidTransition("job ended before every episode produced a terminal")
    if snapshot.last_terminal is None:
        raise InvalidTransition("complete requires a terminal outcome")
    return replace(snapshot, phase=Phase.COMPLETE)


def _tool_label(tool: ToolEvent) -> str:
    cmd = tool.args.get("cmd") or tool.args.get("url") or tool.name
    return f"{tool.name} {cmd}".strip()


def _log_line(event: JobEvent) -> str:
    stamp = _fmt_clock(getattr(event, "at_s", 0.0))
    if isinstance(event, RestoreStarted):
        return f"{stamp}  restore {event.guest_id}  {event.image_ref}"
    if isinstance(event, RestoreFinished):
        return f"{stamp}  restored {event.guest_id}  {event.host_address}  {_fmt_dur(event.duration_s)}"
    if isinstance(event, TurnStarted):
        return f"{stamp}  {event.role.value} turn {event.turn_index} started"
    if isinstance(event, TurnEnded):
        return f"{stamp}  {event.role.value} turn {event.turn_index} done  {_fmt_dur(event.duration_s)}"
    if isinstance(event, ToolObserved):
        code = "—" if event.tool.exit_code is None else str(event.tool.exit_code)
        return (
            f"{stamp}  {event.role.value}  {_tool_label(event.tool)}  "
            f"exit {code}  {event.tool.duration_ms}ms"
        )
    if isinstance(event, ProbeStarted):
        return f"{stamp}  probe episode {event.episode_index}"
    if isinstance(event, ProbeFinished):
        return (
            f"{stamp}  probe euid={event.result.guest_attacker_euid} "
            f"host_root={event.result.host_confirmed_root} "
            f"avail={event.result.availability_ok}"
        )
    if isinstance(event, EpisodeEnded):
        t = event.terminal
        return (
            f"{stamp}  episode {event.episode_index}  {t.reason_code.value}  "
            f"a:{t.attacker_reward:g} d:{t.defender_reward:g}"
        )
    if isinstance(event, JobEnded):
        return f"{stamp}  job complete  {_fmt_dur(event.duration_s)}"
    if isinstance(event, JobError):
        return f"{stamp}  error {event.operation}  {event.message}"
    return f"{stamp}  {event.kind}"


def _fmt_clock(at_s: float) -> str:
    total = max(0, int(at_s))
    return f"{total // 60:02d}:{total % 60:02d}"


def _fmt_dur(duration_s: float) -> str:
    if duration_s < 1:
        return f"{int(duration_s * 1000)}ms"
    return f"{duration_s:.1f}s"
