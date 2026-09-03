from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Union

from ultron.env.backend import IsolationBackend
from ultron.train.schema_v1 import ReasonCode, Role, TerminalOutcome

from .spec import JobSpec

AgentStance = Literal["idle", "acting", "waiting", "won", "held", "failed"]
GuestHealth = Literal["unknown", "restoring", "live", "quarantined", "stopped"]
STROKE_LIMIT = 12


@dataclass(frozen=True)
class TurnCursor:
    index: int
    side: Role
    per_side: int

    @classmethod
    def of(cls, index: int, per_side: int) -> TurnCursor:
        if per_side < 0:
            raise ValueError("per_side must be >= 0")
        budget = per_side * 2
        if budget == 0:
            raise ValueError("no turns; do not build a cursor")
        if not 0 <= index < budget:
            raise ValueError("turn index out of episode budget")
        side = Role.ATTACKER if index % 2 == 0 else Role.DEFENDER
        return cls(index=index, side=side, per_side=per_side)

    @property
    def side_ordinal(self) -> int:
        return self.index // 2

    @property
    def budget(self) -> int:
        return self.per_side * 2


@dataclass(frozen=True)
class Progress:
    episode_index: int
    episode_count: int
    turns_per_side: int
    cursor: TurnCursor | None
    started_mono: float
    episode_started_mono: float
    finished_durations_s: tuple[float, ...]
    now_mono: float

    def __post_init__(self) -> None:
        if not 0 <= self.episode_index <= self.episode_count:
            raise ValueError("episode_index")
        if self.cursor is not None and self.cursor.per_side != self.turns_per_side:
            raise ValueError("cursor/spec turns_per_side disagree")

    @property
    def episodes_done(self) -> int:
        return min(self.episode_index, self.episode_count)

    @property
    def fraction(self) -> float:
        total = float(self.episode_count)
        done = float(self.episodes_done)
        if self.cursor is not None and self.episodes_done < self.episode_count:
            done += (self.cursor.index + 1) / float(self.cursor.budget)
        return min(1.0, done / total)

    @property
    def eta_s(self) -> float | None:
        if not self.finished_durations_s:
            return None
        mean = sum(self.finished_durations_s) / len(self.finished_durations_s)
        remaining = self.episode_count - self.episodes_done
        if self.cursor is not None and remaining > 0:
            remaining = max(0, remaining - 1)
        return mean * remaining

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.now_mono - self.started_mono)


@dataclass(frozen=True)
class Stroke:
    at_mono: float
    actor: Literal["attacker", "defender", "guest", "job"]
    verb: str
    detail: str
    exit_code: int | None
    duration_ms: int | None


@dataclass(frozen=True)
class AgentFace:
    role: Role
    stance: AgentStance
    turns_done: int
    last: Stroke | None


@dataclass(frozen=True)
class GuestFace:
    guest_id: str
    isolation: IsolationBackend
    host_address: str
    image_ref: str
    health: GuestHealth
    quarantine_reason: str | None


@dataclass(frozen=True)
class Cast:
    attacker: AgentFace
    defender: AgentFace
    guest: GuestFace | None


@dataclass(frozen=True)
class ProcessReport:
    headline: str
    strokes: tuple[Stroke, ...]


@dataclass(frozen=True)
class Idle:
    kind: Literal["idle"] = "idle"


@dataclass(frozen=True)
class Restoring:
    guest: GuestFace
    sha256: str
    kind: Literal["restoring"] = "restoring"


@dataclass(frozen=True)
class Trading:
    guest: GuestFace
    cursor: TurnCursor
    kind: Literal["trading"] = "trading"


@dataclass(frozen=True)
class Probing:
    guest: GuestFace
    kind: Literal["probing"] = "probing"


@dataclass(frozen=True)
class Settled:
    guest: GuestFace
    terminal: TerminalOutcome
    kind: Literal["settled"] = "settled"


@dataclass(frozen=True)
class Failed:
    at: Literal["restore", "turn", "probe", "loop"]
    error: str
    guest: GuestFace | None
    kind: Literal["failed"] = "failed"


@dataclass(frozen=True)
class Done:
    last: Settled
    kind: Literal["done"] = "done"


Phase = Union[Idle, Restoring, Trading, Probing, Settled, Failed, Done]


@dataclass(frozen=True)
class Board:
    spec: JobSpec
    progress: Progress
    phase: Phase
    cast: Cast
    process: ProcessReport
    quarantine: Mapping[str, str] | None

    @property
    def acting_role(self) -> Role | None:
        if isinstance(self.phase, Trading):
            return self.phase.cursor.side
        return None

    @property
    def reason_code(self) -> ReasonCode | None:
        if isinstance(self.phase, Settled):
            return self.phase.terminal.reason_code
        if isinstance(self.phase, Done):
            return self.phase.last.terminal.reason_code
        return None
