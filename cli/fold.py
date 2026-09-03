from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from threading import Condition, Lock

from ultron.train.adjudicator import ProbeResult, adjudicate
from ultron.train.episode_runner import GuestVm
from ultron.train.schema_v1 import (
    ReasonCode,
    Role,
    TerminalOutcome,
    ToolEvent,
    TrajectoryStep,
    TrajectoryV1,
)

from .board import (
    STROKE_LIMIT,
    AgentFace,
    Board,
    Cast,
    Done,
    Failed,
    GuestFace,
    GuestHealth,
    Idle,
    Phase,
    Probing,
    ProcessReport,
    Progress,
    Restoring,
    Settled,
    Stroke,
    Trading,
    TurnCursor,
)
from .spec import JobSpec


class IllegalTransition(RuntimeError):
    pass


def _guest_face(
    vm: GuestVm,
    health: GuestHealth,
    reason: str | None,
) -> GuestFace:
    return GuestFace(
        guest_id=vm.vm_id,
        isolation=vm.isolation,
        host_address=vm.host_address,
        image_ref=vm.image_ref,
        health=health,
        quarantine_reason=reason,
    )


def _stroke_from_tool(event: ToolEvent, role: Role, at_mono: float) -> Stroke:
    detail = ""
    cmd = event.args.get("cmd") if event.args else None
    if isinstance(cmd, str):
        detail = cmd
    elif event.stdout_head:
        detail = event.stdout_head.splitlines()[0] if event.stdout_head else ""
    return Stroke(
        at_mono=at_mono,
        actor=role.value,
        verb=event.name,
        detail=detail[:48],
        exit_code=event.exit_code,
        duration_ms=event.duration_ms,
    )


def _stances_from_reason(reason: ReasonCode) -> tuple[str, str]:
    if reason is ReasonCode.ATTACKER_ROOT:
        return "won", "failed"
    if reason in (ReasonCode.DEFENDER_HOLD, ReasonCode.TIMEOUT):
        return "held", "won"
    return "failed", "failed"


class BoardFold:
    def __init__(self, spec: JobSpec, *, clock: Callable[[], float]) -> None:
        self._spec = spec
        self._clock = clock
        self._lock = Lock()
        self._changed = Condition(self._lock)
        self._generation = 0
        started = clock()
        self._started = started
        self._episode_started = started
        self._episode_index = 0
        self._finished: list[float] = []
        self._phase: Phase = Idle()
        self._guest: GuestFace | None = None
        self._sha = spec.snapshot_sha256
        self._attacker = AgentFace(Role.ATTACKER, "idle", 0, None)
        self._defender = AgentFace(Role.DEFENDER, "idle", 0, None)
        self._strokes: list[Stroke] = []
        self._headline = "idle"
        self._quarantine: Mapping[str, str] | None = None
        self._vm: GuestVm | None = None

    def snapshot(self) -> Board:
        with self._lock:
            return self._board()

    def subscribe(self) -> Iterator[Board]:
        last = -1
        while True:
            with self._changed:
                self._changed.wait_for(lambda: self._generation != last, timeout=0.1)
                last = self._generation
                board = self._board()
            yield board


    def poll_quarantine(self, mapping: Mapping[str, str] | None) -> None:
        with self._lock:
            if mapping == self._quarantine:
                return
            self._quarantine = None if mapping is None else dict(mapping)
            if self._guest is not None and mapping is not None:
                reason = mapping.get(self._guest.guest_id)
                health: GuestHealth = "quarantined" if reason else self._guest.health
                if reason:
                    health = "quarantined"
                self._guest = GuestFace(
                    guest_id=self._guest.guest_id,
                    isolation=self._guest.isolation,
                    host_address=self._guest.host_address,
                    image_ref=self._guest.image_ref,
                    health=health,
                    quarantine_reason=reason,
                )
            self._bump()

    def apply_restore_start(self, vm: GuestVm, sha256: str) -> None:
        with self._lock:
            if not isinstance(self._phase, (Idle, Settled)):
                raise IllegalTransition(f"restore from {type(self._phase).__name__}")
            self._vm = vm
            reason = None
            health: GuestHealth = "restoring"
            if self._quarantine and vm.vm_id in self._quarantine:
                reason = self._quarantine[vm.vm_id]
                health = "quarantined"
            self._guest = _guest_face(vm, health, reason)
            self._sha = sha256
            self._episode_started = self._clock()
            self._attacker = AgentFace(Role.ATTACKER, "waiting", 0, self._attacker.last)
            self._defender = AgentFace(Role.DEFENDER, "waiting", 0, self._defender.last)
            self._phase = Restoring(guest=self._guest, sha256=sha256)
            self._push(
                Stroke(self._clock(), "guest", "restore", sha256[:12], None, None),
                "restoring " + vm.vm_id,
            )
            self._bump()

    def apply_restore_done(self, vm: GuestVm) -> None:
        with self._lock:
            if not isinstance(self._phase, Restoring):
                raise IllegalTransition("restore_done")
            self._vm = vm
            self._guest = _guest_face(vm, "live", None)
            self._phase = Restoring(guest=self._guest, sha256=self._sha)
            self._headline = "guest live"
            self._bump()

    def apply_turn_start(self, role: Role, turn_index: int) -> None:
        with self._lock:
            cursor = TurnCursor.of(turn_index, self._spec.turns_per_side)
            if cursor.side is not role:
                raise IllegalTransition("role/index disagree")
            if self._guest is None:
                raise IllegalTransition("turn without guest")
            if isinstance(self._phase, Restoring):
                pass
            elif isinstance(self._phase, Trading):
                if turn_index != self._phase.cursor.index + 1:
                    raise IllegalTransition("turn skip")
            else:
                raise IllegalTransition(f"turn from {type(self._phase).__name__}")
            self._phase = Trading(guest=self._guest, cursor=cursor)
            acting, waiting = (
                (Role.ATTACKER, Role.DEFENDER)
                if role is Role.ATTACKER
                else (Role.DEFENDER, Role.ATTACKER)
            )
            self._set_stance(acting, "acting")
            self._set_stance(waiting, "waiting")
            self._headline = f"{role.value} turn {turn_index}"
            self._bump()

    def apply_turn_done(
        self, role: Role, turn_index: int, steps: list[TrajectoryStep]
    ) -> None:
        with self._lock:
            if not isinstance(self._phase, Trading):
                raise IllegalTransition("turn_done")
            if self._phase.cursor.index != turn_index or self._phase.cursor.side is not role:
                raise IllegalTransition("turn_done mismatch")
            now = self._clock()
            last: Stroke | None = None
            for step in steps:
                for event in step.tool_events:
                    last = _stroke_from_tool(event, role, now)
                    self._strokes.append(last)
            self._strokes = self._strokes[-STROKE_LIMIT:]
            if role is Role.ATTACKER:
                self._attacker = AgentFace(
                    Role.ATTACKER, "waiting", self._attacker.turns_done + 1, last
                )
            else:
                self._defender = AgentFace(
                    Role.DEFENDER, "waiting", self._defender.turns_done + 1, last
                )
            self._headline = f"{role.value} turn {turn_index} done"
            self._bump()

    def apply_probe_start(self) -> None:
        with self._lock:
            if self._guest is None:
                raise IllegalTransition("probe without guest")
            if isinstance(self._phase, Restoring) and self._spec.turns_per_side == 0:
                pass
            elif isinstance(self._phase, Trading):
                pass
            else:
                raise IllegalTransition(f"probe from {type(self._phase).__name__}")
            self._phase = Probing(guest=self._guest)
            self._attacker = AgentFace(
                Role.ATTACKER, "waiting", self._attacker.turns_done, self._attacker.last
            )
            self._defender = AgentFace(
                Role.DEFENDER, "waiting", self._defender.turns_done, self._defender.last
            )
            self._headline = "final probe"
            self._bump()

    def apply_probe_done(self, probe: ProbeResult) -> None:
        with self._lock:
            if not isinstance(self._phase, Probing) or self._guest is None:
                raise IllegalTransition("probe_done")
            reason, atk, dfn = adjudicate(probe)
            terminal = TerminalOutcome(
                reason_code=reason,
                attacker_euid=probe.guest_attacker_euid,
                host_confirmed_root=probe.host_confirmed_root,
                availability_ok=probe.availability_ok,
                attacker_reward=atk,
                defender_reward=dfn,
            )
            atk_stance, dfn_stance = _stances_from_reason(reason)
            self._attacker = AgentFace(
                Role.ATTACKER, atk_stance, self._attacker.turns_done, self._attacker.last
            )
            self._defender = AgentFace(
                Role.DEFENDER, dfn_stance, self._defender.turns_done, self._defender.last
            )
            self._phase = Settled(guest=self._guest, terminal=terminal)
            self._headline = reason.value
            self._push(
                Stroke(self._clock(), "job", "probe", reason.value, None, None),
                reason.value,
            )
            self._bump()

    def apply_episode_done(self, trajectories: list[TrajectoryV1] | None = None) -> None:
        with self._lock:
            if not isinstance(self._phase, Settled):
                raise IllegalTransition("episode_done")
            elapsed = max(0.0, self._clock() - self._episode_started)
            self._finished.append(elapsed)
            self._episode_index += 1
            self._attacker = AgentFace(Role.ATTACKER, "idle", 0, self._attacker.last)
            self._defender = AgentFace(Role.DEFENDER, "idle", 0, self._defender.last)
            _ = trajectories
            self._bump()

    def apply_fail(self, at: str, error: BaseException) -> None:
        with self._lock:
            allowed = ("restore", "turn", "probe", "loop")
            where = at if at in allowed else "loop"
            self._phase = Failed(at=where, error=str(error), guest=self._guest)  # type: ignore[arg-type]
            self._headline = f"failed {where}"
            self._bump()

    def apply_job_done(self) -> None:
        with self._lock:
            if isinstance(self._phase, Failed):
                return
            if not isinstance(self._phase, Settled):
                raise IllegalTransition("job_done")
            self._phase = Done(last=self._phase)
            self._headline = "done"
            self._bump()

    def _set_stance(self, role: Role, stance: str) -> None:
        if role is Role.ATTACKER:
            self._attacker = AgentFace(
                Role.ATTACKER, stance, self._attacker.turns_done, self._attacker.last
            )
        else:
            self._defender = AgentFace(
                Role.DEFENDER, stance, self._defender.turns_done, self._defender.last
            )

    def _push(self, stroke: Stroke, headline: str) -> None:
        self._strokes.append(stroke)
        self._strokes = self._strokes[-STROKE_LIMIT:]
        self._headline = headline

    def _bump(self) -> None:
        self._generation += 1
        self._changed.notify_all()

    def _board(self) -> Board:
        cursor = self._phase.cursor if isinstance(self._phase, Trading) else None
        now = self._clock()
        progress = Progress(
            episode_index=self._episode_index,
            episode_count=self._spec.episode_count,
            turns_per_side=self._spec.turns_per_side,
            cursor=cursor,
            started_mono=self._started,
            episode_started_mono=self._episode_started,
            finished_durations_s=tuple(self._finished),
            now_mono=now,
        )
        return Board(
            spec=self._spec,
            progress=progress,
            phase=self._phase,
            cast=Cast(
                attacker=self._attacker,
                defender=self._defender,
                guest=self._guest,
            ),
            process=ProcessReport(self._headline, tuple(self._strokes)),
            quarantine=self._quarantine,
        )
