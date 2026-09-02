from ultron.cli.model import (
    EpisodeEnded,
    InvalidTransition,
    JobEnded,
    JobError,
    JobMeta,
    Phase,
    ProbeFinished,
    ProbeStarted,
    RestoreFinished,
    RestoreStarted,
    ToolObserved,
    TurnEnded,
    TurnStarted,
    apply,
    estimate_eta_s,
    initial_snapshot,
    progress,
)
from ultron.env.backend import IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.schema_v1 import ReasonCode, Role, TerminalOutcome, ToolEvent


def _meta(**kwargs) -> JobMeta:
    values = dict(
        generation=3,
        profile_id="web",
        isolation=IsolationBackend.DOCKER,
        episodes_planned=2,
        turns_per_side=2,
    )
    values.update(kwargs)
    return JobMeta(**values)


def _tool(name: str = "bash", cmd: str = "id", exit_code: int = 0) -> ToolEvent:
    return ToolEvent(
        name=name,
        args={"cmd": cmd},
        stdout_head=f"{cmd} ok\n",
        stdout_tail="",
        exit_code=exit_code,
        duration_ms=12,
    )


def _probe(*, euid: int = 1000, root: bool = False) -> ProbeResult:
    return ProbeResult(
        guest_attacker_euid=euid,
        host_confirmed_root=root,
        availability_ok=True,
        infra_ok=True,
        timed_out=False,
    )


def _terminal(reason: ReasonCode = ReasonCode.DEFENDER_HOLD) -> TerminalOutcome:
    attacker = 1.0 if reason is ReasonCode.ATTACKER_ROOT else 0.0
    defender = 1.0 if reason is ReasonCode.DEFENDER_HOLD else 0.0
    return TerminalOutcome(
        reason_code=reason,
        attacker_euid=0 if reason is ReasonCode.ATTACKER_ROOT else 1000,
        host_confirmed_root=reason is ReasonCode.ATTACKER_ROOT,
        availability_ok=True,
        attacker_reward=attacker,
        defender_reward=defender,
    )


def _one_episode(snap, episode: int, *, at: float, root: bool = False):
    snap = apply(
        snap,
        RestoreStarted(
            episode_index=episode,
            guest_id=f"vm-{episode}",
            image_ref="img:golden",
            isolation=IsolationBackend.DOCKER,
            at_s=at,
        ),
    )
    snap = apply(
        snap,
        RestoreFinished(
            episode_index=episode,
            guest_id=f"vm-{episode}",
            host_address=f"vm-{episode}",
            duration_s=0.2,
            at_s=at + 0.2,
        ),
    )
    for turn, role in ((0, Role.ATTACKER), (1, Role.DEFENDER), (2, Role.ATTACKER), (3, Role.DEFENDER)):
        snap = apply(
            snap,
            TurnStarted(episode_index=episode, turn_index=turn, role=role, at_s=at + 1 + turn),
        )
        snap = apply(
            snap,
            ToolObserved(
                episode_index=episode,
                turn_index=turn,
                role=role,
                sequence=0,
                tool=_tool(cmd=f"{role.value}-{turn}"),
                at_s=at + 1.1 + turn,
            ),
        )
        snap = apply(
            snap,
            TurnEnded(
                episode_index=episode,
                turn_index=turn,
                role=role,
                duration_s=0.4,
                step_count=1,
                at_s=at + 1.4 + turn,
            ),
        )
    snap = apply(snap, ProbeStarted(episode_index=episode, at_s=at + 6))
    snap = apply(
        snap,
        ProbeFinished(
            episode_index=episode,
            result=_probe(euid=0 if root else 1000, root=root),
            duration_s=0.1,
            at_s=at + 6.1,
        ),
    )
    reason = ReasonCode.ATTACKER_ROOT if root else ReasonCode.DEFENDER_HOLD
    snap = apply(
        snap,
        EpisodeEnded(
            episode_index=episode,
            duration_s=8.0,
            terminal=_terminal(reason),
            guest_id=f"vm-{episode}",
            at_s=at + 6.2,
        ),
    )
    return snap


def test_initial_is_restoring() -> None:
    snap = initial_snapshot(_meta(), started_at_s=0.0)
    assert snap.phase is Phase.RESTORING
    assert snap.active_role is None
    assert estimate_eta_s(snap) is None
    assert progress(snap).completed_episodes == 0


def test_full_job_reaches_complete_with_eta() -> None:
    snap = initial_snapshot(_meta(), started_at_s=0.0)
    snap = _one_episode(snap, 0, at=0.0, root=False)
    assert snap.phase is Phase.RESTORING
    assert snap.episode_index == 1
    assert snap.last_terminal is not None
    assert snap.last_terminal.reason_code is ReasonCode.DEFENDER_HOLD
    assert estimate_eta_s(snap) == 8.0
    snap = _one_episode(snap, 1, at=10.0, root=True)
    snap = apply(snap, JobEnded(duration_s=18.0, at_s=18.0))
    assert snap.phase is Phase.COMPLETE
    assert estimate_eta_s(snap) == 0.0
    assert progress(snap).completed_episodes == 2
    assert snap.completed[1].terminal.reason_code is ReasonCode.ATTACKER_ROOT


def test_both_sides_cannot_be_active() -> None:
    snap = initial_snapshot(_meta(), started_at_s=0.0)
    snap = apply(
        snap,
        RestoreStarted(
            episode_index=0,
            guest_id="vm-0",
            image_ref="img",
            isolation=IsolationBackend.DOCKER,
            at_s=0.0,
        ),
    )
    snap = apply(
        snap,
        RestoreFinished(
            episode_index=0,
            guest_id="vm-0",
            host_address="vm-0",
            duration_s=0.1,
            at_s=0.1,
        ),
    )
    snap = apply(
        snap,
        TurnStarted(episode_index=0, turn_index=0, role=Role.ATTACKER, at_s=0.2),
    )
    assert snap.phase is Phase.ATTACKING
    assert snap.active_role is Role.ATTACKER
    snap = apply(
        snap,
        TurnStarted(episode_index=0, turn_index=1, role=Role.DEFENDER, at_s=0.3),
    )
    assert snap.phase is Phase.DEFENDING
    assert snap.active_role is Role.DEFENDER


def test_complete_without_terminals_is_illegal() -> None:
    snap = initial_snapshot(_meta(), started_at_s=0.0)
    try:
        apply(snap, JobEnded(duration_s=1.0, at_s=1.0))
    except InvalidTransition:
        return
    raise AssertionError("expected InvalidTransition")


def test_tool_on_wrong_side_is_illegal() -> None:
    snap = initial_snapshot(_meta(), started_at_s=0.0)
    snap = apply(
        snap,
        RestoreStarted(
            episode_index=0,
            guest_id="vm-0",
            image_ref="img",
            isolation=IsolationBackend.DOCKER,
            at_s=0.0,
        ),
    )
    snap = apply(
        snap,
        RestoreFinished(
            episode_index=0,
            guest_id="vm-0",
            host_address="vm-0",
            duration_s=0.1,
            at_s=0.1,
        ),
    )
    snap = apply(
        snap,
        TurnStarted(episode_index=0, turn_index=0, role=Role.ATTACKER, at_s=0.2),
    )
    try:
        apply(
            snap,
            ToolObserved(
                episode_index=0,
                turn_index=0,
                role=Role.DEFENDER,
                sequence=0,
                tool=_tool(),
                at_s=0.3,
            ),
        )
    except InvalidTransition:
        return
    raise AssertionError("expected InvalidTransition")


def test_error_moves_to_failed() -> None:
    snap = initial_snapshot(_meta(), started_at_s=0.0)
    snap = apply(snap, JobError(message="boom", operation="restore", at_s=0.4))
    assert snap.phase is Phase.FAILED
    assert snap.error == "boom"
    try:
        apply(
            snap,
            TurnStarted(episode_index=0, turn_index=0, role=Role.ATTACKER, at_s=0.5),
        )
    except InvalidTransition:
        return
    raise AssertionError("expected InvalidTransition")


def test_log_records_process_lines() -> None:
    snap = initial_snapshot(_meta(), started_at_s=0.0)
    snap = apply(
        snap,
        RestoreStarted(
            episode_index=0,
            guest_id="vm-0",
            image_ref="img:golden",
            isolation=IsolationBackend.DOCKER,
            at_s=5.0,
        ),
    )
    assert snap.log[-1].startswith("00:05  restore vm-0")
