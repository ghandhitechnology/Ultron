import pytest

from ultron.env.backend import IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.schema_v1 import ReasonCode, Role, TerminalOutcome
from ultron.cli.model import (
    AttackingSnapshot,
    CompleteSnapshot,
    DefendingSnapshot,
    EpisodeEndedEvent,
    InvalidTransition,
    JobEndedEvent,
    JobSpec,
    ProbeEvent,
    ProbingSnapshot,
    RestoreEvent,
    RestoringSnapshot,
    TurnEndedEvent,
    TurnStartedEvent,
    apply,
    estimate_eta_s,
    initial_snapshot,
)


def _spec(episodes: int = 2, turns_per_side: int = 1) -> JobSpec:
    return JobSpec(
        version="0.1.0",
        generation=0,
        profile_id="web",
        isolation_backend=IsolationBackend.DOCKER,
        episode_count=episodes,
        turns_per_side=turns_per_side,
    )


def _probe() -> ProbeResult:
    return ProbeResult(
        guest_attacker_euid=1000,
        host_confirmed_root=False,
        availability_ok=True,
        infra_ok=True,
        timed_out=False,
    )


def _terminal() -> TerminalOutcome:
    return TerminalOutcome(
        reason_code=ReasonCode.DEFENDER_HOLD,
        attacker_euid=1000,
        host_confirmed_root=False,
        availability_ok=True,
        attacker_reward=0.0,
        defender_reward=1.0,
    )


def _run_one_episode(snapshot, *, index: int, duration_s: float = 5.0):
    snapshot = apply(snapshot, RestoreEvent(index, "web-01", "img", 0.2, 1.0))
    snapshot = apply(snapshot, TurnStartedEvent(index, 0, Role.ATTACKER, 1.1))
    snapshot = apply(snapshot, TurnEndedEvent(index, 0, Role.ATTACKER, 0.3, 1, 1.4))
    snapshot = apply(snapshot, TurnStartedEvent(index, 1, Role.DEFENDER, 1.5))
    snapshot = apply(snapshot, TurnEndedEvent(index, 1, Role.DEFENDER, 0.3, 1, 1.8))
    snapshot = apply(snapshot, ProbeEvent(index, _probe(), 0.1, 1.9))
    snapshot = apply(snapshot, EpisodeEndedEvent(index, duration_s, _terminal(), 2.0))
    return snapshot


def test_happy_path_reaches_complete_with_terminal():
    snapshot = initial_snapshot(_spec(episodes=1), started_at_s=0.0)
    assert isinstance(snapshot, RestoringSnapshot)

    snapshot = apply(snapshot, RestoreEvent(0, "web-01", "img", 0.2, 1.0))
    assert isinstance(snapshot, AttackingSnapshot)

    snapshot = apply(snapshot, TurnStartedEvent(0, 0, Role.ATTACKER, 1.1))
    assert isinstance(snapshot, AttackingSnapshot)
    snapshot = apply(snapshot, TurnEndedEvent(0, 0, Role.ATTACKER, 0.3, 1, 1.4))

    snapshot = apply(snapshot, TurnStartedEvent(0, 1, Role.DEFENDER, 1.5))
    assert isinstance(snapshot, DefendingSnapshot)
    snapshot = apply(snapshot, TurnEndedEvent(0, 1, Role.DEFENDER, 0.3, 1, 1.8))

    snapshot = apply(snapshot, ProbeEvent(0, _probe(), 0.1, 1.9))
    assert isinstance(snapshot, ProbingSnapshot)

    snapshot = apply(snapshot, EpisodeEndedEvent(0, 5.0, _terminal(), 2.0))
    assert isinstance(snapshot, RestoringSnapshot)

    snapshot = apply(snapshot, JobEndedEvent(2.1, 2.1))
    assert isinstance(snapshot, CompleteSnapshot)
    assert isinstance(snapshot.final_episode.terminal, TerminalOutcome)


def test_job_end_before_any_episode_is_illegal():
    snapshot = initial_snapshot(_spec(), started_at_s=0.0)
    with pytest.raises(InvalidTransition):
        apply(snapshot, JobEndedEvent(1.0, 1.0))


def test_turn_start_while_restoring_is_illegal():
    snapshot = initial_snapshot(_spec(), started_at_s=0.0)
    with pytest.raises(InvalidTransition):
        apply(snapshot, TurnStartedEvent(0, 0, Role.ATTACKER, 1.0))


def test_eta_none_until_episode_completed_then_remaining_times_mean():
    snapshot = initial_snapshot(_spec(episodes=2), started_at_s=0.0)
    assert estimate_eta_s(snapshot) is None

    snapshot = apply(snapshot, RestoreEvent(0, "web-01", "img", 0.2, 1.0))
    assert estimate_eta_s(snapshot) is None

    snapshot = _run_one_episode(
        initial_snapshot(_spec(episodes=2), started_at_s=0.0),
        index=0,
        duration_s=8.0,
    )
    assert estimate_eta_s(snapshot) == pytest.approx(8.0)
