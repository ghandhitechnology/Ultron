from dataclasses import dataclass, replace

from ultron.env.backend import IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.episode_runner import EpisodeConfig, EpisodeRunner
from ultron.train.schema_v1 import ReasonCode, Role, ToolEvent, TrajectoryStep


@dataclass(frozen=True)
class FakeVm:
    vm_id: str
    isolation: IsolationBackend
    host_address: str
    image_ref: str


def _runner(probe: ProbeResult, *, run_turn=None, turns=0, turn_probe=None) -> EpisodeRunner:
    return EpisodeRunner(
        snapshot_sha256="a" * 64,
        load_profile=lambda profile_id: {},
        run_turn=run_turn or (lambda *args: []),
        final_probe=lambda vm, profile: probe,
        restore=lambda vm, sha: None,
        turns_per_side=turns,
        turn_probe=turn_probe,
    )


def _cfg(*, shaping: bool = True, prep_turns: int = 0) -> EpisodeConfig:
    return EpisodeConfig(
        profile_id="web",
        generation=0,
        group_id="group-1",
        opponent_checkpoint_id="defender-gen0",
        attacker_ckpt="attacker-gen0",
        defender_ckpt="defender-gen0",
        shaping=shaping,
        prep_turns=prep_turns,
    )


def _vm() -> FakeVm:
    return FakeVm("vm-1", IsolationBackend.KVM, "10.0.0.2", "img:golden")


def test_run_restores_with_expected_sha() -> None:
    calls: list[tuple[str, str]] = []
    probe = ProbeResult(
        guest_attacker_euid=1000,
        host_confirmed_root=False,
        availability_ok=True,
        infra_ok=True,
        timed_out=False,
    )
    runner = EpisodeRunner(
        snapshot_sha256="a" * 64,
        load_profile=lambda profile_id: {},
        run_turn=lambda *args: [],
        final_probe=lambda vm, profile: probe,
        restore=lambda vm, sha: calls.append((vm.vm_id, sha)),
        turns_per_side=0,
    )
    cfg = EpisodeConfig(
        profile_id="web",
        generation=0,
        group_id="group-1",
        opponent_checkpoint_id="defender-gen0",
        attacker_ckpt="attacker-gen0",
        defender_ckpt="defender-gen0",
    )
    trajectories = runner.run(cfg, _vm())
    assert calls == [("vm-1", "a" * 64)]
    assert len(trajectories) == 2
    assert all(traj.isolation_backend is IsolationBackend.KVM for traj in trajectories)


def test_timeout_is_unresolved() -> None:
    probe = ProbeResult(1000, False, True, True, True)
    trajectories = _runner(probe).run(_cfg(), _vm())
    assert trajectories[0].terminal.reason_code is ReasonCode.TIMEOUT
    assert trajectories[0].terminal.attacker_reward == 0.0
    assert trajectories[0].terminal.defender_reward == 0.0
    assert trajectories[0].terminal.unresolved is True


def test_attacker_finish_stops_the_trial() -> None:
    finish = TrajectoryStep(
        turn_index=0,
        side=Role.ATTACKER,
        prompt_token_ids=[],
        assistant_token_ids=[1],
        assistant_mask=[1],
        tool_events=[ToolEvent("finish", {}, "", "", 0, 1)],
    )
    calls: list[Role] = []

    def run_turn(vm, side, profile, turn):
        calls.append(side)
        return [finish] if side is Role.ATTACKER else []

    probe = ProbeResult(1000, False, True, True, False)
    _runner(probe, run_turn=run_turn, turns=8).run(_cfg(), _vm())
    assert calls == [Role.ATTACKER]


def test_shaping_off_skips_subgoal_credit() -> None:
    hit = TrajectoryStep(
        turn_index=0,
        side=Role.ATTACKER,
        prompt_token_ids=[],
        assistant_token_ids=[1],
        assistant_mask=[1],
        tool_events=[],
        subgoal_hits=["suid_bin_found"],
    )
    probe = ProbeResult(1000, False, True, True, False)

    def run_turn(vm, side, profile, turn):
        return [replace(hit, turn_index=turn, side=side)]

    trajectories = _runner(probe, run_turn=run_turn, turns=1).run(_cfg(shaping=False), _vm())
    attacker = next(traj for traj in trajectories if traj.role is Role.ATTACKER)
    assert attacker.steps[0].turn_reward == 0.0
