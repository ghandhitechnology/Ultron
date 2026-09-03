from dataclasses import dataclass

from ultron.env.backend import IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.episode_runner import EpisodeConfig, EpisodeRunner
from ultron.train.schema_v1 import Role, ToolEvent, TrajectoryStep
from ultron.cli.model import EpisodeCase, JobEvent
from ultron.cli.observe import instrument_runner, run_job


@dataclass(frozen=True)
class FakeVm:
    vm_id: str
    isolation: IsolationBackend
    host_address: str
    image_ref: str


def _tool() -> ToolEvent:
    return ToolEvent(
        name="bash",
        args={"cmd": "id"},
        stdout_head="uid=1000",
        stdout_tail="",
        exit_code=0,
        duration_ms=12,
    )


def _step(role: Role, turn: int) -> TrajectoryStep:
    return TrajectoryStep(
        turn_index=turn,
        side=role,
        prompt_token_ids=[1],
        assistant_token_ids=[2],
        assistant_mask=[1],
        tool_events=[_tool()],
    )


def _runner() -> EpisodeRunner:
    probe = ProbeResult(
        guest_attacker_euid=1000,
        host_confirmed_root=False,
        availability_ok=True,
        infra_ok=True,
        timed_out=False,
    )
    return EpisodeRunner(
        snapshot_sha256="a" * 64,
        load_profile=lambda profile_id: {},
        run_turn=lambda vm, role, profile, turn: [_step(role, turn)],
        final_probe=lambda vm, profile: probe,
        restore=lambda vm, sha: None,
        turns_per_side=1,
    )


def _case() -> EpisodeCase:
    config = EpisodeConfig(
        profile_id="web",
        generation=0,
        group_id="g0-demo",
        opponent_checkpoint_id="defender-gen0",
        attacker_ckpt="attacker-gen0",
        defender_ckpt="defender-gen0",
    )
    vm = FakeVm("web-01", IsolationBackend.DOCKER, "10.0.0.10", "img:golden")
    return EpisodeCase(config=config, vm=vm)


def test_run_job_emits_full_episode_contract():
    events: list[JobEvent] = []
    clock = iter(float(n) for n in range(1000))
    run_job(_runner(), [_case()], emit=events.append, clock=lambda: next(clock))

    kinds = [event.kind for event in events]
    for expected in (
        "restore",
        "turn_started",
        "tool",
        "turn_ended",
        "probe",
        "episode_ended",
        "job_ended",
    ):
        assert expected in kinds, f"missing {expected} in {kinds}"


def test_instrument_clones_without_mutating_original():
    runner = _runner()
    original_restore = runner.restore
    original_run_turn = runner.run_turn
    original_probe = runner.final_probe

    events: list[JobEvent] = []
    instrumented = instrument_runner(
        runner, episode_index=0, emit=events.append, clock=lambda: 0.0
    )

    assert instrumented is not runner
    assert instrumented.restore is not runner.restore
    assert runner.restore is original_restore
    assert runner.run_turn is original_run_turn
    assert runner.final_probe is original_probe
