from ultron.cli.demo import make_demo, origin_clock
from ultron.cli.model import JobMeta, Phase, apply, initial_snapshot
from ultron.cli.observe import drive_job, instrument_runner
from ultron.env.backend import IsolationBackend
from ultron.train.schema_v1 import ReasonCode


def test_drive_demo_through_real_episode_runner(tmp_path) -> None:
    meta = JobMeta(
        generation=0,
        profile_id="web",
        isolation=IsolationBackend.DOCKER,
        episodes_planned=2,
        turns_per_side=2,
    )
    runner, cases = make_demo(meta, delay_s=0.0, sleep=lambda _s: None)
    events: list = []
    drive_job(meta, runner, cases, emit=events.append, clock=origin_clock(), responses_dir=tmp_path)
    snap = initial_snapshot(meta, started_at_s=0.0)
    for event in events:
        snap = apply(snap, event)
    assert snap.phase is Phase.COMPLETE
    assert len(snap.completed) == 2
    assert snap.completed[0].terminal.reason_code is ReasonCode.DEFENDER_HOLD
    assert snap.completed[1].terminal.reason_code is ReasonCode.ATTACKER_ROOT
    kinds = [event.kind for event in events]
    assert kinds[:2] == ["response_archive_opened", "restore_started"]
    assert kinds[-1] == "job_ended"
    assert "tool" in kinds
    assert "probe_finished" in kinds


def test_short_case_list_fails_instead_of_illegal_complete(tmp_path) -> None:
    meta = JobMeta(
        generation=0,
        profile_id="web",
        isolation=IsolationBackend.DOCKER,
        episodes_planned=2,
        turns_per_side=1,
    )
    runner, cases = make_demo(meta, delay_s=0.0, sleep=lambda _s: None)
    events: list = []
    drive_job(meta, runner, cases[:1], emit=events.append, clock=origin_clock(), responses_dir=tmp_path)
    snap = initial_snapshot(meta, started_at_s=0.0)
    for event in events:
        snap = apply(snap, event)
    assert snap.phase is Phase.FAILED
    assert snap.error is not None
    assert "completed 1 of 2" in snap.error


def test_instrumentation_preserves_per_turn_compromise_probes():
    from ultron.train.adjudicator import ProbeResult
    meta = JobMeta(generation=0, profile_id="web", isolation=IsolationBackend.DOCKER, episodes_planned=1, turns_per_side=1)
    runner, cases = make_demo(meta, delay_s=0.0, sleep=lambda _: None)
    runner.turn_probe = lambda *_: ProbeResult(0, True, True, True, False)
    observed = instrument_runner(runner, episode_index=0, emit=lambda _: None, clock=origin_clock())
    trajectories = observed.run(cases[0].config, cases[0].vm)
    assert trajectories[0].terminal.reason_code is ReasonCode.ATTACKER_ROOT
    assert trajectories[0].terminal.recovered
