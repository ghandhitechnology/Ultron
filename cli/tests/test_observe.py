from ultron.cli.demo import make_demo, origin_clock
from ultron.cli.model import JobMeta, Phase, apply, initial_snapshot
from ultron.cli.observe import drive_job
from ultron.env.backend import IsolationBackend
from ultron.train.schema_v1 import ReasonCode


def test_drive_demo_through_real_episode_runner() -> None:
    meta = JobMeta(
        generation=0,
        profile_id="web",
        isolation=IsolationBackend.DOCKER,
        episodes_planned=2,
        turns_per_side=2,
    )
    runner, cases = make_demo(meta, delay_s=0.0, sleep=lambda _s: None)
    events: list = []
    drive_job(meta, runner, cases, emit=events.append, clock=origin_clock())
    snap = initial_snapshot(meta, started_at_s=0.0)
    for event in events:
        snap = apply(snap, event)
    assert snap.phase is Phase.COMPLETE
    assert len(snap.completed) == 2
    assert snap.completed[0].terminal.reason_code is ReasonCode.DEFENDER_HOLD
    assert snap.completed[1].terminal.reason_code is ReasonCode.ATTACKER_ROOT
    kinds = [event.kind for event in events]
    assert kinds[0] == "restore_started"
    assert kinds[-1] == "job_ended"
    assert "tool" in kinds
    assert "probe_finished" in kinds
