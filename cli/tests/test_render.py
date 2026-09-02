from ultron.cli.demo import make_demo
from ultron.cli.model import JobMeta, apply, initial_snapshot
from ultron.cli.observe import drive_job
from ultron.cli.render import arena_block, format_eta, header_line, progress_block
from ultron.env.backend import IsolationBackend


def test_header_includes_profile_and_eta_placeholder() -> None:
    meta = JobMeta(
        generation=3,
        profile_id="hardened-server",
        isolation=IsolationBackend.DOCKER,
        episodes_planned=2,
        turns_per_side=2,
    )
    snap = initial_snapshot(meta, started_at_s=0.0)
    line = header_line(snap)
    assert "LIVE GUEST GYM" in line
    assert "hardened-server" in line
    assert "ETA --" in line
    assert format_eta(None) == "ETA --"
    assert format_eta(125) == "ETA 02:05"


def test_arena_names_both_agents_and_sandbox() -> None:
    meta = JobMeta(
        generation=0,
        profile_id="web",
        isolation=IsolationBackend.DOCKER,
        episodes_planned=1,
        turns_per_side=1,
    )
    runner, cases = make_demo(meta, delay_s=0.0, sleep=lambda _s: None)
    events: list = []
    drive_job(meta, runner, cases, emit=events.append, clock=lambda: 0.0)
    snap = initial_snapshot(meta, started_at_s=0.0)
    for event in events:
        snap = apply(snap, event)
    art = arena_block(snap)
    assert "ATTACKER" in art
    assert "DEFENDER" in art
    assert "SANDBOX" in art
    assert "docker" in art
    bars = progress_block(snap)
    assert "EPISODES" in bars
    assert "TURNS" in bars
