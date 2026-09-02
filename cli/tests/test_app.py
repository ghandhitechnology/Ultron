import pytest

textual = pytest.importorskip("textual")

from ultron.cli.demo import make_demo
from ultron.cli.model import JobMeta, Phase
from ultron.cli.tui import run_live_job
from ultron.env.backend import IsolationBackend


def test_demo_job_paints_and_expands_sandbox(tmp_path) -> None:
    meta = JobMeta(
        generation=1,
        profile_id="web",
        isolation=IsolationBackend.DOCKER,
        episodes_planned=1,
        turns_per_side=1,
        version="0.1.0",
    )
    runner, cases = make_demo(meta, delay_s=0.0, sleep=lambda _s: None)
    shot = tmp_path / "sim.svg"
    snap = run_live_job(meta, runner, cases, screenshot=shot)
    assert snap.phase is Phase.COMPLETE
    svg = shot.read_text()
    assert "LIVE GUEST GYM" in svg or "ultron" in svg.lower()
    expanded = tmp_path / "sim_sandbox.svg"
    assert expanded.is_file()
    assert "SANDBOX" in expanded.read_text() or "guest" in expanded.read_text().lower()
