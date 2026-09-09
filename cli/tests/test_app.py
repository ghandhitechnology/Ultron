import pytest

textual = pytest.importorskip("textual")

from ultron.cli.demo import make_demo
from ultron.cli.model import JobMeta, Phase
from ultron.cli.tui import run_live_job
from ultron.env.backend import IsolationBackend


@pytest.mark.parametrize("size", [(100, 36), (80, 24)])
def test_model_text_is_visible_before_response_finishes(size) -> None:
    import asyncio
    from queue import Queue
    from textual.widgets import RichLog, Static
    from ultron.cli.app import SimApp
    from ultron.cli.model import TurnStarted, ModelResponseStarted, ModelResponseDelta, ModelResponseFinished
    from ultron.train.schema_v1 import Role

    async def check() -> None:
        meta = JobMeta(0, "web", IsolationBackend.DOCKER, 1, 1)
        events = Queue()
        app = SimApp(meta, events, sentinel=object())
        async with app.run_test(size=size) as pilot:
            events.put(TurnStarted(0, 0, Role.ATTACKER, 0.0))
            events.put(ModelResponseStarted(0, 0, Role.ATTACKER, "r1", "test-model", 0.1))
            events.put(ModelResponseDelta(0, 0, Role.ATTACKER, "r1", "Hello [bold]literal[/bold]", 0.2))
            await pilot.pause(0.1)
            assert app.snapshot.attacker_response.status == "streaming"
            panel = app.query_one("#response", Static)
            assert panel.display
            assert "Hello [bold]literal[/bold]" in str(panel.render())
            assert app.query_one("#status", Static).region.bottom <= size[1]
            app.action_transcript_page_up()
            events.put(ModelResponseDelta(0, 0, Role.ATTACKER, "r1", " world", 0.3))
            events.put(ModelResponseFinished(0, 0, Role.ATTACKER, "r1", "complete", 0.4))
            await pilot.pause(0.1)
            assert app.snapshot.attacker_response.text.endswith(" world")
            log = app.query_one("#log", RichLog)
            assert log.auto_scroll is False
            assert "Hello [bold]literal[/bold] world" in "\n".join(line.text for line in log.lines)
            app.action_transcript_live()
            assert log.auto_scroll is True

    asyncio.run(check())


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
    assert "INTERACTION" in svg
    assert "COMPLETE" in svg
    expanded = tmp_path / "sim_sandbox.svg"
    assert expanded.is_file()
    assert "SANDBOX" in expanded.read_text() or "guest" in expanded.read_text().lower()
