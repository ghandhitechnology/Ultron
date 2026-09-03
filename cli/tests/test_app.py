import asyncio
from queue import Queue

import pytest

textual = pytest.importorskip("textual")

from ultron.cli.app import SimApp
from ultron.cli.demo import make_demo
from ultron.cli.help import HelpBar, ShortcutChip
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
    assert "exploiter" in svg.lower()
    assert "vision" in svg.lower()
    expanded = tmp_path / "sim_sandbox.svg"
    assert expanded.is_file()
    assert "SANDBOX" in expanded.read_text() or "guest" in expanded.read_text().lower()
    svg = shot.read_text()
    assert "help" in svg.lower() or "attacker" in svg.lower()


def test_gym_help_bar_and_mouse_expand_fold() -> None:
    meta = JobMeta(
        generation=1,
        profile_id="web",
        isolation=IsolationBackend.DOCKER,
        episodes_planned=1,
        turns_per_side=1,
        version="0.1.0",
    )
    app = SimApp(meta, Queue(), sentinel=object(), sim=True)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app.query_one("#help", HelpBar)
            keys = [chip.shortcut.key for chip in bar.query(ShortcutChip)]
            assert keys == ["a", "s", "d", "t", "q"]
            await pilot.click("#sandbox")
            await pilot.pause()
            assert app.expanded == "sandbox"
            folded = [chip.shortcut.action for chip in bar.query(ShortcutChip)]
            assert folded[0] == "collapse"
            await pilot.click("#detail")
            await pilot.pause()
            assert app.expanded is None
            await pilot.click(".act-t-expand-tool")
            await pilot.pause()
            assert app.expanded == "tool"
            await pilot.click("#header")
            await pilot.pause()
            assert app.expanded is None
            await pilot.click("#log")
            await pilot.pause()
            log_keys = [chip.shortcut.action for chip in bar.query(ShortcutChip)]
            assert "expand:tool" in log_keys

    asyncio.run(run())
