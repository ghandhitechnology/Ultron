import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

textual = pytest.importorskip("textual")

from textual.widgets import DataTable, OptionList, RichLog, Select

from ultron.cli.catalog import ActionId, TmuxPlan, all_actions, plan
from ultron.cli.console import MAX_LOG_LINES, ConsoleApp, View
from ultron.cli.jobs import LogChunk, SessionInfo, SessionState
from ultron.train.family import FamilyName

ROOT = Path(__file__).resolve().parents[2]


def _session(name: str, state: SessionState, *, exit_code: int | None = None) -> SessionInfo:
    return SessionInfo(
        name=name,
        state=state,
        pid=4321 if state is SessionState.RUNNING else None,
        command="bash run",
        log_path=ROOT / "data" / "logs" / f"{name}.log",
        exit_code=exit_code,
    )


def test_console_lists_every_catalog_action() -> None:
    app = ConsoleApp(root=ROOT)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            listing = app.query_one("#actions", OptionList)
            ids = {option.id for option in listing.options if option.id is not None}
            expected = {spec.id.value for spec in all_actions(root=ROOT)}
            assert expected <= ids
            assert app.view is View.CATALOG
            sprites = app.query_one("#sprites")
            assert sprites.display is True
            body = str(sprites.content)
            assert "exploiter" in body
            assert "vision" in body

    asyncio.run(run())


def test_console_pixel_idle_advances() -> None:
    app = ConsoleApp(root=ROOT)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            start = app._pixel_tick
            await pilot.pause(0.4)
            assert app._pixel_tick > start
            body = str(app.query_one("#sprites").content)
            assert "exploiter" in body
            assert "vision" in body

    asyncio.run(run())


def test_console_switches_to_jobs_and_results() -> None:
    app = ConsoleApp(root=ROOT)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("j")
            await pilot.pause()
            assert app.view is View.JOBS
            await pilot.press("r")
            await pilot.pause()
            assert app.view is View.RESULTS
            await pilot.press("t")
            await pilot.pause()
            assert app.view is View.CATALOG
            assert app.selected is ActionId.TESTS

    asyncio.run(run())


def test_console_family_selector_pins_launches() -> None:
    app = ConsoleApp(root=ROOT, family="gemma")

    async def run() -> None:
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            listing = app.query_one("#family", Select)
            assert listing.value == "gemma"
            assert app.family is FamilyName.GEMMA
            app._select_action(ActionId.GENERATION)
            built = plan(
                ActionId.GENERATION,
                app._field_values(),
                root=ROOT,
                family=app.family.value,
            )
            assert isinstance(built, TmuxPlan)
            assert ("ULTRON_MODEL_FAMILY", "gemma") in built.env
            listing.value = "qwen-8b"
            await pilot.pause()
            assert app.family is FamilyName.QWEN_8B
            assert app.pack.base_model == "Qwen/Qwen3-8B"
            listing.value = "gemma-abliterated"
            await pilot.pause()
            assert app.family is FamilyName.GEMMA_ABLITERATED
            assert app.pack.base_model == "huihui-ai/Huihui-gemma-4-12B-it-abliterated"

    asyncio.run(run())


def test_console_runs_archive_list() -> None:
    app = ConsoleApp(root=ROOT)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._select_action(ActionId.ARCHIVE_LIST)
            app._run_selected()
            for _ in range(40):
                await pilot.pause(0.05)
                if app.view is View.RUN and app._done:
                    break
            assert app.view is View.RUN
            assert app._done

    asyncio.run(run())


def test_jobs_show_exit_outcome_and_open_logs_back_to_list() -> None:
    sessions = (
        _session("running-job", SessionState.RUNNING),
        _session("failed-job", SessionState.DEAD, exit_code=23),
    )
    app = ConsoleApp(root=ROOT)

    async def run() -> None:
        with (
            patch("ultron.cli.console.list_sessions", return_value=sessions),
            patch("ultron.cli.console.session_status", side_effect=lambda name, **_: sessions[0] if name == "running-job" else sessions[1]),
            patch("ultron.cli.console.read_log_chunk", return_value=LogChunk("line\n", 5)),
        ):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("j")
                await pilot.pause()
                table = app.query_one("#job-table", DataTable)
                assert table.get_row("failed-job")[1:3] == ["finished", "failed (23)"]

                table.move_cursor(row=table.get_row_index("failed-job"))
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(10):
                    await pilot.pause(0.05)
                    if "failed (23)" in str(app.query_one("#run-header").content):
                        break
                assert app.view is View.RUN
                assert "failed (23)" in str(app.query_one("#run-header").content)

                await pilot.press("escape")
                await pilot.pause()
                assert app.view is View.JOBS
                assert app._selected_session() == "failed-job"

    asyncio.run(run())


def test_job_refresh_preserves_selection_and_log_scroll() -> None:
    sessions = tuple(_session(f"job-{index:02d}", SessionState.RUNNING) for index in range(20))
    app = ConsoleApp(root=ROOT)

    async def run() -> None:
        with (
            patch("ultron.cli.console.list_sessions", return_value=sessions),
            patch("ultron.cli.console.session_status", return_value=sessions[15]),
            patch("ultron.cli.console.read_log_chunk", return_value=LogChunk("", 0)),
        ):
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.press("j")
                await pilot.pause()
                table = app.query_one("#job-table", DataTable)
                table.move_cursor(row=15)
                await pilot.pause()
                table.scroll_to(y=5, animate=False, immediate=True, force=True)

                log = app.query_one("#job-log", RichLog)
                for index in range(100):
                    log.write(f"line {index}")
                await pilot.pause()
                log.scroll_to(y=10, animate=False, immediate=True, force=True)
                app._poll_after = float("inf")
                line_count = len(log.lines)

                app._refresh_jobs()
                await pilot.pause()

                assert app._selected_session() == "job-15"
                assert table.scroll_y == 5
                assert len(log.lines) == line_count
                assert log.scroll_y == 10

    asyncio.run(run())


def test_console_log_views_have_a_hard_line_limit() -> None:
    app = ConsoleApp(root=ROOT)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.query_one("#job-log", RichLog).max_lines == MAX_LOG_LINES
            assert app.query_one("#run-log", RichLog).max_lines == MAX_LOG_LINES

    asyncio.run(run())
