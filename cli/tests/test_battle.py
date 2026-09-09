import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable, RichLog, Static

from ultron.cli import battle
from ultron.cli.battle import BattleApp, _scan_archives, run_battle
from ultron.cli.model import JobMeta, ModelResponseDelta, ModelResponseFinished, ModelResponseStarted
from ultron.cli.responses import ResponseArchive
from ultron.env.backend import IsolationBackend
from ultron.train.schema_v1 import Role


def _meta(generation: int = 0) -> JobMeta:
    return JobMeta(generation, "web", IsolationBackend.DOCKER, 1, 1)


def _response(
    root: Path,
    *,
    generation: int,
    role: Role,
    response_id: str,
    text: str,
    status: str = "complete",
    model: str = "fixture-model",
) -> ResponseArchive:
    archive = ResponseArchive(_meta(generation), root)
    archive.record(ModelResponseStarted(0, 0, role, response_id, model, 0.0))
    archive.record(ModelResponseDelta(0, 0, role, response_id, text, 0.1))
    archive.record(ModelResponseFinished(0, 0, role, response_id, status, 0.2, "provider failed" if status == "error" else None))
    return archive


def _stamp(archive: ResponseArchive, value: int) -> None:
    for path in (archive.path, archive.journal_path):
        os.utime(path, ns=(value, value))


def _log_text(app: BattleApp, selector: str) -> str:
    return "\n".join(line.text for line in app.query_one(selector, RichLog).lines)


async def _wait_for(pilot, predicate, *, attempts: int = 80) -> None:
    for _ in range(attempts):
        await pilot.pause(0.05)
        if predicate():
            return
    raise AssertionError("viewer did not reach the expected state")


def test_live_partial_text_is_visible_before_response_finishes(tmp_path: Path) -> None:
    archive = ResponseArchive(_meta(4), tmp_path)
    archive.record(ModelResponseStarted(0, 0, Role.ATTACKER, "live", "model-a", 0.0))
    archive.record(
        ModelResponseDelta(0, 0, Role.ATTACKER, "live", "hello [bold]literal[/bold]", 0.1)
    )
    before = archive.path.read_bytes(), archive.journal_path.read_bytes()
    app = BattleApp(tmp_path)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_for(pilot, lambda: len(app._items) == 1)
            assert app._items[0].status == "streaming"
            assert "hello [bold]literal[/bold]" in _log_text(app, "#attacker-response")
            assert "streaming 1" in str(app.query_one("#header", Static).content)

            archive.record(ModelResponseDelta(0, 0, Role.ATTACKER, "live", " world", 0.2))
            archive.record(ModelResponseFinished(0, 0, Role.ATTACKER, "live", "complete", 0.3))
            await _wait_for(
                pilot,
                lambda: app._items[0].status == "complete" and app._items[0].text.endswith(" world"),
            )
            assert "completed 1" in str(app.query_one("#header", Static).content)
            assert "hello [bold]literal[/bold] world" in _log_text(app, "#selected-response")

    asyncio.run(run())
    assert before[0] != archive.path.read_bytes()
    assert before[1] != archive.journal_path.read_bytes()


def test_selecting_older_request_stays_pinned_as_new_requests_arrive(tmp_path: Path) -> None:
    older = _response(
        tmp_path,
        generation=1,
        role=Role.ATTACKER,
        response_id="older",
        text="\n".join(f"older response line {index}" for index in range(100)),
    )
    current = _response(
        tmp_path,
        generation=2,
        role=Role.ATTACKER,
        response_id="current",
        text="current response",
    )
    _stamp(older, 1_000_000_000)
    _stamp(current, 2_000_000_000)
    app = BattleApp(tmp_path)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_for(pilot, lambda: len(app._items) == 2)
            table = app.query_one("#requests", DataTable)
            assert app._items[0].response_id == "current"
            await pilot.press("down")
            await _wait_for(
                pilot,
                lambda: app._selected_key is not None
                and app._items_by_key[app._selected_key].response_id == "older",
            )
            selected = app._selected_key
            assert app._follow_latest is False
            log = app.query_one("#selected-response", RichLog)
            app.action_response_page_up()
            await pilot.wait_for_scheduled_animations()
            old_scroll = log.scroll_y
            assert log.auto_scroll is False
            assert old_scroll < log.max_scroll_y

            newest = _response(
                tmp_path,
                generation=3,
                role=Role.ATTACKER,
                response_id="newest",
                text="newest response",
            )
            _stamp(newest, 3_000_000_000)
            await _wait_for(pilot, lambda: table.row_count == 3)

            assert app._items[0].response_id == "newest"
            assert app._selected_key == selected
            assert "older response" in _log_text(app, "#selected-response")
            assert log.scroll_y == pytest.approx(old_scroll, abs=1)
            assert log.auto_scroll is False
            assert "newest response" in _log_text(app, "#attacker-response")

    asyncio.run(run())


def test_concurrent_roles_and_error_counts_are_shown_truthfully(tmp_path: Path) -> None:
    attacker = _response(
        tmp_path,
        generation=5,
        role=Role.ATTACKER,
        response_id="attacker-r",
        text="attacker output",
        model="attacker-model",
    )
    defender = _response(
        tmp_path,
        generation=5,
        role=Role.DEFENDER,
        response_id="defender-r",
        text="defender partial",
        status="error",
        model="defender-model",
    )
    _stamp(attacker, 1_000_000_000)
    _stamp(defender, 2_000_000_000)
    app = BattleApp(tmp_path)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_for(pilot, lambda: len(app._items) == 2)
            header = str(app.query_one("#header", Static).content)
            assert "streaming 0" in header
            assert "completed 1" in header
            assert "errors 1" in header
            assert "attacker output" in _log_text(app, "#attacker-response")
            assert "defender partial" in _log_text(app, "#defender-response")
            assert "ERROR: provider failed" in _log_text(app, "#selected-response")
            selected = _log_text(app, "#selected-response").replace("\n", "")
            assert str(defender.path.resolve()) in selected

    asyncio.run(run())


def test_unchanged_archives_are_not_reloaded(tmp_path: Path) -> None:
    archive = _response(
        tmp_path,
        generation=1,
        role=Role.DEFENDER,
        response_id="cached",
        text="cached response",
    )

    with patch("ultron.cli.battle.load_response_archive", wraps=battle.load_response_archive) as load:
        first = _scan_archives(tmp_path, {})
        second = _scan_archives(tmp_path, first.cache)

    assert first.items == second.items
    assert first.items[0].path == archive.path.resolve()
    assert load.call_count == 1


def test_snapshot_waits_for_archive_load_and_exports_plain_text(tmp_path: Path) -> None:
    _response(
        tmp_path,
        generation=7,
        role=Role.ATTACKER,
        response_id="snapshot",
        text="snapshot [bold]plain[/bold]",
    )
    screenshot = tmp_path / "battle.svg"

    run_battle(tmp_path, screenshot=screenshot)

    svg = screenshot.read_text()
    assert "RESPONSE&#160;ARCHIVE" in svg
    assert "snapshot&#160;[bold]plain[/bold]" in svg


def test_compact_layout_keeps_status_visible(tmp_path: Path) -> None:
    _response(
        tmp_path,
        generation=2,
        role=Role.DEFENDER,
        response_id="compact",
        text="This response should wrap inside its role pane. " * 5,
    )
    app = BattleApp(tmp_path)

    async def run() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for(pilot, lambda: app._loaded_once)
            status = app.query_one("#status", Static)
            assert status.region.bottom <= 24
            assert "response(s)" in str(status.content)
            assert app.query_one("#defender-response", RichLog).max_scroll_x == 0

    asyncio.run(run())
