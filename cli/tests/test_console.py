import asyncio
from pathlib import Path

import pytest

textual = pytest.importorskip("textual")

from textual.widgets import OptionList

from ultron.cli.catalog import ActionId, all_actions
from ultron.cli.console import ConsoleApp, View

ROOT = Path(__file__).resolve().parents[2]


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
