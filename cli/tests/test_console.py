import asyncio
from pathlib import Path

import pytest

textual = pytest.importorskip("textual")

from textual.widgets import OptionList, Select

from ultron.cli.catalog import ActionId, TmuxPlan, all_actions, plan
from ultron.cli.console import ConsoleApp, View
from ultron.train.family import FamilyName

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
