import asyncio
from pathlib import Path

import pytest

textual = pytest.importorskip("textual")

from textual.widgets import Input, OptionList, Select

from ultron.cli.catalog import ActionId, TmuxPlan, all_actions, plan
from ultron.cli.console import ConsoleApp, View
from ultron.cli.help import HelpBar, ShortcutChip
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

    asyncio.run(run())


def test_console_help_bar_follows_view_and_form_focus() -> None:
    app = ConsoleApp(root=ROOT)

    async def run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app.query_one("#help", HelpBar)
            catalog = [chip.shortcut.key for chip in bar.query(ShortcutChip)]
            assert catalog == ["enter", "m", "j", "r", "t", "q"]
            await pilot.click(".act-j-show_jobs")
            await pilot.pause()
            assert app.view is View.JOBS
            jobs = [chip.shortcut.action for chip in bar.query(ShortcutChip)]
            assert "back" in jobs
            assert "refresh" in jobs
            await pilot.press("escape")
            await pilot.pause()
            assert app.view is View.CATALOG
            app._select_action(ActionId.GENERATION)
            first = next(iter(app._inputs.values()))
            first.focus()
            await pilot.pause()
            form = [chip.shortcut.key for chip in bar.query(ShortcutChip)]
            assert form[0] in {"tab", "enter"}
            assert "j" not in form
            assert "click" in form
            first.value = "3"
            await pilot.pause()
            filled = [chip.shortcut.key for chip in bar.query(ShortcutChip)]
            assert filled[0] == "enter"
            assert isinstance(app.focused, Input)

    asyncio.run(run())


def test_console_mouse_focuses_family_and_actions() -> None:
    app = ConsoleApp(root=ROOT)

    async def run() -> None:
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await pilot.click("#family")
            await pilot.pause()
            keys = [chip.shortcut.key for chip in app.query(ShortcutChip)]
            assert keys[0] == "enter"
            assert "a" in keys
            await pilot.click("#header-title")
            await pilot.pause()
            assert app.view is View.CATALOG
            listing = app.query_one("#actions", OptionList)
            await pilot.click("#actions")
            await pilot.pause()
            assert listing.has_focus or app.focused is listing

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
