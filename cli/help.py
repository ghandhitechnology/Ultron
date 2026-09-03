from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Input, Label, Static

from ultron.cli.shortcuts import Shortcut


class MouseStatic(Static):
    can_focus = True

    def on_click(self) -> None:
        self.focus()


class FieldLabel(Label):
    def __init__(self, text: str, target: Input, **kwargs) -> None:
        super().__init__(text, classes="field-label", **kwargs)
        self._target = target

    def on_click(self) -> None:
        self._target.focus()


def chip_class(shortcut: Shortcut) -> str:
    return f"act-{shortcut.key}-{shortcut.action.replace(':', '-')}"


class ShortcutChip(Static):
    can_focus = False

    def __init__(self, shortcut: Shortcut) -> None:
        super().__init__(
            f"{shortcut.key} {shortcut.label}",
            classes=f"shortcut {chip_class(shortcut)}",
        )
        self.shortcut = shortcut

    def apply(self, shortcut: Shortcut) -> None:
        self.shortcut = shortcut
        self.set_classes(f"shortcut {chip_class(shortcut)}")
        self.update(f"{shortcut.key} {shortcut.label}")

    def on_click(self) -> None:
        name, _, arg = self.shortcut.action.partition(":")
        method = getattr(self.app, f"action_{name}")
        if arg:
            method(arg)
        else:
            method()


class HelpBar(Horizontal):
    def compose(self) -> ComposeResult:
        yield Static(id="help-status")
        yield Horizontal(id="help-keys")

    def set_status(self, text: str) -> None:
        self.query_one("#help-status", Static).update(text)

    def set_shortcuts(self, shortcuts: tuple[Shortcut, ...]) -> None:
        keys = self.query_one("#help-keys", Horizontal)
        chips = list(keys.query(ShortcutChip))
        visible = tuple(chip.shortcut for chip in chips if chip.display)
        if visible == shortcuts:
            return
        for chip, item in zip(chips, shortcuts):
            chip.apply(item)
            chip.display = True
        extra = shortcuts[len(chips) :]
        if extra:
            keys.mount(*[ShortcutChip(item) for item in extra])
        for chip in chips[len(shortcuts) :]:
            chip.display = False
            chip.set_classes("shortcut")
