from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import RichLog, Static

from ultron.cli.help import HelpBar, MouseStatic
from ultron.cli.model import InvalidTransition, JobMeta, JobSnapshot, Phase, apply, initial_snapshot
from ultron.cli.pixel import mascot_strip
from ultron.cli.render import (
    attacker_pane,
    defender_pane,
    detail_block,
    footer_line,
    header_line,
    progress_block,
    sandbox_pane,
)
from ultron.cli.shortcuts import GymFocus, gym_shortcuts

CSS_PATH = Path(__file__).with_name("sim.tcss")


class HotPane(MouseStatic):
    def __init__(self, pane: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pane = pane

    def on_click(self) -> None:
        self.focus()
        self.app.action_expand(self.pane)


class HotDetail(MouseStatic):
    def on_click(self) -> None:
        self.focus()
        self.app.action_collapse()


class GymHeader(MouseStatic):
    def on_click(self) -> None:
        self.focus()
        if getattr(self.app, "expanded", None):
            self.app.action_collapse()


class SimApp(App[None]):
    CSS_PATH = CSS_PATH
    TITLE = "ultron"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("q", "quit", "quit", show=True),
        Binding("escape", "collapse", "fold", show=True),
        Binding("a", "expand('attacker')", "attacker", show=True),
        Binding("s", "expand('sandbox')", "sandbox", show=True),
        Binding("d", "expand('defender')", "defender", show=True),
        Binding("t", "expand('tool')", "tool", show=True),
        Binding("l", "collapse", "log", show=False),
    ]

    def __init__(
        self,
        meta: JobMeta,
        events: Queue,
        *,
        sentinel: object,
        sim: bool = True,
    ) -> None:
        super().__init__()
        self.meta = meta
        self.events = events
        self.sentinel = sentinel
        self.sim = sim
        self.snapshot: JobSnapshot = initial_snapshot(meta, started_at_s=0.0)
        self.expanded: str | None = None
        self._log_index = 0
        self._done = False
        self._pixel_tick = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="frame"):
            yield GymHeader(id="header")
            yield Static(id="sprites")
            with Horizontal(id="arena"):
                yield HotPane("attacker", id="attacker")
                yield HotPane("sandbox", id="sandbox")
                yield HotPane("defender", id="defender")
            yield HotDetail(id="detail")
            yield RichLog(id="log", highlight=False, markup=False, wrap=True)
            yield MouseStatic(id="progress")
            yield HelpBar(id="help")

    def on_mount(self) -> None:
        self.query_one("#detail", Static).display = False
        self.set_interval(0.05, self._drain)
        self.set_interval(0.16, self._tick_pixels)
        self._paint()

    def action_expand(self, pane: str) -> None:
        self.expanded = pane
        self._paint()
        self.query_one("#detail").focus()

    def action_collapse(self) -> None:
        self.expanded = None
        self._paint()
        self.query_one("#attacker").focus()

    def on_descendant_focus(self, event) -> None:
        self._refresh_help()

    def _drain(self) -> None:
        drained = False
        while True:
            try:
                item = self.events.get_nowait()
            except Empty:
                break
            drained = True
            if item is self.sentinel:
                self._done = True
                break
            try:
                self.snapshot = apply(self.snapshot, item)
            except InvalidTransition as exc:
                if self.snapshot.phase in (Phase.COMPLETE, Phase.FAILED):
                    break
                self.snapshot = replace(self.snapshot, phase=Phase.FAILED, error=str(exc))
        if drained:
            self._paint()

    def _tick_pixels(self) -> None:
        self._pixel_tick += 1
        self.query_one("#sprites", Static).update(mascot_strip(self._pixel_tick))

    def _paint(self) -> None:
        snap = self.snapshot
        self.query_one("#header", Static).update(header_line(snap))
        self.query_one("#sprites", Static).update(mascot_strip(self._pixel_tick))
        self.query_one("#progress", Static).update(progress_block(snap))
        self.query_one("#help", HelpBar).set_status(footer_line(snap, sim=self.sim))
        self._refresh_help()
        arena = self.query_one("#arena", Horizontal)
        detail = self.query_one("#detail", Static)
        if self.expanded:
            arena.display = False
            detail.display = True
            detail.update(detail_block(snap, self.expanded))
        else:
            arena.display = True
            detail.display = False
            self.query_one("#attacker", HotPane).update(attacker_pane(snap))
            self.query_one("#sandbox", HotPane).update(sandbox_pane(snap))
            self.query_one("#defender", HotPane).update(defender_pane(snap))
        log = self.query_one("#log", RichLog)
        while self._log_index < len(snap.log):
            log.write(snap.log[self._log_index])
            self._log_index += 1

    def _refresh_help(self) -> None:
        self.query_one("#help", HelpBar).set_shortcuts(
            gym_shortcuts(expanded=self.expanded, focus=self._gym_focus())
        )

    def _gym_focus(self) -> GymFocus:
        widget = self.focused
        if widget is not None and widget.id == "log":
            return GymFocus.LOG
        if self.expanded:
            return GymFocus.DETAIL
        if widget is None or (widget.id or "").startswith("help"):
            return GymFocus.ARENA
        if widget.id in {"attacker", "sandbox", "defender", "header", "progress"}:
            return GymFocus.ARENA
        return GymFocus.ARENA
