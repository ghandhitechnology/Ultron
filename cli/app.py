from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import RichLog, Static

from ultron.cli.model import JobMeta, JobSnapshot, apply, initial_snapshot
from ultron.cli.render import (
    attacker_pane,
    defender_pane,
    detail_block,
    footer_line,
    header_line,
    progress_block,
    sandbox_pane,
)

CSS_PATH = Path(__file__).with_name("sim.tcss")


class HotPane(Static):
    def __init__(self, pane: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pane = pane

    def on_click(self) -> None:
        self.app.action_expand(self.pane)


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

    def compose(self) -> ComposeResult:
        with Vertical(id="frame"):
            yield Static(id="header")
            with Horizontal(id="arena"):
                yield HotPane("attacker", id="attacker")
                yield HotPane("sandbox", id="sandbox")
                yield HotPane("defender", id="defender")
            yield Static(id="detail")
            yield RichLog(id="log", highlight=False, markup=False, wrap=True)
            yield Static(id="progress")
            yield Static(id="status")

    def on_mount(self) -> None:
        self.query_one("#detail", Static).display = False
        self.set_interval(0.05, self._drain)
        self._paint()

    def action_expand(self, pane: str) -> None:
        self.expanded = pane
        self._paint()

    def action_collapse(self) -> None:
        self.expanded = None
        self._paint()

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
            self.snapshot = apply(self.snapshot, item)
        if drained:
            self._paint()

    def _paint(self) -> None:
        snap = self.snapshot
        self.query_one("#header", Static).update(header_line(snap))
        self.query_one("#progress", Static).update(progress_block(snap))
        self.query_one("#status", Static).update(footer_line(snap, sim=self.sim))
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
