from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from time import monotonic

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import RichLog, Static

from ultron.cli.model import InvalidTransition, JobMeta, JobSnapshot, Phase, apply, initial_snapshot
from ultron.cli.render import (
    attacker_pane,
    defender_pane,
    detail_block,
    footer_line,
    header_line,
    progress_block,
    response_block,
    sandbox_pane,
    transcript_entry,
    transcript_header,
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
        Binding("pageup", "transcript_page_up", "older", show=True),
        Binding("pagedown", "transcript_page_down", "newer", show=True),
        Binding("end", "transcript_live", "live", show=True),
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
        self._done = False

    def compose(self) -> ComposeResult:
        with Vertical(id="frame"):
            yield Static(id="header", markup=False)
            with Horizontal(id="arena"):
                yield HotPane("attacker", id="attacker", markup=False)
                yield HotPane("sandbox", id="sandbox", markup=False)
                yield HotPane("defender", id="defender", markup=False)
            yield Static(id="detail", markup=False)
            yield Static(id="response", markup=False)
            yield Static(id="transcript-header", markup=False)
            yield RichLog(id="log", highlight=False, markup=True, wrap=True, max_lines=5000, min_width=1)
            yield Static(id="progress", markup=False)
            yield Static(id="status", markup=False)

    def on_mount(self) -> None:
        self.query_one("#detail", Static).display = False
        self.query_one("#response", Static).display = False
        self.set_interval(0.05, self._drain)
        self._paint()

    def on_resize(self) -> None:
        if self.query("#header"):
            self._paint()

    def action_expand(self, pane: str) -> None:
        self.expanded = pane
        self._paint()

    def action_collapse(self) -> None:
        self.expanded = None
        self._paint()

    def action_transcript_page_up(self) -> None:
        log = self.query_one("#log", RichLog)
        log.auto_scroll = False
        log.scroll_page_up()

    def action_transcript_page_down(self) -> None:
        self.query_one("#log", RichLog).scroll_page_down()

    def action_transcript_live(self) -> None:
        log = self.query_one("#log", RichLog)
        log.auto_scroll = True
        log.scroll_end(animate=False)

    def _drain(self) -> None:
        drained = False
        log = self.query_one("#log", RichLog)
        # Yield to input and painting even when the producer stays ahead of us.
        deadline = monotonic() + 0.015
        for _ in range(256):
            if monotonic() >= deadline:
                break
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
            entry = transcript_entry(item, self.snapshot)
            if entry:
                log.write(entry)
        if drained:
            self._paint()

    def _paint(self) -> None:
        snap = self.snapshot
        response = self.query_one("#response", Static)
        content = response_block(snap, width=max(20, self.size.width - 8))
        response.display = bool(content) and self.expanded is None
        response.update(content)
        self.query_one("#header", Static).update(header_line(snap))
        self.query_one("#transcript-header", Static).update(transcript_header(snap))
        self.query_one("#progress", Static).update(progress_block(snap))
        self.query_one("#status", Static).update(footer_line(snap, sim=self.sim))
        arena = self.query_one("#arena", Horizontal)
        detail = self.query_one("#detail", Static)
        detail.styles.height = min(9, max(4, self.size.height // 4))
        if self.expanded:
            arena.display = False
            detail.display = True
            detail.update(detail_block(snap, self.expanded))
        else:
            arena.display = self.size.height >= 32
            detail.display = False
            self.query_one("#attacker", HotPane).update(attacker_pane(snap))
            self.query_one("#sandbox", HotPane).update(sandbox_pane(snap))
            self.query_one("#defender", HotPane).update(defender_pane(snap))
