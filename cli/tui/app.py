from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Thread

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Static

from ultron.cli.ascii import render
from ultron.cli.board import Board
from ultron.cli.fold import BoardFold

ViewFocus = str


class TheaterApp(App[None]):
    CSS = """
    Screen {
        background: #0b1020;
        color: #c9d1e8;
    }
    #canvas {
        height: 1fr;
        border: solid #3d2a5c;
        padding: 0 1;
    }
    """
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("a", "focus_attacker", "Attacker"),
        Binding("d", "focus_defender", "Defender"),
        Binding("g", "focus_guest", "Guest"),
        Binding("p", "focus_process", "Process"),
        Binding("escape", "focus_none", "Collapse"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        boards: Iterator[Board],
        *,
        screenshot: str | None = None,
    ) -> None:
        super().__init__()
        self._boards = boards
        self._screenshot = screenshot
        self._focus: ViewFocus = "none"
        self._last: Board | None = None
        try:
            self._use_alternate_screen = False
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield Vertical(Static(id="canvas"), Footer())

    def on_mount(self) -> None:
        self.set_interval(0.15, self._tick)
        if self._screenshot:
            self.call_after_refresh(self._shot)

    def _tick(self) -> None:
        try:
            board = next(self._boards)
        except StopIteration:
            if self._last is None:
                return
            board = self._last
        self._last = board
        canvas = self.query_one("#canvas", Static)
        size = self.size
        canvas.update(render(board, cols=max(72, size.width), rows=max(20, size.height - 2)))

    def action_focus_attacker(self) -> None:
        self._focus = "attacker"

    def action_focus_defender(self) -> None:
        self._focus = "defender"

    def action_focus_guest(self) -> None:
        self._focus = "guest"

    def action_focus_process(self) -> None:
        self._focus = "process"

    def action_focus_none(self) -> None:
        self._focus = "none"

    def _shot(self) -> None:
        if self._screenshot:
            self.save_screenshot(self._screenshot)
            self.exit()

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        x = event.x
        width = max(1, self.size.width)
        if event.y <= 2:
            return
        if event.y >= self.size.height - 8:
            self._focus = "process"
            return
        third = width // 3
        if x < third:
            self._focus = "attacker"
        elif x > 2 * third:
            self._focus = "defender"
        else:
            self._focus = "guest"


def run_theater(fold: BoardFold, worker: Callable[[], None] | None = None) -> Board:
    if worker is not None:
        thread = Thread(target=worker, daemon=True)
        thread.start()
    app = TheaterApp(fold.subscribe())
    app.run()
    return fold.snapshot()


def screenshot_board(board: Board, path: Path) -> None:
    class Shot(App[None]):
        CSS = TheaterApp.CSS

        def compose(self) -> ComposeResult:
            yield Static(render(board, cols=100, rows=28))

        def on_mount(self) -> None:
            self.call_after_refresh(self._save)

        def _save(self) -> None:
            self.save_screenshot(str(path))
            self.exit()

    Shot().run()
