"""Read-only live and replay viewer for recorded model responses."""
from __future__ import annotations

import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, RichLog, Static

from ultron.cli.responses import load_response_archive

MAX_ARCHIVES = 200
MAX_REQUESTS = 500
MAX_RESPONSE_CHARS = 64 * 1024
MAX_PANEL_CHARS = 12 * 1024
MAX_LOG_LINES = 2_000
SCAN_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class ResponseItem:
    key: str
    path: Path
    run_id: str
    created_at: str
    generation: str
    role: str
    model: str
    response_id: str
    status: str
    text: str
    error: str | None
    order: tuple[int, int]


@dataclass(frozen=True)
class _CachedArchive:
    signature: tuple[int, int, int, int]
    items: tuple[ResponseItem, ...]
    total_items: int


@dataclass(frozen=True)
class _ScanResult:
    cache: dict[Path, _CachedArchive]
    items: tuple[ResponseItem, ...]
    discovered: int
    scan_error: str | None = None


class BattleApp(App[None]):
    TITLE = "ultron responses"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen {
        background: #1e1e2e;
        color: #cdd6f4;
    }

    #frame {
        height: 100%;
        border: heavy #45475a;
        background: #1e1e2e;
    }

    #header {
        height: 3;
        padding: 0 1;
        content-align: left middle;
        background: #181825;
        border-bottom: solid #45475a;
        text-style: bold;
    }

    #latest {
        height: 9;
        background: #181825;
        border-bottom: solid #45475a;
    }

    .role-pane {
        width: 1fr;
        height: 100%;
    }

    #attacker-pane {
        border-right: solid #45475a;
    }

    .role-title {
        height: 2;
        padding: 0 1;
        content-align: left middle;
        background: #24243b;
        text-style: bold;
    }

    #attacker-title {
        color: #f38ba8;
    }

    #defender-title {
        color: #a6e3a1;
    }

    .role-log {
        height: 1fr;
        padding: 0 1;
        background: #11111b;
        color: #cdd6f4;
    }

    #requests {
        height: 10;
        background: #181825;
    }

    #selected-title {
        height: 2;
        padding: 0 1;
        content-align: left middle;
        background: #302f55;
        color: #e5e7f2;
        text-style: bold;
    }

    #selected-response {
        height: 1fr;
        min-height: 4;
        padding: 0 1;
        background: #11111b;
        color: #e5e7f2;
        scrollbar-background: #181825;
        scrollbar-color: #585b70;
        scrollbar-color-hover: #6c5ce7;
        scrollbar-color-active: #89b4fa;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: #313244;
        color: #cdd6f4;
        text-style: bold;
    }

    Screen.compact #latest {
        height: 6;
    }

    Screen.compact #requests {
        height: 6;
    }

    Screen.compact .role-title,
    Screen.compact #selected-title {
        height: 1;
    }
    """
    BINDINGS = [
        Binding("q", "quit", "quit", show=True),
        Binding("pageup", "response_page_up", "older text", show=False),
        Binding("pagedown", "response_page_down", "newer text", show=False),
        Binding("end", "follow_latest", "latest", show=True),
        Binding("g", "refresh", "refresh", show=False),
    ]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.source = path.expanduser().resolve(strict=False)
        self._results: queue.Queue[_ScanResult] = queue.Queue()
        self._cache: dict[Path, _CachedArchive] = {}
        self._items: tuple[ResponseItem, ...] = ()
        self._items_by_key: dict[str, ResponseItem] = {}
        self._selected_key: str | None = None
        self._rendered_selected: ResponseItem | None = None
        self._follow_latest = True
        self._scan_inflight = False
        self._next_scan_at = 0.0
        self._discovered = 0
        self._loaded_once = False

    def compose(self) -> ComposeResult:
        with Vertical(id="frame"):
            yield Static(id="header", markup=False)
            with Horizontal(id="latest"):
                with Vertical(id="attacker-pane", classes="role-pane"):
                    yield Static("ATTACKER  waiting", id="attacker-title", classes="role-title", markup=False)
                    yield RichLog(
                        id="attacker-response",
                        classes="role-log",
                        highlight=False,
                        markup=False,
                        wrap=True,
                        min_width=1,
                        max_lines=300,
                    )
                with Vertical(id="defender-pane", classes="role-pane"):
                    yield Static("DEFENDER  waiting", id="defender-title", classes="role-title", markup=False)
                    yield RichLog(
                        id="defender-response",
                        classes="role-log",
                        highlight=False,
                        markup=False,
                        wrap=True,
                        min_width=1,
                        max_lines=300,
                    )
            yield DataTable(id="requests", cursor_type="row", zebra_stripes=True)
            yield Static("SELECTED RESPONSE", id="selected-title", markup=False)
            yield RichLog(
                id="selected-response",
                highlight=False,
                markup=False,
                wrap=True,
                max_lines=MAX_LOG_LINES,
                min_width=1,
            )
            yield Static(id="status", markup=False)

    def on_mount(self) -> None:
        table = self.query_one("#requests", DataTable)
        for label, key in (
            ("generation", "generation"),
            ("role", "role"),
            ("state", "state"),
            ("model", "model"),
            ("request", "request"),
        ):
            table.add_column(label, key=key)
        table.focus()
        self._paint_header()
        self._paint_status("waiting for responses")
        self.set_interval(0.1, self._tick)
        self._request_scan(force=True)

    def on_resize(self) -> None:
        self.screen.set_class(self.size.height < 31, "compact")

    def action_quit(self) -> None:
        self.exit()

    def action_response_page_up(self) -> None:
        log = self.query_one("#selected-response", RichLog)
        log.auto_scroll = False
        log.scroll_page_up()

    def action_response_page_down(self) -> None:
        self.query_one("#selected-response", RichLog).scroll_page_down()

    def action_follow_latest(self) -> None:
        self._follow_latest = True
        if self._items:
            self._select(self._items[0].key, move_cursor=True)

    def action_refresh(self) -> None:
        self._request_scan(force=True)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "requests" or event.row_key.value is None:
            return
        key = event.row_key.value
        if key != self._current_table_key(event.data_table):
            return
        if key != self._selected_key:
            self._follow_latest = False
            self._select(key)

    def _tick(self) -> None:
        self._drain_results()
        self._request_scan()

    def _request_scan(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if self._scan_inflight or (not force and now < self._next_scan_at):
            return
        source = self.source
        cache = dict(self._cache)
        current_items = self._items
        discovered = self._discovered
        self._scan_inflight = True
        self._next_scan_at = now + SCAN_INTERVAL_SECONDS

        def worker() -> None:
            try:
                result = _scan_archives(source, cache)
            except Exception as exc:
                result = _ScanResult(cache, current_items, discovered, str(exc))
            self._results.put(result)

        threading.Thread(target=worker, name="ultron-response-viewer", daemon=True).start()

    def _drain_results(self) -> None:
        latest: _ScanResult | None = None
        while True:
            try:
                latest = self._results.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return
        self._scan_inflight = False
        changed = latest.items != self._items or latest.discovered != self._discovered
        self._cache = latest.cache
        self._discovered = latest.discovered
        if changed:
            self._items = latest.items
            self._items_by_key = {item.key: item for item in latest.items}
            self._render_items()
        elif not self._loaded_once and latest.items:
            self._loaded_once = True
        if latest.scan_error:
            self._paint_status(f"scan error: {latest.scan_error}")
        elif not latest.items:
            self._paint_status("waiting for responses")

    def _render_items(self) -> None:
        table = self.query_one("#requests", DataTable)
        old_scroll = table.scroll_y
        table.clear()
        for item in self._items:
            table.add_row(
                item.generation,
                item.role,
                _display_status(item.status),
                item.model,
                _short_id(item.response_id),
                key=item.key,
            )

        if not self._items:
            self._selected_key = None
            self._clear_response_panels()
            self._paint_header()
            self._paint_status("waiting for responses")
            return

        if self._follow_latest or self._selected_key not in self._items_by_key:
            target = self._items[0].key
        else:
            target = self._selected_key
        self._select(target, move_cursor=True)
        target_scroll = 0 if self._follow_latest else old_scroll
        table.call_after_refresh(table.scroll_to, y=target_scroll, animate=False)
        self._paint_latest_roles()
        self._paint_header()
        self._paint_status()
        self._loaded_once = True

    def _select(self, key: str, *, move_cursor: bool = False) -> None:
        item = self._items_by_key.get(key)
        if item is None:
            return
        previous = self._rendered_selected
        same_response = previous is not None and previous.key == key
        self._selected_key = key
        if move_cursor:
            table = self.query_one("#requests", DataTable)
            table.move_cursor(row=table.get_row_index(key), scroll=True)
        title = (
            f"SELECTED  gen {item.generation} · {item.role} · {_display_status(item.status)} · "
            f"{item.model} · {_short_id(item.response_id)}"
        )
        self.query_one("#selected-title", Static).update(title)
        if previous == item:
            self._paint_status()
            return
        lines = [f"Path: {item.path}", ""]
        if item.error:
            lines.extend((f"ERROR: {item.error}", ""))
        lines.append(item.text or "(no response text)")
        log = self.query_one("#selected-response", RichLog)
        old_scroll = log.scroll_y
        was_at_end = log.is_vertical_scroll_end
        log.clear()
        log.write("\n".join(lines), scroll_end=not same_response or was_at_end)
        if same_response and not was_at_end:
            log.call_after_refresh(log.scroll_to, y=old_scroll, animate=False)
        elif not same_response:
            log.auto_scroll = True
        self._rendered_selected = item
        self._paint_status()

    def _paint_latest_roles(self) -> None:
        for role in ("attacker", "defender"):
            item = next((candidate for candidate in self._items if candidate.role == role), None)
            title = self.query_one(f"#{role}-title", Static)
            log = self.query_one(f"#{role}-response", RichLog)
            log.clear()
            if item is None:
                title.update(f"{role.upper()}  waiting")
                log.write("(no response yet)")
                continue
            title.update(
                f"{role.upper()}  gen {item.generation} · {_display_status(item.status)} · {item.model}"
            )
            log.write(_bounded_text(item.text, MAX_PANEL_CHARS) or "(no response text)", scroll_end=True)

    def _clear_response_panels(self) -> None:
        for role in ("attacker", "defender"):
            self.query_one(f"#{role}-title", Static).update(f"{role.upper()}  waiting")
            log = self.query_one(f"#{role}-response", RichLog)
            log.clear()
            log.write("(no response yet)")
        self.query_one("#selected-title", Static).update("SELECTED RESPONSE")
        self.query_one("#selected-response", RichLog).clear()
        self._rendered_selected = None

    def _paint_header(self) -> None:
        counts = Counter(item.status for item in self._items)
        text = (
            f"RESPONSE ARCHIVE  streaming {counts['streaming']} · completed {counts['complete']} · "
            f"errors {counts['error']} · cancelled {counts['cancelled']}\n{self.source}"
        )
        self.query_one("#header", Static).update(text)

    def _paint_status(self, message: str | None = None) -> None:
        if message is None:
            follow = "following latest" if self._follow_latest else "selection pinned"
            message = (
                f"{len(self._items)} response(s) from {self._discovered} request archive(s) · {follow} · "
                "↑/↓ select · end latest · pgup/pgdn text · q quit"
            )
        self.query_one("#status", Static).update(f"  {message}")

    @staticmethod
    def _current_table_key(table: DataTable) -> str | None:
        if table.row_count == 0:
            return None
        return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value


def run_battle(path: Path, screenshot: Path | None = None) -> None:
    """Open a read-only archive viewer, or wait for data and export one SVG snapshot."""
    app = BattleApp(path)
    if screenshot is None:
        app.run()
        return
    _export_screenshot(app, screenshot)


def _export_screenshot(app: BattleApp, path: Path) -> None:
    import asyncio

    async def capture() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(500):
                await pilot.pause(0.05)
                if app._loaded_once:
                    await pilot.pause(0.2)
                    break
            else:
                raise TimeoutError(f"no response archives appeared under {app.source}")
            destination = path if path.suffix else path.with_suffix(".svg")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(app.export_screenshot(), encoding="utf-8")

    asyncio.run(capture())


def _scan_archives(source: Path, cache: dict[Path, _CachedArchive]) -> _ScanResult:
    candidates = _response_paths(source)
    stamped: list[tuple[int, Path, tuple[int, int, int, int]]] = []
    for path in candidates:
        try:
            signature = _archive_signature(path)
        except OSError:
            continue
        stamped.append((max(signature[0], signature[2]), path, signature))
    stamped.sort(key=lambda entry: (entry[0], entry[1].as_posix()), reverse=True)
    recent = stamped[:MAX_ARCHIVES]

    next_cache: dict[Path, _CachedArchive] = {}
    remaining = MAX_REQUESTS
    for modified_ns, path, signature in recent:
        if remaining == 0:
            break
        cached = cache.get(path)
        cache_has_enough = cached is not None and (
            cached.total_items == len(cached.items) or remaining <= len(cached.items)
        )
        if cached is not None and cached.signature == signature and cache_has_enough:
            kept = cached.items[-remaining:]
            next_cache[path] = _CachedArchive(signature, kept, cached.total_items)
            remaining -= len(kept)
            continue
        try:
            document = load_response_archive(path)
            loaded = _archive_items(path, document, modified_ns)
            kept = loaded[-remaining:]
            next_cache[path] = _CachedArchive(signature, kept, len(loaded))
            remaining -= len(kept)
        except Exception as exc:
            item = _archive_error_item(path, modified_ns, exc)
            next_cache[path] = _CachedArchive((-1, -1, -1, -1), (item,), 1)
            remaining -= 1

    items = sorted(
        (item for archive in next_cache.values() for item in archive.items),
        key=lambda item: item.order,
        reverse=True,
    )[:MAX_REQUESTS]
    return _ScanResult(next_cache, tuple(items), len(candidates))


def _response_paths(source: Path) -> tuple[Path, ...]:
    if source.name == "responses.json" or source.is_file():
        return (source,) if source.exists() else ()
    if not source.is_dir():
        return ()
    return tuple(source.rglob("responses.json"))


def _archive_signature(path: Path) -> tuple[int, int, int, int]:
    snapshot = path.stat()
    journal_path = path.with_name("response-events.jsonl")
    try:
        journal = journal_path.stat()
    except FileNotFoundError:
        return (snapshot.st_mtime_ns, snapshot.st_size, 0, 0)
    return (snapshot.st_mtime_ns, snapshot.st_size, journal.st_mtime_ns, journal.st_size)


def _archive_items(path: Path, document: Mapping[str, Any], modified_ns: int) -> tuple[ResponseItem, ...]:
    raw_responses = document.get("responses")
    if not isinstance(raw_responses, list):
        raise ValueError("responses.json has no response list")
    meta = document.get("meta") if isinstance(document.get("meta"), Mapping) else {}
    capture = document.get("capture") if isinstance(document.get("capture"), Mapping) else {}
    generation = _one_line(meta.get("generation", capture.get("generation", "?")))
    default_role = _one_line(capture.get("role", meta.get("role", "unknown")))
    run_id = _one_line(document.get("run_id", path.parent.name))
    created_at = _one_line(document.get("created_at", ""))
    archive_error = document.get("error")
    items: list[ResponseItem] = []
    for index, raw in enumerate(raw_responses):
        if not isinstance(raw, Mapping):
            continue
        response_id = _one_line(raw.get("response_id", f"response-{index}"))
        role = _one_line(raw.get("role", default_role)).lower() or "unknown"
        status = _normal_status(raw.get("status", document.get("status", "streaming")))
        error = raw.get("error") or (archive_error if status in {"error", "cancelled"} else None)
        items.append(
            ResponseItem(
                key=f"{path.as_posix()}::{response_id}::{index}",
                path=path,
                run_id=run_id,
                created_at=created_at,
                generation=generation or "?",
                role=role,
                model=_one_line(raw.get("model", "unknown")) or "unknown",
                response_id=response_id,
                status=status,
                text=_bounded_text(_plain(raw.get("text", "")), MAX_RESPONSE_CHARS),
                error=None if error is None else _bounded_text(_plain(error), MAX_RESPONSE_CHARS),
                order=(modified_ns, index),
            )
        )
    if items:
        return tuple(items)
    if document.get("status") in {"error", "cancelled"} or archive_error:
        return (_archive_document_error(path, document, modified_ns),)
    return ()


def _archive_document_error(
    path: Path, document: Mapping[str, Any], modified_ns: int
) -> ResponseItem:
    meta = document.get("meta") if isinstance(document.get("meta"), Mapping) else {}
    capture = document.get("capture") if isinstance(document.get("capture"), Mapping) else {}
    status = _normal_status(document.get("status", "error"))
    return ResponseItem(
        key=f"{path.as_posix()}::archive",
        path=path,
        run_id=_one_line(document.get("run_id", path.parent.name)),
        created_at=_one_line(document.get("created_at", "")),
        generation=_one_line(meta.get("generation", "?")),
        role=_one_line(capture.get("role", meta.get("role", "unknown"))).lower(),
        model="unknown",
        response_id="archive",
        status=status,
        text="",
        error=_bounded_text(_plain(document.get("error") or "request ended without a response"), MAX_RESPONSE_CHARS),
        order=(modified_ns, 0),
    )


def _archive_error_item(path: Path, modified_ns: int, error: Exception) -> ResponseItem:
    return ResponseItem(
        key=f"{path.as_posix()}::load-error",
        path=path,
        run_id=path.parent.name,
        created_at="",
        generation="?",
        role="unknown",
        model="unknown",
        response_id="archive",
        status="error",
        text="",
        error=_bounded_text(_plain(f"Could not read {path}: {error}"), MAX_RESPONSE_CHARS),
        order=(modified_ns, 0),
    )


def _normal_status(value: Any) -> str:
    status = _one_line(value).lower()
    return status if status in {"streaming", "complete", "error", "cancelled"} else "error"


def _display_status(status: str) -> str:
    return "completed" if status == "complete" else status


def _short_id(value: str) -> str:
    return value if len(value) <= 12 else value[:12]


def _one_line(value: Any) -> str:
    return " ".join(_plain(value).splitlines()).strip()


def _plain(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    return "".join(character if character in "\n\t" or ord(character) >= 32 else "�" for character in text)


def _bounded_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n\n[... {omitted:,} characters omitted ...]\n\n{text[-tail:]}"
