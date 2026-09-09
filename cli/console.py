from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Input, Label, OptionList, RichLog, Select, Static
from textual.widgets.option_list import Option

from ultron.cli.catalog import (
    ActionGroup,
    ActionId,
    BattlePlan,
    CatalogError,
    ForegroundPlan,
    GymPlan,
    LaunchPlan,
    TmuxPlan,
    all_actions,
    family_options,
    plan,
    repo_root,
    resolve_pack,
    spec_for,
)
from ultron.cli.jobs import (
    AlreadyRunning,
    JobsError,
    LogChunk,
    SessionInfo,
    SessionState,
    list_sessions,
    read_log_chunk,
    session_status,
    start_session,
    stop_session,
)
from ultron.cli.pixel import mascot_strip
from ultron.cli.results import (
    ResultsError,
    discover_generations,
    fetch_review,
    read_markdown,
)
from ultron.train.family import FamilyName, FamilyPack
from ultron.cli.responses import default_response_directory

CSS_PATH = Path(__file__).with_name("console.tcss")
SENTINEL = object()
MAX_LOG_LINES = 2_000
MAX_LOG_READ_BYTES = 64 * 1024
MAX_FOREGROUND_EVENTS_PER_TICK = 200
SESSION_POLL_SECONDS = 0.75


@dataclass
class _LogContext:
    session: str | None = None
    offset: int | None = None


@dataclass(frozen=True)
class _SessionUpdate:
    generation: int
    log_id: str
    session: str
    info: SessionInfo | None
    chunk: LogChunk | None
    status_error: str | None = None
    log_error: str | None = None


class View(str, Enum):
    CATALOG = "catalog"
    JOBS = "jobs"
    RESULTS = "results"
    RUN = "run"


class ConsoleApp(App[GymPlan | BattlePlan | None]):
    CSS_PATH = CSS_PATH
    TITLE = "ultron"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("q", "quit", "quit", show=True),
        Binding("escape", "back", "back", show=True),
        Binding("enter", "confirm", "run", show=True),
        Binding("a", "show_catalog", "actions", show=True),
        Binding("j", "show_jobs", "jobs", show=True),
        Binding("b", "show_battle", "battle", show=True),
        Binding("r", "show_results", "results", show=True),
        Binding("t", "focus_tests", "tests", show=True),
        Binding("m", "focus_family", "model", show=True),
        Binding("s", "stop_job", "stop", show=True),
        Binding("g", "refresh", "refresh", show=False),
    ]

    def __init__(self, *, root: Path | None = None, family: str | None = None, initial_view: View | str | None = None, initial_session: str | None = None) -> None:
        super().__init__()
        self.root = root or repo_root()
        self.pack: FamilyPack = resolve_pack(family, root=self.root)
        self.family: FamilyName = self.pack.name
        self.view = View.CATALOG
        if isinstance(initial_view, str):
            initial_view = View(initial_view)
        self._initial_view = initial_view
        self._initial_session = initial_session
        self.selected = ActionId.GENERATION
        self._inputs: dict[str, Input] = {}
        self._run_session: str | None = None
        self._run_title = ""
        self._events: queue.Queue[object] = queue.Queue()
        self._session_updates: queue.Queue[_SessionUpdate] = queue.Queue()
        self._log_contexts = {
            "job-log": _LogContext(),
            "run-log": _LogContext(),
        }
        self._active_log_id: str | None = None
        self._poll_generation = 0
        self._poll_inflight = False
        self._poll_after = 0.0
        self._job_count = 0
        self._run_return_view = View.CATALOG
        self._done = False
        self._pixel_tick = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="frame"):
            with Horizontal(id="header"):
                yield Static(id="header-title")
                yield Select[str](
                    family_options(root=self.root),
                    value=self.family.value,
                    prompt="model",
                    allow_blank=False,
                    compact=True,
                    type_to_search=False,
                    id="family",
                )
            yield Static(id="sprites")
            with Horizontal(id="catalog"):
                yield OptionList(id="actions")
                with Vertical(id="detail"):
                    yield Static(id="summary")
                    yield Vertical(id="form")
            with Vertical(id="jobs"):
                yield DataTable(id="job-table", cursor_type="row", zebra_stripes=True)
                yield RichLog(
                    id="job-log",
                    highlight=False,
                    markup=False,
                    wrap=True,
                    max_lines=MAX_LOG_LINES,
                )
            with Vertical(id="results"):
                yield DataTable(id="result-table")
                yield RichLog(
                    id="review-log", highlight=False, markup=False, wrap=True, max_lines=MAX_LOG_LINES
                )
            with Vertical(id="run"):
                yield Static(id="run-header")
                yield RichLog(
                    id="run-log",
                    highlight=False,
                    markup=False,
                    wrap=True,
                    max_lines=MAX_LOG_LINES,
                )
            yield Static(id="status")

    def on_mount(self) -> None:
        self._fill_actions()
        job_table = self.query_one("#job-table", DataTable)
        for label, key in (
            ("session", "session"),
            ("state", "state"),
            ("outcome", "outcome"),
            ("pid", "pid"),
            ("command", "command"),
        ):
            job_table.add_column(label, key=key)
        self.query_one("#result-table", DataTable).add_columns("gen", "verdict", "episodes", "asr", "review")
        self.set_interval(0.2, self._tick)
        self.set_interval(0.16, self._tick_pixels)
        self._show(View.CATALOG)
        if not self.query("#form Input"):
            self._select_action(self.selected)
        if self._initial_session:
            self._open_session(self._initial_session, f"job {self._initial_session}")
            return
        if self._initial_view is View.JOBS:
            self._show(View.JOBS)
            self._refresh_jobs()
            return
        if self._initial_view is View.RESULTS:
            self._show(View.RESULTS)
            self._refresh_results()
            return

    def action_quit(self) -> None:
        self.exit(None)

    def action_show_catalog(self) -> None:
        self._show(View.CATALOG)

    def action_show_battle(self) -> None:
        directory = default_response_directory() if os.environ.get("ULTRON_RESPONSES_DIR") else self.root / "data" / "responses"
        self.exit(BattlePlan(directory))

    def action_show_jobs(self) -> None:
        self._show(View.JOBS)
        self._refresh_jobs()

    def action_show_results(self) -> None:
        self._show(View.RESULTS)
        self._refresh_results()

    def action_focus_tests(self) -> None:
        self._show(View.CATALOG)
        self._select_action(ActionId.TESTS)
        self._highlight_action(ActionId.TESTS)

    def action_focus_family(self) -> None:
        self.query_one("#family", Select).focus()

    def action_back(self) -> None:
        if self.view is View.RUN:
            self._show(self._run_return_view)
            if self.view is View.JOBS:
                self._refresh_jobs()
            return
        if self.view in (View.JOBS, View.RESULTS):
            self._show(View.CATALOG)

    def action_refresh(self) -> None:
        if self.view is View.JOBS:
            self._refresh_jobs()
        elif self.view is View.RESULTS:
            self._refresh_results()
        elif self.view is View.RUN:
            self._poll_after = 0.0
            self._request_session_poll()

    def action_stop_job(self) -> None:
        session = self._selected_session()
        if session is None:
            self._set_status("no job selected")
            return
        try:
            stop_session(session, root=self.root)
        except JobsError as exc:
            self._set_status(str(exc))
            return
        if self.view is View.JOBS:
            self._refresh_jobs()
        else:
            self._poll_after = 0.0
            self._request_session_poll()
        self._set_status(f"stopped {session}")

    def action_confirm(self) -> None:
        if self.view is View.CATALOG:
            self._run_selected()
            return
        if self.view is View.JOBS:
            session = self._selected_session()
            if session is None:
                return
            self._open_session(session, f"job {session}")
            return
        if self.view is View.RESULTS:
            self._fetch_selected()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "actions" or event.option.id is None:
            return
        if event.option.id.startswith("group-"):
            return
        self._select_action(ActionId(event.option.id))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "actions" or event.option.id is None:
            return
        if event.option.id.startswith("group-"):
            return
        self._select_action(ActionId(event.option.id))
        self._run_selected()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "family" or event.value is Select.NULL:
            return
        self._set_family(str(event.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "job-table" or self.view is not View.JOBS:
            return
        if event.row_key.value is not None and event.row_key.value == self._table_session(event.data_table):
            self._activate_session_log(event.row_key.value, "job-log")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "job-table" or self.view is not View.JOBS:
            return
        if event.row_key.value is not None:
            self._open_session(event.row_key.value, f"job {event.row_key.value}")

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        if event.data_table.id == "result-table" and self.view is View.RESULTS:
            self._fetch_selected()

    def _fill_actions(self) -> None:
        options: list[Option] = []
        last: ActionGroup | None = None
        for spec in all_actions(root=self.root):
            if spec.group is not last:
                options.append(Option(spec.group.value.upper(), id=f"group-{spec.group.value}", disabled=True))
                last = spec.group
            options.append(Option(spec.title, id=spec.id.value))
        listing = self.query_one("#actions", OptionList)
        listing.clear_options()
        listing.add_options(options)
        self._highlight_action(self.selected)

    def _highlight_action(self, action_id: ActionId) -> None:
        listing = self.query_one("#actions", OptionList)
        listing.highlighted = listing.get_option_index(action_id.value)

    def _select_action(self, action_id: ActionId) -> None:
        if action_id is self.selected and self._inputs:
            return
        self.selected = action_id
        spec = spec_for(action_id, root=self.root)
        self.query_one("#summary", Static).update(f"{spec.title}\n{spec.summary}")
        form = self.query_one("#form", Vertical)
        form.remove_children()
        self._inputs = {}
        for field in spec.fields:
            widget = Input(value=field.default, placeholder=field.help or field.label)
            self._inputs[field.key] = widget
            form.mount(Label(field.label, classes="field-label"))
            form.mount(widget)

    def _field_values(self) -> dict[str, str]:
        spec = spec_for(self.selected, root=self.root)
        return {field.key: self._inputs[field.key].value for field in spec.fields}

    def _set_family(self, name: str) -> None:
        try:
            pack = resolve_pack(name, root=self.root)
        except CatalogError as exc:
            self._set_status(str(exc))
            return
        if pack.name is self.family:
            return
        self.pack = pack
        self.family = pack.name
        if self.view is View.RESULTS:
            self._refresh_results()
        self._set_status(f"family {pack.name.value}  {pack.base_model}")

    def _run_selected(self) -> None:
        try:
            built = plan(self.selected, self._field_values(), root=self.root, family=self.family.value)
        except CatalogError as exc:
            self._set_status(str(exc))
            return
        self._launch(built)

    def _launch(self, built: LaunchPlan) -> None:
        match built:
            case GymPlan() | BattlePlan():
                self.exit(built)
            case TmuxPlan():
                try:
                    result = start_session(
                        built.session,
                        built.argv,
                        extra_env=dict(built.env),
                        root=self.root,
                    )
                except JobsError as exc:
                    self._set_status(str(exc))
                    return
                note = "already running" if isinstance(result, AlreadyRunning) else "started"
                self._open_session(built.session, f"{note} {built.session}")
            case ForegroundPlan():
                self._start_foreground(built)
            case _:
                self._set_status(f"unhandled launch {type(built)!r}")

    def _start_foreground(self, built: ForegroundPlan) -> None:
        if self.view is not View.RUN:
            self._run_return_view = self.view
        self._run_session = None
        self._run_title = built.title
        self._events = queue.Queue()
        events = self._events
        self._done = False
        self._deactivate_session_log()
        self.query_one("#run-log", RichLog).clear()
        self.query_one("#run-header", Static).update(f"  RUN  {built.title}")

        def worker() -> None:
            try:
                env = os.environ.copy()
                env.update(dict(built.env))
                proc = subprocess.Popen(
                    list(built.argv),
                    cwd=str(built.cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    events.put(line.rstrip("\n"))
                code = proc.wait()
                events.put(f"[exit {code}]")
            except Exception as exc:
                events.put(f"error {exc}")
            finally:
                events.put(SENTINEL)

        threading.Thread(target=worker, name="ultron-console-run", daemon=True).start()
        self._show(View.RUN)
        self._set_status(f"running {built.title}")

    def _open_session(self, session: str, title: str) -> None:
        if self.view is not View.RUN:
            self._run_return_view = self.view
        self._run_session = session
        self._run_title = title
        self._done = False
        self.query_one("#run-log", RichLog).clear()
        self.query_one("#run-header", Static).update(f"  RUN  {title}")
        self._show(View.RUN)
        self._activate_session_log(session, "run-log", reset=True)
        self._request_session_poll()

    def _tick(self) -> None:
        if self.view is View.RUN and self._run_session is None:
            self._drain_events()
        self._drain_session_updates()
        self._request_session_poll()

    def _tick_pixels(self) -> None:
        self._pixel_tick += 1
        self._paint_sprites()

    def _paint_sprites(self) -> None:
        sprites = self.query("#sprites")
        if sprites:
            sprites.first(Static).update(mascot_strip(self._pixel_tick))

    def _drain_events(self) -> None:
        log = self.query_one("#run-log", RichLog)
        for _ in range(MAX_FOREGROUND_EVENTS_PER_TICK):
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                return
            if item is SENTINEL:
                self._done = True
                self._set_status(f"finished {self._run_title}")
                return
            self._write_log(log, str(item))

    def _activate_session_log(self, session: str, log_id: str, *, reset: bool = False) -> None:
        context = self._log_contexts[log_id]
        switched = context.session != session
        if switched or reset:
            context.session = session
            context.offset = None
            self.query_one(f"#{log_id}", RichLog).clear()
        if self._active_log_id == log_id and not switched and not reset:
            return
        self._active_log_id = log_id
        self._poll_generation += 1
        self._poll_inflight = False
        self._poll_after = 0.0

    def _deactivate_session_log(self) -> None:
        if self._active_log_id is None:
            return
        self._active_log_id = None
        self._poll_generation += 1
        self._poll_inflight = False

    def _request_session_poll(self) -> None:
        log_id = self._active_log_id
        if log_id is None or self._poll_inflight or time.monotonic() < self._poll_after:
            return
        context = self._log_contexts[log_id]
        if context.session is None:
            return
        generation = self._poll_generation
        session = context.session
        offset = context.offset
        self._poll_inflight = True

        def worker() -> None:
            info: SessionInfo | None = None
            chunk: LogChunk | None = None
            status_error: str | None = None
            log_error: str | None = None
            try:
                info = session_status(session, root=self.root)
            except Exception as exc:
                status_error = str(exc)
            try:
                chunk = read_log_chunk(
                    session,
                    offset=offset,
                    max_bytes=MAX_LOG_READ_BYTES,
                    root=self.root,
                )
            except Exception as exc:
                log_error = str(exc)
            self._session_updates.put(
                _SessionUpdate(
                    generation=generation,
                    log_id=log_id,
                    session=session,
                    info=info,
                    chunk=chunk,
                    status_error=status_error,
                    log_error=log_error,
                )
            )

        threading.Thread(target=worker, name="ultron-console-log", daemon=True).start()

    def _drain_session_updates(self) -> None:
        while True:
            try:
                update = self._session_updates.get_nowait()
            except queue.Empty:
                return
            if update.generation != self._poll_generation or update.log_id != self._active_log_id:
                continue
            context = self._log_contexts[update.log_id]
            if context.session != update.session:
                continue
            self._poll_inflight = False
            self._poll_after = time.monotonic() + SESSION_POLL_SECONDS
            if update.chunk is not None:
                context.offset = update.chunk.next_offset
                log = self.query_one(f"#{update.log_id}", RichLog)
                if update.chunk.reset:
                    log.clear()
                if update.chunk.skipped_bytes:
                    self._write_log(log, f"[skipped {update.chunk.skipped_bytes:,} earlier log bytes]")
                for line in update.chunk.text.splitlines():
                    self._write_log(log, line)
            if update.info is not None:
                self._show_session_status(update.log_id, update.info)
            elif update.status_error:
                self._set_status(update.status_error)
            elif update.log_error:
                self._set_status(update.log_error)

    @staticmethod
    def _write_log(log: RichLog, line: str) -> None:
        log.write(line, scroll_end=log.is_vertical_scroll_end)

    def _show_session_status(self, log_id: str, info: SessionInfo) -> None:
        state = _state_label(info)
        outcome = _outcome_label(info)
        detail = state if not outcome else f"{state} · {outcome}"
        if log_id == "run-log":
            self.query_one("#run-header", Static).update(f"  JOB  {info.name}   {detail}")
            self._set_status(f"{detail} · pgup/pgdn scroll · end follow · s stop · esc back")
            return
        table = self.query_one("#job-table", DataTable)
        if info.name in table.rows:
            table.update_cell(info.name, "state", state)
            table.update_cell(info.name, "outcome", outcome or "—")
            table.update_cell(info.name, "pid", "—" if info.pid is None else str(info.pid))
        self._set_status(
            f"{self._job_count} job(s) · {info.name} {detail} · enter log · s stop · g refresh"
        )

    def _refresh_jobs(self) -> None:
        table = self.query_one("#job-table", DataTable)
        selected = self._table_session(table)
        old_row = table.cursor_row
        old_scroll_y = table.scroll_y
        try:
            sessions = list_sessions(root=self.root)
        except JobsError as exc:
            self._set_status(str(exc))
            return
        table.clear()
        if not sessions:
            self._job_count = 0
            self._deactivate_session_log()
            self.query_one("#job-log", RichLog).clear()
            self._set_status("no tmux jobs · g refresh · esc back")
            return
        for item in sessions:
            table.add_row(
                item.name,
                _state_label(item),
                _outcome_label(item) or "—",
                "—" if item.pid is None else str(item.pid),
                item.command,
                key=item.name,
            )
        names = {item.name for item in sessions}
        target = selected if selected in names else sessions[min(old_row, len(sessions) - 1)].name
        table.move_cursor(row=table.get_row_index(target), scroll=False)
        table.call_after_refresh(table.scroll_to, y=old_scroll_y, animate=False)
        self._job_count = len(sessions)
        self._activate_session_log(target, "job-log")
        selected_info = next(item for item in sessions if item.name == target)
        detail = _state_label(selected_info)
        outcome = _outcome_label(selected_info)
        if outcome:
            detail = f"{detail} · {outcome}"
        self._set_status(
            f"{len(sessions)} job(s) · {target} {detail} · enter log · s stop · g refresh"
        )

    def _refresh_results(self) -> None:
        table = self.query_one("#result-table", DataTable)
        table.clear()
        found = discover_generations(root=self.root, archive_dir=self.pack.archive_root)
        if not found:
            self._set_status("no generations in data/traces or data/archives")
            return
        for item in found:
            verdict = "—" if item.review is None else item.review.verdict
            episodes = "—" if item.review is None else str(item.review.episodes)
            asr = "—" if item.review is None or item.review.asr is None else f"{item.review.asr:.3f}"
            review = "yes" if item.review is not None else "no"
            table.add_row(str(item.generation), verdict, episodes, asr, review, key=str(item.generation))
        self._set_status(f"{len(found)} generation(s)")

    def _fetch_selected(self) -> None:
        table = self.query_one("#result-table", DataTable)
        if table.row_count == 0:
            self._set_status("no generation selected")
            return
        generation = int(str(table.get_row_at(table.cursor_row)[0]))
        traces = self.root / "data" / "traces" / f"gen{generation}"
        try:
            summary = fetch_review(
                traces,
                generation=generation,
                phase="complete",
                eval_dir=self.root / "data" / "eval",
                archive_dir=self.pack.archive_root,
                pfsp_path=self.pack.pfsp_manifest,
            )
        except ResultsError as exc:
            self._set_status(str(exc))
            return
        log = self.query_one("#review-log", RichLog)
        log.clear()
        log.write(read_markdown(summary))
        self._refresh_results()
        self._set_status(f"gen{generation} {summary.verdict}")

    def _selected_session(self) -> str | None:
        if self._run_session and self.view is View.RUN:
            return self._run_session
        table = self.query_one("#job-table", DataTable)
        if self.view is not View.JOBS or table.row_count == 0:
            return None
        return str(table.get_row_at(table.cursor_row)[0])

    @staticmethod
    def _table_session(table: DataTable) -> str | None:
        if table.row_count == 0:
            return None
        return str(table.get_row_at(table.cursor_row)[0])

    def _show(self, view: View) -> None:
        self.view = view
        self.query_one("#catalog").display = view is View.CATALOG
        self.query_one("#jobs").display = view is View.JOBS
        self.query_one("#results").display = view is View.RESULTS
        self.query_one("#run").display = view is View.RUN
        self.query_one("#sprites").display = view is View.CATALOG
        self.query_one("#header-title", Static).update(_header(view))
        self._paint_sprites()
        self._set_status(_footer(view))
        if view is View.CATALOG:
            self._deactivate_session_log()
            self.query_one("#actions", OptionList).focus()
        elif view is View.JOBS:
            table = self.query_one("#job-table", DataTable)
            table.focus()
            session = self._table_session(table)
            if session is not None:
                self._activate_session_log(session, "job-log")
        elif view is View.RESULTS:
            self._deactivate_session_log()
            self.query_one("#result-table", DataTable).focus()
        else:
            self.query_one("#run-log", RichLog).focus()

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(f"  {text}")


def run_console(*, root: Path | None = None, family: str | None = None, initial_view: View | str | None = None, initial_session: str | None = None) -> GymPlan | BattlePlan | None:
    return ConsoleApp(root=root, family=family, initial_view=initial_view, initial_session=initial_session).run()


def _header(view: View) -> str:
    match view:
        case View.CATALOG:
            return "  ULTRON EXPERIMENT   select an action"
        case View.JOBS:
            return "  ULTRON EXPERIMENT   tmux jobs"
        case View.RESULTS:
            return "  ULTRON EXPERIMENT   generation results"
        case View.RUN:
            return "  ULTRON EXPERIMENT   run output"
        case _:
            raise ValueError(f"unhandled view {view!r}")


def _footer(view: View) -> str:
    match view:
        case View.CATALOG:
            return "enter run · b battle · m model · j jobs · r results · t tests · q quit"
        case View.JOBS:
            return "↑/↓ select · enter log · s stop · g refresh · esc back"
        case View.RESULTS:
            return "enter fetch review · g refresh · esc back · q quit"
        case View.RUN:
            return "pgup/pgdn scroll · end follow · s stop · esc back"
        case _:
            raise ValueError(f"unhandled view {view!r}")


def _state_label(info: SessionInfo) -> str:
    if info.state is SessionState.DEAD:
        return "finished"
    return info.state.value


def _outcome_label(info: SessionInfo) -> str:
    if info.state is not SessionState.DEAD or info.exit_code is None:
        return ""
    if info.exit_code == 0:
        return "success (0)"
    return f"failed ({info.exit_code})"
