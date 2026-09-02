from __future__ import annotations

import os
import queue
import subprocess
import threading
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
    list_sessions,
    read_logs,
    start_session,
    stop_session,
)
from ultron.cli.pixel import PixelStyle, advance_style_tick, mascot_strip, style_at
from ultron.cli.results import (
    ResultsError,
    discover_generations,
    fetch_review,
    read_markdown,
)
from ultron.train.family import FamilyName, FamilyPack

CSS_PATH = Path(__file__).with_name("console.tcss")
SENTINEL = object()


class View(str, Enum):
    CATALOG = "catalog"
    JOBS = "jobs"
    RESULTS = "results"
    RUN = "run"


class ConsoleApp(App[GymPlan | None]):
    CSS_PATH = CSS_PATH
    TITLE = "ultron"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("q", "quit", "quit", show=True),
        Binding("escape", "back", "back", show=True),
        Binding("enter", "confirm", "run", show=True),
        Binding("a", "show_catalog", "actions", show=True),
        Binding("j", "show_jobs", "jobs", show=True),
        Binding("r", "show_results", "results", show=True),
        Binding("t", "focus_tests", "tests", show=True),
        Binding("m", "focus_family", "model", show=True),
        Binding("s", "stop_job", "stop", show=True),
        Binding("p", "cycle_pixel", "pixel", show=True),
        Binding("g", "refresh", "refresh", show=False),
    ]

    def __init__(self, *, root: Path | None = None, family: str | None = None) -> None:
        super().__init__()
        self.root = root or repo_root()
        self.pack: FamilyPack = resolve_pack(family, root=self.root)
        self.family: FamilyName = self.pack.name
        self.view = View.CATALOG
        self.selected = ActionId.GENERATION
        self._inputs: dict[str, Input] = {}
        self._run_session: str | None = None
        self._run_title = ""
        self._events: queue.Queue[object] = queue.Queue()
        self._log_cursor = 0
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
                yield DataTable(id="job-table")
                yield RichLog(id="job-log", highlight=False, markup=False, wrap=True)
            with Vertical(id="results"):
                yield DataTable(id="result-table")
                yield RichLog(id="review-log", highlight=False, markup=False, wrap=True)
            with Vertical(id="run"):
                yield Static(id="run-header")
                yield RichLog(id="run-log", highlight=False, markup=False, wrap=True)
            yield Static(id="status")

    def on_mount(self) -> None:
        self._fill_actions()
        self.query_one("#job-table", DataTable).add_columns("session", "state", "pid", "command")
        self.query_one("#result-table", DataTable).add_columns("gen", "verdict", "episodes", "asr", "review")
        self.set_interval(0.2, self._tick)
        self.set_interval(0.16, self._tick_pixels)
        self._show(View.CATALOG)
        if not self.query("#form Input"):
            self._select_action(self.selected)

    def action_quit(self) -> None:
        self.exit(None)

    def action_show_catalog(self) -> None:
        self._show(View.CATALOG)

    def action_show_jobs(self) -> None:
        self._refresh_jobs()
        self._show(View.JOBS)

    def action_show_results(self) -> None:
        self._refresh_results()
        self._show(View.RESULTS)

    def action_focus_tests(self) -> None:
        self._show(View.CATALOG)
        self._select_action(ActionId.TESTS)
        self._highlight_action(ActionId.TESTS)

    def action_focus_family(self) -> None:
        self.query_one("#family", Select).focus()

    def action_back(self) -> None:
        if self.view is View.RUN:
            self._show(View.CATALOG)
            return
        if self.view in (View.JOBS, View.RESULTS):
            self._show(View.CATALOG)

    def action_cycle_pixel(self) -> None:
        self._pixel_tick = advance_style_tick(self._pixel_tick)
        self._paint_sprites()

    def action_refresh(self) -> None:
        if self.view is View.JOBS:
            self._refresh_jobs()
        elif self.view is View.RESULTS:
            self._refresh_results()
        elif self.view is View.RUN:
            self._poll_run_logs()

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
        self._set_status(f"stopped {session}")
        self._refresh_jobs()

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
            case GymPlan():
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
        self._run_session = None
        self._run_title = built.title
        self._events = queue.Queue()
        self._done = False
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
                    self._events.put(line.rstrip("\n"))
                code = proc.wait()
                self._events.put(f"[exit {code}]")
            except Exception as exc:
                self._events.put(f"error {exc}")
            finally:
                self._events.put(SENTINEL)

        threading.Thread(target=worker, name="ultron-console-run", daemon=True).start()
        self._show(View.RUN)
        self._set_status(f"running {built.title}")

    def _open_session(self, session: str, title: str) -> None:
        self._run_session = session
        self._run_title = title
        self._done = False
        self.query_one("#run-log", RichLog).clear()
        self.query_one("#run-header", Static).update(f"  RUN  {title}")
        self._show(View.RUN)
        self._poll_run_logs()

    def _tick(self) -> None:
        if self.view is View.RUN:
            self._drain_events()
            self._poll_run_logs()

    def _tick_pixels(self) -> None:
        self._pixel_tick += 1
        self._paint_sprites()

    def _paint_sprites(self) -> None:
        self.query_one("#sprites", Static).update(mascot_strip(self._pixel_tick))
        if self.view is View.CATALOG:
            self._set_status(_footer(self.view, pixel_style=style_at(self._pixel_tick)))

    def _drain_events(self) -> None:
        log = self.query_one("#run-log", RichLog)
        while True:
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                break
            if item is SENTINEL:
                self._done = True
                self._set_status(f"finished {self._run_title}")
                break
            log.write(str(item))

    def _poll_run_logs(self) -> None:
        if self._run_session is None:
            return
        try:
            text = read_logs(self._run_session, tail=0, root=self.root)
        except JobsError as exc:
            self._set_status(str(exc))
            return
        log = self.query_one("#run-log", RichLog)
        lines = text.splitlines()
        while self._log_cursor < len(lines):
            log.write(lines[self._log_cursor])
            self._log_cursor += 1

    def _refresh_jobs(self) -> None:
        table = self.query_one("#job-table", DataTable)
        table.clear()
        try:
            sessions = list_sessions(root=self.root)
        except JobsError as exc:
            self._set_status(str(exc))
            return
        if not sessions:
            self._set_status("no tmux jobs")
            return
        for item in sessions:
            table.add_row(
                item.name,
                item.state.value,
                "—" if item.pid is None else str(item.pid),
                item.command,
                key=item.name,
            )
        self._set_status(f"{len(sessions)} job(s)")

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

    def _show(self, view: View) -> None:
        self.view = view
        if view is View.RUN:
            self._log_cursor = 0
        self.query_one("#catalog").display = view is View.CATALOG
        self.query_one("#jobs").display = view is View.JOBS
        self.query_one("#results").display = view is View.RESULTS
        self.query_one("#run").display = view is View.RUN
        self.query_one("#sprites").display = view is View.CATALOG
        self.query_one("#header-title", Static).update(_header(view))
        self._paint_sprites()
        self._set_status(_footer(view, pixel_style=style_at(self._pixel_tick)))

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(f"  {text}")


def run_console(*, root: Path | None = None, family: str | None = None) -> GymPlan | None:
    return ConsoleApp(root=root, family=family).run()


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


def _footer(view: View, *, pixel_style: PixelStyle | None = None) -> str:
    pixel = "" if pixel_style is None else f" · pixel {pixel_style.value}"
    match view:
        case View.CATALOG:
            return f"enter run · m model · j jobs · r results · t tests · p pixel{pixel} · q quit"
        case View.JOBS:
            return "enter logs · s stop · g refresh · esc back · q quit"
        case View.RESULTS:
            return "enter fetch review · g refresh · esc back · q quit"
        case View.RUN:
            return "s stop · esc back · q quit"
        case _:
            raise ValueError(f"unhandled view {view!r}")
