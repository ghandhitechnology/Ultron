from __future__ import annotations

import queue
import threading
import time
from typing import Iterable

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import ListItem, ListView, Static

from ultron.train.schema_v1 import Role

from .model import (
    AttackingSnapshot,
    CompleteSnapshot,
    DefendingSnapshot,
    FailedSnapshot,
    JobEvent,
    JobSnapshot,
    JobSpec,
    ProbeEvent,
    ProbingSnapshot,
    RestoreEvent,
    ToolObservedEvent,
    active_role,
    apply,
    estimate_eta_s,
    initial_snapshot,
    progress,
)
from .observe import run_job

BG_BASE = "#1c1c2b"
BG_PANE = "#26263a"
BORDER = "#5f5f87"

C_TEXT = "#d0d0d0"
C_DIM = "#8a8a8a"
C_ACCENT = "#af87ff"
C_ATTACKER = "#d75f87"
C_DEFENDER = "#5fafaf"
C_SANDBOX = "#d7af87"
C_OK = "#87af87"
C_WARN = "#d7af5f"
C_FAIL = "#d75f5f"
C_BADGE_FG = "#080808"

NODE_W = 22


def _c(color: str, text: str) -> str:
    return f"[{color}]{text}[/]"


def _rt(markup: str) -> Text:
    return Text.from_markup(markup)


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _bar(fraction: float, width: int) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def _pad(text: str, width: int) -> str:
    return text[:width].ljust(width)


def _box(title: str, title_color: int, lines: list[str], width: int) -> list[str]:
    inner = width - 2
    top = _c(title_color, "┌" + "─" * inner + "┐")
    bottom = _c(title_color, "└" + "─" * inner + "┘")
    side = _c(title_color, "│")
    body = [f"{side} {line} {side}" for line in lines]
    return [_c(title_color, _pad(title, width)), top, *body, bottom]


class HeaderPane(Static):
    pass


class ArenaPane(Static):
    def on_click(self, event: events.Click) -> None:
        third = max(1, self.size.width // 3)
        target = ("attacker", "sandbox", "defender")[min(2, event.x // third)]
        self.app.expand(target)


class ProcessLog(ListView):
    pass


class ProgressPane(Static):
    pass


class FooterPane(Static):
    pass


class DetailPane(Static):
    def on_click(self, event: events.Click) -> None:
        self.app.collapse()


class SimApp(App):
    CSS = f"""
    Screen {{ background: {BG_BASE}; }}
    #header {{ height: 2; padding: 0 1; background: {BG_BASE}; color: white; }}
    #arena {{
        height: 11; border: round {BORDER}; background: {BG_PANE};
        padding: 0 1; margin: 0 1;
    }}
    #process {{
        border: round {BORDER}; background: {BG_PANE};
        margin: 0 1; height: 1fr;
    }}
    #process > ListItem {{ background: {BG_PANE}; padding: 0 1; }}
    #process > ListItem.--highlight {{ background: #3a3a4f; }}
    #progress {{ height: 4; border: round {BORDER}; background: {BG_PANE};
        padding: 0 1; margin: 0 1; }}
    #footer {{ height: 2; padding: 0 1; background: {BG_BASE}; }}
    #detail {{
        display: none; layer: overlay; width: 90%; height: 80%;
        offset: 5% 10%; border: round {BORDER}; background: {BG_PANE};
        padding: 1 2;
    }}
    #detail.shown {{ display: block; }}
    """

    BINDINGS = [
        ("q", "quit", "quit"),
        ("a", "expand_attacker", "attacker"),
        ("s", "expand_sandbox", "sandbox"),
        ("d", "expand_defender", "defender"),
        ("l", "focus_log", "log"),
        ("escape", "collapse", "collapse"),
    ]

    def __init__(
        self,
        spec: JobSpec,
        runner,
        cases: Iterable,
    ) -> None:
        super().__init__()
        self._spec = spec
        self._runner = runner
        self._cases = tuple(cases)
        self._queue: "queue.Queue[JobEvent]" = queue.Queue()
        self.snapshot: JobSnapshot = initial_snapshot(spec, started_at_s=time.monotonic())
        self._log_count = 0
        self._detail: tuple[str, object] | None = None
        self._worker: threading.Thread | None = None

    def compose(self) -> ComposeResult:
        yield HeaderPane(id="header")
        yield ArenaPane(id="arena")
        log = ProcessLog(id="process")
        log.border_title = "PROCESS"
        yield log
        yield ProgressPane(id="progress")
        yield FooterPane(id="footer")
        yield DetailPane(id="detail")

    def on_mount(self) -> None:
        self.query_one("#arena", ArenaPane).border_title = "ARENA"
        self.query_one("#progress", ProgressPane).border_title = "PROGRESS"
        self._render_all()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        self.set_interval(0.05, self._drain)
        self.set_interval(1.0, self._render_all)

    def _run_worker(self) -> None:
        run_job(self._runner, self._cases, emit=self._queue.put, clock=time.monotonic)

    def _drain(self) -> None:
        changed = False
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            self.snapshot = apply(self.snapshot, event)
            self._append_log(event)
            changed = True
        if changed:
            self._render_all()

    def expand(self, target: str) -> None:
        self._detail = (target, None)
        self._show_detail()

    def collapse(self) -> None:
        self._detail = None
        self.query_one("#detail", DetailPane).remove_class("shown")

    def action_expand_attacker(self) -> None:
        self.expand("attacker")

    def action_expand_sandbox(self) -> None:
        self.expand("sandbox")

    def action_expand_defender(self) -> None:
        self.expand("defender")

    def action_collapse(self) -> None:
        self.collapse()

    def action_focus_log(self) -> None:
        self.query_one("#process", ProcessLog).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        payload = getattr(event.item, "job_event", None)
        if payload is not None:
            self._detail = ("tool", payload)
            self._show_detail()

    def _append_log(self, event: JobEvent) -> None:
        log = self.query_one("#process", ProcessLog)
        item = ListItem(Static(_rt(self._log_line(event))))
        item.job_event = event
        log.append(item)
        self._log_count += 1
        log.index = self._log_count - 1

    def _log_line(self, event: JobEvent) -> str:
        stamp = _c(C_DIM, time.strftime("%H:%M:%S"))
        if isinstance(event, RestoreEvent):
            body = _c(C_DIM, f"restore ok  guest {event.guest_id}  {event.duration_s:4.2f}s")
            return f"{_c(C_ACCENT, '▌')} {stamp} ep {event.episode_index:02d}  {body}"
        if isinstance(event, ToolObservedEvent):
            accent = C_ATTACKER if event.role is Role.ATTACKER else C_DEFENDER
            exit_code = event.tool.exit_code
            exit_txt = _c(C_OK if exit_code == 0 else C_WARN, f"exit {exit_code}")
            cmd = _pad(str(event.tool.args.get("cmd", event.tool.name)), 38)
            dur = _c(C_DIM, f"{event.tool.duration_ms / 1000:4.2f}s")
            return f"{_c(accent, '▌')} {stamp} t{event.turn_index:02d} {event.role.value[:1].upper()}  {cmd} {exit_txt} {dur}"
        if event.kind == "turn_started":
            accent = C_ATTACKER if event.role is Role.ATTACKER else C_DEFENDER
            return f"{_c(accent, '▌')} {stamp} t{event.turn_index:02d} {event.role.value[:1].upper()}  {_c(C_DIM, 'turn started')}"
        if event.kind == "turn_ended":
            accent = C_ATTACKER if event.role is Role.ATTACKER else C_DEFENDER
            body = f"turn finished  {event.trajectory_step_count} steps  {event.duration_s:4.2f}s"
            return f"{_c(accent, '▌')} {stamp} t{event.turn_index:02d} {event.role.value[:1].upper()}  {_c(C_DIM, body)}"
        if isinstance(event, ProbeEvent):
            body = f"probe  euid {event.result.guest_attacker_euid}  root {event.result.host_confirmed_root}"
            return f"{_c(C_SANDBOX, '▌')} {stamp} ep {event.episode_index:02d}  {_c(C_SANDBOX, body)}"
        if event.kind == "episode_ended":
            reason = event.terminal.reason_code.value
            color = C_FAIL if reason == "ATTACKER_ROOT" else C_OK
            return f"{_c(color, '▌')} {stamp} ep {event.episode_index:02d}  {_c(color, 'episode ' + reason)}  {_fmt_dur(event.duration_s)}"
        if event.kind == "job_ended":
            return f"{_c(C_ACCENT, '▌')} {stamp} {_c(C_ACCENT, 'job finished')}  {_fmt_dur(event.duration_s)}"
        return f"{_c(C_FAIL, '▌')} {stamp} {_c(C_FAIL, 'error ' + getattr(event, 'message', ''))}"

    def _render_all(self) -> None:
        self.query_one("#header", HeaderPane).update(_rt(self._render_header()))
        self.query_one("#arena", ArenaPane).update(_rt(self._render_arena()))
        self.query_one("#progress", ProgressPane).update(_rt(self._render_progress()))
        self.query_one("#footer", FooterPane).update(_rt(self._render_footer()))
        if self._detail is not None:
            self._show_detail()

    def _render_header(self) -> str:
        prog = progress(self.snapshot)
        spec = self._spec
        eta = estimate_eta_s(self.snapshot)
        eta_txt = f"eta {_fmt_dur(eta)}" if eta is not None else _c(C_WARN, "eta --")
        pct = 0
        if prog.total_episodes:
            pct = int(100 * prog.completed_episodes / prog.total_episodes)
        current = min(self._current_episode_index() + 1, prog.total_episodes)
        elapsed = _fmt_dur(time.monotonic() - self.snapshot.data.started_at_s)
        line1 = (
            f"{_c(C_ACCENT, '▌')} {_c(C_TEXT, 'generation')} {_c(C_ACCENT, str(spec.generation))}"
            f"  {_c(C_TEXT, 'profile')} {_c(C_ACCENT, spec.profile_id)}"
            f"  {_c(C_DIM, 'episode')} {_c(C_TEXT, f'{current:02d}/{prog.total_episodes:02d}')}"
            f"   {_c(C_TEXT, f'{pct}%')}   {eta_txt}"
        )
        line2 = (
            f"{_c(C_ACCENT, '▌')} {_c(C_DIM, 'isolation ' + spec.isolation_backend.value)}"
            f"   {_c(C_DIM, str(spec.turns_per_side) + ' turns/side')}"
            f"      {_c(C_DIM, 'elapsed ' + elapsed)}"
        )
        return f"{line1}\n{line2}"

    def _render_arena(self) -> str:
        snapshot = self.snapshot
        prog = progress(snapshot)
        role = active_role(snapshot)
        guest_id, image_ref = self._guest_facts()
        euid, root, avail = self._probe_facts()
        a_active = role is Role.ATTACKER
        d_active = role is Role.DEFENDER

        inner = NODE_W - 4
        turn_label = ""
        if isinstance(snapshot, (AttackingSnapshot, DefendingSnapshot)):
            turn_label = f"turn {snapshot.turn_index + 1}"

        def side_lines(role_of: Role, is_active: bool, color: int) -> list[str]:
            state = "█ ACTIVE" if is_active else "░ WAITING"
            state_line = _pad(f"{state}  {turn_label if is_active else ''}", inner)
            return [
                _c(color if is_active else C_DIM, state_line),
                _c(C_TEXT, _pad(self._last_tool(role_of), inner)),
                _c(C_DIM, _pad(self._spec.isolation_backend.value, inner)),
            ]

        avail_color = C_OK if avail == "ok" else C_WARN if avail == "--" else C_FAIL
        sandbox_lines = [
            _c(C_DIM, _pad(image_ref, inner)),
            _c(C_TEXT, _pad(f"euid {euid}  root {root}", inner)),
            _c(avail_color, _pad(f"avail {avail}", inner)),
        ]

        attacker = _box(
            "ATTACKER", C_ATTACKER, side_lines(Role.ATTACKER, a_active, C_ATTACKER), NODE_W
        )
        sandbox = _box(f"SANDBOX {guest_id}", C_SANDBOX, sandbox_lines, NODE_W)
        defender = _box(
            "DEFENDER", C_DEFENDER, side_lines(Role.DEFENDER, d_active, C_DEFENDER), NODE_W
        )

        left = _c(C_ATTACKER if a_active else C_DIM, "───>" if a_active else "┄┄┄>")
        right = _c(C_DEFENDER if d_active else C_DIM, "<───" if d_active else "<┄┄┄")
        blank = "    "

        rows = []
        for i, (a, s, d) in enumerate(zip(attacker, sandbox, defender)):
            link_l = left if i == 3 else blank
            link_r = right if i == 3 else blank
            rows.append(f" {a}  {link_l}  {s}  {link_r}  {d}")

        tape = self._turn_tape(prog)
        rows.append(
            f" {_c(C_DIM, 'turns')}  {tape}  "
            f"{_c(C_TEXT, f'{prog.completed_turns}/{prog.total_turns}')}"
        )
        return "\n".join(rows)

    def _turn_tape(self, prog) -> str:
        cells = []
        for turn in range(prog.total_turns):
            role_letter = "A" if turn % 2 == 0 else "D"
            color = C_ATTACKER if turn % 2 == 0 else C_DEFENDER
            glyph = "█" if turn < prog.completed_turns else "░"
            cells.append(_c(color if turn < prog.completed_turns else C_DIM, f"{role_letter}{glyph}"))
        return "\\[" + "".join(cells) + "]"

    def _render_progress(self) -> str:
        prog = progress(self.snapshot)
        ep_frac = prog.completed_episodes / prog.total_episodes if prog.total_episodes else 0
        turn_frac = prog.completed_turns / prog.total_turns if prog.total_turns else 0
        eta = estimate_eta_s(self.snapshot)
        eta_txt = _fmt_dur(eta) if eta is not None else "--"
        ep_line = (
            f"{_c(C_DIM, 'episodes')}  {_c(C_ACCENT, _bar(ep_frac, 34))}  "
            f"{_c(C_TEXT, f'{prog.completed_episodes}/{prog.total_episodes}')}  "
            f"{_c(C_DIM, f'{int(ep_frac * 100)}%')}   {_c(C_DIM, 'eta ' + eta_txt)}"
        )
        turn_line = (
            f"{_c(C_DIM, 'turns   ')}  {_c(C_DEFENDER, _bar(turn_frac, 34))}  "
            f"{_c(C_TEXT, f'{prog.completed_turns}/{prog.total_turns}')}  "
            f"{_c(C_DIM, f'{int(turn_frac * 100)}%')}"
        )
        return f"{ep_line}\n{turn_line}"

    def _render_footer(self) -> str:
        spec = self._spec
        hints = _c(
            C_DIM,
            "a/s/d expand  l log  click line detail  esc collapse  q quit",
        )
        badge = f"[on {C_ACCENT}][{C_BADGE_FG}] SIM MODE [/][/]"
        left = f"{_c(C_TEXT, 'ultron v0.1.0')}  {_c(C_DIM, f'gen {spec.generation} | {spec.profile_id} | {spec.isolation_backend.value}')}"
        return f"{hints}\n{left}          {badge}"

    def _current_episode_index(self) -> int:
        snapshot = self.snapshot
        if isinstance(snapshot, CompleteSnapshot):
            return snapshot.final_episode.episode_index
        if isinstance(snapshot, FailedSnapshot):
            return len(snapshot.data.prior_episodes)
        return snapshot.episode_index

    def _guest_facts(self) -> tuple[str, str]:
        episode = self._current_episode_index()
        for event in reversed(self.snapshot.data.recent_events):
            if isinstance(event, RestoreEvent) and event.episode_index == episode:
                return event.guest_id, event.image_ref
        return "pending", "pending"

    def _probe_facts(self) -> tuple[str, str, str]:
        episode = self._current_episode_index()
        probe = None
        if isinstance(self.snapshot, CompleteSnapshot):
            probe = self.snapshot.final_episode.probe
        elif isinstance(self.snapshot, ProbingSnapshot):
            probe = self.snapshot.result
        else:
            for event in reversed(self.snapshot.data.recent_events):
                if isinstance(event, ProbeEvent) and event.episode_index == episode:
                    probe = event.result
                    break
        if probe is None:
            return "--", "--", "--"
        return (
            str(probe.guest_attacker_euid),
            "yes" if probe.host_confirmed_root else "no",
            "ok" if probe.availability_ok else "no",
        )

    def _last_tool(self, role: Role) -> str:
        episode = self._current_episode_index()
        for event in reversed(self.snapshot.data.recent_events):
            if (
                isinstance(event, ToolObservedEvent)
                and event.role is role
                and event.episode_index == episode
            ):
                return f"{event.tool.name} {event.tool.args.get('cmd', '')}"
        return "idle"

    def _show_detail(self) -> None:
        if self._detail is None:
            return
        target, payload = self._detail
        detail = self.query_one("#detail", DetailPane)
        detail.add_class("shown")
        if target == "tool" and isinstance(payload, ToolObservedEvent):
            tool = payload.tool
            args = "\n".join(f"   {k}  {v}" for k, v in tool.args.items())
            body = (
                f"{_c(C_ACCENT, 'TOOL')}  {_c(C_TEXT, tool.name)}   {_c(C_DIM, 'seq ' + str(payload.sequence))}\n"
                f"{_c(C_DIM, 'turn ' + str(payload.turn_index) + '  ' + payload.role.value)}   "
                f"{_c(C_OK if tool.exit_code == 0 else C_WARN, 'exit ' + str(tool.exit_code))}   "
                f"{_c(C_DIM, str(tool.duration_ms) + 'ms')}\n\n"
                f"{_c(C_TEXT, 'args')}\n{args}\n\n"
                f"{_c(C_TEXT, 'stdout head')}\n   {tool.stdout_head}\n\n"
                f"{_c(C_TEXT, 'stdout tail')}\n   {tool.stdout_tail}\n\n"
                f"{_c(C_DIM, 'esc back')}"
            )
        elif target == "sandbox":
            guest_id, image_ref = self._guest_facts()
            euid, root, avail = self._probe_facts()
            body = (
                f"{_c(C_SANDBOX, 'SANDBOX ' + guest_id)}\n\n"
                f"{_c(C_TEXT, 'isolation')}  {self._spec.isolation_backend.value}\n"
                f"{_c(C_TEXT, 'image')}      {image_ref}\n"
                f"{_c(C_TEXT, 'euid')}       {euid}\n"
                f"{_c(C_TEXT, 'root')}       {root}\n"
                f"{_c(C_TEXT, 'avail')}      {avail}\n\n"
                f"{_c(C_DIM, 'esc back')}"
            )
        else:
            role = Role.ATTACKER if target == "attacker" else Role.DEFENDER
            body = (
                f"{_c(C_ATTACKER if role is Role.ATTACKER else C_DEFENDER, target.upper())}\n\n"
                f"{_c(C_TEXT, 'last tool')}  {self._last_tool(role)}\n\n"
                f"{_c(C_DIM, 'esc back')}"
            )
        detail.update(_rt(body))


def run_live_job(spec: JobSpec, runner, cases: Iterable) -> JobSnapshot:
    app = SimApp(spec, runner, cases)
    app.run()
    return app.snapshot
