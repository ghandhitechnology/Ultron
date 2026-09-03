from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NoReturn


class ConsoleScreen(str, Enum):
    CATALOG = "catalog"
    JOBS = "jobs"
    RESULTS = "results"
    RUN = "run"


class ConsoleFocus(str, Enum):
    CATALOG = "catalog"
    FAMILY = "family"
    FORM = "form"
    JOBS = "jobs"
    JOB_LOG = "job_log"
    RESULTS = "results"
    REVIEW = "review"
    RUN = "run"


class GymFocus(str, Enum):
    ARENA = "arena"
    DETAIL = "detail"
    LOG = "log"


@dataclass(frozen=True)
class Shortcut:
    key: str
    label: str
    action: str


def console_shortcuts(
    screen: ConsoleScreen,
    focus: ConsoleFocus,
    *,
    typing: bool = False,
    filled: bool = False,
    has_job: bool = False,
    has_generation: bool = False,
    can_stop: bool = False,
) -> tuple[Shortcut, ...]:
    match screen:
        case ConsoleScreen.CATALOG:
            return _catalog_shortcuts(focus, typing=typing, filled=filled)
        case ConsoleScreen.JOBS:
            return _jobs_shortcuts(focus, has_job=has_job)
        case ConsoleScreen.RESULTS:
            return _results_shortcuts(focus, has_generation=has_generation)
        case ConsoleScreen.RUN:
            return _run_shortcuts(can_stop=can_stop)
        case _:
            _unreachable(screen)


def gym_shortcuts(*, expanded: str | None, focus: GymFocus) -> tuple[Shortcut, ...]:
    fold = (Shortcut("esc", "fold", "collapse"),) if expanded else ()
    quit_s = (Shortcut("q", "quit", "quit"),)
    match focus:
        case GymFocus.LOG:
            common = (
                Shortcut("t", "tool", "expand:tool"),
                Shortcut("esc", "fold", "collapse") if expanded else Shortcut("a", "attacker", "expand:attacker"),
            )
            return common + quit_s
        case GymFocus.DETAIL:
            return (
                Shortcut("esc", "fold", "collapse"),
                Shortcut("a", "attacker", "expand:attacker"),
                Shortcut("s", "sandbox", "expand:sandbox"),
                Shortcut("d", "defender", "expand:defender"),
                Shortcut("t", "tool", "expand:tool"),
            ) + quit_s
        case GymFocus.ARENA:
            return (
                Shortcut("a", "attacker", "expand:attacker"),
                Shortcut("s", "sandbox", "expand:sandbox"),
                Shortcut("d", "defender", "expand:defender"),
                Shortcut("t", "tool", "expand:tool"),
            ) + fold + quit_s
        case _:
            _unreachable(focus)


def _catalog_shortcuts(focus: ConsoleFocus, *, typing: bool, filled: bool) -> tuple[Shortcut, ...]:
    match focus:
        case ConsoleFocus.FORM:
            run = Shortcut("enter", "run", "confirm")
            nxt = Shortcut("tab", "next field", "focus_next")
            ordered = (run, nxt) if filled else (nxt, run)
            return ordered + (
                Shortcut("esc", "catalog", "show_catalog"),
                Shortcut("click", "quit", "quit"),
            )
        case ConsoleFocus.FAMILY:
            return (
                Shortcut("enter", "pick", "focus_family"),
                Shortcut("esc", "catalog", "show_catalog"),
                Shortcut("a", "actions", "show_catalog"),
                Shortcut("q", "quit", "quit"),
            )
        case ConsoleFocus.CATALOG | ConsoleFocus.JOBS | ConsoleFocus.JOB_LOG | ConsoleFocus.RESULTS | ConsoleFocus.REVIEW | ConsoleFocus.RUN:
            if typing:
                return _catalog_shortcuts(ConsoleFocus.FORM, typing=True, filled=filled)
            return (
                Shortcut("enter", "run", "confirm"),
                Shortcut("m", "model", "focus_family"),
                Shortcut("j", "jobs", "show_jobs"),
                Shortcut("r", "results", "show_results"),
                Shortcut("t", "tests", "focus_tests"),
                Shortcut("q", "quit", "quit"),
            )
        case _:
            _unreachable(focus)


def _jobs_shortcuts(focus: ConsoleFocus, *, has_job: bool) -> tuple[Shortcut, ...]:
    refresh = Shortcut("g", "refresh", "refresh")
    back = Shortcut("esc", "back", "back")
    quit_s = Shortcut("q", "quit", "quit")
    stop = (Shortcut("s", "stop", "stop_job"),) if has_job else ()
    match focus:
        case ConsoleFocus.JOB_LOG:
            return (back, refresh) + stop + (quit_s,)
        case ConsoleFocus.JOBS | ConsoleFocus.CATALOG | ConsoleFocus.FAMILY | ConsoleFocus.FORM | ConsoleFocus.RESULTS | ConsoleFocus.REVIEW | ConsoleFocus.RUN:
            primary = (Shortcut("enter", "logs", "confirm"),) if has_job else ()
            return primary + stop + (refresh, back, quit_s)
        case _:
            _unreachable(focus)


def _results_shortcuts(focus: ConsoleFocus, *, has_generation: bool) -> tuple[Shortcut, ...]:
    refresh = Shortcut("g", "refresh", "refresh")
    back = Shortcut("esc", "back", "back")
    quit_s = Shortcut("q", "quit", "quit")
    match focus:
        case ConsoleFocus.REVIEW:
            return (back, refresh, quit_s)
        case ConsoleFocus.RESULTS | ConsoleFocus.CATALOG | ConsoleFocus.FAMILY | ConsoleFocus.FORM | ConsoleFocus.JOBS | ConsoleFocus.JOB_LOG | ConsoleFocus.RUN:
            primary = (Shortcut("enter", "review", "confirm"),) if has_generation else ()
            return primary + (refresh, back, quit_s)
        case _:
            _unreachable(focus)


def _run_shortcuts(*, can_stop: bool) -> tuple[Shortcut, ...]:
    stop = (Shortcut("s", "stop", "stop_job"),) if can_stop else ()
    return stop + (
        Shortcut("esc", "back", "back"),
        Shortcut("q", "quit", "quit"),
    )


def _unreachable(value: object) -> NoReturn:
    raise ValueError(f"unhandled {value!r}")
