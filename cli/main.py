from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultron import __version__
from ultron.cli.catalog import CatalogError, GymPlan, resolve_pack
from ultron.cli.demo import make_demo
from ultron.cli.model import JobMeta
from ultron.env.backend import IsolationBackend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manage Ultron generations, jobs, tests, and live guest-gym rollouts.",
    )
    parser.add_argument(
        "--family",
        help="Base-model family for console launches: qwen-4b, qwen-8b, or gemma.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print running tmux jobs and recent logs without opening the TUI.",
    )
    parser.add_argument(
        "--session",
        help="With --check, show only this session. Without --check, open it in the progress view.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=50,
        help="Log lines to show with --check (default 50).",
    )
    sub = parser.add_subparsers(dest="cmd")
    demo = sub.add_parser("demo", help="Run a fake episode loop in the live TUI.")
    demo.add_argument("--episodes", type=int, default=2)
    demo.add_argument("--turns-per-side", type=int, default=2)
    demo.add_argument("--generation", type=int, default=0)
    demo.add_argument("--profile", default="web")
    demo.add_argument("--delay", type=float, default=0.12)
    demo.add_argument(
        "--screenshot",
        type=Path,
        help="Write a Textual SVG screenshot after the demo job finishes.",
    )
    console = sub.add_parser("console", help="Experiment control TUI for jobs, tests, and results.")
    console.add_argument(
        "--family",
        help="Base-model family for console launches: qwen-4b, qwen-8b, or gemma.",
    )
    console.add_argument("--check", action="store_true", help="Same as top-level --check.")
    console.add_argument("--session", help="Open this session in the progress view.")
    console.add_argument("--tail", type=int, default=50)
    check = sub.add_parser("check", help="Print running tmux jobs and recent logs.")
    check.add_argument("--session", help="Show only this session.")
    check.add_argument("--tail", type=int, default=50)
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return _run_demo(args)
    if getattr(args, "check", False):
        return _run_check(session=args.session, tail=args.tail)
    if args.cmd == "check":
        return _run_check(session=args.session, tail=args.tail)
    if args.cmd not in {None, "console"}:
        parser.error("unknown command")
    family = getattr(args, "family", None)
    session = getattr(args, "session", None)
    if session:
        return _run_console(family=family, initial_session=session)
    return _run_smart_console(family=family)


def _run_demo(args: argparse.Namespace) -> int:
    if args.episodes < 1 or args.turns_per_side < 1:
        sys.stderr.write("--episodes and --turns-per-side must be >= 1\n")
        return 2
    if args.generation < 0:
        sys.stderr.write("--generation must be >= 0\n")
        return 2
    if args.delay < 0:
        sys.stderr.write("--delay must be >= 0\n")
        return 2
    meta = JobMeta(
        generation=args.generation,
        profile_id=args.profile,
        isolation=IsolationBackend.DOCKER,
        episodes_planned=args.episodes,
        turns_per_side=args.turns_per_side,
        version=__version__,
        snapshot_sha256="demo-sha",
    )
    try:
        from ultron.cli.tui import run_live_job
    except ImportError as exc:
        sys.stderr.write(_tui_install_hint(exc))
        return 2
    runner, cases = make_demo(meta, delay_s=args.delay)
    run_live_job(meta, runner, cases, screenshot=args.screenshot)
    return 0


def _run_check(*, session: str | None, tail: int) -> int:
    if tail < 0:
        sys.stderr.write("--tail must be >= 0\n")
        return 2
    try:
        from ultron.cli.jobs import JobsError, SessionState, list_sessions, read_logs, session_status
    except ImportError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    try:
        if session:
            info = session_status(session)
            pid = "-" if info.pid is None else str(info.pid)
            sys.stdout.write(f"{info.name}\t{info.state.value}\t{pid}\t{info.command}\n")
            if info.state is not SessionState.RUNNING:
                sys.stdout.write(f"{info.name} is {info.state.value}\n")
                return 1
            _print_logs(session, tail, read_logs)
            return 0
        sessions = list_sessions()
    except JobsError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    running = [item for item in sessions if item.state is SessionState.RUNNING]
    if not running:
        if not sessions:
            sys.stdout.write("no tmux jobs\n")
        else:
            for item in sessions:
                pid = "-" if item.pid is None else str(item.pid)
                sys.stdout.write(f"{item.name}\t{item.state.value}\t{pid}\t{item.command}\n")
            sys.stdout.write("no running jobs\n")
        return 1
    for item in running:
        pid = "-" if item.pid is None else str(item.pid)
        sys.stdout.write(f"{item.name}\t{item.state.value}\t{pid}\t{item.command}\n")
    for item in running:
        _print_logs(item.name, tail, read_logs)
    return 0


def _print_logs(session: str, tail: int, read_logs) -> None:
    from ultron.cli.jobs import JobsError

    if tail == 0:
        return
    sys.stdout.write(f"--- {session} last {tail} ---\n")
    try:
        text = read_logs(session, tail=tail)
    except JobsError as exc:
        sys.stdout.write(f"{exc}\n")
        return
    if not text.strip():
        sys.stdout.write("(empty log)\n")
        return
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


def _run_smart_console(*, family: str | None = None) -> int:
    try:
        from ultron.cli.jobs import running_sessions
    except ImportError:
        return _run_console(family=family)
    try:
        running = running_sessions()
    except Exception:
        return _run_console(family=family)
    if not running:
        return _run_console(family=family)
    if len(running) == 1:
        return _run_console(family=family, initial_session=running[0].name)
    return _run_console(family=family, initial_view="jobs")


def _run_console(*, family: str | None = None, initial_view: str | None = None, initial_session: str | None = None) -> int:
    try:
        resolve_pack(family)
    except CatalogError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    try:
        from ultron.cli.console import run_console
        from ultron.cli.tui import run_live_job
    except ImportError as exc:
        sys.stderr.write(_tui_install_hint(exc))
        return 2
    while True:
        result = run_console(family=family, initial_view=initial_view, initial_session=initial_session)
        initial_view = None
        initial_session = None
        if result is None:
            return 0
        if not isinstance(result, GymPlan):
            return 0
        runner, cases = make_demo(result.meta, delay_s=result.delay_s)
        run_live_job(result.meta, runner, cases)


def _tui_install_hint(exc: ImportError) -> str:
    return (
        "ultron needs the tui extra. Install with: pip install -e '.[tui]'\n"
        f"{exc}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
