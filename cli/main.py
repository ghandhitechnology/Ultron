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
        prog="ultron-sim",
        description="Manage Ultron generations, jobs, tests, and live guest-gym rollouts.",
    )
    parser.add_argument(
        "--family",
        help="Base-model family for console launches: qwen-4b, qwen-8b, gemma, or gemma-abliterated.",
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
        help="Base-model family for console launches: qwen-4b, qwen-8b, gemma, or gemma-abliterated.",
    )
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return _run_demo(args)
    if args.cmd not in {None, "console"}:
        parser.error("unknown command")
    return _run_console(family=args.family)


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


def _run_console(*, family: str | None = None) -> int:
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
        result = run_console(family=family)
        if result is None:
            return 0
        if not isinstance(result, GymPlan):
            return 0
        runner, cases = make_demo(result.meta, delay_s=result.delay_s)
        run_live_job(result.meta, runner, cases)


def _tui_install_hint(exc: ImportError) -> str:
    return (
        "ultron-sim needs the tui extra. Install with: pip install -e '.[tui]'\n"
        f"{exc}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
