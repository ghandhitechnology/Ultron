"""Child process for `ultron preview`: drive one demo job in the battle TUI.

textual-serve launches this module once per browser session with the web driver
configured through the environment, so a plain `App.run()` serves that session.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultron import __version__
from ultron.cli.model import JobMeta
from ultron.cli.responses import default_response_directory
from ultron.env.backend import IsolationBackend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one preview demo job in the live battle view.")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--turns-per-side", type=int, default=2)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--profile", default="web")
    parser.add_argument("--delay", type=float, default=0.12)
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=default_response_directory(),
    )
    args = parser.parse_args(argv)
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
        from ultron.cli.demo import make_demo
        from ultron.cli.tui import run_live_job
    except ImportError as exc:
        sys.stderr.write(_tui_install_hint(exc))
        return 2
    runner, cases = make_demo(meta, delay_s=args.delay)
    snapshot = run_live_job(meta, runner, cases, responses_dir=args.responses_dir)
    if snapshot.responses_path:
        sys.stderr.write(f"Responses saved to {snapshot.responses_path}\n")
    if snapshot.error:
        sys.stderr.write(f"{snapshot.error}\n")
        return 1
    return 0


def _tui_install_hint(exc: ImportError) -> str:
    return (
        "ultron preview needs the tui extra. Install with: pip install -e '.[tui]'\n"
        f"{exc}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
