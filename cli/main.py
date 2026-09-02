from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultron import __version__
from ultron.cli.demo import make_demo
from ultron.cli.model import JobMeta
from ultron.env.backend import IsolationBackend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ultron-sim",
        description="Full-screen live view of an Ultron guest-gym rollout.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
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
    args = parser.parse_args(argv)
    if args.cmd != "demo":
        parser.error("unknown command")
    if args.episodes < 1 or args.turns_per_side < 1:
        parser.error("--episodes and --turns-per-side must be >= 1")
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
        sys.stderr.write(
            "ultron-sim needs the tui extra. Install with: pip install -e '.[tui]'\n"
            f"{exc}\n"
        )
        return 2
    runner, cases = make_demo(meta, delay_s=args.delay)
    run_live_job(meta, runner, cases, screenshot=args.screenshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
