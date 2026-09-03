from __future__ import annotations

import argparse

from ultron.env.backend import IsolationBackend

from .demo import make_demo
from .model import JobSpec

VERSION = "0.1.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ultron-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a deterministic fake rollout")
    demo.add_argument("--episodes", type=int, default=2)
    demo.add_argument("--generation", type=int, default=0)
    demo.add_argument("--profile", default="web")
    demo.add_argument("--turns-per-side", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "demo":
        return 2

    spec = JobSpec(
        version=VERSION,
        generation=args.generation,
        profile_id=args.profile,
        isolation_backend=IsolationBackend.DOCKER,
        episode_count=args.episodes,
        turns_per_side=args.turns_per_side,
    )
    runner, cases = make_demo(spec)

    try:
        from .app import run_live_job
    except ImportError:
        print("install ultron[tui]")
        return 2

    run_live_job(spec, runner, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
