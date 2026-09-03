from __future__ import annotations

import argparse
from pathlib import Path

from .demo import SCRIPTS
from .job import LiveJob


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ultron live simulation theater.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="scripted job, no GPU or guests")
    demo.add_argument("--script", choices=SCRIPTS, default="full-16")
    demo.add_argument("--ascii", action="store_true")
    demo.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    job = LiveJob.demo(script=args.script)
    if args.ascii:
        print(job.ascii())
        return 0
    if args.screenshot is not None:
        try:
            from ultron.cli.tui.app import screenshot_board
        except ImportError:
            print("install the TUI extra: pip install 'ultron[tui]'")
            return 2
        screenshot_board(job.board_at_end(), args.screenshot)
        return 0
    job.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
