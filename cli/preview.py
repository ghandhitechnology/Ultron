"""Serve the live battle TUI as a browser preview."""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from ultron.cli.responses import default_response_directory

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8008


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def add_preview_parser(sub: argparse._SubParsersAction) -> None:
    preview = sub.add_parser("preview", help="Serve the demo battle view in a web browser.")
    preview.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Web server bind address, default %(default)s.",
    )
    preview.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help="Web server port, default %(default)s.",
    )
    preview.add_argument("--episodes", type=int, default=2)
    preview.add_argument("--turns-per-side", type=int, default=2)
    preview.add_argument("--generation", type=int, default=0)
    preview.add_argument("--profile", default="web")
    preview.add_argument("--delay", type=float, default=0.12)
    preview.add_argument(
        "--responses-dir",
        type=Path,
        default=default_response_directory(),
        help="Directory for per-run response JSON files, default data/responses.",
    )


def preview_command(args: argparse.Namespace, python: str = sys.executable) -> str:
    """Shell command textual-serve runs for each browser session."""
    return shlex.join(
        [
            python,
            "-m",
            "ultron.cli.preview_app",
            "--episodes", str(args.episodes),
            "--turns-per-side", str(args.turns_per_side),
            "--generation", str(args.generation),
            "--profile", args.profile,
            "--delay", str(args.delay),
            "--responses-dir", str(args.responses_dir),
        ]
    )


def run_preview(args: argparse.Namespace) -> int:
    problem = _validate(args)
    if problem:
        sys.stderr.write(f"{problem}\n")
        return 2
    try:
        from textual_serve.server import Server
    except ImportError as exc:
        sys.stderr.write(_web_install_hint(exc))
        return 2
    server = Server(
        preview_command(args),
        host=args.host,
        port=args.port,
        title="ultron preview",
    )
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    return 0


def _validate(args: argparse.Namespace) -> str | None:
    if args.episodes < 1:
        return "--episodes must be >= 1"
    if args.turns_per_side < 1:
        return "--turns-per-side must be >= 1"
    if args.generation < 0:
        return "--generation must be >= 0"
    if args.delay < 0:
        return "--delay must be >= 0"
    return None


def _web_install_hint(exc: ImportError) -> str:
    return (
        "ultron preview needs the web extra. Install with: pip install -e '.[web]'\n"
        f"{exc}\n"
    )
