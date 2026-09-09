import argparse
import sys
from unittest.mock import patch

from ultron.cli.main import main
from ultron.cli.preview import DEFAULT_HOST, DEFAULT_PORT, add_preview_parser, preview_command, run_preview


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    add_preview_parser(sub)
    return parser.parse_args(argv)


def test_preview_defaults_bind_all_interfaces() -> None:
    args = _args(["preview"])
    assert args.host == DEFAULT_HOST == "0.0.0.0"
    assert args.port == DEFAULT_PORT == 8008


def test_preview_command_launches_child_module() -> None:
    argv = _args(["preview", "--episodes", "3", "--responses-dir", "/tmp/r dir"])
    command = preview_command(argv, python="/opt/venv/bin/python")
    assert command.startswith("/opt/venv/bin/python -m ultron.cli.preview_app")
    assert "--episodes 3" in command
    assert "'/tmp/r dir'" in command


def test_preview_rejects_invalid_options() -> None:
    assert main(["preview", "--episodes", "0"]) == 2
    assert main(["preview", "--turns-per-side", "0"]) == 2
    assert main(["preview", "--generation", "-1"]) == 2
    assert main(["preview", "--delay", "-0.5"]) == 2


def test_preview_requires_web_extra() -> None:
    with patch.dict(sys.modules, {"textual_serve.server": None}):
        assert run_preview(_args(["preview"])) == 2
