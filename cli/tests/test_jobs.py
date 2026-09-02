from pathlib import Path

import pytest

from ultron.cli.jobs import (
    JobsError,
    SessionState,
    log_path,
    parse_status_output,
)


def test_parse_status_running_and_missing(tmp_path: Path) -> None:
    env = {"ULTRON_TMUX_LOG_DIR": str(tmp_path / "logs")}
    text = "ultron-gen-0\t4321\t0\tbash\nultron-vllm-attacker\tmissing\n"
    items = parse_status_output(text, root=tmp_path, env=env)
    assert len(items) == 2
    assert items[0].name == "ultron-gen-0"
    assert items[0].state is SessionState.RUNNING
    assert items[0].pid == 4321
    assert items[0].command == "bash"
    assert items[1].state is SessionState.MISSING
    assert items[1].log_path == tmp_path / "logs" / "ultron-vllm-attacker.log"


def test_parse_status_empty_list() -> None:
    assert parse_status_output("No Ultron tmux sessions.\n") == ()


def test_parse_status_rejects_garbage() -> None:
    with pytest.raises(JobsError, match="unreadable"):
        parse_status_output("not-a-status-line")


def test_log_path_uses_env(tmp_path: Path) -> None:
    env = {"ULTRON_TMUX_LOG_DIR": str(tmp_path / "custom")}
    assert log_path("ultron-gen-1", env=env) == tmp_path / "custom" / "ultron-gen-1.log"
