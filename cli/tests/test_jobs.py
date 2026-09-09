import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ultron.cli.jobs import (
    JobsError,
    SessionState,
    list_sessions,
    log_path,
    parse_status_output,
    read_log_chunk,
    session_status,
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


def test_parse_status_reports_exit_outcome(tmp_path: Path) -> None:
    text = "ok\t4321\t1\tbash\t0\nfailed\t4322\t1\tbash\t23\n"
    items = parse_status_output(text, root=tmp_path)

    assert items[0].state is SessionState.DEAD
    assert items[0].exit_code == 0
    assert items[1].state is SessionState.DEAD
    assert items[1].exit_code == 23


def test_parse_status_rejects_invalid_exit_code() -> None:
    with pytest.raises(JobsError, match="exit code"):
        parse_status_output("job\t4321\t1\tbash\tnot-a-number\n")


def test_parse_status_empty_list() -> None:
    assert parse_status_output("No Ultron tmux sessions.\n") == ()


def test_parse_status_rejects_garbage() -> None:
    with pytest.raises(JobsError, match="unreadable"):
        parse_status_output("not-a-status-line")


def test_log_path_uses_env(tmp_path: Path) -> None:
    env = {"ULTRON_TMUX_LOG_DIR": str(tmp_path / "custom")}
    assert log_path("ultron-gen-1", env=env) == tmp_path / "custom" / "ultron-gen-1.log"


def test_read_log_chunk_tails_and_then_reads_only_new_bytes(tmp_path: Path) -> None:
    env = {"ULTRON_TMUX_LOG_DIR": str(tmp_path)}
    path = tmp_path / "job.log"
    path.write_text("first\nsecond\nthird\n")

    initial = read_log_chunk("job", offset=None, max_bytes=13, env=env)
    assert initial.text == "second\nthird\n"
    assert initial.skipped_bytes == 6
    assert initial.next_offset == path.stat().st_size

    with path.open("a") as handle:
        handle.write("fourth\n")
    update = read_log_chunk("job", offset=initial.next_offset, max_bytes=13, env=env)
    assert update.text == "fourth\n"
    assert update.skipped_bytes == 0


def test_read_log_chunk_resets_after_log_is_truncated(tmp_path: Path) -> None:
    env = {"ULTRON_TMUX_LOG_DIR": str(tmp_path)}
    path = tmp_path / "job.log"
    path.write_text("old output\n")
    offset = path.stat().st_size
    path.write_text("new\n")

    chunk = read_log_chunk("job", offset=offset, max_bytes=64, env=env)

    assert chunk.text == "new\n"
    assert chunk.reset is True


def test_list_sessions_reports_a_tmux_helper_timeout(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "tmux_job.sh"
    script.parent.mkdir()
    script.touch()

    with patch(
        "ultron.cli.jobs.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="tmux_job.sh list", timeout=10),
    ):
        with pytest.raises(JobsError, match="list timed out"):
            list_sessions(root=tmp_path)


def test_status_error_is_distinct_from_missing_session() -> None:
    failed = subprocess.CompletedProcess([], 1, stdout="", stderr="Cannot inspect tmux socket")
    with patch("ultron.cli.jobs._run", return_value=failed):
        with pytest.raises(JobsError, match="Cannot inspect tmux socket"):
            session_status("job")
    missing = subprocess.CompletedProcess([], 1, stdout="job\tmissing\n", stderr="")
    with patch("ultron.cli.jobs._run", return_value=missing):
        assert session_status("job").state is SessionState.MISSING
