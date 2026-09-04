from pathlib import Path
from unittest.mock import patch

from ultron.cli.jobs import SessionInfo, SessionState
from ultron.cli.main import main


def _info(name: str, state: SessionState) -> SessionInfo:
    return SessionInfo(name=name, state=state, pid=4321 if state is SessionState.RUNNING else None, command="bash run", log_path=Path(f"/tmp/{name}.log"))


def test_demo_rejects_negative_generation() -> None:
    assert main(["demo", "--generation", "-1"]) == 2


def test_demo_rejects_zero_episodes() -> None:
    assert main(["demo", "--episodes", "0"]) == 2


def test_console_rejects_unknown_family() -> None:
    assert main(["--family", "llama-8b"]) == 2
    assert main(["console", "--family", "llama-8b"]) == 2


def test_check_reports_no_jobs(capsys) -> None:
    with patch("ultron.cli.jobs.list_sessions", return_value=()):
        assert main(["--check"]) == 1
    assert "no tmux jobs" in capsys.readouterr().out


def test_check_lists_running(capsys) -> None:
    items = (_info("ultron-gen-0", SessionState.RUNNING),)
    with patch("ultron.cli.jobs.list_sessions", return_value=items):
        with patch("ultron.cli.jobs.read_logs", return_value="a\nb"):
            assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "ultron-gen-0\trunning" in out


def test_bare_console_opens_single_running() -> None:
    import ultron.cli.main as m

    with patch("ultron.cli.jobs.running_sessions", return_value=(_info("ultron-gen-0", SessionState.RUNNING),)):
        with patch.object(m, "_run_console", return_value=0) as rc:
            assert main([]) == 0
            rc.assert_called_once_with(family=None, initial_session="ultron-gen-0")


def test_bare_console_opens_jobs_for_many() -> None:
    import ultron.cli.main as m

    items = (_info("ultron-gen-0", SessionState.RUNNING), _info("ultron-gen-1", SessionState.RUNNING))
    with patch("ultron.cli.jobs.running_sessions", return_value=items):
        with patch.object(m, "_run_console", return_value=0) as rc:
            assert main([]) == 0
            rc.assert_called_once_with(family=None, initial_view="jobs")
