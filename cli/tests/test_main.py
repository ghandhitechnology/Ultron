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


def test_battle_command_opens_requested_archive(tmp_path) -> None:
    import pytest
    pytest.importorskip("textual")
    with patch("ultron.cli.battle.run_battle") as run:
        assert main(["battle", str(tmp_path)]) == 0
    run.assert_called_once_with(tmp_path, screenshot=None)


def test_battle_uses_shared_response_directory(tmp_path, monkeypatch) -> None:
    import pytest
    pytest.importorskip("textual")
    monkeypatch.setenv("ULTRON_RESPONSES_DIR", str(tmp_path))
    with patch("ultron.cli.battle.run_battle") as run:
        assert main(["battle"]) == 0
    run.assert_called_once_with(tmp_path, screenshot=None)


def test_console_returns_after_battle_view(tmp_path) -> None:
    import pytest
    pytest.importorskip("textual")
    from ultron.cli.catalog import BattlePlan
    from ultron.cli.main import _run_console

    with patch("ultron.cli.console.run_console", side_effect=[BattlePlan(tmp_path), None]):
        with patch("ultron.cli.main._run_battle", return_value=0) as battle:
            assert _run_console() == 0
    battle.assert_called_once_with(tmp_path)


def test_demo_reports_failed_run_and_response_path(tmp_path, capsys) -> None:
    import pytest
    pytest.importorskip("textual")
    from types import SimpleNamespace
    from ultron.cli.model import Phase

    result = SimpleNamespace(phase=Phase.FAILED, error="Provider disconnected", responses_path=str(tmp_path / "responses.json"))
    with patch("ultron.cli.tui.run_live_job", return_value=result) as run:
        assert main(["demo", "--responses-dir", str(tmp_path)]) == 1
    assert run.call_args.kwargs["responses_dir"] == tmp_path
    output = capsys.readouterr()
    assert str(tmp_path / "responses.json") in output.out
    assert "Provider disconnected" in output.err


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
