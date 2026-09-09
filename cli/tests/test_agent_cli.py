import json
from unittest.mock import patch

import pytest

from ultron.cli.main import main


def _snapshot(state="running", code=0):
    return {"schema_version": 1, "observed_at": "2026-09-09T00:00:00Z",
            "jobs": [{"name": "ultron-test", "state": state, "exit_code": None if state == "running" else code}],
            "issues": [], "errors": []}


@pytest.mark.parametrize("state,code,reason", [
    ("succeeded", 0, "completed"), ("failed", 1, "failed"),
    ("missing", 1, "failed"), ("exited", 0, "unknown_exit"),
])
def test_watch_stops_on_terminal_target(state, code, reason, capsys):
    with patch("ultron.cli.monitor.collect_status", return_value=_snapshot(state, code)):
        with patch("ultron.cli.monitor.status_exit_code", return_value=code):
            assert main(["watch", "--session", "ultron-test", "--json", "--no-gpu"]) == (2 if state == "exited" else code)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["watch"]["reason"] == reason


def test_watch_timeout_is_distinct_from_success_and_does_not_stop_job(capsys):
    with patch("ultron.cli.monitor.collect_status", side_effect=lambda **kw: _snapshot()) as collect:
        with patch("ultron.cli.monitor.status_exit_code", return_value=0):
            with patch("ultron.cli.agent_cli.time.monotonic", side_effect=[0, 0, 1]):
                with patch("ultron.cli.agent_cli.time.sleep") as sleep:
                    assert main(["watch", "--session", "ultron-test", "--timeout", "1", "--interval", "0.5", "--json"]) == 124
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["watch"]["reason"] for r in records] == ["poll", "timeout"]
    assert records[-1]["watch"]["exit_code"] == 124
    assert collect.call_count == 1
    sleep.assert_called_once_with(0.5)


def test_status_scopes_collection_and_returns_json(capsys, tmp_path):
    with patch("ultron.cli.monitor.collect_status", return_value=_snapshot()) as collect:
        with patch("ultron.cli.monitor.status_exit_code", return_value=0):
            assert main(["status", "--session", "ultron-test", "--json", "--state-dir", str(tmp_path), "--no-gpu"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1
    assert collect.call_args.kwargs["state_dir"] == tmp_path
    assert collect.call_args.kwargs["include_gpu"] is False


@pytest.mark.parametrize("argv", [
    ["watch", "--timeout", "nan"], ["watch", "--interval", "0"],
    ["status", "--stale-seconds", "inf"], ["status", "--session", "../../secret"],
    ["job", "start", "ultron-test"],
])
def test_invalid_inputs_return_structured_errors(argv, capsys):
    # job options precede the positional session because remaining argv is the command.
    if argv[:2] == ["job", "start"]:
        argv = argv[:2] + ["--json"] + argv[2:]
    else:
        argv += ["--json"]
    assert main(argv) == 2
    assert json.loads(capsys.readouterr().out)["error"]


def test_job_start_preserves_argv_without_shell_expansion(capsys):
    from ultron.cli.jobs import Started

    command = ["python", "-c", "print('$HOME; `true`')"]
    with patch("ultron.cli.jobs.start_session", return_value=Started("ultron-test")) as start:
        assert main(["job", "start", "--json", "ultron-test", "--", *command]) == 0
    start.assert_called_once_with("ultron-test", command, extra_env={"PYTHONUNBUFFERED": "1"})
    assert json.loads(capsys.readouterr().out)["result"] == "started"


def test_job_error_uses_single_json_object(capsys):
    from ultron.cli.jobs import JobsError

    with patch("ultron.cli.jobs.restart_session", side_effect=JobsError("Session is still running")):
        assert main(["job", "restart", "--json", "ultron-test"]) == 2
    output = capsys.readouterr()
    assert json.loads(output.out)["error"] == "Session is still running"
    assert output.err == ""
