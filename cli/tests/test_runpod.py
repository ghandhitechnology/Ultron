from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from ultron.cli.main import main


def completed(argv, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def pod_document(identity: Path, **fields):
    return {
        "id": "pod-1",
        "name": "ultron-train",
        "desiredStatus": "RUNNING",
        "runtimeStatus": "running",
        "gpuCount": 2,
        "env": {"RUNPOD_API_KEY": "must-not-leak"},
        "ssh": {
            "ip": "203.0.113.7",
            "port": 41022,
            "ssh_key": {"path": str(identity), "exists": True},
        },
        **fields,
    }


def remote_status(state="running", exit_code=0):
    return {
        "schema_version": 1,
        "jobs": [{"name": "ultron-gen-0", "state": state}],
        "resources": {"gpu": {"devices": []}},
        "issues": [],
        "errors": [],
        "exit_code": exit_code,
    }


def composite(state: str | None = None, exit_code=2):
    ultron = None if state is None else remote_status(state, exit_code)
    return {
        "schema_version": 1,
        "observed_at": "2026-09-09T00:00:00Z",
        "pod": {"id": "pod-1", "runtimeStatus": "running", "desiredStatus": "RUNNING"},
        "transport": {"state": "connected" if ultron else "unavailable"},
        "ultron": ultron,
        "provider_logs": {"available": False, "entries": []},
        "issues": [],
        "exit_code": exit_code,
    }


def test_status_combines_provider_and_remote_state_without_provider_env(tmp_path, capsys):
    identity = tmp_path / "id_ed25519"
    identity.write_text("private key is never read by the command")
    pod = pod_document(identity)
    remote = remote_status()

    with patch(
        "ultron.cli.runpod._command",
        side_effect=[completed([], stdout=json.dumps(pod)), completed([], stdout=json.dumps(remote))],
    ) as run:
        assert main(["runpod", "status", "pod-1", "--session", "ultron-gen-0", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["pod"]["runtimeStatus"] == "running"
    assert output["ultron"]["jobs"][0]["state"] == "running"
    assert "env" not in output["pod"]
    assert "must-not-leak" not in json.dumps(output)
    ssh_argv = run.call_args_list[1].args[0]
    assert ssh_argv[0] == "ssh"
    assert "BatchMode=yes" in ssh_argv
    assert "ultron-gen-0" in ssh_argv[-1]
    assert str(identity) in ssh_argv


def test_status_uses_runpod_system_logs_when_ssh_is_unavailable(tmp_path, capsys):
    identity = tmp_path / "missing"
    pod = pod_document(
        identity,
        runtimeStatus="initializing",
        runtimeStatusReason="awaiting_container",
        ssh={"error": "pod is still initializing"},
    )
    system_log = {"source": "system", "line": "Pulling image", "ts": "2026-09-09T00:00:00Z"}
    with patch(
        "ultron.cli.runpod._command",
        side_effect=[completed([], stdout=json.dumps(pod)), completed([], stdout=json.dumps(system_log) + "\n")],
    ):
        assert main(["runpod", "status", "pod-1", "--json"]) == 2

    output = json.loads(capsys.readouterr().out)
    assert output["transport"]["state"] == "unavailable"
    assert output["provider_logs"]["entries"][0]["line"] == "Pulling image"
    assert output["exit_code"] == 2


def test_stopped_pod_is_a_provider_failure_without_remote_probe(tmp_path, capsys):
    identity = tmp_path / "id"
    pod = pod_document(
        identity,
        desiredStatus="EXITED",
        runtimeStatus="stopped",
        ssh={"error": "pod is stopped"},
    )
    with patch("ultron.cli.runpod._command", return_value=completed([], stdout=json.dumps(pod))) as run:
        assert main(["runpod", "status", "pod-1", "--system-log-tail", "0", "--json"]) == 1
    assert run.call_count == 1
    assert json.loads(capsys.readouterr().out)["pod"]["runtimeStatus"] == "stopped"


def test_watch_retries_transport_misses_records_each_observation_and_completes(tmp_path, capsys):
    record = tmp_path / "observations.jsonl"
    with patch(
        "ultron.cli.runpod.collect_remote_status",
        side_effect=[composite(), composite(), composite("succeeded", 0)],
    ), patch("ultron.cli.runpod.time.sleep"), patch(
        "ultron.cli.runpod.time.monotonic", side_effect=[0, 0, 1, 2]
    ):
        assert main([
            "runpod", "watch", "pod-1", "--session", "ultron-gen-0", "--json",
            "--record", str(record), "--max-misses", "3",
        ]) == 0

    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    retained = [json.loads(line) for line in record.read_text().splitlines()]
    assert [item["watch"]["reason"] for item in output] == ["poll", "poll", "completed"]
    assert [item["watch"]["consecutive_misses"] for item in output] == [1, 2, 0]
    assert len(retained) == 3


def test_watch_stops_after_bounded_monitoring_failures(capsys):
    with patch("ultron.cli.runpod.collect_remote_status", side_effect=[composite(), composite()]), patch(
        "ultron.cli.runpod.time.sleep"
    ), patch("ultron.cli.runpod.time.monotonic", side_effect=[0, 0, 1]):
        assert main([
            "runpod", "watch", "pod-1", "--session", "ultron-gen-0", "--json",
            "--no-record", "--max-misses", "2",
        ]) == 2
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[-1]["watch"]["reason"] == "monitoring_unavailable"


def test_watch_does_not_hide_remote_failure_behind_a_succeeded_tmux_job(capsys):
    with patch(
        "ultron.cli.runpod.collect_remote_status",
        return_value=composite("succeeded", 1),
    ), patch("ultron.cli.runpod.time.monotonic", side_effect=[0, 0]):
        assert main([
            "runpod", "watch", "pod-1", "--session", "ultron-gen-0", "--json", "--no-record",
        ]) == 1
    assert json.loads(capsys.readouterr().out)["watch"]["reason"] == "failed"


def test_logs_reuses_and_updates_a_local_cursor(tmp_path, capsys):
    identity = tmp_path / "id_ed25519"
    identity.write_text("key")
    cursor_file = tmp_path / "cursor.json"
    cursor_file.write_text(json.dumps({
        "schema_version": 1,
        "pod_id": "pod-1",
        "session": "ultron-gen-0",
        "cursor": "old-cursor",
    }))
    pod = pod_document(identity)
    log_result = {"schema_version": 1, "text": "next line\n", "next_cursor": "new-cursor", "has_more": False}
    with patch(
        "ultron.cli.runpod._command",
        side_effect=[completed([], stdout=json.dumps(pod)), completed([], stdout=json.dumps(log_result))],
    ) as run:
        assert main([
            "runpod", "logs", "pod-1", "--session", "ultron-gen-0", "--json",
            "--cursor-file", str(cursor_file),
        ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["text"] == "next line\n"
    assert "old-cursor" in run.call_args_list[1].args[0][-1]
    assert json.loads(cursor_file.read_text())["cursor"] == "new-cursor"


def test_failed_remote_log_command_does_not_advance_cursor(tmp_path, capsys):
    identity = tmp_path / "id_ed25519"
    identity.write_text("key")
    cursor_file = tmp_path / "cursor.json"
    original = {
        "schema_version": 1,
        "pod_id": "pod-1",
        "session": "ultron-gen-0",
        "cursor": "old-cursor",
    }
    cursor_file.write_text(json.dumps(original))
    with patch(
        "ultron.cli.runpod._command",
        side_effect=[
            completed([], stdout=json.dumps(pod_document(identity))),
            completed([], returncode=1, stdout=json.dumps({"next_cursor": "bad"}), stderr="remote failed"),
        ],
    ):
        assert main([
            "runpod", "logs", "pod-1", "--session", "ultron-gen-0", "--json",
            "--cursor-file", str(cursor_file),
        ]) == 2
    assert "remote failed" in json.loads(capsys.readouterr().out)["error"]
    assert json.loads(cursor_file.read_text()) == original


def test_broken_output_pipe_does_not_advance_cursor(tmp_path):
    identity = tmp_path / "id_ed25519"
    identity.write_text("key")
    cursor_file = tmp_path / "cursor.json"
    original = {
        "schema_version": 1,
        "pod_id": "pod-1",
        "session": "ultron-gen-0",
        "cursor": "old-cursor",
    }
    cursor_file.write_text(json.dumps(original))
    log_result = {"schema_version": 1, "text": "unread\n", "next_cursor": "new-cursor", "has_more": False}
    with patch(
        "ultron.cli.runpod._command",
        side_effect=[
            completed([], stdout=json.dumps(pod_document(identity))),
            completed([], stdout=json.dumps(log_result)),
        ],
    ), patch("builtins.print", side_effect=BrokenPipeError):
        assert main([
            "runpod", "logs", "pod-1", "--session", "ultron-gen-0", "--json",
            "--cursor-file", str(cursor_file),
        ]) == 0
    assert json.loads(cursor_file.read_text()) == original


def test_cursor_file_must_contain_an_object(tmp_path, capsys):
    identity = tmp_path / "id_ed25519"
    identity.write_text("key")
    cursor_file = tmp_path / "cursor.json"
    cursor_file.write_text("null")
    with patch(
        "ultron.cli.runpod._command",
        return_value=completed([], stdout=json.dumps(pod_document(identity))),
    ):
        assert main([
            "runpod", "logs", "pod-1", "--session", "ultron-gen-0", "--json",
            "--cursor-file", str(cursor_file),
        ]) == 2
    assert "JSON object" in json.loads(capsys.readouterr().out)["error"]


def test_human_status_includes_remote_pipeline_and_monitoring_errors(capsys):
    snapshot = composite("succeeded", 1)
    snapshot["ultron"]["pipelines"] = [{"name": "generation-0", "stages": [{"name": "train", "state": "failed"}]}]
    snapshot["ultron"]["issues"] = [{"code": "stage_failed", "message": "train failed"}]
    with patch("ultron.cli.runpod.collect_remote_status", return_value=snapshot):
        assert main(["runpod", "status", "pod-1"]) == 1
    text = capsys.readouterr().out
    assert "pipeline generation-0: train=failed" in text
    assert "ultron stage_failed: train failed" in text


def test_missing_pod_id_and_unsafe_remote_path_return_structured_errors(capsys, monkeypatch):
    monkeypatch.delenv("ULTRON_RUNPOD_ID", raising=False)
    assert main(["runpod", "status", "--json"]) == 2
    assert "pod ID" in json.loads(capsys.readouterr().out)["error"]

    assert main(["runpod", "status", "pod-1", "--repo", "relative", "--json"]) == 2
    assert "absolute path" in json.loads(capsys.readouterr().out)["error"]


def test_remote_command_quotes_operator_supplied_paths():
    from ultron.cli.runpod import _remote_command

    command = _remote_command("/workspace/project $(touch nope)", None, ["status", "--json"])
    assert command.startswith("bash -lc ")
    assert "cd -- '/workspace/project $(touch nope)'" in shlex_split_once(command)


def shlex_split_once(command: str) -> str:
    import shlex

    outer = shlex.split(command)
    assert outer[:2] == ["bash", "-lc"]
    return outer[2]
