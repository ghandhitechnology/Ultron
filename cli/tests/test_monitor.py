import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ultron.cli import jobs, monitor


@pytest.fixture(autouse=True)
def isolated_jobs(monkeypatch):
    monkeypatch.setattr(jobs, "list_sessions", lambda **kwargs: ())
    monkeypatch.delenv("ULTRON_PIPELINE_STATE_DIR", raising=False)
    monkeypatch.delenv("ULTRON_RESPONSES_DIR", raising=False)


def collect(tmp_path, **kwargs):
    return monitor.collect_status(root=tmp_path, include_gpu=False, **kwargs)


def write_stage(tmp_path, name="train", state="running", **fields):
    directory = tmp_path / "data" / "job-state" / "generation-0"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.{state}"
    path.write_text("\n".join(f"{key}={value}" for key, value in {"attempt": 1, **fields}.items()) + "\n")
    return path


def write_archive(tmp_path, document, run="run-1"):
    path = tmp_path / "data" / "responses" / run / "responses.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize("state,exit_code,expected,code", [
    (jobs.SessionState.RUNNING, None, "running", 0),
    (jobs.SessionState.DEAD, 0, "succeeded", 0),
    (jobs.SessionState.DEAD, 17, "failed", 1),
    (jobs.SessionState.MISSING, None, "missing", 1),
])
def test_target_session_outcomes(tmp_path, monkeypatch, state, exit_code, expected, code):
    info = jobs.SessionInfo("job", state, None, "bash", tmp_path / "missing.log", exit_code)
    monkeypatch.setattr(jobs, "session_status", lambda *args, **kwargs: info)
    snapshot = collect(tmp_path, session="job")
    assert snapshot["jobs"][0]["state"] == expected
    assert monitor.status_exit_code(snapshot) == code
    assert snapshot["exit_code"] == code
    json.dumps(snapshot, allow_nan=False)


def test_monitoring_failure_is_not_missing_session(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise jobs.JobsError("Cannot inspect socket")
    monkeypatch.setattr(jobs, "list_sessions", fail)
    snapshot = collect(tmp_path)
    assert snapshot["jobs"] == []
    assert snapshot["exit_code"] == 2
    assert snapshot["errors"][0]["source"] == "jobs"
    assert not snapshot["issues"]


def test_quiet_log_is_information_and_tail_is_bounded(tmp_path, monkeypatch):
    log = tmp_path / "job.log"
    log.write_bytes(b"x" * (monitor.MAX_LOG_BYTES + 300))
    os.utime(log, (1, 1))
    info = jobs.SessionInfo("job", jobs.SessionState.RUNNING, os.getpid(), "bash", log)
    monkeypatch.setattr(jobs, "list_sessions", lambda **kwargs: (info,))
    snapshot = collect(tmp_path)
    output = snapshot["jobs"][0]["log"]
    assert output["quiet"] is True
    assert output["truncated"] is True
    assert len(output["tail"]) == monitor.MAX_LOG_BYTES
    assert snapshot["issues"][0]["code"] == "log_quiet"
    assert snapshot["exit_code"] == 0


def test_orphaned_running_stage_requires_dead_owner(tmp_path, monkeypatch):
    path = write_stage(tmp_path, pid=987654)
    os.utime(path, (1, 1))
    monkeypatch.setattr(monitor, "_pid_alive", lambda pid: False)
    snapshot = collect(tmp_path)
    stage = snapshot["pipelines"][0]["stages"][0]
    assert stage["state"] == "orphaned"
    assert snapshot["exit_code"] == 1
    assert any(issue["code"] == "stage_orphaned" for issue in snapshot["issues"])


def test_old_stage_with_live_owner_is_running(tmp_path):
    path = write_stage(tmp_path, pid=os.getpid())
    os.utime(path, (1, 1))
    snapshot = collect(tmp_path)
    assert snapshot["pipelines"][0]["stages"][0]["state"] == "running"
    assert snapshot["exit_code"] == 0


def test_legacy_stage_without_owner_does_not_claim_orphaned(tmp_path):
    write_stage(tmp_path)
    snapshot = collect(tmp_path)
    stage = snapshot["pipelines"][0]["stages"][0]
    assert stage["owner_alive"] is None
    assert stage["state"] == "running"
    assert snapshot["exit_code"] == 0


def test_retrying_stage_is_not_terminal_failure(tmp_path):
    write_stage(tmp_path, pid=os.getpid(), state="running", max_attempts=3, status=1)
    path = tmp_path / "data/job-state/generation-0/train.running"
    with path.open("a") as handle:
        handle.write("state=retrying\nretry_at=12345\n")
    snapshot = collect(tmp_path)
    assert snapshot["pipelines"][0]["stages"][0]["state"] == "retrying"
    assert any(issue["code"] == "stage_retrying" for issue in snapshot["issues"])
    assert snapshot["exit_code"] == 0


def test_target_ignores_unrelated_and_legacy_stage_failures(tmp_path, monkeypatch):
    info = jobs.SessionInfo("current", jobs.SessionState.RUNNING, os.getpid(), "bash", tmp_path / "missing.log")
    monkeypatch.setattr(jobs, "session_status", lambda *args, **kwargs: info)
    write_stage(tmp_path, name="other", state="failed", status=1, session="other")
    write_stage(tmp_path, name="legacy", state="failed", status=2)
    snapshot = collect(tmp_path, session="current")
    assert not any(stage["in_scope"] for stage in snapshot["pipelines"][0]["stages"])
    assert snapshot["exit_code"] == 0
    write_stage(tmp_path, name="current", state="failed", status=1, session="current")
    assert collect(tmp_path, session="current")["exit_code"] == 1


def test_newest_state_wins_over_leftover_failed_file(tmp_path):
    old = write_stage(tmp_path, state="failed", status=1)
    os.utime(old, (1, 1))
    write_stage(tmp_path, state="running", pid=os.getpid())
    snapshot = collect(tmp_path)
    assert len(snapshot["pipelines"][0]["stages"]) == 1
    assert snapshot["pipelines"][0]["stages"][0]["state"] == "running"
    assert snapshot["exit_code"] == 0


@pytest.mark.parametrize("text", ["garbage", "attempt=bad\n", "pid=0\n", "fingerprint=123\n", "state=alien\n"])
def test_corrupt_stage_reports_monitoring_error(tmp_path, text):
    path = write_stage(tmp_path)
    path.write_text(text)
    snapshot = collect(tmp_path)
    assert snapshot["exit_code"] == 2
    assert snapshot["errors"]


def test_stage_read_is_bounded(tmp_path):
    path = write_stage(tmp_path)
    path.write_bytes(b"x" * (monitor.MAX_STATE_BYTES + 1))
    snapshot = collect(tmp_path)
    assert snapshot["exit_code"] == 2
    assert "exceeds" in snapshot["errors"][0]["message"]


def test_directory_scan_is_bounded_and_reports_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "MAX_DIRECTORY_ENTRIES", 3)
    directory = tmp_path / "data/job-state"
    directory.mkdir(parents=True)
    for index in range(10):
        (directory / f"{index}.done").write_text("status=0\n")
    snapshot = collect(tmp_path)
    assert len(snapshot["pipelines"][0]["stages"]) <= 3
    assert snapshot["pipeline_scan"]["truncated"] is True
    assert any(issue["code"] == "state_scan_truncated" for issue in snapshot["issues"])


def test_response_history_has_bounded_previews_and_does_not_fail_job(tmp_path):
    path = write_archive(tmp_path, {"status": "error", "error": "old failure", "responses": [
        {"response_id": str(index), "status": "error", "text": "a" * 1000} for index in range(8)
    ]})
    path.with_name("response-events.jsonl").write_text("a journal deliberately not parsed")
    snapshot = collect(tmp_path)
    run = snapshot["responses"]["runs"][0]
    assert run["response_count"] == 8
    assert run["counts"] == {"error": 8}
    assert len(run["previews"]) == 3
    assert len(run["previews"][0]["text"]) == monitor.MAX_PREVIEW_CHARS
    assert run["previews"][0]["truncated"] is True
    assert run["journal_replayed"] is False
    assert run["journal_present"] is True
    assert snapshot["exit_code"] == 0


def test_oversized_archive_is_explicitly_omitted(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "MAX_RESPONSE_BYTES", 10)
    write_archive(tmp_path, {"responses": [], "status": "running"})
    snapshot = collect(tmp_path)
    run = snapshot["responses"]["runs"][0]
    assert run["available"] is False
    assert run["reason"] == "snapshot_exceeds_read_limit"
    assert snapshot["responses"]["truncated"] is True


def test_corrupt_archive_reports_error(tmp_path):
    path = write_archive(tmp_path, {})
    path.write_text('{"responses": [')
    assert collect(tmp_path)["exit_code"] == 2


def test_no_gpu_is_unavailable_not_failed(tmp_path):
    with patch.object(monitor.shutil, "which", return_value=None):
        snapshot = monitor.collect_status(root=tmp_path)
    assert snapshot["resources"]["gpu"]["reason"] == "nvidia-smi_not_found"
    assert snapshot["exit_code"] == 0


def test_gpu_timeout_is_bounded_and_truthful():
    with patch.object(monitor.shutil, "which", return_value="/usr/bin/nvidia-smi"), patch.object(
        monitor.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 3)
    ) as run:
        result = monitor._gpu()
    assert result["available"] is False
    assert result["reason"] == "query_timeout"
    assert run.call_args.kwargs["timeout"] == monitor.GPU_TIMEOUT_SECONDS


def test_gpu_parses_unavailable_fields_without_nan():
    def fake_run(command, **kwargs):
        kwargs["stdout"].write(b"0, GPU, 83, 1000, 24000, 61\n1, GPU2, N/A, nan, 24000, [Not Supported]\n")
        return subprocess.CompletedProcess(command, 0)
    with patch.object(monitor.shutil, "which", return_value="/usr/bin/nvidia-smi"), patch.object(monitor.subprocess, "run", fake_run):
        result = monitor._gpu()
    assert result["devices"][0]["utilization_percent"] == 83
    assert result["devices"][1]["memory_used_mib"] is None
    assert result["devices"][1]["temperature_c"] is None
    json.dumps(result, allow_nan=False)


def test_symlinked_directories_are_not_scanned(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "train.failed").write_text("status=1\n")
    directory = tmp_path / "data/job-state"
    directory.mkdir(parents=True)
    (directory / "escape").symlink_to(outside, target_is_directory=True)
    assert collect(tmp_path)["pipelines"] == []


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_quiet_threshold_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="stale_seconds"):
        collect(tmp_path, stale_seconds=value)


def test_latest_pointer_bypasses_history_scan_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "MAX_DIRECTORY_ENTRIES", 3)
    for index in range(8):
        write_archive(tmp_path, {"responses": [], "status": "complete"}, run=f"old-{index}")
    current = write_archive(tmp_path, {"responses": [], "status": "running", "meta": {"generation": 9}}, run="current")
    directory = current.parent.parent
    (directory / ".latest-attacker.json").write_text(json.dumps({"path": "current/responses.json"}))
    snapshot = collect(tmp_path)
    run = snapshot["responses"]["runs"][0]
    assert run["run_id"] == "current"
    assert run["generation"] == 9
    assert run["latest_for_role"] == "attacker"
    assert snapshot["responses"]["truncated"] is True


def test_latest_pointer_cannot_read_outside_archive_directory(tmp_path):
    outside = tmp_path / "outside" / "responses.json"
    outside.parent.mkdir()
    outside.write_text('{"responses": [], "status": "running"}')
    directory = tmp_path / "data/responses"
    directory.mkdir(parents=True)
    (directory / ".latest-attacker.json").write_text(json.dumps({"path": str(outside)}))
    snapshot = collect(tmp_path)
    assert snapshot["responses"]["runs"] == []
    assert "escapes" in snapshot["errors"][0]["message"]
    assert snapshot["exit_code"] == 2


def test_response_directory_can_be_one_run(tmp_path):
    path = write_archive(tmp_path, {"responses": [], "status": "complete"})
    snapshot = collect(tmp_path, responses_dir=path.parent)
    assert snapshot["responses"]["runs"][0]["run_id"] == path.parent.name


def test_current_manifest_excludes_failure_from_previous_invocation(tmp_path):
    old = write_stage(tmp_path, state="failed", status=1, run_id="old-run")
    (old.parent / ".pipeline").write_text("pipeline=generation-0\nrun_id=current-run\npid=123\n")
    snapshot = collect(tmp_path)
    pipeline = snapshot["pipelines"][0]
    assert pipeline["run_id"] == "current-run"
    assert pipeline["stages"][0]["historical"] is True
    assert pipeline["stages"][0]["in_scope"] is False
    assert snapshot["exit_code"] == 0
    write_stage(tmp_path, state="failed", status=1, run_id="current-run")
    assert collect(tmp_path)["exit_code"] == 1


def test_pipeline_manifest_is_reported_before_first_stage(tmp_path):
    directory = tmp_path / "data/job-state/pipeline"
    directory.mkdir(parents=True)
    (directory / ".pipeline").write_text("run_id=current-run\npipeline=pipeline\n")
    snapshot = collect(tmp_path)
    assert snapshot["pipelines"][0]["run_id"] == "current-run"
    assert snapshot["pipelines"][0]["stages"] == []


def test_state_disappearing_during_transition_is_not_monitor_error(tmp_path, monkeypatch):
    path = write_stage(tmp_path, pid=os.getpid())
    original = monitor._read
    def disappear(target, *args, **kwargs):
        if target == path:
            raise FileNotFoundError(target)
        return original(target, *args, **kwargs)
    monkeypatch.setattr(monitor, "_read", disappear)
    snapshot = collect(tmp_path)
    assert snapshot["errors"] == []
    assert snapshot["exit_code"] == 0


def test_dead_job_with_unknown_exit_does_not_report_success(tmp_path, monkeypatch):
    info = jobs.SessionInfo("job", jobs.SessionState.DEAD, 123, "bash", tmp_path / "missing.log")
    monkeypatch.setattr(jobs, "list_sessions", lambda **kwargs: (info,))
    snapshot = collect(tmp_path)
    assert snapshot["jobs"][0]["state"] == "exited"
    assert snapshot["exit_code"] == 2


def test_all_storage_locations_are_measured(tmp_path):
    state = tmp_path / "state-volume"
    response = tmp_path / "response-volume"
    state.mkdir()
    response.mkdir()
    snapshot = collect(tmp_path, state_dir=state, responses_dir=response)
    assert {item["path"] for item in snapshot["resources"]["disks"]} == {str(tmp_path), str(state), str(response)}
