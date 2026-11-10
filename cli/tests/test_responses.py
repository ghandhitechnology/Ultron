from __future__ import annotations

import json
import os
import threading
from concurrent.futures import CancelledError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ultron.cli.demo import make_demo
from ultron.cli.model import (
    InvalidTransition, JobEnded, JobMeta, ModelResponseDelta, ModelResponseFinished,
    ModelResponseStarted, Phase, ProbeFinished, ProbeStarted, RestoreFinished,
    RestoreStarted, TurnEnded, TurnStarted, apply, initial_snapshot,
)
from ultron.cli.observe import drive_job
from ultron.env.backend import IsolationBackend
from ultron.response_stream import ModelResponse, stream_chat_completion
from ultron.train.adjudicator import ProbeResult
from ultron.train.schema_v1 import Role


def meta():
    return JobMeta(0, "web", IsolationBackend.DOCKER, 1, 1)


def archive(tmp_path):
    paths = list(tmp_path.glob("*/responses.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def test_archive_respects_shared_response_directory(tmp_path, monkeypatch):
    from ultron.cli.responses import ResponseArchive
    monkeypatch.setenv("ULTRON_RESPONSES_DIR", str(tmp_path))
    store = ResponseArchive(meta())
    assert store.path.parent.parent == tmp_path


def test_recent_role_pointers_track_activity_independently(tmp_path):
    from ultron.cli.responses import ResponseArchive

    attacker = ResponseArchive(meta(), tmp_path)
    attacker.record(ModelResponseStarted(0, 0, Role.ATTACKER, "a", "model", 0))
    defender = ResponseArchive(meta(), tmp_path)
    defender.record(ModelResponseStarted(0, 0, Role.DEFENDER, "d", "model", 0))
    attacker.record(ModelResponseDelta(0, 0, Role.ATTACKER, "a", "latest attacker", 1))
    for role, store in (("attacker", attacker), ("defender", defender)):
        pointer = json.loads((tmp_path / f".latest-{role}.json").read_text())
        assert pointer["role"] == role
        assert tmp_path / pointer["path"] == store.path
        assert pointer["run_id"] == store.run_id


def test_demo_persists_every_response_and_allocates_unique_run(tmp_path):
    for _ in range(2):
        runner, cases = make_demo(meta(), delay_s=0, sleep=lambda _: None)
        drive_job(meta(), runner, cases, emit=lambda _: None, clock=lambda: 0, responses_dir=tmp_path)
    paths = list(tmp_path.glob("*/responses.json"))
    assert len(paths) == 2
    for path in paths:
        saved = json.loads(path.read_text())
        assert saved["status"] == "complete"
        assert {r["role"] for r in saved["responses"]} == {"attacker", "defender"}
        assert len(saved["responses"]) == 2
        assert all(r["status"] == "complete" and r["text"].startswith("Demo response.") for r in saved["responses"])


@pytest.mark.parametrize("broken", [False, True])
def test_provider_stream_reaches_snapshot_and_disk_before_completion(tmp_path, broken):
    first_visible = threading.Event()
    release = threading.Event()
    errors = []
    events = []
    snapshot = initial_snapshot(meta(), started_at_s=0)
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(request)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"hello "}}]}\n\n')
            self.wfile.flush()
            release.wait(5)
            if not broken:
                self.wfile.write('data: {"choices":[{"index":0,"delta":{"content":"세계"}}]}\n\ndata: [DONE]\n\n'.encode())
                self.wfile.flush()
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    runner, cases = make_demo(meta(), delay_s=0, sleep=lambda _: None)

    def run_turn(*_):
        stream_chat_completion(endpoint, model="fixture-model", messages=[{"role": "user", "content": "hello"}])
        return []

    runner.run_turn = run_turn

    def emit(event):
        nonlocal snapshot
        snapshot = apply(snapshot, event)
        events.append(event)
        if isinstance(event, ModelResponseDelta):
            first_visible.set()

    def drive():
        try:
            drive_job(meta(), runner, cases, emit=emit, clock=lambda: 0, responses_dir=tmp_path)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=drive)
    worker.start()
    try:
        assert first_visible.wait(5)
        assert worker.is_alive()
        assert snapshot.attacker_response.text == "hello "
        assert snapshot.attacker_response.status == "streaming"
        assert not any(isinstance(e, ModelResponseFinished) for e in events)
        saved = archive(tmp_path)
        assert saved["responses"][0]["text"] == "hello "
        assert saved["responses"][0]["status"] == "streaming"
    finally:
        release.set()
        worker.join(5)
        server.shutdown()
        server.server_close()
        server_thread.join(5)
    assert not worker.is_alive()
    saved = archive(tmp_path)
    assert received[0]["stream"] is True
    assert saved["responses"][0]["model"] == "fixture-model"
    if broken:
        assert len(errors) == 1
        assert "before [DONE]" in str(errors[0])
        assert saved["status"] == "error"
        assert saved["responses"][0]["text"] == "hello "
        assert saved["responses"][0]["status"] == "error"
        assert snapshot.phase is Phase.FAILED
    else:
        assert errors == []
        assert saved["status"] == "complete"
        assert len(saved["responses"]) == 2
        assert all(r["text"] == "hello 세계" for r in saved["responses"])
        assert snapshot.phase is Phase.COMPLETE


def test_cancel_preserves_received_partial_text(tmp_path):
    cancel = threading.Event()
    runner, cases = make_demo(meta(), delay_s=0, sleep=lambda _: None)

    def run_turn(*_):
        with ModelResponse("test") as response:
            response.write("first ")
            cancel.set()
            response.write("last received")
        return []

    runner.run_turn = run_turn
    with pytest.raises(CancelledError):
        drive_job(meta(), runner, cases, emit=lambda _: None, clock=lambda: 0, responses_dir=tmp_path, cancel=cancel)
    saved = archive(tmp_path)
    assert saved["status"] == "cancelled"
    assert saved["responses"][0]["status"] == "cancelled"
    assert saved["responses"][0]["text"] == "first last received"


def test_cancelled_job_does_not_start_guest_operations(tmp_path):
    cancel = threading.Event()
    cancel.set()
    runner, cases = make_demo(meta(), delay_s=0, sleep=lambda _: None)
    restored = []
    runner.restore = lambda *_: restored.append(True)
    with pytest.raises(CancelledError):
        drive_job(meta(), runner, cases, emit=lambda _: None, clock=lambda: 0, responses_dir=tmp_path, cancel=cancel)
    assert restored == []
    assert archive(tmp_path)["status"] == "cancelled"


def test_finished_response_cannot_accept_late_delta():
    snapshot = apply(initial_snapshot(meta(), started_at_s=0), TurnStarted(0, 0, Role.ATTACKER, 0))
    snapshot = apply(snapshot, ModelResponseStarted(0, 0, Role.ATTACKER, "r", "model", 0))
    snapshot = apply(snapshot, ModelResponseFinished(0, 0, Role.ATTACKER, "r", "complete", 0))
    with pytest.raises(InvalidTransition, match="matching active response"):
        apply(snapshot, ModelResponseDelta(0, 0, Role.ATTACKER, "r", "late", 0))


def test_journal_recovers_chunks_newer_than_json_snapshot(tmp_path):
    from ultron.cli.responses import ResponseArchive, load_response_archive

    store = ResponseArchive(meta(), tmp_path)
    store.record(ModelResponseStarted(0, 0, Role.ATTACKER, "r", "model", 0))
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "first ", 0))
    before = store.path.read_bytes()
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "second", 0))
    assert store.path.read_bytes() == before
    assert load_response_archive(store.path)["responses"][0]["text"] == "first second"
    with store.journal_path.open("ab") as handle:
        handle.write(b'{"sequence":4,"text":"' + "한".encode("utf-8")[:2])
    assert load_response_archive(store.path)["responses"][0]["text"] == "first second"


def test_delta_journal_groups_fsyncs_and_completion_is_a_barrier(tmp_path, monkeypatch):
    from ultron.cli import responses
    from ultron.cli.responses import ResponseArchive, load_response_archive

    store = ResponseArchive(meta(), tmp_path, sync_interval_s=60, sync_bytes=1_000_000)
    store.record(ModelResponseStarted(0, 0, Role.ATTACKER, "r", "model", 0))
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "first", 0))
    journal_fd = store._journal._handle.fileno()
    original_fsync = os.fsync
    journal_syncs = []

    def count_journal_fsync(fd):
        if fd == journal_fd:
            journal_syncs.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(responses.os, "fsync", count_journal_fsync)
    for _ in range(100):
        store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "x", 0))
    assert journal_syncs == []

    store.record(ModelResponseFinished(0, 0, Role.ATTACKER, "r", "complete", 1))
    assert journal_syncs == [journal_fd]
    assert load_response_archive(store.path)["responses"][0]["text"] == "first" + "x" * 100

    store.record(JobEnded(1, 1))
    assert journal_syncs == [journal_fd, journal_fd]
    assert load_response_archive(store.path)["status"] == "complete"


def test_zero_interval_provides_strict_per_write_fsync_baseline(tmp_path, monkeypatch):
    from ultron.cli import responses
    from ultron.cli.responses import DurableGroupWriter

    writer = DurableGroupWriter(tmp_path / "strict.jsonl", "w", sync_interval_s=0)
    writer_fd = writer._handle.fileno()
    original_fsync = os.fsync
    syncs = []

    def count_writer_fsync(fd):
        if fd == writer_fd:
            syncs.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(responses.os, "fsync", count_writer_fsync)
    for index in range(10):
        writer.write(f"{index}\n")
    assert syncs == [writer_fd] * 10
    writer.close()


def test_group_writer_syncs_a_burst_once_after_target_interval(tmp_path, monkeypatch):
    from ultron.cli import responses
    from ultron.cli.responses import DurableGroupWriter

    writer = DurableGroupWriter(
        tmp_path / "grouped.jsonl", "w", sync_interval_s=0.02, sync_bytes=1_000_000,
    )
    writer_fd = writer._handle.fileno()
    original_fsync = os.fsync
    synced = threading.Event()
    syncs = []

    def count_writer_fsync(fd):
        if fd == writer_fd:
            syncs.append(fd)
            synced.set()
        return original_fsync(fd)

    monkeypatch.setattr(responses.os, "fsync", count_writer_fsync)
    for index in range(50):
        writer.write(f"{index}\n")
    assert synced.wait(0.5)
    assert syncs == [writer_fd]
    writer.close()


def test_group_writers_share_one_daemon_scheduler(tmp_path):
    from ultron.cli import responses
    from ultron.cli.responses import DurableGroupWriter

    writers = [
        DurableGroupWriter(tmp_path / f"stream-{index}", "w", sync_interval_s=60)
        for index in range(24)
    ]
    for writer in writers:
        writer.write("pending\n")
    scheduler_thread = responses._GROUP_COMMIT_SCHEDULER._thread
    assert scheduler_thread is not None and scheduler_thread.is_alive()
    assert scheduler_thread.daemon
    assert sum(
        thread.name == "response-group-commit" for thread in threading.enumerate()
    ) == 1
    for writer in writers:
        writer.close()


def test_concurrent_delta_writes_have_contiguous_journal_and_complete_text(tmp_path):
    from ultron.cli.responses import ResponseArchive, load_response_archive

    store = ResponseArchive(meta(), tmp_path, sync_interval_s=60, sync_bytes=1_000_000)
    response_ids = [f"r-{index}" for index in range(4)]
    for response_id in response_ids:
        store.record(ModelResponseStarted(0, 0, Role.ATTACKER, response_id, "model", 0))

    def write(response_id):
        for _ in range(50):
            store.record(ModelResponseDelta(0, 0, Role.ATTACKER, response_id, "x", 0))
        store.record(ModelResponseFinished(
            0, 0, Role.ATTACKER, response_id, "complete", 1,
        ))

    workers = [threading.Thread(target=write, args=(response_id,)) for response_id in response_ids]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    store.record(JobEnded(1, 1))

    document = load_response_archive(store.path)
    assert {item["response_id"]: item["text"] for item in document["responses"]} == {
        response_id: "x" * 50 for response_id in response_ids
    }
    entries = [json.loads(line) for line in store.journal_path.read_text().splitlines()]
    assert [entry["sequence"] for entry in entries] == list(range(1, len(entries) + 1))


def test_journal_sync_failure_is_reported_and_archive_stops_accepting_events(tmp_path, monkeypatch):
    from ultron.cli import responses
    from ultron.cli.responses import ResponseArchive

    store = ResponseArchive(meta(), tmp_path, sync_interval_s=60, sync_bytes=1_000_000)
    store.record(ModelResponseStarted(0, 0, Role.ATTACKER, "r", "model", 0))
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "first", 0))
    journal_fd = store._journal._handle.fileno()
    original_fsync = os.fsync

    def fail_journal_fsync(fd):
        if fd == journal_fd:
            raise OSError("No space left on device")
        return original_fsync(fd)

    monkeypatch.setattr(responses.os, "fsync", fail_journal_fsync)
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "pending", 0))
    with pytest.raises(OSError, match="No space left on device"):
        store.record(ModelResponseFinished(0, 0, Role.ATTACKER, "r", "complete", 1))
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "late", 2))
    document = json.loads(store.path.read_text())
    assert document["status"] == "error"
    assert "Response archive write failed" in document["error"]
    assert not document["responses"][0]["text"].endswith("late")


def test_background_sync_failure_reaches_next_archive_write(tmp_path, monkeypatch):
    from ultron.cli import responses
    from ultron.cli.responses import ResponseArchive

    store = ResponseArchive(meta(), tmp_path, sync_interval_s=0.01, sync_bytes=1_000_000)
    store.record(ModelResponseStarted(0, 0, Role.ATTACKER, "r", "model", 0))
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "first", 0))
    journal_fd = store._journal._handle.fileno()
    original_fsync = os.fsync
    failed = threading.Event()

    def fail_background_sync(fd):
        if fd == journal_fd:
            failed.set()
            raise OSError("background disk failure")
        return original_fsync(fd)

    monkeypatch.setattr(responses.os, "fsync", fail_background_sync)
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "pending", 0))
    assert failed.wait(0.5)
    with pytest.raises(OSError, match="background disk failure"):
        store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "late", 1))
    assert json.loads(store.path.read_text())["status"] == "error"


def test_failed_terminal_snapshot_is_overridden_by_durable_error(tmp_path, monkeypatch):
    from ultron.cli import responses
    from ultron.cli.responses import ResponseArchive, load_response_archive

    store = ResponseArchive(meta(), tmp_path, sync_interval_s=60)
    store.record(ModelResponseStarted(0, 0, Role.ATTACKER, "r", "model", 0))
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "complete text", 0))
    original_atomic_write = responses.atomic_write_text
    failed = False

    def fail_complete_snapshot(path, content, **kwargs):
        nonlocal failed
        if path == store.path and '"status": "complete"' in content and not failed:
            failed = True
            raise OSError("snapshot device failure")
        return original_atomic_write(path, content, **kwargs)

    monkeypatch.setattr(responses, "atomic_write_text", fail_complete_snapshot)
    with pytest.raises(OSError, match="snapshot device failure"):
        store.record_many((
            ModelResponseFinished(0, 0, Role.ATTACKER, "r", "complete", 1),
            JobEnded(1, 1),
        ))

    recovered = load_response_archive(store.path)
    assert recovered["status"] == "error"
    assert "Response archive write failed: snapshot device failure" in recovered["error"]
    assert recovered["responses"][0]["text"] == "complete text"
    journal = [json.loads(line)["event"]["kind"] for line in store.journal_path.read_text().splitlines()]
    assert journal[-2:] == ["job_ended", "error"]


def test_archive_preserves_restore_and_turn_timings(tmp_path):
    from ultron.cli.responses import ResponseArchive, load_response_archive

    store = ResponseArchive(meta(), tmp_path, sync_interval_s=60)
    store.record(RestoreStarted(0, "vm-0", "image", IsolationBackend.DOCKER, 0.1))
    store.record(RestoreFinished(0, "vm-0", "127.0.0.1", 0.25, 0.35))
    store.record(TurnStarted(0, 0, Role.ATTACKER, 0.4))
    store.record(TurnEnded(0, 0, Role.ATTACKER, 1.5, 3, 1.9))
    store.record(ProbeStarted(0, 2.0))
    store.record(ProbeFinished(0, ProbeResult(1000, False, True, True, False), 0.2, 2.2))
    store.record(JobEnded(2.3, 2.3))

    events = load_response_archive(store.path)["events"]
    assert [event["kind"] for event in events] == [
        "restore_started", "restore_finished", "turn_started", "turn_ended",
        "probe_started", "probe_finished", "job_ended",
    ]
    assert events[1]["duration_s"] == 0.25
    assert events[3]["duration_s"] == 1.5


def test_external_cancel_closes_archive_before_blocked_worker_returns(tmp_path):
    from ultron.cli.responses import ResponseArchive

    store = ResponseArchive(meta(), tmp_path)
    store.record(ModelResponseStarted(0, 0, Role.ATTACKER, "r", "model", 0))
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "partial", 0))
    store.finish("cancelled", "Battle view closed")
    store.record(ModelResponseDelta(0, 0, Role.ATTACKER, "r", "late", 0))
    saved = archive(tmp_path)
    assert saved["status"] == "cancelled"
    assert saved["responses"][0]["status"] == "cancelled"
    assert saved["responses"][0]["text"] == "partial"


def test_quitting_battle_returns_failure_and_preserves_archive(tmp_path, monkeypatch):
    pytest.importorskip("textual")
    import time
    from ultron.cli import tui

    produced = threading.Event()
    runner, cases = make_demo(meta(), delay_s=0, sleep=lambda _: None)

    def run_turn(*_):
        with ModelResponse("test") as response:
            response.write("visible before quitting")
            produced.set()
            time.sleep(0.05)
            response.write(" next chunk")
        return []

    class ClosedApp:
        def __init__(self, meta, *_args, **_kwargs):
            self.snapshot = initial_snapshot(meta, started_at_s=0)

        def run(self):
            assert produced.wait(5)

    runner.run_turn = run_turn
    monkeypatch.setattr(tui, "SimApp", ClosedApp)
    result = tui.run_live_job(meta(), runner, cases, responses_dir=tmp_path)
    assert result.phase is Phase.FAILED
    assert result.error == "Battle view closed"
    assert result.responses_path
    assert result.attacker_response.status == "cancelled"
    assert archive(tmp_path)["status"] == "cancelled"


def test_archive_write_failure_still_reaches_ui(tmp_path, monkeypatch):
    from ultron.cli.responses import ResponseArchive

    runner, cases = make_demo(meta(), delay_s=0, sleep=lambda _: None)
    original_record = ResponseArchive.record
    failed = False

    def disk_full(store, event):
        nonlocal failed
        if isinstance(event, ModelResponseDelta):
            failed = True
        if failed:
            raise OSError("No space left on device")
        original_record(store, event)

    monkeypatch.setattr(ResponseArchive, "record", disk_full)
    events = []
    with pytest.raises(OSError, match="No space left on device"):
        drive_job(meta(), runner, cases, emit=events.append, clock=lambda: 0, responses_dir=tmp_path)
    snapshot = initial_snapshot(meta(), started_at_s=0)
    for event in events:
        snapshot = apply(snapshot, event)
    assert snapshot.phase is Phase.FAILED
    assert snapshot.error == "No space left on device"
    assert snapshot.attacker_response.status == "error"
    assert [event.kind for event in events].count("error") == 1
