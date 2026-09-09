from __future__ import annotations

import json
import threading
from concurrent.futures import CancelledError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ultron.cli.demo import make_demo
from ultron.cli.model import (
    InvalidTransition, JobMeta, ModelResponseDelta, ModelResponseFinished,
    ModelResponseStarted, Phase, TurnStarted, apply, initial_snapshot,
)
from ultron.cli.observe import drive_job
from ultron.env.backend import IsolationBackend
from ultron.response_stream import ModelResponse, stream_chat_completion
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
