from __future__ import annotations

import gzip
import http.client
import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ultron.cli.responses import load_response_archive
from ultron.response_proxy import CaptureProxy
from ultron.train.schema_v1 import Role


@pytest.fixture
def backend():
    release = threading.Event()
    requests = []
    first = b'data: {"choices":[{"index":0,"delta":{"content":"hello ","reasoning_content":"reason","tool_calls":[{"index":0,"function":{"name":"example","arguments":"{}"}}]}}]}\n\n'
    last = 'data: {"choices":[{"index":0,"delta":{"content":"세계"}}]}\n\ndata: [DONE]\n\n'.encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", "123")
            self.end_headers()

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = None
            if not isinstance(payload, dict):
                response = b'{"error":{"message":"invalid request"}}'
                requests.append((self.path, dict(self.headers), raw))
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
                return
            requests.append((self.path, dict(self.headers), raw))
            scenario = payload.get("scenario", "normal")
            stream = payload.get("stream", False)
            if scenario in ("http_error", "http_500"):
                body = b'{"error":{"message":"rate limited","type":"rate_limit"}}'
                self.send_response(500 if scenario == "http_500" else 429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", "5")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.send_header("X-Provider-Test", "preserved")
            if scenario == "gzip":
                self.send_header("Content-Encoding", "gzip")
            if stream:
                if scenario == "chunked":
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Connection", "X-Internal")
                    self.send_header("X-Internal", "not forwarded")
                    self.end_headers()
                    # Split transport chunks inside a UTF-8 character and SSE CRLF.
                    full = (first + last).replace(b"\n", b"\r\n")
                    split = full.index("세계".encode()) + 1
                    for piece in (full[:3], full[3:split], full[split:]):
                        self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return
                if scenario == "length_mismatch":
                    self.send_header("Content-Length", str(len(first) + 1000))
                self.end_headers()
                try:
                    if scenario == "gzip":
                        self.wfile.write(gzip.compress(first + last))
                        self.wfile.flush()
                        return
                    self.wfile.write(first)
                    self.wfile.flush()
                    release.wait(5)
                    if scenario not in ("premature", "length_mismatch"):
                        if scenario == "disconnect":
                            for _ in range(30):
                                self.wfile.write(first)
                                self.wfile.flush()
                                time.sleep(0.01)
                        self.wfile.write(last)
                        self.wfile.flush()
                except OSError:
                    pass
            else:
                message = {"content": "full reply", "reasoning_content": "full reason",
                           "tool_calls": [{"id": "tool-1", "function": {"name": "example", "arguments": "{}"}}]}
                body = json.dumps({"choices": [{"index": index, "message": message}
                                              for index in range(payload.get("n", 1))]}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, release, requests, first, last
    release.set()
    server.shutdown()
    server.server_close()
    thread.join(5)


@pytest.fixture
def proxy(backend, tmp_path):
    server, *_ = backend
    proxy = CaptureProxy(f"http://127.0.0.1:{server.server_port}", Role.ATTACKER,
                         generation=3, responses_dir=tmp_path)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    yield proxy
    proxy.shutdown()
    proxy.server_close()
    thread.join(5)


def request(proxy, payload, *, headers=None, path="/v1/chat/completions"):
    connection = http.client.HTTPConnection(*proxy.server_address, timeout=5)
    raw = json.dumps(payload).encode()
    connection.request("POST", path, body=raw, headers={"Content-Type": "application/json", **(headers or {})})
    return connection, connection.getresponse(), raw


def saved(tmp_path, *, count=1, terminal=True):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        paths = list(tmp_path.glob("*/responses.json"))
        if len(paths) == count:
            documents = [load_response_archive(path) for path in paths]
            if not terminal or all(doc["status"] != "running" for doc in documents):
                return list(zip(paths, documents))
        time.sleep(0.01)
    pytest.fail("response archives did not reach expected state")


def test_stream_is_forwarded_and_durable_before_provider_finishes(proxy, backend, tmp_path):
    _, release, requests, first, last = backend
    headers = {"Authorization": "Bearer secret-token", "X-Ultron-Run-Id": "run-123",
               "X-Ultron-Episode": "4", "X-Ultron-Turn": "9"}
    connection, response, raw_request = request(proxy, {"model": "test", "stream": True}, headers=headers)
    try:
        first_line = response.readline()
        assert first_line == first.splitlines(keepends=True)[0]
        path, document = saved(tmp_path, terminal=False)[0]
        assert document["status"] == "running"
        assert document["responses"][0]["text"] == "hello "
        assert document["responses"][0]["status"] == "streaming"
        assert document["meta"]["generation"] == 3
        assert document["capture"]["role"] == "attacker"
        assert document["capture"]["run_id"] == "run-123"
        assert document["responses"][0]["episode_index"] == 4
        assert document["responses"][0]["turn_index"] == 9
        assert path.with_name("raw-response.body").read_bytes() == first
        provider_event = json.loads(path.with_name("provider-events.jsonl").read_text().splitlines()[0])
        assert provider_event["choices"][0]["delta"]["reasoning_content"] == "reason"
        assert provider_event["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "example"
        assert response.status == 200
        assert response.getheader("X-Provider-Test") == "preserved"
        assert requests[0][2] == raw_request
        assert requests[0][1]["Authorization"] == "Bearer secret-token"
        release.set()
        assert first_line + response.read() == first + last
    finally:
        release.set()
        connection.close()
    path, document = saved(tmp_path)[0]
    assert document["status"] == "complete"
    assert document["responses"][0]["text"] == "hello 세계"
    assert path.with_name("raw-response.body").read_bytes() == first + last
    assert b"secret-token" not in b"".join(item.read_bytes() for item in path.parent.iterdir())


def test_nonstream_and_concurrent_same_role_requests_have_full_json(proxy, tmp_path):
    results = []

    def run():
        connection, response, _ = request(proxy, {"model": "test", "n": 2})
        results.append(json.loads(response.read()))
        connection.close()

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(results) == 2
    for path, document in saved(tmp_path, count=2):
        assert document["status"] == "complete"
        assert len(document["responses"]) == 2
        assert all(record["text"] == "full reply" for record in document["responses"])
        assert document["provider_response"] == results[0]
        assert json.loads(path.with_name("raw-response.body").read_bytes()) == results[0]


@pytest.mark.parametrize("scenario", ["premature", "length_mismatch"])
def test_premature_upstream_eof_preserves_partial_response(proxy, backend, tmp_path, scenario):
    backend[1].set()
    connection, response, _ = request(proxy, {"model": "test", "stream": True, "scenario": scenario})
    try:
        try:
            body = response.read()
        except http.client.IncompleteRead as exc:
            body = exc.partial
        assert body == backend[3]
    finally:
        connection.close()
    path, document = saved(tmp_path)[0]
    assert document["status"] == "error"
    assert document["responses"][0]["text"] == "hello "
    assert document["responses"][0]["status"] == "error"
    assert path.with_name("raw-response.body").read_bytes() == body


@pytest.mark.parametrize("status,scenario", [(429, "http_error"), (500, "http_500")])
def test_http_error_status_and_body_are_preserved(proxy, tmp_path, status, scenario):
    connection, response, _ = request(proxy, {"model": "test", "scenario": scenario})
    assert response.status == status
    assert response.getheader("Retry-After") == "5"
    body = response.read()
    connection.close()
    path, document = saved(tmp_path)[0]
    assert document["capture"]["http_status"] == status
    assert document["status"] == "error"
    assert document["provider_response"] == json.loads(body)
    assert path.with_name("raw-response.body").read_bytes() == body


def test_gzip_stream_forwarded_unchanged_and_decoded_for_json(proxy, backend, tmp_path):
    connection, response, _ = request(proxy, {"model": "test", "stream": True, "scenario": "gzip"})
    assert response.getheader("Content-Encoding") == "gzip"
    body = response.read()
    connection.close()
    assert gzip.decompress(body) == backend[3] + backend[4]
    path, document = saved(tmp_path)[0]
    assert document["status"] == "complete"
    assert document["responses"][0]["text"] == "hello 세계"
    assert path.with_name("raw-response.body").read_bytes() == body


def test_health_requests_do_not_create_response_archives(proxy, tmp_path):
    connection = http.client.HTTPConnection(*proxy.server_address, timeout=5)
    connection.request("GET", "/health")
    response = connection.getresponse()
    assert response.status == 200
    assert response.read() == b"ok"
    connection.close()
    assert list(tmp_path.iterdir()) == []


def test_client_disconnect_marks_partial_response_cancelled(proxy, backend, tmp_path):
    sock = socket.create_connection(proxy.server_address, timeout=5)
    body = json.dumps({"model": "test", "stream": True, "scenario": "disconnect"}).encode()
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    received = b""
    while b"hello " not in received:
        received += sock.recv(4096)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()
    backend[1].set()
    _, document = saved(tmp_path)[0]
    assert document["status"] == "cancelled"
    assert document["responses"][0]["text"].startswith("hello ")


def test_shutdown_cancels_archive_while_provider_is_blocked(proxy, backend, tmp_path):
    connection, response, _ = request(proxy, {"model": "test", "stream": True})
    assert b"hello " in response.readline()
    started = time.monotonic()
    proxy.shutdown()
    assert time.monotonic() - started < 2
    _, document = saved(tmp_path)[0]
    assert document["status"] == "cancelled"
    assert document["responses"][0]["text"] == "hello "
    connection.close()
    backend[1].set()


def test_unsafe_attribution_headers_are_not_used_as_paths_or_metadata(proxy, tmp_path):
    connection, response, _ = request(proxy, {"model": "test"}, headers={
        "X-Ultron-Run-Id": "../../escape", "X-Ultron-Episode": "-3", "X-Ultron-Turn": "abc",
    })
    response.read()
    connection.close()
    path, document = saved(tmp_path)[0]
    assert path.parent.parent == tmp_path
    assert document["capture"]["run_id"] is None
    assert document["capture"]["episode_index"] is None
    assert document["capture"]["turn_index"] is None


def test_chunked_crlf_stream_and_split_utf8_preserve_body(proxy, backend, tmp_path):
    connection, response, _ = request(proxy, {"model": "test", "stream": True, "scenario": "chunked"})
    assert response.getheader("Transfer-Encoding") is None
    assert response.getheader("X-Internal") is None
    assert response.getheader("Connection") == "close"
    body = response.read()
    connection.close()
    assert body == (backend[3] + backend[4]).replace(b"\n", b"\r\n")
    path, document = saved(tmp_path)[0]
    assert document["status"] == "complete"
    assert document["responses"][0]["text"] == "hello 세계"
    assert path.with_name("raw-response.body").read_bytes() == body


def test_chunked_request_body_reaches_provider_unchanged(proxy, backend, tmp_path):
    connection = http.client.HTTPConnection(*proxy.server_address, timeout=5)
    raw = b'{"model":"test","stream":false}'
    connection.request("POST", "/v1/chat/completions", body=iter([raw[:5], raw[5:]]),
                       headers={"Content-Type": "application/json"}, encode_chunked=True)
    response = connection.getresponse()
    assert response.status == 200
    response.read()
    connection.close()
    assert backend[2][0][2] == raw
    assert saved(tmp_path)[0][1]["status"] == "complete"


def test_invalid_request_json_is_forwarded_for_provider_validation(proxy, backend, tmp_path):
    connection = http.client.HTTPConnection(*proxy.server_address, timeout=5)
    raw = b'{invalid json'
    connection.request("POST", "/v1/chat/completions", body=raw, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    assert response.status == 400
    body = response.read()
    connection.close()
    assert backend[2][0][2] == raw
    _, document = saved(tmp_path)[0]
    assert document["status"] == "error"
    assert document["provider_response"] == json.loads(body)


def test_head_preserves_content_length_without_waiting_for_body(proxy):
    connection = http.client.HTTPConnection(*proxy.server_address, timeout=5)
    connection.request("HEAD", "/health")
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("Content-Length") == "123"
    assert response.read() == b""
    connection.close()


def test_recording_failure_before_headers_is_explicit_503(proxy, backend, monkeypatch, capsys):
    from ultron.cli.responses import ResponseArchive

    def no_space(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(ResponseArchive, "flush", no_space)
    connection, response, _ = request(proxy, {"model": "test"})
    assert response.status == 503
    assert b"Response recording failed" in response.read()
    connection.close()
    assert backend[2] == []
    assert "Response recording failed" in capsys.readouterr().err


def test_shutdown_still_stops_server_when_recording_fails(proxy, backend, monkeypatch, capsys):
    from ultron.cli.responses import ResponseArchive

    connection, response, _ = request(proxy, {"model": "test", "stream": True})
    assert b"hello " in response.readline()

    def no_space(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(ResponseArchive, "record", no_space)
    started = time.monotonic()
    proxy.shutdown()
    assert time.monotonic() - started < 2
    assert "Response recording failed" in capsys.readouterr().err
    connection.close()
    backend[1].set()


@pytest.mark.parametrize("upstream_host", ["localhost", "127.0.0.1"])
def test_proxy_rejects_its_own_loopback_port_before_binding(upstream_host):
    # Keeping this socket open also proves validation precedes the attempted bind.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        with pytest.raises(ValueError, match="must not point back to itself"):
            CaptureProxy(f"http://{upstream_host}:{port}", Role.ATTACKER, listen_port=port)
