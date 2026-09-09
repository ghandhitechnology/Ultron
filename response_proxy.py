"""Passively record model HTTP responses while forwarding provider bytes live."""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ultron.cli.model import (
    JobEnded, JobMeta, ModelResponseDelta, ModelResponseFinished, ModelResponseStarted,
)
from ultron.cli.responses import ResponseArchive
from ultron.env.backend import IsolationBackend
from ultron.train.schema_v1 import Role

_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade"}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def _index(value: str | None) -> int | None:
    if value is not None and re.fullmatch(r"[0-9]{1,9}", value):
        return int(value)
    return None


class _RecordingError(Exception):
    pass


def _recording(operation, *args):
    try:
        return operation(*args)
    except OSError as exc:
        raise _RecordingError(str(exc)) from exc


def _finish_safely(capture, status: str, error: str) -> bool:
    if capture is None:
        return False
    try:
        capture.finish(status, error)
        return False
    except OSError as exc:
        print(f"Response recording failed: {exc}", file=sys.stderr)
        return True


class _ClientDisconnected(Exception):
    pass


class _Capture:
    def __init__(self, proxy, path: str, headers, request: bytes) -> None:
        try:
            payload = json.loads(request)
        except (ValueError, UnicodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        run = headers.get("X-Ultron-Run-Id", "")
        run = run if _SAFE_ID.fullmatch(run) else None
        episode = _index(headers.get("X-Ultron-Episode"))
        turn = _index(headers.get("X-Ultron-Turn"))
        self.role = proxy.role
        self.episode = episode or 0
        self.turn = turn or 0
        self.model = payload.get("model") if isinstance(payload.get("model"), str) else ""
        self.archive = ResponseArchive(JobMeta(
            generation=proxy.generation, profile_id="model-response",
            isolation=IsolationBackend.DOCKER, episodes_planned=1, turns_per_side=1,
            group_id=run or "capture",
        ), proxy.responses_dir)
        self.archive.document["capture"] = {
            "role": self.role.value,
            "request_path": path,
            "run_id": run,
            "episode_index": episode,
            "turn_index": turn,
            "http_status": None,
            "content_type": None,
            "content_encoding": None,
            "raw_response_file": "raw-response.body",
            "provider_events_file": "provider-events.jsonl",
        }
        self.raw = self.archive.path.with_name("raw-response.body").open("wb")
        self.provider_events = self.archive.path.with_name("provider-events.jsonl").open("w", encoding="utf-8")
        self.started = time.monotonic()
        self.ids: dict[int, str] = {}
        self.lock = threading.RLock()
        self.closed = False
        self.sse = False
        self.done = False
        self.error: str | None = None
        self.pending = b""
        self.data: list[bytes] = []
        self.body = bytearray()
        self.decoder = None
        self._start(0)

    def _clock(self) -> float:
        return time.monotonic() - self.started

    def _start(self, choice: int) -> str:
        if choice not in self.ids:
            response_id = f"{self.archive.run_id}-{choice}"
            self.ids[choice] = response_id
            self.archive.record(ModelResponseStarted(
                self.episode, self.turn, self.role, response_id, self.model, self._clock(),
            ))
        return self.ids[choice]

    def headers(self, status: int, headers) -> None:
        with self.lock:
            if self.closed:
                return
            content_type = headers.get("Content-Type", "")
            encoding = headers.get("Content-Encoding", "").lower()
            self.sse = "text/event-stream" in content_type.lower()
            self.archive.document["capture"].update(
                http_status=status, content_type=content_type, content_encoding=encoding or None,
            )
            if encoding == "gzip":
                self.decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            elif encoding == "deflate":
                self.decoder = zlib.decompressobj()
            elif encoding and encoding != "identity":
                self.error = f"Cannot decode response content encoding {encoding}"
            if status >= 400:
                self.error = f"Upstream HTTP {status}"
            self.archive.flush()

    def feed(self, chunk: bytes) -> None:
        with self.lock:
            if self.closed:
                return
            self.raw.write(chunk)
            self.raw.flush()
            os.fsync(self.raw.fileno())
            try:
                decoded = self.decoder.decompress(chunk) if self.decoder else chunk
                if self.sse:
                    self.pending += decoded
                    while b"\n" in self.pending:
                        line, self.pending = self.pending.split(b"\n", 1)
                        line = line.rstrip(b"\r")
                        if line.startswith(b"data:"):
                            self.data.append(line[5:].lstrip(b" "))
                        elif not line and self.data:
                            payload = b"\n".join(self.data)
                            self.data.clear()
                            if payload == b"[DONE]":
                                self.done = True
                            else:
                                self._payload(json.loads(payload), streaming=True)
                else:
                    self.body.extend(decoded)
            except (ValueError, UnicodeError, zlib.error) as exc:
                self.error = f"Invalid provider response: {type(exc).__name__}"

    def _payload(self, payload, *, streaming: bool) -> None:
        if not isinstance(payload, dict):
            self.error = "Provider response must be a JSON object"
            return
        if payload.get("error"):
            self.error = "Provider returned an error payload"
        if streaming:
            self.provider_events.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.provider_events.flush()
            os.fsync(self.provider_events.fileno())
        else:
            self.archive.document["provider_response"] = payload
        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            self.error = "Provider choices must be an array"
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            if not isinstance(index, int) or index < 0:
                continue
            response_id = self._start(index)
            message = choice.get("delta" if streaming else "message", {})
            text = message.get("content") if isinstance(message, dict) else None
            if text is None:
                text = choice.get("text")
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text
                               if isinstance(part, dict) and isinstance(part.get("text"), str))
            if isinstance(text, str) and text:
                self.archive.record(ModelResponseDelta(
                    self.episode, self.turn, self.role, response_id, text, self._clock(),
                ))

    def finish(self, status: str = "complete", error: str | None = None) -> None:
        with self.lock:
            if self.closed:
                return
            if status == "complete":
                if self.decoder is not None and not self.decoder.eof:
                    self.error = self.error or "Compressed provider response ended early"
                if self.sse and not self.done:
                    self.error = self.error or "Provider stream ended before [DONE]"
                elif not self.sse:
                    try:
                        self._payload(json.loads(self.body), streaming=False)
                    except (ValueError, UnicodeError):
                        self.error = self.error or "Provider response was not valid JSON"
                if self.error:
                    status, error = "error", self.error
            try:
                for response_id in self.ids.values():
                    self.archive.record(ModelResponseFinished(
                        self.episode, self.turn, self.role, response_id, status,
                        self._clock(), error=error,
                    ))
                if status == "complete":
                    self.archive.record(JobEnded(self._clock(), self._clock()))
                else:
                    self.archive.finish(status, error)
            finally:
                self.closed = True
                self.raw.close()
                self.provider_events.close()


class CaptureProxy:
    """Loopback reverse proxy with one durable response archive per model request."""

    def __init__(self, upstream_url: str, role: Role, generation: int = 0,
                 responses_dir: Path | None = None, listen_host: str = "127.0.0.1",
                 listen_port: int = 0) -> None:
        upstream = urlsplit(upstream_url)
        if upstream.scheme not in ("http", "https") or not upstream.hostname:
            raise ValueError("upstream_url must be an HTTP or HTTPS URL")
        if upstream.username or upstream.password or upstream.query or upstream.fragment:
            raise ValueError("upstream_url must not contain credentials, query, or fragment")
        if listen_host not in ("127.0.0.1", "localhost"):
            raise ValueError("capture proxy must listen on loopback")
        upstream_port = upstream.port or (443 if upstream.scheme == "https" else 80)
        upstream_is_local = upstream.hostname in ("127.0.0.1", "localhost")
        if listen_port and upstream_is_local and upstream_port == listen_port:
            raise ValueError("capture proxy upstream must not point back to itself")
        self.upstream = upstream
        self.role = Role(role)
        self.generation = generation
        self.responses_dir = responses_dir
        self.stopping = threading.Event()
        self._lock = threading.Lock()
        self._active: dict[http.client.HTTPConnection, _Capture | None] = {}
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args) -> None:
                pass

            def do_GET(self):
                proxy._forward(self)

            do_POST = do_GET
            do_HEAD = do_GET
            do_OPTIONS = do_GET
            do_PUT = do_GET
            do_DELETE = do_GET
            do_PATCH = do_GET

        self.server = ThreadingHTTPServer((listen_host, listen_port), Handler)
        if upstream_is_local and upstream_port == self.server.server_address[1]:
            self.server.server_close()
            raise ValueError("capture proxy upstream must not point back to itself")
        self.server.daemon_threads = True
        self.server_address = self.server.server_address

    def serve_forever(self, poll_interval: float = 0.1) -> None:
        self.server.serve_forever(poll_interval=poll_interval)

    def shutdown(self) -> None:
        self.stopping.set()
        with self._lock:
            active = list(self._active.items())
        try:
            for connection, capture in active:
                upstream_socket = getattr(connection, "capture_socket", None) or connection.sock
                if upstream_socket:
                    try:
                        upstream_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                _finish_safely(capture, "cancelled", "Capture proxy stopped")
        finally:
            self.server.shutdown()

    def server_close(self) -> None:
        self.server.server_close()

    @staticmethod
    def _body(handler: BaseHTTPRequestHandler) -> bytes:
        if handler.headers.get("Transfer-Encoding", "").lower() == "chunked":
            body = bytearray()
            while True:
                size = int(handler.rfile.readline().split(b";", 1)[0], 16)
                if not size:
                    while handler.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return bytes(body)
                chunk = handler.rfile.read(size)
                if len(chunk) != size or handler.rfile.read(2) != b"\r\n":
                    raise ValueError("Incomplete chunked request body")
                body.extend(chunk)
        size = int(handler.headers.get("Content-Length", "0"))
        if size < 0:
            raise ValueError("Invalid request Content-Length")
        body = handler.rfile.read(size)
        if len(body) != size:
            raise ValueError("Incomplete request body")
        return body

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        capture = None
        connection = None
        headers_sent = False
        handler.close_connection = True
        try:
            body = self._body(handler)
            path = urlsplit(handler.path).path
            if handler.command == "POST" and path.rstrip("/").endswith(("/chat/completions", "/completions")):
                capture = _recording(_Capture, self, path, handler.headers, body)
            connection_type = http.client.HTTPSConnection if self.upstream.scheme == "https" else http.client.HTTPConnection
            connection = connection_type(self.upstream.hostname, self.upstream.port, timeout=300)
            with self._lock:
                self._active[connection] = capture
            if self.stopping.is_set():
                raise InterruptedError("Capture proxy stopped")
            hop = _HOP_HEADERS | {h.strip().lower() for h in handler.headers.get("Connection", "").split(",")}
            request_headers = [(name, value) for name, value in handler.headers.items()
                               if name.lower() not in hop | {"host", "content-length"}]
            # Transport framing may change; the request body and application headers do not.
            target = self.upstream.path.rstrip("/") + handler.path
            connection.putrequest(handler.command, target, skip_accept_encoding=True)
            for name, value in request_headers:
                connection.putheader(name, value)
            if body or "Content-Length" in handler.headers or "Transfer-Encoding" in handler.headers:
                connection.putheader("Content-Length", str(len(body)))
            connection.endheaders(body)
            connection.capture_socket = connection.sock
            upstream = connection.getresponse()
            if capture:
                _recording(capture.headers, upstream.status, upstream.headers)
            handler.send_response_only(upstream.status, upstream.reason)
            response_hop = _HOP_HEADERS | {h.strip().lower() for h in upstream.headers.get("Connection", "").split(",")}
            for name, value in upstream.getheaders():
                if name.lower() not in response_hop:
                    handler.send_header(name, value)
            handler.send_header("Connection", "close")
            try:
                handler.end_headers()
            except OSError as exc:
                raise _ClientDisconnected() from exc
            headers_sent = True
            expected = None if handler.command == "HEAD" or upstream.status in (204, 304) else upstream.length
            received = 0
            while True:
                chunk = upstream.read1(65536)
                if not chunk:
                    break
                received += len(chunk)
                if capture:
                    _recording(capture.feed, chunk)
                try:
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
                except OSError as exc:
                    raise _ClientDisconnected() from exc
            if expected is not None and received != expected:
                raise http.client.IncompleteRead(b"", expected - received)
            if capture:
                _recording(capture.finish)
        except _ClientDisconnected:
            _finish_safely(capture, "cancelled", "Client disconnected")
        except _RecordingError as exc:
            print(f"Response recording failed: {exc}", file=sys.stderr)
            _finish_safely(capture, "error", f"Response recording failed: {exc}")
            if not headers_sent:
                self._send_error(handler, 503, "Response recording failed")
        except Exception as exc:
            recording_failed = _finish_safely(
                capture, "cancelled" if self.stopping.is_set() else "error",
                "Capture proxy stopped" if self.stopping.is_set() else f"{type(exc).__name__}: {exc}",
            )
            if not headers_sent:
                self._send_error(handler, 503 if recording_failed else 502,
                                 "Response recording failed" if recording_failed else "Upstream request failed")
        finally:
            if connection:
                connection.close()
                with self._lock:
                    self._active.pop(connection, None)

    @staticmethod
    def _send_error(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
        try:
            handler.send_error(status, message)
        except OSError:
            pass
