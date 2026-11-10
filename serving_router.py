"""Local streaming router for replicas serving identical immutable adapters."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Iterator, Mapping, Sequence
from urllib.parse import urlsplit

_HOP = {"connection", "transfer-encoding", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade"}


class ReplicaRouter:
    """Prefer cached episode/model prefixes, spill to the less busy replica."""

    def __init__(self, upstreams: Sequence[str], models: Sequence[Mapping], *, aliases: Mapping[str, str] | None = None, skew: int = 1):
        if not upstreams or skew < 0:
            raise ValueError("router needs at least one replica and non-negative load skew")
        for url in upstreams:
            parsed = urlsplit(url)
            if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.path not in {"", "/"}:
                raise ValueError("replicas must use local HTTP origins")
        self.upstreams = tuple(upstreams)
        self.models = {model["id"]: dict(model) for model in models}
        self.aliases = dict(aliases or {})
        if any(value not in self.models for value in self.aliases.values()):
            raise ValueError("alias points to an unknown immutable model")
        self.skew = skew
        self.active = [0] * len(upstreams)
        self._lock = threading.Lock()

    def resolve_model(self, value: str) -> str:
        model = self.aliases.get(value, value)
        if model not in self.models:
            raise ValueError("request must select an immutable model advertised by /v1/models")
        return model

    @contextmanager
    def lease(self, key: str) -> Iterator[str]:
        preferred = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % len(self.upstreams)
        with self._lock:
            least = min(range(len(self.active)), key=lambda index: (self.active[index], index))
            selected = preferred if self.active[preferred] <= self.active[least] + self.skew else least
            self.active[selected] += 1
        try:
            yield self.upstreams[selected]
        finally:
            with self._lock:
                self.active[selected] -= 1

    def health(self) -> bool:
        for origin in self.upstreams:
            parsed = urlsplit(origin)
            connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
            try:
                connection.request("GET", "/v1/models")
                response = connection.getresponse()
                payload = json.loads(response.read())
                present = {row["id"] for row in payload.get("data", [])}
                if response.status != 200 or not set(self.models).issubset(present):
                    return False
            except (OSError, ValueError, http.client.HTTPException):
                return False
            finally:
                connection.close()
        return True


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], router: ReplicaRouter, *, instance_id: str | None = None):
        self.router = router
        self.instance_id = instance_id
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:
        pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            healthy = self.server.router.health()
            self._json(200 if healthy else 503, {"ready": healthy, "instance_id": self.server.instance_id})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": list(self.server.router.models.values())})
        else:
            self._json(404, {"error": "unknown route"})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/v1/completions"}:
            self._json(404, {"error": "unknown route"})
            return
        sent = False
        connection = None
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("router requires Content-Length request framing")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64 * 1024 * 1024:
                raise ValueError("invalid request length")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
                raise ValueError("model is required")
            model = self.server.router.resolve_model(payload["model"])
            payload["model"] = model
            body = json.dumps(payload, separators=(",", ":")).encode()
            episode = self.headers.get("X-Ultron-Episode", "")
            run_id = self.headers.get("X-Ultron-Run-Id", "")
            key = f"{run_id}:{episode}:{model}"
            with self.server.router.lease(key) as origin:
                parsed = urlsplit(origin)
                connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)
                headers = {name: value for name, value in self.headers.items() if name.lower() not in _HOP | {"host", "content-length"}}
                headers["Content-Length"] = str(len(body))
                connection.request("POST", self.path, body, headers)
                upstream = connection.getresponse()
                self.send_response_only(upstream.status, upstream.reason)
                for name, value in upstream.getheaders():
                    if name.lower() not in _HOP:
                        self.send_header(name, value)
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                sent = True
                expected = upstream.length
                received = 0
                while chunk := upstream.read1(65536):
                    received += len(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
                if expected is not None and received != expected:
                    raise http.client.IncompleteRead(b"", expected - received)
        except (ValueError, OSError, http.client.HTTPException) as exc:
            # Never retry a submitted generation: a retry may sample another action.
            self.close_connection = True
            if not sent:
                try:
                    self._json(400 if isinstance(exc, ValueError) else 502, {"error": str(exc)})
                except OSError:
                    pass
        finally:
            if connection is not None:
                connection.close()
