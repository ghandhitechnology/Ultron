"""Verify real HTTP transport through the CLI recorder into the independent UI."""
from __future__ import annotations

import asyncio
import json
import selectors
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

pytest.importorskip("textual")

from textual.widgets import RichLog, Static
from ultron.cli.battle import BattleApp


def test_capture_command_streams_to_client_archive_and_battle(tmp_path: Path) -> None:
    first = b'data: {"choices":[{"index":0,"delta":{"content":"First chunk"}}]}\n\n'
    second = b'data: {"choices":[{"index":0,"delta":{"content":" and the rest"}}]}\n\ndata: [DONE]\n\n'
    release = threading.Event()
    received_first = threading.Event()
    client_body = bytearray()
    client_errors = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert request["stream"] is True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            if release.wait(15):
                self.wfile.write(second)
                self.wfile.flush()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    capture = subprocess.Popen(
        [sys.executable, "-m", "ultron.cli.main", "capture", "--role", "attacker",
         "--upstream", f"http://127.0.0.1:{upstream.server_port}", "--port", "0",
         "--responses-dir", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    client = None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(capture.stdout, selectors.EVENT_READ)
            assert selector.select(10), "Capture command did not report its listening endpoint"
        line = capture.stdout.readline().strip()
        assert line.startswith("Capture listening on "), line
        endpoint = line.removeprefix("Capture listening on ") + "/chat/completions"

        def receive():
            try:
                request = Request(endpoint, data=json.dumps({
                    "model": "fixture-model", "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                }).encode(), headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=15) as response:
                    for chunk in response:
                        client_body.extend(chunk)
                        if b"First chunk" in client_body:
                            received_first.set()
            except Exception as exc:
                client_errors.append(exc)

        client = threading.Thread(target=receive, daemon=True)
        client.start()
        assert received_first.wait(10), client_errors
        assert client.is_alive(), "Provider should still be waiting to finish"

        async def inspect_live_view():
            app = BattleApp(tmp_path)
            def response_text():
                return "\n".join(line.text for line in app.query_one("#attacker-response", RichLog).lines)
            async with app.run_test(size=(120, 40)) as pilot:
                for _ in range(50):
                    await pilot.pause(0.05)
                    if "First chunk" in response_text():
                        break
                assert "First chunk" in response_text()
                assert "streaming" in str(app.query_one("#attacker-title", Static).render()).lower()
                release.set()
                for _ in range(60):
                    await pilot.pause(0.05)
                    if "and the rest" in response_text() and "complete" in str(app.query_one("#attacker-title", Static).render()).lower():
                        break
                assert "First chunk and the rest" in response_text()
                assert "complete" in str(app.query_one("#attacker-title", Static).render()).lower()

        asyncio.run(inspect_live_view())
        client.join(5)
        assert not client.is_alive()
        assert client_errors == []
        assert bytes(client_body) == first + second
        assert capture.poll() is None, "Closing the viewer must leave the recorder running"
        archives = list(tmp_path.glob("*/responses.json"))
        assert len(archives) == 1
        document = json.loads(archives[0].read_text())
        assert document["status"] == "complete"
        assert document["responses"][0]["text"] == "First chunk and the rest"
        assert archives[0].with_name("raw-response.body").read_bytes() == first + second
    finally:
        release.set()
        if client is not None:
            client.join(5)
        if capture.poll() is None:
            capture.send_signal(signal.SIGTERM)
        try:
            capture.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            capture.kill()
            capture.communicate(timeout=5)
            raise
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(5)
    assert capture.returncode == 143
