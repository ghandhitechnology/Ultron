from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from ultron.cli import serve

ROOT = Path(__file__).resolve().parents[2]

FAKE_SERVER = '''
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BODY = json.dumps({
    "model": "fixture-model",
    "choices": [{"index": 0, "message": {"content": "captured text"}}],
}).encode()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

server = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
Path(sys.argv[2]).write_text(str(os.getpid()))
server.serve_forever()
'''


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def free_port_pair() -> tuple[int, int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as first:
        first.bind(("127.0.0.1", 0))
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as second:
            second.bind(("127.0.0.1", 0))
            return first.getsockname()[1], second.getsockname()[1]


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met before timeout")


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def supervisor_command(
    listen_port: int,
    upstream_port: int,
    responses_dir: Path,
    child_command: list[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ultron.cli.serve",
        "--role",
        "attacker",
        "--generation",
        "4",
        "--listen-port",
        str(listen_port),
        "--upstream-port",
        str(upstream_port),
        "--responses-dir",
        str(responses_dir),
        "--",
        *child_command,
    ]


def test_supervisor_forwards_captures_and_stops_model_on_sigterm(tmp_path: Path) -> None:
    listen_port, upstream_port = free_port_pair()
    responses_dir = tmp_path / "responses"
    child_pid_path = tmp_path / "child-pid"
    fake_server = tmp_path / "fake_server.py"
    fake_server.write_text(FAKE_SERVER)
    process = subprocess.Popen(
        supervisor_command(
            listen_port,
            upstream_port,
            responses_dir,
            [sys.executable, str(fake_server), str(upstream_port), str(child_pid_path)],
        ),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child_pid = 0

    def proxy_is_ready() -> bool:
        try:
            with urlopen(f"http://127.0.0.1:{listen_port}/health", timeout=0.2) as response:
                return response.status == 200
        except (OSError, URLError):
            return False

    try:
        wait_until(proxy_is_ready)
        child_pid = int(child_pid_path.read_text())
        request = Request(
            f"http://127.0.0.1:{listen_port}/v1/chat/completions",
            data=json.dumps({"model": "fixture-model", "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        assert payload["choices"][0]["message"]["content"] == "captured text"

        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, (stdout, stderr)
        assert f"Responses: {responses_dir.resolve()}" in stdout
        assert not process_exists(child_pid)

        archives = list(responses_dir.glob("*/responses.json"))
        assert len(archives) == 1
        recorded = json.loads(archives[0].read_text())
        assert recorded["meta"]["generation"] == 4
        assert recorded["responses"][0]["role"] == "attacker"
        assert recorded["responses"][0]["text"] == "captured text"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        if child_pid and process_exists(child_pid):
            os.killpg(child_pid, signal.SIGKILL)


def test_supervisor_preserves_model_exit_code(tmp_path: Path) -> None:
    listen_port, upstream_port = free_port_pair()
    result = subprocess.run(
        supervisor_command(
            listen_port,
            upstream_port,
            tmp_path / "responses",
            [sys.executable, "-c", "raise SystemExit(23)"],
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 23, result.stderr


def test_port_collisions_fail_before_model_spawn(tmp_path: Path) -> None:
    marker = tmp_path / "spawned"
    child = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).touch()",
    ]
    for occupied_side in ("listen", "upstream"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            occupied_port = occupied.getsockname()[1]
            other_port = free_port()
            listen_port = occupied_port if occupied_side == "listen" else other_port
            upstream_port = occupied_port if occupied_side == "upstream" else other_port
            result = subprocess.run(
                supervisor_command(
                    listen_port,
                    upstream_port,
                    tmp_path / "responses",
                    child,
                ),
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=5,
            )
            assert result.returncode == 2, result.stderr
            assert not marker.exists()


def test_invalid_responses_directory_fails_before_model_spawn(tmp_path: Path) -> None:
    responses_file = tmp_path / "responses-file"
    responses_file.write_text("not a directory")
    marker = tmp_path / "spawned"
    listen_port, upstream_port = free_port_pair()
    result = subprocess.run(
        supervisor_command(
            listen_port,
            upstream_port,
            responses_file,
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ],
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 2
    assert "is not writable" in result.stderr
    assert not marker.exists()


def test_proxy_failure_terminates_model_process(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child-pid"

    class FailingProxy:
        server_address = ("127.0.0.1", 0)

        def __init__(self) -> None:
            self.shutdown_called = False
            self.close_called = False

        def serve_forever(self) -> None:
            wait_until(child_pid_path.exists)
            raise RuntimeError("fixture proxy failure")

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.close_called = True

    proxy = FailingProxy()
    result = serve.supervise(
        proxy,
        [
            sys.executable,
            "-c",
            (
                "import os, time; from pathlib import Path; "
                f"Path({str(child_pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)"
            ),
        ],
    )
    child_pid = int(child_pid_path.read_text())

    assert result == 1
    assert not process_exists(child_pid)
    assert proxy.shutdown_called
    assert proxy.close_called


def test_vllm_scripts_keep_client_ports_and_use_private_backends() -> None:
    cases = (
        (ROOT / "scripts" / "serve_vllm_attacker.sh", "ATTACKER", "8001", "8101"),
        (ROOT / "scripts" / "serve_vllm_defender.sh", "DEFENDER", "8002", "8102"),
    )
    for script, role, listen_port, upstream_port in cases:
        text = script.read_text()
        assert "-m ultron.cli.serve" in text
        assert f"--role {role.lower()}" in text
        assert f'LISTEN_PORT="${{ULTRON_{role}_PORT:-{listen_port}}}"' in text
        assert (
            f'UPSTREAM_PORT="${{ULTRON_{role}_UPSTREAM_PORT:-{upstream_port}}}"'
            in text
        )
        assert "--host 127.0.0.1" in text
