"""Run a model server behind the passive response capture proxy."""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import FrameType
from typing import Protocol, Sequence

from ultron.response_proxy import CaptureProxy
from ultron.cli.responses import default_response_directory
from ultron.train.schema_v1 import Role

_LOOPBACK = "127.0.0.1"


class Proxy(Protocol):
    server_address: tuple

    def serve_forever(self) -> None: ...

    def shutdown(self) -> None: ...

    def server_close(self) -> None: ...


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _generation(value: str) -> int:
    generation = int(value)
    if generation < 0:
        raise argparse.ArgumentTypeError("generation must be non-negative")
    return generation


def _backend_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind((_LOOPBACK, port))


def _prepare_responses_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path):
        pass


def _group_exists(process: subprocess.Popen) -> bool:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(process: subprocess.Popen, timeout: float = 3.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_exists(process):
            return
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _exit_status(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def supervise(
    proxy: Proxy,
    command: Sequence[str],
    *,
    responses_dir: Path | None = None,
) -> int:
    """Run command until it exits or the already-bound proxy fails."""
    try:
        child = subprocess.Popen(command, start_new_session=True)
    except OSError as exc:
        proxy.server_close()
        sys.stderr.write(f"Could not start model server: {exc}\n")
        return 127

    child_done = threading.Event()
    proxy_stopped = threading.Event()
    stop_lock = threading.Lock()
    child_returncode: list[int] = []
    requested_status: list[int] = []

    def stop_proxy() -> None:
        if proxy_stopped.is_set():
            return
        with stop_lock:
            if proxy_stopped.is_set():
                return
            proxy_stopped.set()
            proxy.shutdown()

    def watch_child() -> None:
        child_returncode.append(child.wait())
        child_done.set()
        stop_proxy()
        _stop_process_group(child)

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        if not requested_status:
            requested_status.append(128 + signum)
        _stop_process_group(child)

    previous_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    watcher = threading.Thread(target=watch_child, name="ultron-model-server", daemon=True)
    watcher.start()
    host, port = proxy.server_address[:2]
    print(f"Capture proxy listening on http://{host}:{port}", flush=True)
    if responses_dir is not None:
        print(f"Responses: {responses_dir.resolve()}", flush=True)

    proxy_error: BaseException | None = None
    try:
        proxy.serve_forever()
        if not child_done.is_set():
            proxy_error = RuntimeError("capture proxy stopped while the model server was running")
    except BaseException as exc:
        proxy_error = exc
    finally:
        if not child_done.is_set():
            _stop_process_group(child)
        child_done.wait(timeout=6)
        stop_proxy()
        watcher.join(timeout=6)
        proxy.server_close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if proxy_error is not None:
        sys.stderr.write(f"Capture proxy failed: {proxy_error}\n")
        return 1
    if requested_status:
        return requested_status[0]
    return _exit_status(child_returncode[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a model server behind passive response capture.",
    )
    parser.add_argument("--role", choices=("attacker", "defender"), required=True)
    parser.add_argument("--generation", type=_generation, default=0)
    parser.add_argument("--listen-port", type=_port, required=True)
    parser.add_argument("--upstream-port", type=_port, required=True)
    parser.add_argument("--responses-dir", type=Path, default=default_response_directory())
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        build_parser().error("a model server command is required after --")
    if args.listen_port == args.upstream_port:
        build_parser().error("listen and upstream ports must differ")
    try:
        _prepare_responses_dir(args.responses_dir)
    except OSError as exc:
        sys.stderr.write(f"Responses directory {args.responses_dir} is not writable: {exc}\n")
        return 2
    try:
        _backend_port_available(args.upstream_port)
    except OSError as exc:
        sys.stderr.write(
            f"Model backend port {_LOOPBACK}:{args.upstream_port} is unavailable: {exc}\n"
        )
        return 2
    try:
        proxy = CaptureProxy(
            f"http://{_LOOPBACK}:{args.upstream_port}",
            Role(args.role),
            generation=args.generation,
            responses_dir=args.responses_dir,
            listen_host=_LOOPBACK,
            listen_port=args.listen_port,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(
            f"Capture port {_LOOPBACK}:{args.listen_port} is unavailable: {exc}\n"
        )
        return 2
    return supervise(proxy, command, responses_dir=args.responses_dir)


if __name__ == "__main__":
    raise SystemExit(main())
