"""Run passive response capture in front of an existing model server."""
from __future__ import annotations

import signal
import sys
import threading
import tempfile

from ultron.train.schema_v1 import Role


def run_capture(args) -> int:
    from ultron.response_proxy import CaptureProxy

    if not 0 <= args.port <= 65535 or args.generation < 0:
        sys.stderr.write("Port must be between 0 and 65535, and generation must be non-negative.\n")
        return 2
    try:
        args.responses_dir = args.responses_dir.expanduser()
        args.responses_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=args.responses_dir) as probe:
            probe.write(b"capture")
            probe.flush()
        proxy = CaptureProxy(
            args.upstream,
            Role(args.role),
            generation=args.generation,
            responses_dir=args.responses_dir,
            listen_port=args.port,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Could not start capture: {exc}\n")
        return 2

    stopped = threading.Event()
    errors: list[BaseException] = []
    exit_status = 0

    def stop(signum, _frame):
        nonlocal exit_status
        exit_status = 128 + signum
        stopped.set()

    def serve():
        try:
            proxy.serve_forever()
        except BaseException as exc:
            errors.append(exc)
        finally:
            stopped.set()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    thread = threading.Thread(target=serve, name="ultron-capture", daemon=True)
    thread.start()
    host, port = proxy.server_address[:2]
    print(f"Capture listening on http://{host}:{port}/v1", flush=True)
    print(f"Responses: {args.responses_dir.resolve()}", flush=True)
    try:
        stopped.wait()
    finally:
        for close in (proxy.shutdown, proxy.server_close):
            try:
                close()
            except Exception as exc:
                errors.append(exc)
        thread.join(timeout=5)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if errors:
        sys.stderr.write(f"Capture stopped: {errors[0]}\n")
        return 1
    return exit_status
