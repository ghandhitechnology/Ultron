#!/usr/bin/env python3

import json
import os
import pwd
import socket
import subprocess
from pathlib import Path
from typing import Any

PORT = 9910
MAX_REQUEST_BYTES = 1_048_576


def dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "probe_attacker_euid":
        username = require_user(params)
        return {"euid": probe_attacker_euid(username)}
    if method == "exec_as_user":
        username = require_user(params)
        command = params.get("cmd")
        if not isinstance(command, str) or not command:
            raise ValueError("cmd must be a non-empty string")
        completed = subprocess.run(
            ["runuser", "-u", username, "--", "sh", "-lc", command],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return {"stdout": completed.stdout, "exit_code": completed.returncode}
    if method == "subgoal_scan":
        require_user(params)
        return {"hits": []}
    if method == "availability_ping":
        services = params.get("services")
        if not isinstance(services, list) or not all(isinstance(item, str) for item in services):
            raise ValueError("services must be an array of strings")
        details = {service: service_active(service) for service in services}
        return {"ok": all(details.values()), "details": details}
    raise ValueError(f"unknown method: {method}")


def require_user(params: dict[str, Any]) -> str:
    username = params.get("username")
    if not isinstance(username, str) or not username:
        raise ValueError("username must be a non-empty string")
    pwd.getpwnam(username)
    return username


def probe_attacker_euid(username: str) -> int:
    uid = pwd.getpwnam(username).pw_uid
    if uid == 0:
        return 0
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            for line in status_path.read_text().splitlines():
                if not line.startswith("Uid:"):
                    continue
                real, effective, *_ = (int(value) for value in line.split()[1:])
                if real == uid and effective == 0:
                    return 0
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return uid


def service_active(service: str) -> bool:
    service = service.split(":", 1)[0]
    service = {"http": "nginx", "postgres": "postgresql"}.get(service, service)
    if service == "ssh":
        service = "sshd"
    return (
        subprocess.run(
            ["systemctl", "is-active", "--quiet", service],
            timeout=10,
            check=False,
        ).returncode
        == 0
    )


def handle(connection: socket.socket) -> None:
    raw = receive_line(connection)
    request_id: Any = None
    try:
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params")
        if not isinstance(method, str) or not isinstance(params, dict):
            raise ValueError("method and params are required")
        response = {"id": request_id, "result": dispatch(method, params)}
    except Exception as exc:
        response = {"id": request_id, "error": f"{type(exc).__name__}: {exc}"}
    connection.sendall((json.dumps(response) + "\n").encode())


def receive_line(connection: socket.socket) -> bytes:
    data = bytearray()
    while len(data) < MAX_REQUEST_BYTES:
        chunk = connection.recv(min(65536, MAX_REQUEST_BYTES - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if b"\n" in chunk:
            return bytes(data).split(b"\n", 1)[0]
    raise ValueError("request missing newline or exceeds 1 MiB")


def main() -> None:
    family = getattr(socket, "AF_VSOCK", None)
    if family is None:
        raise RuntimeError("AF_VSOCK is unavailable")
    cid_any = getattr(socket, "VMADDR_CID_ANY", -1)
    with socket.socket(family, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((cid_any, int(os.environ.get("ULTRON_GUEST_AGENT_PORT", PORT))))
        server.listen(16)
        while True:
            connection, _ = server.accept()
            with connection:
                handle(connection)


if __name__ == "__main__":
    main()
