from __future__ import annotations

import json
import socket
from itertools import count
from typing import Any


class GuestAgentError(RuntimeError):
    pass


class GuestAgentClient:
    def __init__(
        self,
        vm_id: str,
        vsock_cid: int,
        port: int = 9910,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        if vsock_cid < 3:
            raise ValueError("guest vsock CID must be at least 3")
        self.vm_id = vm_id
        self.vsock_cid = vsock_cid
        self.port = port
        self.timeout_s = timeout_s
        self._request_ids = count(1)

    def probe_attacker_euid(self, username: str) -> int:
        result = self._rpc("probe_attacker_euid", {"username": username})
        euid = result.get("euid")
        if not isinstance(euid, int) or isinstance(euid, bool):
            raise GuestAgentError("probe response has invalid euid")
        return euid

    def subgoal_scan(self, username: str) -> dict[str, Any]:
        return self._rpc("subgoal_scan", {"username": username})

    def exec_as_user(self, username: str, cmd: str) -> tuple[str, int]:
        result = self._rpc("exec_as_user", {"username": username, "cmd": cmd})
        stdout = result.get("stdout")
        exit_code = result.get("exit_code")
        if not isinstance(stdout, str) or not isinstance(exit_code, int):
            raise GuestAgentError("exec response has invalid fields")
        return stdout, exit_code

    def availability_ping(self, services: list[str]) -> dict[str, Any]:
        return self._rpc("availability_ping", {"services": services})

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = next(self._request_ids)
        request = json.dumps({"id": request_id, "method": method, "params": params}) + "\n"
        family = getattr(socket, "AF_VSOCK", None)
        if family is None:
            raise GuestAgentError("this host Python does not support AF_VSOCK")
        try:
            with socket.socket(family, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_s)
                connection.connect((self.vsock_cid, self.port))
                connection.sendall(request.encode())
                response = _read_line(connection)
        except OSError as exc:
            raise GuestAgentError(f"{self.vm_id}: guest-agent RPC failed: {exc}") from exc
        return _parse_response(response, request_id)


def _read_line(connection: socket.socket, limit: int = 1_048_576) -> bytes:
    chunks = bytearray()
    while len(chunks) < limit:
        chunk = connection.recv(min(65536, limit - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
        if b"\n" in chunk:
            return bytes(chunks).split(b"\n", 1)[0]
    raise GuestAgentError("guest-agent response missing newline or exceeds 1 MiB")


def _parse_response(raw: bytes, request_id: int) -> dict[str, Any]:
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GuestAgentError("guest-agent returned invalid JSON") from exc
    if not isinstance(response, dict) or response.get("id") != request_id:
        raise GuestAgentError("guest-agent response id mismatch")
    error = response.get("error")
    if error is not None:
        raise GuestAgentError(str(error))
    result = response.get("result")
    if not isinstance(result, dict):
        raise GuestAgentError("guest-agent result must be an object")
    return result
