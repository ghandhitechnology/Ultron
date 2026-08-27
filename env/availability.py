from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ServiceProbe:
    name: str
    port: int


@dataclass(frozen=True)
class AvailabilityResult:
    ok: bool
    details: dict[str, bool]


def parse_service(value: str) -> ServiceProbe:
    try:
        name, raw_port = value.rsplit(":", 1)
        port = int(raw_port)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid service probe {value!r}, expected name:port") from exc
    if not name or not 1 <= port <= 65535:
        raise ValueError(f"invalid service probe {value!r}")
    return ServiceProbe(name=name, port=port)


def tcp_probe(host: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def check_availability(
    host: str,
    services: list[ServiceProbe],
    *,
    probe: Callable[[str, int], bool] = tcp_probe,
) -> AvailabilityResult:
    details = {service.name: probe(host, service.port) for service in services}
    return AvailabilityResult(ok=all(details.values()), details=details)
