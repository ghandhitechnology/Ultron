from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Protocol


class VmBackend(Protocol):
    def restore(self, vm_id: str, snapshot_path: Path, timeout_s: int) -> None: ...

    def stop(self, vm_id: str) -> None: ...


@dataclass(frozen=True)
class VmHandle:
    vm_id: str
    vsock_cid: int
    host_address: str
    snapshot_path: Path


class VmPool:
    """A fixed-size lease pool. The libvirt backend owns VM lifecycle details."""

    def __init__(
        self,
        handles: list[VmHandle],
        backend: VmBackend,
        *,
        restore_timeout_s: int = 30,
    ) -> None:
        if not handles:
            raise ValueError("VM pool requires at least one handle")
        self.backend = backend
        self.restore_timeout_s = restore_timeout_s
        self._available: Queue[VmHandle] = Queue(maxsize=len(handles))
        self._quarantined: dict[str, str] = {}
        for handle in handles:
            self._available.put(handle)

    def acquire(self, timeout_s: float | None = None) -> VmHandle:
        try:
            return self._available.get(timeout=timeout_s)
        except Empty as exc:
            raise TimeoutError("no VM became available") from exc

    def restore(self, handle: VmHandle) -> None:
        self.backend.restore(
            handle.vm_id,
            handle.snapshot_path,
            self.restore_timeout_s,
        )

    def release(self, handle: VmHandle) -> None:
        if handle.vm_id not in self._quarantined:
            self._available.put(handle)

    def quarantine(self, handle: VmHandle, reason: str) -> None:
        self._quarantined[handle.vm_id] = reason
        self.backend.stop(handle.vm_id)

    @property
    def quarantined(self) -> dict[str, str]:
        return dict(self._quarantined)
