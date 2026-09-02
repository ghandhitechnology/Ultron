from __future__ import annotations

from queue import Empty, Queue

from .backend import GuestBackend, GuestHandle

VmHandle = GuestHandle


def restore_verified(
    handle: GuestHandle,
    expected_id: str,
    backend: GuestBackend,
    *,
    restore_timeout_s: int = 30,
) -> GuestHandle:
    backend.verify_image(handle.image_ref, expected_id)
    return backend.restore(handle.guest_id, handle.image_ref, restore_timeout_s)


class VmPool:
    def __init__(
        self,
        handles: list[GuestHandle],
        backend: GuestBackend,
        *,
        restore_timeout_s: int = 30,
    ) -> None:
        if not handles:
            raise ValueError("VM pool requires at least one handle")
        self.backend = backend
        self.restore_timeout_s = restore_timeout_s
        self._available: Queue[GuestHandle] = Queue(maxsize=len(handles))
        self._quarantined: dict[str, str] = {}
        for handle in handles:
            self._available.put(handle)

    def acquire(self, timeout_s: float | None = None) -> GuestHandle:
        try:
            return self._available.get(timeout=timeout_s)
        except Empty as exc:
            raise TimeoutError("no VM became available") from exc

    def restore(self, handle: GuestHandle) -> GuestHandle:
        return self.backend.restore(
            handle.guest_id,
            handle.image_ref,
            self.restore_timeout_s,
        )

    def restore_verified(self, handle: GuestHandle, expected_id: str) -> GuestHandle:
        return restore_verified(
            handle,
            expected_id,
            self.backend,
            restore_timeout_s=self.restore_timeout_s,
        )

    def release(self, handle: GuestHandle) -> None:
        if handle.guest_id not in self._quarantined:
            self._available.put(handle)

    def quarantine(self, handle: GuestHandle, reason: str) -> None:
        self._quarantined[handle.guest_id] = reason
        self.backend.stop(handle.guest_id)

    @property
    def quarantined(self) -> dict[str, str]:
        return dict(self._quarantined)
