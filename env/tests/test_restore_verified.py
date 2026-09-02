import pytest

from ultron.env.backend import GuestHandle, IsolationBackend
from ultron.env.snapshot import SnapshotDriftError
from ultron.env.vm_pool import VmPool, restore_verified


class RecordingBackend:
    isolation = IsolationBackend.KVM

    def __init__(self, *, accept_id: str | None = None) -> None:
        self.accept_id = accept_id
        self.verified: list[tuple[str, str]] = []
        self.restored: list[tuple[str, str, int]] = []
        self.stopped: list[str] = []

    def verify_image(self, image_ref: str, expected_id: str) -> None:
        self.verified.append((image_ref, expected_id))
        if self.accept_id is not None and expected_id != self.accept_id:
            raise SnapshotDriftError(f"image mismatch for {image_ref}")

    def restore(self, guest_id: str, image_ref: str, timeout_s: int) -> GuestHandle:
        self.restored.append((guest_id, image_ref, timeout_s))
        return GuestHandle(
            guest_id=guest_id,
            isolation=self.isolation,
            host_address="10.0.0.2",
            image_ref=image_ref,
        )

    def stop(self, guest_id: str) -> None:
        self.stopped.append(guest_id)

    def exec_as_user(self, guest: GuestHandle, username: str, cmd: str) -> tuple[str, int]:
        return ("", 1)

    def confirm_root(self, guest: GuestHandle, username: str) -> bool:
        return False


def make_handle() -> GuestHandle:
    return GuestHandle(
        guest_id="vm-1",
        isolation=IsolationBackend.KVM,
        host_address="10.0.0.2",
        image_ref="img:golden",
    )


def test_restore_verified_verifies_then_restores() -> None:
    handle = make_handle()
    backend = RecordingBackend()

    restore_verified(handle, "sha-ok", backend, restore_timeout_s=45)

    assert backend.verified == [("img:golden", "sha-ok")]
    assert backend.restored == [("vm-1", "img:golden", 45)]


def test_restore_verified_rejects_drift() -> None:
    handle = make_handle()
    backend = RecordingBackend(accept_id="sha-ok")

    with pytest.raises(SnapshotDriftError):
        restore_verified(handle, "0" * 64, backend)

    assert backend.restored == []


def test_pool_restore_verified() -> None:
    handle = make_handle()
    backend = RecordingBackend()
    pool = VmPool([handle], backend, restore_timeout_s=30)

    pool.restore_verified(handle, "sha-ok")

    assert backend.verified == [("img:golden", "sha-ok")]
    assert backend.restored == [("vm-1", "img:golden", 30)]
