from pathlib import Path

import pytest

from ultron.env.snapshot import SnapshotDriftError, sha256_file
from ultron.env.vm_pool import VmHandle, VmPool, restore_verified


class RecordingBackend:
    def __init__(self) -> None:
        self.restored: list[tuple[str, Path, int]] = []
        self.stopped: list[str] = []

    def restore(self, vm_id: str, snapshot_path: Path, timeout_s: int) -> None:
        self.restored.append((vm_id, snapshot_path, timeout_s))

    def stop(self, vm_id: str) -> None:
        self.stopped.append(vm_id)


def make_handle(path: Path) -> VmHandle:
    return VmHandle(vm_id="vm-1", vsock_cid=3, host_address="10.0.0.2", snapshot_path=path)


def test_restore_verified_verifies_then_restores(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.qcow2"
    snapshot.write_bytes(b"snapshot-bytes")
    handle = make_handle(snapshot)
    backend = RecordingBackend()

    restore_verified(handle, sha256_file(snapshot), backend, restore_timeout_s=45)

    assert backend.restored == [("vm-1", snapshot, 45)]


def test_restore_verified_rejects_drift(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.qcow2"
    snapshot.write_bytes(b"snapshot-bytes")
    handle = make_handle(snapshot)
    backend = RecordingBackend()

    with pytest.raises(SnapshotDriftError):
        restore_verified(handle, "0" * 64, backend)

    assert backend.restored == []


def test_pool_restore_verified(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.qcow2"
    snapshot.write_bytes(b"snapshot-bytes")
    handle = make_handle(snapshot)
    backend = RecordingBackend()
    pool = VmPool([handle], backend, restore_timeout_s=30)

    pool.restore_verified(handle, sha256_file(snapshot))

    assert backend.restored == [("vm-1", snapshot, 30)]
