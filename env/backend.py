from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class IsolationBackend(str, Enum):
    KVM = "kvm"
    DOCKER = "docker"


@dataclass(frozen=True)
class GuestHandle:
    guest_id: str
    isolation: IsolationBackend
    host_address: str
    image_ref: str

    @property
    def vm_id(self) -> str:
        return self.guest_id


class GuestBackend(Protocol):
    isolation: IsolationBackend

    def verify_image(self, image_ref: str, expected_id: str) -> None: ...

    def restore(self, guest_id: str, image_ref: str, timeout_s: int) -> GuestHandle: ...

    def stop(self, guest_id: str) -> None: ...

    def exec_as_user(self, guest: GuestHandle, username: str, cmd: str) -> tuple[str, int]: ...

    def confirm_root(self, guest: GuestHandle, username: str) -> bool: ...
