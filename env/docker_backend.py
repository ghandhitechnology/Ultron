from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

from .backend import GuestHandle, IsolationBackend
from .snapshot import SnapshotDriftError

Runner = Callable[..., CompletedProcess[str]]

_DEFAULT_NETWORK = "ultron-isolated"


class DockerBackend:
    isolation = IsolationBackend.DOCKER

    def __init__(
        self,
        run: Runner | None = None,
        proc_root: Path = Path("/proc"),
        network: str = _DEFAULT_NETWORK,
        cpus: str = "2",
        memory: str = "4g",
    ) -> None:
        self._run = run if run is not None else subprocess.run
        self.proc_root = proc_root
        self.network = network
        self.cpus = cpus
        self.memory = memory

    def verify_image(self, image_ref: str, expected_id: str) -> None:
        result = self._invoke(["docker", "inspect", "--format", "{{.Id}}", image_ref])
        actual = (result.stdout or "").strip()
        if result.returncode != 0 or actual != expected_id:
            raise SnapshotDriftError(
                f"image id mismatch for {image_ref}: expected {expected_id}, got {actual}"
            )

    def restore(self, guest_id: str, image_ref: str, timeout_s: int) -> GuestHandle:
        self._invoke(["docker", "rm", "-f", guest_id])
        created = self._invoke(
            [
                "docker",
                "create",
                "--name",
                guest_id,
                "--network",
                self.network,
                "--cpus",
                self.cpus,
                "--memory",
                self.memory,
                image_ref,
            ],
            timeout=timeout_s,
        )
        if created.returncode != 0:
            raise RuntimeError(f"docker create failed for {guest_id}: {created.stderr}")
        started = self._invoke(["docker", "start", guest_id], timeout=timeout_s)
        if started.returncode != 0:
            raise RuntimeError(f"docker start failed for {guest_id}: {started.stderr}")
        host_address = self._container_ip(guest_id)
        if host_address is None:
            raise RuntimeError(
                f"docker network {self.network} assigned no IP to {guest_id}"
            )
        return GuestHandle(
            guest_id=guest_id,
            isolation=self.isolation,
            host_address=host_address,
            image_ref=image_ref,
        )

    def stop(self, guest_id: str) -> None:
        self._invoke(["docker", "rm", "-f", guest_id])

    def exec_as_user(self, guest: GuestHandle, username: str, cmd: str) -> tuple[str, int]:
        result = self._invoke(
            ["docker", "exec", "-u", username, guest.guest_id, "sh", "-c", cmd]
        )
        return (result.stdout or "", result.returncode)

    def confirm_root(self, guest: GuestHandle, username: str) -> bool:
        try:
            result = self._invoke(
                ["docker", "inspect", "--format", "{{.State.Pid}}", guest.guest_id]
            )
            pid = int((result.stdout or "").strip())
        except (OSError, TypeError, ValueError):
            return False
        if result.returncode != 0 or pid <= 0:
            return False
        attacker_uid = self._lookup_uid(guest, username, pid)
        if attacker_uid is None:
            return False
        if self._guest_proc_has_euid0(pid, attacker_uid):
            return True
        return self._host_proc_has_euid0(guest, attacker_uid)

    def _lookup_uid(self, guest: GuestHandle, username: str, pid: int) -> int | None:
        passwd_path = self.proc_root / str(pid) / "root" / "etc" / "passwd"
        try:
            return _passwd_uid(passwd_path.read_text(), username)
        except OSError:
            return self._copy_passwd_uid(guest, username)

    def _copy_passwd_uid(self, guest: GuestHandle, username: str) -> int | None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "passwd"
            result = self._invoke(
                ["docker", "cp", f"{guest.guest_id}:/etc/passwd", str(dest)]
            )
            if result.returncode != 0:
                return None
            try:
                return _passwd_uid(dest.read_text(), username)
            except OSError:
                return None

    def _guest_proc_has_euid0(self, pid: int, attacker_uid: int) -> bool:
        proc_dir = self.proc_root / str(pid) / "root" / "proc"
        try:
            status_paths = list(proc_dir.glob("[0-9]*/status"))
        except OSError:
            return False
        return _status_paths_have_euid0(status_paths, attacker_uid)

    def _host_proc_has_euid0(self, guest: GuestHandle, attacker_uid: int) -> bool:
        result = self._invoke(
            ["docker", "inspect", "--format", "{{.Id}}", guest.guest_id]
        )
        container_id = (result.stdout or "").strip()
        if result.returncode != 0 or not container_id:
            return False
        needles = (container_id, container_id[:12])
        try:
            status_paths = list(self.proc_root.glob("[0-9]*/status"))
        except OSError:
            return False
        for status_path in status_paths:
            try:
                cgroup = (status_path.parent / "cgroup").read_text()
            except OSError:
                continue
            if not any(needle in cgroup for needle in needles):
                continue
            try:
                text = status_path.read_text()
            except OSError:
                continue
            if _status_text_has_euid0(text, attacker_uid):
                return True
        return False

    def _container_ip(self, guest_id: str) -> str | None:
        fmt = f'{{{{(index .NetworkSettings.Networks "{self.network}").IPAddress}}}}'
        result = self._invoke(["docker", "inspect", "--format", fmt, guest_id])
        return _usable_ip(result.stdout or "")

    def _invoke(
        self,
        args: Sequence[str],
        *,
        timeout: int | None = None,
    ) -> CompletedProcess[str]:
        kwargs: dict[str, Any] = {"capture_output": True, "text": True}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return self._run(list(args), **kwargs)


def _usable_ip(text: str) -> str | None:
    value = text.strip()
    if not value or value in {"<no value>", "<nil>"}:
        return None
    return value


def _passwd_uid(passwd_text: str, username: str) -> int | None:
    for line in passwd_text.splitlines():
        fields = line.split(":")
        if len(fields) < 3 or fields[0] != username:
            continue
        try:
            return int(fields[2])
        except ValueError:
            return None
    return None


def _parse_uid_line(line: str) -> tuple[int, int] | None:
    if not line.startswith("Uid:"):
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _status_text_has_euid0(text: str, attacker_uid: int) -> bool:
    for line in text.splitlines():
        parsed = _parse_uid_line(line)
        if parsed is None:
            continue
        real_uid, effective_uid = parsed
        if real_uid == attacker_uid and effective_uid == 0:
            return True
    return False


def _status_paths_have_euid0(status_paths: list[Path], attacker_uid: int) -> bool:
    for status_path in status_paths:
        try:
            text = status_path.read_text()
        except OSError:
            continue
        if _status_text_has_euid0(text, attacker_uid):
            return True
    return False
