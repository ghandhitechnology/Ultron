from pathlib import Path
from subprocess import CompletedProcess

import pytest

from ultron.env.backend import GuestHandle, IsolationBackend
from ultron.env.docker_backend import DockerBackend
from ultron.env.snapshot import SnapshotDriftError


class FakeRunner:
    def __init__(self, inspect_stdout: str = "1", inspect_returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.inspect_stdout = inspect_stdout
        self.inspect_returncode = inspect_returncode
        self.other_returncode = 0

    def __call__(self, args: list[str], **kwargs: object) -> CompletedProcess[str]:
        argv = list(args)
        self.calls.append(argv)
        if len(argv) >= 2 and argv[1] == "inspect":
            return CompletedProcess(argv, self.inspect_returncode, self.inspect_stdout)
        return CompletedProcess(argv, self.other_returncode, "")


def _write_guest_proc(
    proc_root: Path,
    init_pid: int,
    *,
    passwd: str,
    statuses: dict[int, str],
) -> None:
    guest_root = proc_root / str(init_pid) / "root"
    (guest_root / "etc").mkdir(parents=True)
    (guest_root / "etc" / "passwd").write_text(passwd)
    for pid, status in statuses.items():
        status_dir = guest_root / "proc" / str(pid)
        status_dir.mkdir(parents=True)
        (status_dir / "status").write_text(status)


def _backend(tmp_path: Path, runner: FakeRunner) -> DockerBackend:
    return DockerBackend(run=runner, proc_root=tmp_path)


def _guest() -> GuestHandle:
    return GuestHandle(
        guest_id="guest-1",
        isolation=IsolationBackend.DOCKER,
        host_address="guest-1",
        image_ref="ultron/golden:latest",
    )


def test_confirm_root_true_when_matching_uid_line(tmp_path: Path) -> None:
    runner = FakeRunner(inspect_stdout="42\n")
    _write_guest_proc(
        tmp_path,
        42,
        passwd="attacker:x:1000:1000::/home/attacker:/bin/sh\n",
        statuses={7: "Name:\tbash\nUid:\t1000\t0\t0\t0\n"},
    )
    backend = _backend(tmp_path, runner)

    assert backend.confirm_root(_guest(), "attacker") is True


def test_confirm_root_false_when_euid_zero_for_other_user(tmp_path: Path) -> None:
    runner = FakeRunner(inspect_stdout="42\n")
    _write_guest_proc(
        tmp_path,
        42,
        passwd="attacker:x:1000:1000::/home/attacker:/bin/sh\n",
        statuses={7: "Name:\tbash\nUid:\t1001\t0\t0\t0\n"},
    )
    backend = _backend(tmp_path, runner)

    assert backend.confirm_root(_guest(), "attacker") is False


def test_confirm_root_false_when_proc_missing(tmp_path: Path) -> None:
    runner = FakeRunner(inspect_stdout="42\n")
    backend = _backend(tmp_path, runner)

    assert backend.confirm_root(_guest(), "attacker") is False


def test_confirm_root_never_invokes_docker_exec(tmp_path: Path) -> None:
    runner = FakeRunner(inspect_stdout="42\n")
    _write_guest_proc(
        tmp_path,
        42,
        passwd="attacker:x:1000:1000::/home/attacker:/bin/sh\n",
        statuses={7: "Uid:\t1000\t0\t0\t0\n"},
    )
    backend = _backend(tmp_path, runner)

    backend.confirm_root(_guest(), "attacker")

    assert all(len(cmd) < 2 or cmd[1] != "exec" for cmd in runner.calls)


class CopyAndIdRunner:
    def __init__(self, container_id: str = "abc123containerid") -> None:
        self.calls: list[list[str]] = []
        self.container_id = container_id

    def __call__(self, args: list[str], **kwargs: object) -> CompletedProcess[str]:
        argv = list(args)
        self.calls.append(argv)
        if len(argv) >= 2 and argv[1] == "inspect":
            fmt = argv[argv.index("--format") + 1] if "--format" in argv else ""
            if ".State.Pid" in fmt:
                return CompletedProcess(argv, 0, "42\n")
            if ".Id" in fmt:
                return CompletedProcess(argv, 0, self.container_id + "\n")
            return CompletedProcess(argv, 0, "42\n")
        if len(argv) >= 2 and argv[1] == "cp":
            Path(argv[3]).write_text("attacker:x:1000:1000::/home/attacker:/bin/sh\n")
            return CompletedProcess(argv, 0, "")
        return CompletedProcess(argv, 1, "")


def test_confirm_root_host_proc_fallback_without_guest_root(tmp_path: Path) -> None:
    runner = CopyAndIdRunner()
    host_proc = tmp_path / "99"
    host_proc.mkdir()
    (host_proc / "status").write_text("Name:\tpython\nUid:\t1000\t0\t0\t0\n")
    (host_proc / "cgroup").write_text("0::/docker/abc123containerid\n")
    backend = DockerBackend(run=runner, proc_root=tmp_path)

    assert backend.confirm_root(_guest(), "attacker") is True
    assert all(len(cmd) < 2 or cmd[1] != "exec" for cmd in runner.calls)
    assert any(len(cmd) >= 2 and cmd[1] == "cp" for cmd in runner.calls)


def test_verify_image_mismatch_raises() -> None:
    runner = FakeRunner(inspect_stdout="sha256:other\n")
    backend = DockerBackend(run=runner, proc_root=Path("/unused"))

    with pytest.raises(SnapshotDriftError):
        backend.verify_image("ultron/golden:latest", "sha256:expected")


def test_restore_issues_rm_create_start() -> None:
    runner = FakeRunner(inspect_stdout="10.0.0.8\n")
    backend = DockerBackend(run=runner, proc_root=Path("/unused"))

    handle = backend.restore("guest-1", "ultron/golden:latest", timeout_s=30)

    prefixes = [tuple(cmd[:2]) for cmd in runner.calls]
    assert ("docker", "rm") in prefixes
    assert ("docker", "create") in prefixes
    assert ("docker", "start") in prefixes
    create = next(cmd for cmd in runner.calls if cmd[:2] == ["docker", "create"])
    inspect = next(cmd for cmd in runner.calls if cmd[:2] == ["docker", "inspect"])
    fmt = inspect[inspect.index("--format") + 1]
    assert "--cpus" in create and "--memory" in create
    assert "index" in fmt and "ultron-isolated" in fmt and "IPAddress" in fmt
    assert handle.guest_id == "guest-1"
    assert handle.host_address == "10.0.0.8"
    assert handle.image_ref == "ultron/golden:latest"


def test_restore_requires_network_ip() -> None:
    runner = FakeRunner(inspect_stdout="<no value>\n")
    backend = DockerBackend(run=runner, proc_root=Path("/unused"))

    with pytest.raises(RuntimeError, match="assigned no IP"):
        backend.restore("guest-1", "ultron/golden:latest", timeout_s=30)
