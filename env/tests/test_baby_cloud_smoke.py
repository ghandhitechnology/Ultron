import shutil
import subprocess

import pytest

from ultron.env.baby_smoke import _python_connect_cmd, docker_daemon_ok, main
from ultron.env.docker_backend import _usable_ip


def test_usable_ip_rejects_placeholders() -> None:
    assert _usable_ip(" 10.0.0.8 \n") == "10.0.0.8"
    assert _usable_ip("") is None
    assert _usable_ip("<no value>") is None
    assert _usable_ip("<nil>") is None


def test_python_connect_cmd_targets_host_port() -> None:
    cmd = _python_connect_cmd("172.18.0.1", 18001)
    assert "172.18.0.1" in cmd
    assert "18001" in cmd
    assert "connect_ex" in cmd


def test_docker_daemon_ok_without_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ultron.env.baby_smoke.shutil.which", lambda name: None)
    assert docker_daemon_ok() is False


@pytest.mark.docker
@pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "info"], capture_output=True).returncode != 0,
    reason="docker daemon required",
)
def test_baby_cloud_smoke_live() -> None:
    assert main([]) == 0
