from collections.abc import Iterator
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TMUX_JOB = ROOT / "scripts" / "tmux_job.sh"
LAUNCH_SCRIPTS = [
    ROOT / "scripts" / "run_generation.sh",
    ROOT / "scripts" / "rollout_worker.sh",
    ROOT / "scripts" / "train_grpo.sh",
    ROOT / "scripts" / "train_dpo.sh",
    ROOT / "scripts" / "serve_vllm_attacker.sh",
    ROOT / "scripts" / "serve_vllm_defender.sh",
]


def tmux_available() -> bool:
    return subprocess.call(["bash", "-lc", "command -v tmux >/dev/null"]) == 0


pytestmark = pytest.mark.skipif(not tmux_available(), reason="tmux is required")


@pytest.fixture
def job_env(tmp_path: Path) -> Iterator[dict[str, str]]:
    env = os.environ.copy()
    env["TMUX_TMPDIR"] = str(tmp_path / "tmux")
    env["ULTRON_TMUX_LOG_DIR"] = str(tmp_path / "logs")
    env["ULTRON_TMUX_SOCKET_PREFIX"] = "ultrontest"
    env["ULTRON_TMUX_DETACH"] = "1"
    env.pop("TMUX", None)
    env.pop("ULTRON_TMUX_SESSION", None)
    env.pop("ULTRON_NO_TMUX", None)
    (tmp_path / "tmux").mkdir()
    (tmp_path / "logs").mkdir()
    yield env
    listed = run_job(["list"], env)
    for line in listed.stdout.splitlines():
        session = line.split("\t")[0]
        if session and session != "No Ultron tmux sessions.":
            run_job(["stop", session], env)


def run_job(args: list[str], env: dict[str, str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(TMUX_JOB), *args],
        env=env,
        text=True,
        capture_output=True,
        **kwargs,
    )


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met before timeout")


def parse_status(env: dict[str, str], session: str) -> tuple[int, bool, str]:
    result = run_job(["status", session], env)
    assert result.returncode == 0, result.stderr
    parts = result.stdout.strip().split("\t")
    assert parts[0] == session
    return int(parts[1]), parts[2] == "1", parts[3]


def test_launch_scripts_opt_into_tmux() -> None:
    for path in LAUNCH_SCRIPTS:
        text = path.read_text()
        assert "lib_tmux.sh" in text
        assert "ultron_maybe_tmux" in text


def test_start_status_logs_and_stop(job_env: dict[str, str], tmp_path: Path) -> None:
    session = "ultrontest-alive"
    marker = tmp_path / "marker"
    started = run_job(
        ["start", session, "--", "bash", "-c", f"echo running >{marker}; sleep 30"],
        job_env,
    )
    assert started.returncode == 0, started.stderr
    assert "Started" in started.stdout
    wait_until(lambda: marker.exists())
    pid, dead, _cmd = parse_status(job_env, session)
    assert pid > 0
    assert not dead
    logs = run_job(["logs", session], job_env)
    assert logs.returncode == 0
    assert "start" in logs.stdout
    listed = run_job(["list"], job_env)
    assert session in listed.stdout
    stopped = run_job(["stop", session], job_env)
    assert stopped.returncode == 0, stopped.stderr
    wait_until(lambda: run_job(["status", session], job_env).returncode != 0)


def test_duplicate_start_fails(job_env: dict[str, str]) -> None:
    session = "ultrontest-dup"
    first = run_job(["start", session, "--", "sleep", "30"], job_env)
    assert first.returncode == 0, first.stderr
    try:
        second = run_job(["start", session, "--", "sleep", "30"], job_env)
        assert second.returncode == 1
        assert "already running" in second.stderr
    finally:
        run_job(["stop", session], job_env)


def test_job_survives_sighup(job_env: dict[str, str], tmp_path: Path) -> None:
    session = "ultrontest-hup"
    beat = tmp_path / "beat"
    started = run_job(
        [
            "start",
            session,
            "--",
            "bash",
            "-c",
            f"while true; do echo x >>{beat}; sleep 0.1; done",
        ],
        job_env,
    )
    assert started.returncode == 0, started.stderr
    try:
        wait_until(lambda: beat.exists() and beat.stat().st_size > 0)
        pid, dead, _cmd = parse_status(job_env, session)
        assert not dead
        os.kill(pid, signal.SIGHUP)
        time.sleep(0.3)
        size_after_hup = beat.stat().st_size
        wait_until(lambda: beat.stat().st_size > size_after_hup)
        _, still_dead, _ = parse_status(job_env, session)
        assert not still_dead
    finally:
        run_job(["stop", session], job_env)


def test_wrap_run_generation_detaches(job_env: dict[str, str]) -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_generation.sh"), "0"],
        env=job_env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "Started ultron-gen-0" in result.stdout
    try:
        wait_until(
            lambda: "Set ULTRON_ROLLOUT_COMMAND"
            in run_job(["logs", "ultron-gen-0"], job_env).stdout
        )
    finally:
        run_job(["stop", "ultron-gen-0"], job_env)


def test_nested_scripts_stay_in_parent_session(job_env: dict[str, str], tmp_path: Path) -> None:
    parent = tmp_path / "parent.sh"
    child = tmp_path / "child.sh"
    seen = tmp_path / "seen"
    parent.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'source "{ROOT}/scripts/lib_tmux.sh"',
                "ultron_maybe_tmux ultrontest-parent",
                f'echo "parent=${{ULTRON_TMUX_SESSION:-}}" > "{seen}"',
                f'bash "{child}"',
                "sleep 20",
            ]
        )
        + "\n"
    )
    child.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'source "{ROOT}/scripts/lib_tmux.sh"',
                "ultron_maybe_tmux ultrontest-child",
                f'echo "child=${{ULTRON_TMUX_SESSION:-}}" >> "{seen}"',
            ]
        )
        + "\n"
    )
    parent.chmod(0o755)
    child.chmod(0o755)
    wrapped = run_job(["wrap", "ultrontest-parent", str(parent)], job_env)
    assert wrapped.returncode == 0, wrapped.stderr
    try:
        wait_until(lambda: seen.exists() and "child=" in seen.read_text())
        text = seen.read_text()
        assert "parent=ultrontest-parent" in text
        assert "child=ultrontest-parent" in text
        listed = run_job(["list"], job_env).stdout
        assert "ultrontest-parent" in listed
        assert "ultrontest-child" not in listed
    finally:
        run_job(["stop", "ultrontest-parent"], job_env)


def test_no_tmux_runs_in_foreground() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "rollout_worker.sh"), "--generation", "0", "--episodes", "1"],
        env={**os.environ, "ULTRON_NO_TMUX": "1"},
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "ULTRON_ROLLOUT_COMMAND" in result.stderr
