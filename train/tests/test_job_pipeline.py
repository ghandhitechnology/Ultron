from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_LIB = ROOT / "scripts" / "lib_pipeline.sh"
RUNTIME_LIB = ROOT / "scripts" / "lib_runtime.sh"


def run_bash(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def wait_for_path(path: Path, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"path was not created before timeout: {path}")


def wait_for_state(path: Path, state: str, timeout: float = 5.0) -> dict[str, str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            values = dict(
                line.split("=", 1)
                for line in path.read_text().splitlines()
                if "=" in line
            )
        except FileNotFoundError:
            values = {}
        if values.get("state") == state:
            return values
        time.sleep(0.05)
    raise AssertionError(f"state {state!r} was not recorded before timeout: {path}")


def read_state(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
    )


def test_stage_retries_then_records_completion(tmp_path: Path) -> None:
    state = tmp_path / "state"
    attempts = tmp_path / "attempts"
    script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_STAGE_MAX_ATTEMPTS=2
    export ULTRON_STAGE_RETRY_DELAY_SECONDS=0
    ultron_pipeline_init pipeline
    flaky() {{
      count=0
      [[ -f {attempts!s} ]] && count="$(cat {attempts!s})"
      count=$((count + 1))
      printf '%s\n' "$count" > {attempts!s}
      [[ "$count" -ge 2 ]]
    }}
    ultron_run_stage rollout flaky
    """

    result = run_bash(script)

    assert result.returncode == 0, result.stderr
    assert attempts.read_text().strip() == "2"
    assert (state / "rollout.done").is_file()
    assert not (state / "rollout.running").exists()
    assert not (state / "rollout.failed").exists()


def test_retry_delay_is_recorded_as_live_retrying_state(tmp_path: Path) -> None:
    state = tmp_path / "state"
    attempts = tmp_path / "attempts"
    script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_TMUX_SESSION=ultron-test-session
    export ULTRON_STAGE_MAX_ATTEMPTS=2
    export ULTRON_STAGE_RETRY_DELAY_SECONDS=2
    ultron_pipeline_init pipeline
    flaky() {{
      count=0
      [[ -f {attempts!s} ]] && count="$(cat {attempts!s})"
      count=$((count + 1))
      printf '%s\n' "$count" > {attempts!s}
      [[ "$count" -ge 2 ]]
    }}
    ultron_run_stage rollout flaky
    """
    process = subprocess.Popen(
        ["bash", "-c", script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        retrying = wait_for_state(state / "rollout.running", "retrying")
        assert retrying["pipeline"] == "pipeline"
        assert retrying["stage"] == "rollout"
        assert retrying["session"] == "ultron-test-session"
        assert retrying["pid"] == str(process.pid)
        assert retrying["attempt"] == "1"
        assert retrying["max_attempts"] == "2"
        assert retrying["status"] == "1"
        assert int(retrying["retry_at"]) >= int(time.time())
        assert retrying["started_at"].endswith("Z")
        assert not (state / "rollout.failed").exists()
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)

    assert process.returncode == 0, stderr
    assert "attempt 2/2" in stdout
    done = wait_for_state(state / "rollout.done", "done")
    assert done["attempt"] == "2"
    assert done["started_at"] == retrying["started_at"]


def test_completed_stage_is_skipped_on_resume(tmp_path: Path) -> None:
    state = tmp_path / "state"
    marker = tmp_path / "ran"
    script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    ultron_pipeline_init pipeline
    ultron_run_stage review bash -c 'printf x >> {marker!s}'
    """

    assert run_bash(script).returncode == 0
    resumed = run_bash(script)

    assert resumed.returncode == 0, resumed.stderr
    assert marker.read_text() == "x"
    assert "already complete" in resumed.stdout


def test_each_pipeline_init_writes_a_new_run_manifest_without_breaking_resume(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    marker = tmp_path / "ran"
    script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_TMUX_SESSION=ultron-test-session
    ultron_pipeline_init pipeline
    ultron_run_stage review bash -c 'printf x >> {marker!s}'
    """

    first = run_bash(script)
    assert first.returncode == 0, first.stderr
    first_manifest = read_state(state / ".pipeline")
    first_done = read_state(state / "review.done")

    second = run_bash(script)
    assert second.returncode == 0, second.stderr
    second_manifest = read_state(state / ".pipeline")
    second_done = read_state(state / "review.done")

    assert first_manifest["run_id"] != second_manifest["run_id"]
    assert second_manifest["pipeline"] == "pipeline"
    assert second_manifest["session"] == "ultron-test-session"
    assert second_manifest["pid"].isdigit()
    assert second_manifest["started_at"].endswith("Z")
    assert first_done["run_id"] == first_manifest["run_id"]
    assert second_done["run_id"] == first_manifest["run_id"]
    assert marker.read_text() == "x"
    assert "already complete" in second.stdout


def test_changed_stage_command_invalidates_completion(tmp_path: Path) -> None:
    state = tmp_path / "state"
    marker = tmp_path / "ran"
    first = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    ultron_pipeline_init pipeline
    ultron_run_stage review bash -c 'printf first > {marker!s}'
    """
    changed = first.replace("printf first", "printf second")

    assert run_bash(first).returncode == 0
    result = run_bash(changed)

    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "second"
    assert "inputs changed" in result.stdout


def test_changed_stage_executable_invalidates_completion(tmp_path: Path) -> None:
    state = tmp_path / "state"
    marker = tmp_path / "ran"
    command = tmp_path / "stage.sh"
    command.write_text(f"#!/usr/bin/env bash\nprintf x >> {marker!s}\n")
    command.chmod(0o755)
    script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    ultron_pipeline_init pipeline
    ultron_run_stage review {command!s}
    """

    assert run_bash(script).returncode == 0
    command.write_text(f"#!/usr/bin/env bash\nprintf y >> {marker!s}\n")
    result = run_bash(script)

    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "xy"
    assert "inputs changed" in result.stdout


def test_pipeline_input_key_invalidates_completion(tmp_path: Path) -> None:
    state = tmp_path / "state"
    marker = tmp_path / "ran"
    script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_PIPELINE_INPUT_KEY=first
    ultron_pipeline_init pipeline
    ultron_run_stage review bash -c 'printf x >> {marker!s}'
    """

    assert run_bash(script).returncode == 0
    changed = run_bash(script.replace("INPUT_KEY=first", "INPUT_KEY=second"))

    assert changed.returncode == 0, changed.stderr
    assert marker.read_text() == "xx"


def test_upstream_rerun_invalidates_downstream_completion_across_resume(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    prepared = tmp_path / "prepared"
    trained = tmp_path / "trained"

    def pipeline(value: str, *, include_train: bool = True) -> str:
        train = (
            f"ultron_run_stage train bash -c 'cat {prepared!s} >> {trained!s}'"
            if include_train
            else ""
        )
        return f"""
        set -euo pipefail
        source {PIPELINE_LIB!s}
        export ULTRON_PIPELINE_STATE_DIR={state!s}
        ultron_pipeline_init pipeline
        ultron_run_stage prepare bash -c 'printf "{value}\\n" > {prepared!s}'
        {train}
        """

    assert run_bash(pipeline("one")).returncode == 0
    assert run_bash(pipeline("two", include_train=False)).returncode == 0
    resumed = run_bash(pipeline("two"))

    assert resumed.returncode == 0, resumed.stderr
    assert trained.read_text().splitlines() == ["one", "two"]
    assert "Stage prepare: already complete" in resumed.stdout
    assert "Stage train: inputs changed" in resumed.stdout


def test_failed_rerun_removes_stale_completion(tmp_path: Path) -> None:
    state = tmp_path / "state"
    marker = tmp_path / "ran"
    complete = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_STAGE_MAX_ATTEMPTS=1
    ultron_pipeline_init pipeline
    ultron_run_stage review bash -c 'printf x >> {marker!s}'
    """
    failed = complete.replace(
        f"printf x >> {marker!s}", f"printf failed > {tmp_path / 'failed'!s}; exit 9"
    )

    assert run_bash(complete).returncode == 0
    assert run_bash(failed).returncode == 9
    assert not (state / "review.done").exists()
    recovered = run_bash(complete)

    assert recovered.returncode == 0, recovered.stderr
    assert marker.read_text() == "xx"


def test_failed_stage_is_recovered_on_next_run(tmp_path: Path) -> None:
    state = tmp_path / "state"
    failed = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_STAGE_MAX_ATTEMPTS=1
    ultron_pipeline_init pipeline
    ultron_run_stage train bash -c 'exit 19'
    """
    recovered = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_STAGE_MAX_ATTEMPTS=1
    ultron_pipeline_init pipeline
    ultron_run_stage train true
    """

    first = run_bash(failed)
    assert first.returncode == 19
    assert (state / "train.failed").is_file()
    assert not (state / "train.done").exists()

    second = run_bash(recovered)
    assert second.returncode == 0, second.stderr
    assert "Recovering unfinished stage" in second.stdout
    assert (state / "train.done").is_file()


def test_recovery_removes_stale_failure_when_attempt_starts(tmp_path: Path) -> None:
    state = tmp_path / "state"
    entered = tmp_path / "entered"
    failed_script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_STAGE_MAX_ATTEMPTS=1
    ultron_pipeline_init pipeline
    ultron_run_stage train bash -c 'exit 19'
    """
    recovery_script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_STAGE_MAX_ATTEMPTS=1
    ultron_pipeline_init pipeline
    ultron_run_stage train bash -c 'touch {entered!s}; sleep 2'
    """

    assert run_bash(failed_script).returncode == 19
    failure = wait_for_state(state / "train.failed", "failed")
    assert failure["status"] == "19"

    process = subprocess.Popen(
        ["bash", "-c", recovery_script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        wait_for_path(entered)
        running = wait_for_state(state / "train.running", "running")
        assert running["pid"] == str(process.pid)
        assert not (state / "train.failed").exists()
        _, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)

    assert process.returncode == 0, stderr


def test_configuration_failure_is_not_retried(tmp_path: Path) -> None:
    state = tmp_path / "state"
    attempts = tmp_path / "attempts"
    script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    export ULTRON_STAGE_MAX_ATTEMPTS=3
    export ULTRON_STAGE_RETRY_DELAY_SECONDS=0
    ultron_pipeline_init pipeline
    invalid() {{ printf x >> {attempts!s}; return 2; }}
    ultron_run_stage rollout invalid
    """

    result = run_bash(script)

    assert result.returncode == 2
    assert attempts.read_text() == "x"
    assert "retry disabled" in result.stderr


def test_concurrent_stage_is_rejected(tmp_path: Path) -> None:
    state = tmp_path / "state"
    entered = tmp_path / "entered"
    first_script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    ultron_pipeline_init pipeline
    ultron_run_stage train bash -c 'touch {entered!s}; sleep 30'
    """
    first = subprocess.Popen(
        ["bash", "-c", first_script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        wait_for_path(entered)
        second = run_bash(
            f"""
            set -euo pipefail
            source {PIPELINE_LIB!s}
            export ULTRON_PIPELINE_STATE_DIR={state!s}
            ultron_pipeline_init pipeline
            ultron_run_stage train true
            """
        )
        assert second.returncode == 75
        assert "already running" in second.stderr
    finally:
        if first.poll() is None:
            os.killpg(first.pid, signal.SIGTERM)
        first.communicate(timeout=5)


def test_stale_stage_lock_is_reclaimed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    lock = state / "train.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text("99999999\n")
    result = run_bash(
        f"""
        set -euo pipefail
        source {PIPELINE_LIB!s}
        export ULTRON_PIPELINE_STATE_DIR={state!s}
        ultron_pipeline_init pipeline
        ultron_run_stage train true
        """
    )

    assert result.returncode == 0, result.stderr
    assert (state / "train.done").exists()
    assert not lock.exists()


def test_interrupted_stage_records_failure_and_releases_lock(tmp_path: Path) -> None:
    state = tmp_path / "state"
    script = f"""
    set -euo pipefail
    source {PIPELINE_LIB!s}
    export ULTRON_PIPELINE_STATE_DIR={state!s}
    ultron_pipeline_init pipeline
    ultron_run_stage train sleep 30
    """
    process = subprocess.Popen(
        ["bash", "-c", script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        wait_for_path(state / "train.running")
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)

    assert process.returncode == 143
    assert not (state / "train.running").exists()
    assert not (state / "train.lock").exists()
    failed = (state / "train.failed").read_text()
    assert "state=failed" in failed
    assert "pipeline=pipeline" in failed
    assert "stage=train" in failed
    assert f"pid={process.pid}" in failed
    assert "max_attempts=2" in failed
    assert "status=143" in failed
    assert "signal=TERM" in failed


def test_runtime_prefers_project_virtualenv(tmp_path: Path) -> None:
    project = tmp_path / "project"
    python = project / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env bash\nexit 0\n")
    python.chmod(0o755)
    script = f"""
    set -euo pipefail
    source {RUNTIME_LIB!s}
    ultron_load_runtime {project!s}
    printf '%s\n' "$ULTRON_PYTHON"
    """

    result = run_bash(script, env={"PATH": os.environ["PATH"]})

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(python)
