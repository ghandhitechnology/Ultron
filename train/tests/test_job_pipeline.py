from __future__ import annotations

import os
import subprocess
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
