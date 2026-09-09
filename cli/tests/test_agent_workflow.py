"""Run the agent commands against real isolated tmux jobs, without a GPU."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is required")


@pytest.mark.parametrize("exit_code", [0, 7])
def test_agent_can_launch_observe_read_and_clean_up_job(tmp_path, exit_code):
    env = dict(os.environ, TMUX_TMPDIR=str(tmp_path / "tmux"),
               ULTRON_TMUX_LOG_DIR=str(tmp_path / "logs"),
               ULTRON_PIPELINE_STATE_DIR=str(tmp_path / "states"),
               ULTRON_RESPONSES_DIR=str(tmp_path / "responses"))
    for key in ("TMUX", "ULTRON_TMUX_SESSION", "ULTRON_NO_TMUX"):
        env.pop(key, None)
    session = "ultrontest-agent"

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "ultron.cli.main", *args],
                              cwd=ROOT, env=env, text=True, capture_output=True, timeout=15)

    try:
        launched = cli("job", "start", "--json", session, "--", sys.executable, "-c",
                       f"import time; print('training started', flush=True); time.sleep(0.5); print('training ended', flush=True); raise SystemExit({exit_code})")
        assert launched.returncode == 0, launched.stderr + launched.stdout
        assert json.loads(launched.stdout)["result"] == "started"
        watched = cli("watch", "--session", session, "--json", "--no-gpu", "--interval", "0.1", "--timeout", "5")
        assert watched.returncode == (0 if exit_code == 0 else 1), watched.stderr + watched.stdout
        observations = [json.loads(line) for line in watched.stdout.splitlines()]
        assert observations[-1]["watch"]["reason"] == ("completed" if exit_code == 0 else "failed")
        assert observations[-1]["jobs"][0]["exit_code"] == exit_code
        logs = cli("logs", "--session", session, "--json")
        assert logs.returncode == 0, logs.stderr + logs.stdout
        document = json.loads(logs.stdout)
        assert "training ended" in document["text"]
        next_read = cli("logs", "--session", session, "--json", "--cursor", document["next_cursor"])
        assert next_read.returncode == 0
        assert json.loads(next_read.stdout)["text"] == ""
    finally:
        cli("job", "stop", "--json", session)
