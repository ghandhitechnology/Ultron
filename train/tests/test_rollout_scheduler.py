from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import pytest

from ultron.env.backend import GuestHandle, IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.episode_runner import EpisodeConfig, EpisodeRunner
from ultron.train.rollout_scheduler import EpisodeTask, _pilot_tasks, episode_seed, prepare, run
from ultron.train.schema_v1 import Role, TrajectoryStep


class Runtime:
    def __init__(self, episodes=12, *, fail=None):
        self.request = {"generation": 0, "episodes": episodes, "group_size": 6, "seed": 71, "turns_per_side": 1}
        self.fail = fail
        self.calls = []
        self.closed = 0
        self.opened = 0
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def preflight(self):
        return {"isolation": "kvm", "pi_ready": True, "restore_verified": True}

    def plan(self):
        return [EpisodeTask(index, f"episode-{index}", episode_seed(71, index), EpisodeConfig(
            "web", 0, f"group-{index // 6}", "defender-v1", "attacker-v1", "defender-v1",
            episode_id=f"episode-{index}"), snapshot_sha256="f" * 64) for index in range(self.request["episodes"])]

    def open_worker(self, index):
        runtime = self
        self.opened += 1

        class Worker:
            guest = GuestHandle(f"guest-{index}", IsolationBackend.KVM, "127.0.0.1", "golden")

            def runner_for(self, task):
                runtime.calls.append(task.index)

                def restore(*_args):
                    if task.index == runtime.fail:
                        raise RuntimeError("restore failed")

                def turn(_guest, role, _profile, turn_index):
                    with runtime.lock:
                        runtime.active += 1
                        runtime.peak = max(runtime.peak, runtime.active)
                    time.sleep(0.005)
                    with runtime.lock:
                        runtime.active -= 1
                    return [TrajectoryStep(turn_index, role, [1], [task.seed], [1], [])]

                return EpisodeRunner(snapshot_sha256="f" * 64, load_profile=lambda _: {},
                                     restore=restore, run_turn=turn, turns_per_side=1,
                                     turn_probe=lambda *_: ProbeResult(1000, False, True, True, False),
                                     final_probe=lambda *_: ProbeResult(1000, False, True, True, False))

            def close(self):
                runtime.closed += 1

        return Worker()


def test_parallel_and_serial_keep_identical_episodes_seeds_and_order(tmp_path):
    serial = Runtime()
    parallel = Runtime()
    for name, runtime, workers in (("serial", serial, 1), ("parallel", parallel, 4)):
        tasks = prepare(runtime, runtime.request)
        report = run(runtime, tasks, runtime.request, tmp_path / name, workers=workers)
        assert report["completed_episodes"] == 12
        assert runtime.closed == runtime.opened
    assert serial.peak == 1
    assert parallel.peak > 1
    for role in ("attacker", "defender"):
        assert (tmp_path / "serial" / f"{role}.jsonl").read_bytes() == (tmp_path / "parallel" / f"{role}.jsonl").read_bytes()


def test_resume_uses_only_committed_episode_pairs_and_skips_serving(tmp_path):
    runtime = Runtime(fail=3)
    tasks = prepare(runtime, runtime.request)
    with pytest.raises(RuntimeError, match="restore failed"):
        run(runtime, tasks, runtime.request, tmp_path, workers=1)
    assert runtime.calls == [0, 1, 2, 3]
    runtime.fail = None
    runtime.calls.clear()
    report = run(runtime, tasks, runtime.request, tmp_path, workers=2)
    assert report["resumed_episodes"] == 3
    assert set(runtime.calls) == set(range(3, 12))
    opened = runtime.opened
    report = run(runtime, tasks, runtime.request, tmp_path, serving_factory=lambda: pytest.fail("completed run must not allocate GPUs"))
    assert report["resumed_episodes"] == 12
    assert runtime.opened == opened


def test_resume_rejects_modified_commit(tmp_path):
    runtime = Runtime(6)
    tasks = prepare(runtime, runtime.request)
    run(runtime, tasks, runtime.request, tmp_path, workers=1)
    next((tmp_path / "metadata" / "episodes").glob("*.json")).write_text("[]")
    with pytest.raises(ValueError, match="missing or changed"):
        run(runtime, tasks, runtime.request, tmp_path)


def test_plan_rejects_opponent_changes_within_group():
    runtime = Runtime(6)
    tasks = runtime.plan()
    tasks[1] = replace(tasks[1], config=replace(tasks[1].config, defender_ckpt="other"))
    runtime.plan = lambda: tasks
    with pytest.raises(ValueError, match="frozen within a group"):
        prepare(runtime, runtime.request)


def test_scheduler_preserves_transient_compromise_probe(tmp_path):
    runtime = Runtime(6)
    original = runtime.open_worker

    def open_worker(index):
        worker = original(index)
        runner_for = worker.runner_for

        def runner(task):
            value = runner_for(task)
            value.turn_probe = lambda *_: ProbeResult(0, True, True, True, False)
            return value
        worker.runner_for = runner
        return worker

    runtime.open_worker = open_worker
    run(runtime, prepare(runtime, runtime.request), runtime.request, tmp_path, workers=2)
    rows = [json.loads(line) for line in (tmp_path / "attacker.jsonl").read_text().splitlines()]
    assert all(row["terminal"]["reason_code"] == "ATTACKER_ROOT" and row["terminal"]["recovered"] for row in rows)
    meta = json.loads(next((tmp_path / "metadata" / "commits").glob("*.json")).read_text())
    assert [phase["phase"] for phase in meta["phases"]] == ["restore", "turn", "turn_probe", "turn", "turn_probe", "final_probe"]


def test_preflight_refuses_unverified_runtime():
    runtime = Runtime(6)
    runtime.preflight = lambda: {"isolation": "docker", "pi_ready": True, "restore_verified": True}
    with pytest.raises(ValueError, match="KVM"):
        prepare(runtime, runtime.request)


def test_scheduler_accepts_protocol_episode_identifiers(tmp_path):
    from ultron.train.study import PROTOCOL_VERSION, RunIdentity
    runtime = Runtime(6)
    identity = RunIdentity(PROTOCOL_VERSION, "qwen-8b", "adaptive-history+added-dpo", 0, "fixture")
    tasks = [replace(task, task_id=identity.episode_id(task.index),
                     config=replace(task.config, episode_id=identity.episode_id(task.index)))
             for task in runtime.plan()]
    runtime.plan = lambda: tasks
    run(runtime, prepare(runtime, runtime.request), runtime.request, tmp_path, workers=2)
    assert len(list((tmp_path / "metadata" / "commits").glob("*.json"))) == 6


def test_pilot_keeps_whole_groups_and_covers_roles_even_when_plan_is_role_sorted():
    runtime = Runtime(48)
    tasks = [replace(task, config=replace(task.config, learner_role=Role.ATTACKER if task.index < 24 else Role.DEFENDER)) for task in runtime.plan()]
    pilot = _pilot_tasks(tasks, 12)
    assert len(pilot) == 12
    assert {task.config.learner_role for task in pilot} == {Role.ATTACKER, Role.DEFENDER}
    assert len({task.config.group_id for task in pilot}) == 2


def test_early_gate_uses_real_role_transcripts_and_retains_failure_report(tmp_path):
    runtime = Runtime(24)
    runtime.request.update(early_gate_episodes=12, policies={"attacker": {"checkpoint_id": "attacker-v1"}, "defender": {"checkpoint_id": "defender-v1"}})
    original_plan = runtime.plan
    runtime.plan = lambda: [replace(task, config=replace(task.config,
        learner_role=Role.ATTACKER if task.index < 12 else Role.DEFENDER,
        opponent_checkpoint_id="defender-v1" if task.index < 12 else "attacker-v1")) for task in original_plan()]
    tasks = prepare(runtime, runtime.request)
    with pytest.raises(ValueError, match="no learning signal"):
        run(runtime, tasks, runtime.request, tmp_path, workers=3)
    assert len(runtime.calls) == 12
    report = json.loads((tmp_path / "metadata" / "early-gate.json").read_text())
    assert report["status"] == "failed"
    assert report["roles"]["attacker"]["groups"] == 1
    assert report["roles"]["defender"]["groups"] == 1
    assert not (tmp_path / "metadata" / "rollout.json").exists()
