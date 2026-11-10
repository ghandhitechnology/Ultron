"""CPU contract tests; the fake adapter never loads a model or claims GPU parity."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ultron.env.backend import IsolationBackend
from ultron.train.convert_verl import trajectory_to_verl_records
from ultron.train.gates import DataGateError, validate_training_data
from ultron.train.io import atomic_write_json
from ultron.train.pipeline import build_plan, canonical_hash, file_hash, prepare
from ultron.train.schema_v1 import ReasonCode, Role, TerminalOutcome, TrajectoryStep, TrajectoryV1
from ultron.train.study import StudyError
from ultron.train.training import (
    ADAPTER_SCHEMA, RESULT_SCHEMA, TrainingError, adapter_hash,
    build_training_request, policy_for_generation, preflight_adapter,
    receipt_path, run_training, training_config, tree_hash, validate_result,
)


@pytest.fixture
def installed_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    command = tmp_path / "fake-training-adapter"
    command.write_text(
        f"#!{sys.executable}\n"
        "import argparse, json\n"
        "from pathlib import Path\n"
        "from ultron.train.pipeline import canonical_hash, file_hash\n"
        "from ultron.train.training import adapter_hash, tree_hash\n"
        "p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()\n"
        "q=json.loads(Path(a.request).read_text()); output=Path(q['output_dir']); adapter=output/'adapter'\n"
        "adapter.mkdir(parents=True,exist_ok=True)\n"
        "(adapter/'adapter_config.json').write_text(json.dumps({'r':64,'lora_alpha':64,'task_type':'CAUSAL_LM'}))\n"
        "(adapter/'adapter_model.safetensors').write_bytes(q['checkpoint_id'].encode())\n"
        "state=output/'optimizer.bin'; state.write_bytes(('optimizer:'+q['checkpoint_id']).encode())\n"
        "result={key:q[key] for key in ['identity','generation','role','objective','checkpoint_id','model','model_revision','optimizer_strategy']}\n"
        "result.update(schema='ultron.training-result.v1',request_sha256=file_hash(Path(a.request)),input_data_sha256=q['data']['sha256'],consumed_records=q['data']['records'],trained_tokens=q['data']['masked_response_tokens'],applied_config_sha256=canonical_hash(q['config']),parent_policy_sha256=q['parent_policy']['sha256'],input_optimizer_sha256=q['optimizer_state']['sha256'] if q['optimizer_state'] else None,fresh_rollouts=0,optimizer_steps=1,final_loss=0.25,adapter_path=str(adapter),adapter_sha256=adapter_hash(adapter),optimizer_state={'path':str(state),'sha256':tree_hash(state)})\n"
        "Path(a.result).write_text(json.dumps(result))\n"
        "with (output/'invocations').open('a') as f: f.write('one\\n')\n"
    )
    command.chmod(0o755)
    manifest = tmp_path / "adapter-manifest.json"
    atomic_write_json(manifest, {
        "schema": ADAPTER_SCHEMA, "implementation": "cpu-contract-test-only", "revision": "fixture-v1",
        "command_sha256": file_hash(command), "objectives": ["grpo", "dpo"],
        "input_mode": "recorded_tokens", "generates_rollouts": False,
        "behavior_log_probs": "recorded_or_recompute",
    })
    for kind in ("GRPO", "DPO"):
        monkeypatch.setenv(f"ULTRON_{kind}_COMMAND", str(command))
        monkeypatch.setenv(f"ULTRON_{kind}_ADAPTER_MANIFEST", str(manifest))
    monkeypatch.setenv("ULTRON_BASE_MODEL_REVISION", "a" * 40)
    monkeypatch.delenv("ULTRON_RUN_ROOT", raising=False)
    monkeypatch.delenv("ULTRON_REFERENCE_POOL_MANIFEST", raising=False)
    return command


def write_rollout(plan, request: dict) -> dict[str, list[TrajectoryV1]]:
    roles = {"attacker": [], "defender": []}
    for index in range(plan.episodes):
        won = index % 2 == 0
        terminal = TerminalOutcome(
            ReasonCode.ATTACKER_ROOT if won else ReasonCode.DEFENDER_HOLD,
            0 if won else 1000, won, True, float(won), float(not won),
        )
        for role in Role:
            reward = terminal.attacker_reward if role is Role.ATTACKER else terminal.defender_reward
            trajectory = TrajectoryV1(
                "v1", f"episode-{index}", plan.generation, "suid_helper", role, f"{role.value}_lora",
                "frozen-opponent", f"group-{index // 6}",
                [TrajectoryStep(0, role, [1, 2], [3, 4], [1, 0], [], turn_reward=reward, behavior_log_probs=[-0.5, -0.8])],
                terminal, IsolationBackend.KVM,
                behavior_policy_id=request["policies"][role.value]["checkpoint_id"],
            )
            roles[role.value].append(trajectory)
    materialize(plan, request, roles)
    return roles


def materialize(plan, request: dict, roles: dict[str, list[TrajectoryV1]]) -> None:
    plan.trace_dir.mkdir(parents=True, exist_ok=True)
    for role, trajectories in roles.items():
        (plan.trace_dir / f"{role}.jsonl").write_text("".join(json.dumps(item.to_dict()) + "\n" for item in trajectories))
    atomic_write_json(plan.trace_dir / "metadata/rollout.json", {
        "request_sha256": file_hash(plan.request_path), "input_policy": request["policies"],
        "files": [{"path": f"{role}.jsonl", "sha256": file_hash(plan.trace_dir / f"{role}.jsonl")} for role in roles],
    })


def test_matrix_namespaces_rounding_and_final_generation(tmp_path: Path) -> None:
    base = build_plan(generation=0, study_root=tmp_path)
    assert base.episodes == 2052 and base.episodes // base.group_size == 342
    other = build_plan(generation=0, variant="fixed-diverse", seed=2, run_id="replicate", study_root=tmp_path)
    assert base.trace_dir != other.trace_dir
    assert base.environ()["ULTRON_CHECKPOINT_ROOT"] != other.environ()["ULTRON_CHECKPOINT_ROOT"]
    assert build_plan(generation=3).eval_mode == "full"
    assert not build_plan(generation=2).dpo
    assert build_plan(generation=2, variant="fixed-diverse+added-dpo").dpo
    with pytest.raises(StudyError, match="0..3"):
        build_plan(generation=4)
    with pytest.raises(StudyError, match="safe"):
        build_plan(generation=0, run_id="../../escape")
    with pytest.raises(StudyError, match="explicit --prep-turns"):
        build_plan(generation=0, pilot=True)
    pilot = build_plan(generation=0, episodes=240, pilot=True, prep_turns=0)
    assert pilot.purpose == "performance-pilot"
    assert pilot.environ()["ULTRON_COMPUTE_SCOPE"] != build_plan(generation=0, episodes=240).environ()["ULTRON_COMPUTE_SCOPE"]
    assert replace(pilot, generation=3).eval_mode is None


def test_unverified_or_fresh_rollout_adapter_fails_before_input_creation(tmp_path: Path, installed_adapter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = build_plan(generation=0, episodes=6, study_root=tmp_path / "study")
    monkeypatch.delenv("ULTRON_GRPO_ADAPTER_MANIFEST")
    with pytest.raises(TrainingError, match="before rollout"):
        prepare(plan)
    assert not plan.run_root.exists()
    with pytest.raises(TrainingError, match="before rollout"):
        preflight_adapter("grpo", environ={})


def test_recorded_bridge_resume_and_cross_generation_continuity(tmp_path: Path, installed_adapter: Path) -> None:
    plan = build_plan(generation=0, episodes=6, study_root=tmp_path / "study")
    request = prepare(plan)
    write_rollout(plan, request)
    result = run_training(plan, "attacker", "grpo")
    assert result["fresh_rollouts"] == 0
    assert result["trained_tokens"] == 6
    assert result["input_optimizer_sha256"] is None
    assert run_training(plan, "attacker", "grpo")["adapter_sha256"] == result["adapter_sha256"]
    assert (receipt_path(plan, "attacker", "grpo").parent / "invocations").read_text() == "one\n"
    run_training(plan, "defender", "grpo")
    following = replace(plan, generation=1)
    next_request = prepare(following)
    write_rollout(following, next_request)
    update = build_training_request(following, "attacker", "grpo")
    assert update["parent_policy"]["sha256"] == result["adapter_sha256"]
    assert update["optimizer_state"] == result["optimizer_state"]
    (Path(result["adapter_path"]) / "adapter_model.safetensors").write_bytes(b"changed-after-receipt")
    with pytest.raises(TrainingError, match="weights changed"):
        policy_for_generation(following, "attacker")


def test_training_rejects_rewritten_data_and_stale_behavior(tmp_path: Path, installed_adapter: Path) -> None:
    plan = build_plan(generation=0, episodes=6, study_root=tmp_path / "study")
    request = prepare(plan)
    trajectories = write_rollout(plan, request)
    trajectories["attacker"][0].behavior_policy_id = "older-policy"
    materialize(plan, request, trajectories)
    with pytest.raises(DataGateError, match="stale"):
        build_training_request(plan, "attacker", "grpo")
    trajectories["attacker"][0].behavior_policy_id = request["policies"]["attacker"]["checkpoint_id"]
    materialize(plan, request, trajectories)
    with (plan.trace_dir / "attacker.jsonl").open("a") as output:
        output.write("\n")
    with pytest.raises(TrainingError, match="changed after"):
        build_training_request(plan, "attacker", "grpo")


def test_opponent_transcripts_are_kept_but_do_not_become_learner_data(tmp_path: Path, installed_adapter: Path) -> None:
    plan = build_plan(generation=0, episodes=12, study_root=tmp_path / "study")
    request = prepare(plan)
    trajectories = write_rollout(plan, request)
    for index in range(12):
        opponent_role = "defender" if index < 6 else "attacker"
        opponent = trajectories[opponent_role][index]
        opponent.trainable = False
        opponent.behavior_policy_id = "historical-opponent"
    materialize(plan, request, trajectories)
    report = validate_training_data(plan.trace_dir, request=request)
    assert report["episodes"] == 12
    assert report["roles"]["attacker"]["learner_episodes"] == 6
    assert report["roles"]["defender"]["learner_episodes"] == 6
    assert trajectory_to_verl_records(trajectories["defender"][0], 0) == []
    trajectories["attacker"][0].trainable = False
    materialize(plan, request, trajectories)
    with pytest.raises(DataGateError, match="5 eligible members"):
        validate_training_data(plan.trace_dir, request=request)


def test_h100_packing_keeps_the_algorithmic_configuration() -> None:
    plan = build_plan(generation=0)
    config = training_config(plan, "grpo")
    assert config["performance"]["use_remove_padding"] is True
    assert config["performance"]["use_dynamic_bsz"] is True
    assert config["algorithm"]["data"]["train_batch_size"] == 64
    assert config["algorithm"]["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 32
    assert config["algorithm"]["actor_rollout_ref"]["actor"]["ppo_epochs"] == 1
    assert config["invariants"]["fresh_rollout_count"] == 0
    assert config["invariants"]["group_advantages_before_packing"] is True


def test_reject_dpo_on_main_method_before_work(tmp_path: Path, installed_adapter: Path) -> None:
    plan = build_plan(generation=2, episodes=6, study_root=tmp_path / "study")
    with pytest.raises(TrainingError, match="restricted"):
        build_training_request(plan, "attacker", "dpo")


def test_prepare_emits_serving_manifest_with_pinned_base(tmp_path: Path, installed_adapter: Path) -> None:
    from ultron.shared_serving import validate_manifest
    plan = build_plan(generation=0, episodes=6, study_root=tmp_path / "study")
    first = prepare(plan)
    assert prepare(plan) == first
    manifest = validate_manifest(Path(plan.environ()["ULTRON_SERVING_MANIFEST"]))
    assert manifest["base_model_revision"] == "a" * 40
    assert len(manifest["base_policies"]) == 1
    assert manifest["adapters"] == []


@pytest.mark.parametrize("field,value,reason", [
    ("fresh_rollouts", 6, "replacement rollouts"),
    ("trained_tokens", 5, "masked response tokens"),
    ("parent_policy_sha256", "wrong", "different policy"),
])
def test_training_receipt_rejects_changed_learning_inputs(tmp_path: Path, installed_adapter: Path, field, value, reason) -> None:
    plan = build_plan(generation=0, episodes=6, study_root=tmp_path / "study")
    rollout = prepare(plan)
    write_rollout(plan, rollout)
    result = run_training(plan, "attacker", "grpo")
    request_path = Path(result["training_request"])
    request = json.loads(request_path.read_text())
    result[field] = value
    with pytest.raises(TrainingError, match=reason):
        validate_result(request, result, request_path=request_path)


def test_generation_planning_is_read_only_and_full_execution_keeps_protocol_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ultron.train.pipeline import main
    monkeypatch.delenv("ULTRON_RUN_ROOT", raising=False)
    monkeypatch.delenv("ULTRON_COMPUTE_LEDGER", raising=False)
    monkeypatch.delenv("ULTRON_SPENDING_CAP_USD", raising=False)
    assert main(["plan", "--generation", "0", "--study-root", str(tmp_path), "--pilot", "--prep-turns", "0"]) == 0
    assert not list(tmp_path.iterdir())
    assert main(["execution-check", "--generation", "0", "--pilot", "--prep-turns", "0"]) == 2
    monkeypatch.setenv("ULTRON_SPENDING_CAP_USD", "10")
    assert main(["execution-check", "--generation", "0", "--pilot", "--prep-turns", "0"]) == 0
    assert main(["execution-check", "--generation", "0"]) == 2


def test_generation_shell_runs_and_resumes_verified_cpu_contract(tmp_path: Path, installed_adapter: Path) -> None:
    """Exercise shell/receipt/archive plumbing with no model, guest, or GPU."""
    command = tmp_path / "fake-rollout-adapter"
    command.write_text(
        f"#!{sys.executable}\n"
        "import argparse,json\n"
        "from pathlib import Path\n"
        "from ultron.train.pipeline import plan_from_environ\n"
        "from ultron.train.tests.test_training_bridge import write_rollout\n"
        "p=argparse.ArgumentParser(); p.add_argument('--generation',type=int); p.add_argument('--episodes',type=int); p.add_argument('--output'); p.add_argument('--request'); p.add_argument('--preflight-only',action='store_true'); a=p.parse_args()\n"
        "if not a.preflight_only: write_rollout(plan_from_environ(generation=a.generation),json.loads(Path(a.request).read_text()))\n"
    )
    command.chmod(0o755)
    root = Path(__file__).resolve().parents[2]
    args = ["bash", str(root / "scripts/run_generation.sh"), "--pilot", "--prep-turns", "0", "--episodes", "6", "--study-root", str(tmp_path / "study"), "0"]
    env = {**os.environ, "ULTRON_NO_TMUX": "1", "ULTRON_ALLOCATION_ID": "cpu-contract-test", "ULTRON_SPENDING_CAP_USD": "1", "ULTRON_ROLLOUT_COMMAND": str(command), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("ULTRON_ROLLOUT_RUNTIME", None)
    first = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=30)
    assert first.returncode == 0, first.stderr
    second = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=30)
    assert second.returncode == 0, second.stderr
    assert "Stage grpo-attacker: already complete" in second.stdout
    plan = build_plan(generation=0, episodes=6, pilot=True, prep_turns=0, study_root=tmp_path / "study")
    manifest = json.loads((plan.run_root / "archives/gen0/manifest.json").read_text())
    assert manifest["identity"] == plan.to_dict()["identity"]
    for role in ("attacker", "defender"):
        assert (receipt_path(plan, role, "grpo").parent / "invocations").read_text() == "one\n"
        assert manifest["roles"][role]["checkpoint_id"].endswith(":grpo")
