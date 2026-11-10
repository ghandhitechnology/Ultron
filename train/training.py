"""Recorded-trajectory training adapter boundary.

This module validates and dispatches an installed trainer. It does not implement
GRPO or advertise compatibility with an unpinned veRL release. Adapter manifests
describe supported inputs; only real execution can establish numerical parity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

from .archive import ADAPTER_CONFIG, ADAPTER_WEIGHTS, file_digest, is_adapter_dir
from .artifacts import artifact_hash
from .convert_verl import convert_jsonl, read_trajectories
from .family import resolve
from .io import atomic_write_json
from .pipeline import GenerationPlan, ROOT, canonical_hash, file_hash, plan_from_environ
from .study import StudyError

ADAPTER_SCHEMA = "ultron.recorded-training-adapter.v1"
REQUEST_SCHEMA = "ultron.training-request.v1"
RESULT_SCHEMA = "ultron.training-result.v1"


class TrainingError(StudyError):
    pass


def adapter_hash(path: Path) -> str:
    """Immutable serving identity shared with the two-worker model server."""
    if not is_adapter_dir(path):
        raise TrainingError(f"missing serveable LoRA adapter at {path}")
    return artifact_hash(path)


def tree_hash(path: Path) -> str:
    """Hash optimizer state files, including their relative names."""
    if path.is_file():
        return file_hash(path)
    if not path.is_dir():
        raise TrainingError(f"missing optimizer state at {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise TrainingError(f"optimizer state is empty: {path}")
    digest = hashlib.sha256()
    for source in files:
        digest.update(source.relative_to(path).as_posix().encode() + b"\0" + file_hash(source).encode() + b"\0")
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise TrainingError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingError(f"expected JSON object at {path}")
    return value


@dataclass(frozen=True)
class TrainingAdapter:
    command: Path
    manifest_path: Path
    manifest: dict[str, Any]

    def identity(self) -> dict[str, str]:
        return {
            "command": str(self.command), "command_sha256": file_hash(self.command),
            "manifest": str(self.manifest_path), "manifest_sha256": file_hash(self.manifest_path),
            "implementation": self.manifest["implementation"],
            "revision": self.manifest["revision"],
        }


def preflight_adapter(kind: str, *, environ: dict[str, str] | None = None) -> TrainingAdapter:
    if kind not in {"grpo", "dpo"}:
        raise TrainingError(f"unsupported training kind {kind}")
    env = os.environ if environ is None else environ
    command_name = env.get(f"ULTRON_{kind.upper()}_COMMAND")
    manifest_name = env.get(f"ULTRON_{kind.upper()}_ADAPTER_MANIFEST")
    if not command_name or not manifest_name:
        raise TrainingError(
            f"Configure ULTRON_{kind.upper()}_COMMAND and ULTRON_{kind.upper()}_ADAPTER_MANIFEST "
            "for a pinned recorded-trajectory adapter before rollout; stock PPO fresh-rollout dispatch is unsupported."
        )
    command, manifest_path = Path(command_name).resolve(), Path(manifest_name).resolve()
    if not command.is_file() or not os.access(command, os.X_OK):
        raise TrainingError(f"training command must be an executable file: {command}")
    manifest = read_object(manifest_path)
    if manifest.get("schema") != ADAPTER_SCHEMA:
        raise TrainingError(f"training adapter must implement {ADAPTER_SCHEMA}")
    for field in ("implementation", "revision"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise TrainingError(f"adapter {field} must identify a pinned implementation")
    if manifest.get("command_sha256") != file_hash(command):
        raise TrainingError("training adapter command hash differs from its manifest")
    if kind not in manifest.get("objectives", []):
        raise TrainingError(f"adapter does not support {kind}")
    if manifest.get("input_mode") != "recorded_tokens" or manifest.get("generates_rollouts") is not False:
        raise TrainingError("adapter must consume recorded tokens without generating replacement rollouts")
    if manifest.get("behavior_log_probs") not in {"recorded_or_recompute", "recorded"}:
        raise TrainingError("adapter must use recorded behavior log probabilities or recompute from the frozen behavior policy")
    return TrainingAdapter(command, manifest_path, manifest)


def receipt_path(plan: GenerationPlan, role: str, kind: str, *, generation: int | None = None) -> Path:
    gen = plan.generation if generation is None else generation
    suffix = "_dpo" if kind == "dpo" else ""
    return plan.run_root / "checkpoints" / f"gen{gen}" / f"{role}_lora{suffix}" / "training-receipt.json"


def validated_receipt(path: Path, *, identity: dict[str, Any] | None = None) -> dict[str, Any]:
    result = read_object(path)
    if result.get("schema") != RESULT_SCHEMA:
        raise TrainingError(f"unsupported training result at {path}")
    if identity is not None and result.get("identity") != identity:
        raise TrainingError("checkpoint belongs to a different study run")
    adapter = Path(result.get("adapter_path", ""))
    if result.get("adapter_sha256") != adapter_hash(adapter):
        raise TrainingError("checkpoint weights changed after training receipt was written")
    state = result.get("optimizer_state")
    if not isinstance(state, dict) or state.get("sha256") != tree_hash(Path(state.get("path", ""))):
        raise TrainingError("optimizer checkpoint changed after training receipt was written")
    return result


def _policy_from_receipt(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_id": result["checkpoint_id"], "sha256": result["adapter_sha256"],
        "adapter_path": result["adapter_path"], "model": result["model"],
        "revision": result["model_revision"],
    }


def policy_for_generation(plan: GenerationPlan, role: str) -> dict[str, Any]:
    if role not in {"attacker", "defender"}:
        raise TrainingError("role must be attacker or defender")
    if plan.generation == 0:
        pack = resolve(plan.identity.model)
        revision = os.environ.get("ULTRON_BASE_MODEL_REVISION", "")
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise TrainingError("pin ULTRON_BASE_MODEL_REVISION to the resolved model/tokenizer commit before rollout")
        descriptor = {"model": pack.base_model, "revision": revision}
        return {
            **descriptor, "checkpoint_id": f"{pack.base_model}@{revision}:base",
            "sha256": canonical_hash(descriptor), "adapter_path": None,
        }
    previous = plan.generation - 1
    kind = "dpo" if role == "attacker" and plan.cell.dpo and previous >= 2 else "grpo"
    path = receipt_path(plan, role, kind, generation=previous)
    result = validated_receipt(path, identity=plan.to_dict()["identity"])
    if result.get("generation") != previous or result.get("role") != role or result.get("objective") != kind:
        raise TrainingError("previous policy receipt has incorrect generation, role, or objective")
    return _policy_from_receipt(result)


def _optimizer_parent(plan: GenerationPlan, role: str, kind: str, strategy: str) -> dict[str, str] | None:
    if strategy == "reset_each_generation":
        return None
    previous = plan.generation - 1
    if previous < (2 if kind == "dpo" else 0):
        return None
    result = validated_receipt(receipt_path(plan, role, kind, generation=previous), identity=plan.to_dict()["identity"])
    return result["optimizer_state"]


def training_config(plan: GenerationPlan, kind: str) -> dict[str, Any]:
    pack = resolve(plan.identity.model)
    path = pack.grpo_config_dir / f"{pack.grpo_config_name}.yaml" if kind == "grpo" else pack.dpo_config
    config = yaml.safe_load(path.read_text())
    hardware_path = ROOT / "configs/h100.yaml"
    hardware = yaml.safe_load(hardware_path.read_text())
    if hardware["hardware"]["gpu_count"] != 2 or hardware["hardware"]["gpu_memory_gb"] != 80:
        raise TrainingError("this execution profile requires two 80 GB GPUs")
    performance = hardware["training"]
    return {
        "source": str(path), "source_sha256": file_hash(path), "algorithm": config,
        "hardware_profile": str(hardware_path), "hardware_sha256": file_hash(hardware_path),
        "performance": {
            "gpus": 2, "use_dynamic_bsz": bool(performance["use_dynamic_bsz"]),
            "use_remove_padding": bool(performance["use_remove_padding"]),
            "max_token_len_per_gpu": int(performance["max_token_len_per_gpu"]),
        },
        "invariants": {
            "recorded_token_ids": True, "assistant_masks": True,
            "group_advantages_before_packing": True,
            "effective_batch_unchanged": True, "loss_normalization_unchanged": True,
            "immediate_reward_separate_from_return": True,
            "fresh_rollout_count": 0, "group_size": plan.group_size,
            "role_aware_baseline": False,
        },
    }


def build_training_request(plan: GenerationPlan, role: str, kind: str) -> dict[str, Any]:
    if role not in {"attacker", "defender"}:
        raise TrainingError("role must be attacker or defender")
    if kind == "dpo" and (role != "attacker" or not plan.dpo):
        raise TrainingError("DPO is restricted to the added-DPO attacker cells at generation 2 or later")
    adapter = preflight_adapter(kind)
    rollout = read_object(plan.request_path)
    if rollout["identity"] != plan.to_dict()["identity"] or rollout["generation"] != plan.generation:
        raise TrainingError("rollout request belongs to another run or generation")
    if rollout["training_adapters"].get(kind) != adapter.identity():
        raise TrainingError("training adapter changed after the rollout was planned")
    strategy = rollout["optimizer_strategy"]
    parent = policy_for_generation(plan, role)
    if parent != rollout["policies"][role]:
        raise TrainingError("behavior policy changed after rollout planning")
    source = plan.trace_dir / f"{role}.jsonl"
    provenance = read_object(plan.trace_dir / "metadata/rollout.json")
    if provenance.get("request_sha256") != file_hash(plan.request_path):
        raise TrainingError("rollout provenance does not match the generation request")
    for name in ("attacker", "defender"):
        records = provenance.get("files", [])
        entry = next((item for item in records if item.get("path") == f"{name}.jsonl"), None)
        if entry is None or entry.get("sha256") != file_hash(plan.trace_dir / f"{name}.jsonl"):
            raise TrainingError("role trajectories changed after the rollout receipt was written")
    from .gates import validate_training_data
    gate = validate_training_data(plan.trace_dir, request=rollout, role=role)
    if gate["signal_groups"] == 0:
        raise TrainingError(f"{role} has no complete GRPO group with reward variance; inspect the pilot before training")
    if kind == "grpo":
        destination = plan.run_root / "verl" / f"gen{plan.generation}" / f"{role}.jsonl"
        convert_jsonl(source, destination, plan.generation)
        data_path = destination
        if adapter.manifest["behavior_log_probs"] == "recorded":
            for trajectory in read_trajectories(source):
                if trajectory.trainable and any(step.behavior_log_probs is None for step in trajectory.steps):
                    raise TrainingError("adapter requires recorded behavior log probabilities; none may be omitted")
    else:
        result = validated_receipt(receipt_path(plan, role, "grpo"), identity=plan.to_dict()["identity"])
        parent = _policy_from_receipt(result)
        data_path = plan.run_root / "dpo" / f"gen{plan.generation}" / f"{role}.jsonl"
        validate_dpo_pairs(data_path, generation=plan.generation)
    output_dir = receipt_path(plan, role, kind).parent
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    masked_tokens = (
        sum(sum(row["assistant_mask"]) for row in rows)
        if kind == "grpo" else sum(len(row["chosen_token_ids"]) + len(row["rejected_token_ids"]) for row in rows)
    )
    return {
        "schema": REQUEST_SCHEMA, "identity": plan.to_dict()["identity"],
        "generation": plan.generation, "role": role, "objective": kind,
        "model": parent["model"], "model_revision": parent["revision"],
        "parent_policy": parent, "behavior_policy": rollout["policies"][role],
        "optimizer_strategy": strategy,
        "optimizer_state": _optimizer_parent(plan, role, kind, strategy),
        "data": {"path": str(data_path), "sha256": file_hash(data_path), "mode": "recorded_tokens", "records": len(rows), "masked_response_tokens": masked_tokens},
        "rollout_sha256": file_hash(source), "rollout_request_sha256": file_hash(plan.request_path),
        "group_gate": gate, "config": training_config(plan, kind),
        "adapter": adapter.identity(), "output_dir": str(output_dir),
        "checkpoint_id": plan.identity.checkpoint_id(role, plan.generation) + f":{kind}",
    }


def validate_dpo_pairs(path: Path, *, generation: int) -> int:
    """The branch producer must supply evidence, not just chosen/rejected text."""
    count = 0
    if not path.is_file():
        raise TrainingError(f"DPO branch pairs have not been produced: {path}")
    with path.open() as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("prompt_token_ids", "chosen_token_ids", "rejected_token_ids"):
                tokens = row.get(key)
                if not isinstance(tokens, list) or not tokens or any(type(value) is not int or value < 0 for value in tokens):
                    raise TrainingError(f"{path}:{number}: invalid {key}")
            if row["chosen_token_ids"] == row["rejected_token_ids"]:
                raise TrainingError("DPO pair actions must differ")
            for key in ("group_id", "profile_id", "opponent_checkpoint_id", "observation_hash", "prefix_hash", "winner_episode_id", "loser_episode_id"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise TrainingError(f"DPO pair is missing {key} provenance")
            if row.get("generation") != generation or row.get("role") != "attacker":
                raise TrainingError("DPO pair has a different generation or role")
            winner, loser = row.get("winner_reward"), row.get("loser_reward")
            if not all(type(value) in (float, int) and math.isfinite(value) for value in (winner, loser)) or winner <= loser:
                raise TrainingError("DPO pair must contain finite, strictly ordered verified rewards")
            count += 1
    if count == 0:
        raise TrainingError("DPO pair file is empty")
    return count


def validate_result(request: dict[str, Any], result: dict[str, Any], *, request_path: Path) -> dict[str, Any]:
    if result.get("schema") != RESULT_SCHEMA:
        raise TrainingError(f"trainer result must use {RESULT_SCHEMA}")
    if result.get("request_sha256") != file_hash(request_path):
        raise TrainingError("training result does not match the submitted request")
    for field in ("identity", "generation", "role", "objective", "checkpoint_id", "model", "model_revision"):
        if result.get(field) != request[field]:
            raise TrainingError(f"training result changed {field}")
    if result.get("input_data_sha256") != request["data"]["sha256"]:
        raise TrainingError("training result consumed a different dataset")
    if result.get("consumed_records") != request["data"]["records"] or result.get("trained_tokens") != request["data"]["masked_response_tokens"]:
        raise TrainingError("training result dropped, duplicated, or changed masked response tokens")
    if result.get("applied_config_sha256") != canonical_hash(request["config"]):
        raise TrainingError("training result does not report the requested objective and effective batching")
    if result.get("parent_policy_sha256") != request["parent_policy"]["sha256"]:
        raise TrainingError("training result started from a different policy")
    if result.get("optimizer_strategy") != request["optimizer_strategy"]:
        raise TrainingError("training result used a different optimizer strategy")
    expected_state = request["optimizer_state"]
    if result.get("input_optimizer_sha256") != (expected_state["sha256"] if expected_state else None):
        raise TrainingError("training result used a different optimizer parent")
    if type(result.get("fresh_rollouts")) is not int or result["fresh_rollouts"] != 0:
        raise TrainingError("training adapter generated replacement rollouts")
    for field in ("optimizer_steps", "trained_tokens"):
        if type(result.get(field)) is not int or result[field] <= 0:
            raise TrainingError(f"training result requires positive {field}")
    loss = result.get("final_loss")
    if type(loss) not in (float, int) or not math.isfinite(loss):
        raise TrainingError("training result has no finite final loss")
    destination = Path(request["output_dir"]).resolve()
    adapter = Path(result.get("adapter_path", "")).resolve()
    if adapter == destination or not adapter.is_relative_to(destination):
        raise TrainingError("adapter output is outside the requested checkpoint directory")
    digest = adapter_hash(adapter)
    if result.get("adapter_sha256") != digest:
        raise TrainingError("training result adapter hash does not match exported weights")
    if digest == request["parent_policy"]["sha256"]:
        raise TrainingError("training adapter did not change the policy weights")
    config = read_object(adapter / ADAPTER_CONFIG)
    if config.get("r") != 64 or config.get("task_type") != "CAUSAL_LM":
        raise TrainingError("exported adapter must retain rank 64 and CAUSAL_LM")
    state = result.get("optimizer_state")
    if not isinstance(state, dict):
        raise TrainingError("training adapter must export resumable optimizer state")
    state_path = Path(state.get("path", "")).resolve()
    if not state_path.is_relative_to(destination) or state.get("sha256") != tree_hash(state_path):
        raise TrainingError("optimizer state path or hash does not match exported state")
    # Keep the exact request with its result for inspection and future parents.
    return {**result, "training_request": str(request_path), "adapter": request["adapter"]}


def run_training(plan: GenerationPlan, role: str, kind: str) -> dict[str, Any]:
    request = build_training_request(plan, role, kind)
    adapter = preflight_adapter(kind)
    directory = Path(request["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "training-request.json"
    result_path = directory / "adapter-result.json"
    receipt = directory / "training-receipt.json"
    if receipt.exists():
        result = validated_receipt(receipt, identity=request["identity"])
        if read_object(request_path) != request:
            raise TrainingError("existing checkpoint belongs to different inputs; use a new run identity")
        return result
    if request_path.exists() and read_object(request_path) != request:
        raise TrainingError("unfinished training has different inputs; use a new run identity")
    atomic_write_json(request_path, request)
    # A previous partial adapter result is never accepted as this attempt's output.
    result_path.unlink(missing_ok=True)
    completed = subprocess.run([str(adapter.command), "--request", str(request_path), "--result", str(result_path)], check=False)
    if completed.returncode:
        raise TrainingError(f"recorded-trajectory {kind} adapter exited with status {completed.returncode}")
    if file_hash(Path(request["data"]["path"])) != request["data"]["sha256"]:
        raise TrainingError("training input changed during the update")
    result = validate_result(request, read_object(result_path), request_path=request_path)
    atomic_write_json(receipt, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("grpo", "dpo", "resolve"))
    parser.add_argument("--role", choices=("attacker", "defender"), default="attacker")
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.kind == "resolve":
            plan = plan_from_environ(generation=args.generation)
            kind = "dpo" if args.role == "attacker" and plan.dpo else "grpo"
            result = validated_receipt(receipt_path(plan, args.role, kind), identity=plan.to_dict()["identity"])
            print(result["adapter_path"])
        elif args.preflight_only:
            print(json.dumps(preflight_adapter(args.kind).identity(), indent=2))
        else:
            plan = plan_from_environ(generation=args.generation)
            result = run_training(plan, args.role, args.kind)
            print(json.dumps({"checkpoint_id": result["checkpoint_id"], "adapter_path": result["adapter_path"]}))
    except (StudyError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
