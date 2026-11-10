"""Study-aware plans and immutable inputs for the generation launchers.

Planning is read-only. ``prepare`` writes a request only after the training
adapter and policy parents have passed preflight; it does not start a GPU job.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping

import yaml

from .family import FamilyName, resolve
from .io import atomic_write_json
from .study import PROTOCOL_VERSION, RunIdentity, StudyCell, StudyError, assert_paid_execution_allowed, load_protocol

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "ultron.generation-plan.v1"
ROLLOUT_SCHEMA = "ultron.rollout-request.v1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_episode_count(requested: int, group_size: int = 6) -> int:
    if isinstance(requested, bool) or requested < 1 or group_size < 2:
        raise StudyError("episodes must be positive and group size must be at least two")
    return ((requested + group_size - 1) // group_size) * group_size


@dataclass(frozen=True)
class GenerationPlan:
    identity: RunIdentity
    cell: StudyCell
    generation: int
    requested_episodes: int
    episodes: int
    group_size: int
    generations: int
    run_root: Path
    purpose: str = "study"
    prep_turns: int | None = None

    @property
    def trace_dir(self) -> Path:
        return self.run_root / "traces" / f"gen{self.generation}"

    @property
    def request_path(self) -> Path:
        return self.run_root / "requests" / f"gen{self.generation}.json"

    @property
    def plan_path(self) -> Path:
        return self.run_root / "plans" / f"gen{self.generation}.json"

    @property
    def dpo(self) -> bool:
        return self.cell.dpo and self.generation >= 2

    @property
    def eval_mode(self) -> str | None:
        if self.purpose == "performance-pilot":
            return None
        if self.generation == self.generations - 1:
            return "full"
        return "light" if self.generation == 2 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "purpose": self.purpose,
            "prep_turns": self.prep_turns,
            "identity": asdict(self.identity),
            "generation": self.generation,
            "generations": self.generations,
            "requested_episodes": self.requested_episodes,
            "episodes": self.episodes,
            "group_size": self.group_size,
            "groups": self.episodes // self.group_size,
            "method": self.cell.method.value,
            "shaping": self.cell.shaping,
            "dpo": self.dpo,
            "eval_mode": self.eval_mode,
            "run_root": str(self.run_root),
            "trace_dir": str(self.trace_dir),
            "request_path": str(self.request_path),
            "hardware_profile": str(ROOT / "configs/h100.yaml"),
        }

    def environ(self) -> dict[str, str]:
        # Leave space for the longest role/stage suffix under tmux's 40-char
        # project limit; the complete identity remains in every request/log.
        model_label = self.identity.model.rsplit("-", 1)[-1]
        session = f"ultron-{model_label}-{canonical_hash(asdict(self.identity))[:10]}"
        return {
            "ULTRON_MODEL_FAMILY": self.identity.model,
            "ULTRON_STUDY_VARIANT": self.identity.variant,
            "ULTRON_STUDY_SEED": str(self.identity.seed),
            "ULTRON_STUDY_RUN_ID": self.identity.run_id,
            "ULTRON_STUDY_PURPOSE": self.purpose,
            "ULTRON_PREP_TURNS": "" if self.prep_turns is None else str(self.prep_turns),
            "ULTRON_STUDY_SESSION": session,
            "ULTRON_RUN_ROOT": str(self.run_root),
            "ULTRON_TRACE_ROOT": str(self.run_root / "traces"),
            "ULTRON_VERL_ROOT": str(self.run_root / "verl"),
            "ULTRON_DPO_ROOT": str(self.run_root / "dpo"),
            "ULTRON_CHECKPOINT_ROOT": str(self.run_root / "checkpoints"),
            "ULTRON_ARCHIVE_ROOT": str(self.run_root / "archives"),
            "ULTRON_PFSP_MANIFEST": str(self.run_root / "checkpoints/pfsp_pool.json"),
            "ULTRON_EVAL_ROOT": str(self.run_root / "eval"),
            "ULTRON_RESPONSES_DIR": str(self.run_root / "responses"),
            "ULTRON_PIPELINE_STATE_DIR": str(self.run_root / "job-state" / f"gen{self.generation}"),
            "ULTRON_GENERATION_PLAN": str(self.plan_path),
            "ULTRON_ROLLOUT_REQUEST": str(self.request_path),
            "ULTRON_GENERATION": str(self.generation),
            "ULTRON_REQUESTED_EPISODES": str(self.requested_episodes),
            "ULTRON_EPISODES": str(self.episodes),
            "ULTRON_ROLLOUT_SEED": str(self.identity.seed),
            "ULTRON_STUDY_DPO": "1" if self.dpo else "0",
            "ULTRON_STUDY_EVAL_MODE": self.eval_mode or "",
            "ULTRON_COMPUTE_LEDGER": os.environ.get("ULTRON_COMPUTE_LEDGER", str(self.run_root / "compute.jsonl")),
            "ULTRON_COMPUTE_SCOPE": os.environ.get("ULTRON_SCOPE_KEY") or canonical_hash({
                "protocol_sha256": file_hash(ROOT / "configs/study/final.yaml"),
                "model": self.identity.model, "variant": self.identity.variant,
                "seed": self.identity.seed, "generation": self.generation,
                "episodes": self.episodes, "group_size": self.group_size,
                "purpose": self.purpose, "prep_turns": self.prep_turns,
            }),
            "ULTRON_SERVING_MANIFEST": str(self.run_root / "requests" / f"gen{self.generation}-serving.json"),
        }


def build_plan(
    *, generation: int, model: str = "qwen-8b", variant: str = "adaptive-history",
    seed: int = 0, run_id: str = "default", episodes: int | None = None,
    study_root: Path | None = None, pilot: bool = False, prep_turns: int | None = None,
) -> GenerationPlan:
    protocol = load_protocol()
    identity = RunIdentity(PROTOCOL_VERSION, model, variant, seed, run_id)
    cells = [cell for cell in protocol.cells if cell.model.value == model and cell.variant == variant]
    if not cells:
        raise StudyError(f"{model}/{variant} is not a cell in the locked study")
    cell = cells[0]
    if cell.control:
        raise StudyError("the untrained control is evaluated without generation training")
    config = yaml.safe_load((ROOT / "configs/generation.yaml").read_text())
    generations = int(config["outer_generations"])
    if not 0 <= generation < generations:
        raise StudyError(f"generation must be in 0..{generations - 1}; final evaluation follows generation {generations - 1}")
    group_size = int(config["grpo"]["group_size"])
    requested = int(config["episodes_per_generation"]) if episodes is None else episodes
    effective = complete_episode_count(requested, group_size)
    if prep_turns is not None and (isinstance(prep_turns, bool) or prep_turns < 0):
        raise StudyError("prep turns must be a non-negative integer")
    if not pilot and prep_turns is not None and prep_turns != protocol.prep_turns:
        raise StudyError("full study prep turns are controlled by the locked protocol")
    preparation = protocol.prep_turns if prep_turns is None else prep_turns
    if pilot and preparation is None:
        raise StudyError("a calibration pilot requires explicit --prep-turns while the study value is unset")
    root = study_root or ROOT / "data/study"
    namespace = identity.namespace().relative_to("data/study")
    return GenerationPlan(identity, cell, generation, requested, effective, group_size, generations, root.resolve() / namespace,
                          "performance-pilot" if pilot else "study", preparation)


def plan_from_environ(environ: Mapping[str, str] | None = None, *, generation: int | None = None) -> GenerationPlan:
    env = os.environ if environ is None else environ
    requested = env.get("ULTRON_REQUESTED_EPISODES") or env.get("ULTRON_EPISODES")
    run_root = env.get("ULTRON_RUN_ROOT")
    study_root = Path(env["ULTRON_STUDY_ROOT"]) if env.get("ULTRON_STUDY_ROOT") else None
    if run_root:
        # The namespace has four components: model/variant/seed/run.
        study_root = Path(run_root).parents[3]
    return build_plan(
        generation=int(env.get("ULTRON_GENERATION", "0")) if generation is None else generation,
        model=env.get("ULTRON_MODEL_FAMILY", "qwen-8b"),
        variant=env.get("ULTRON_STUDY_VARIANT", "adaptive-history"),
        seed=int(env.get("ULTRON_STUDY_SEED", "0")),
        run_id=env.get("ULTRON_STUDY_RUN_ID", "default"),
        episodes=None if requested is None else int(requested), study_root=study_root,
        pilot=env.get("ULTRON_STUDY_PURPOSE") == "performance-pilot",
        prep_turns=int(env["ULTRON_PREP_TURNS"]) if env.get("ULTRON_PREP_TURNS") else None,
    )


def prepare(plan: GenerationPlan) -> dict[str, Any]:
    from .training import adapter_hash, preflight_adapter, policy_for_generation, read_object

    preflight_adapter("grpo")
    # Validate the DPO adapter before any expensive work in a DPO generation.
    if plan.dpo:
        preflight_adapter("dpo")
    policies = {role: policy_for_generation(plan, role) for role in ("attacker", "defender")}
    protocol = load_protocol()
    existing = read_object(plan.request_path) if plan.request_path.exists() else None
    if existing is None:
        history = plan.run_root / "checkpoints/pfsp_pool.json"
        reference_name = os.environ.get("ULTRON_REFERENCE_POOL_MANIFEST")
        pools = {
            "history": read_object(history) if history.is_file() else {"entries": []},
            "reference": read_object(Path(reference_name)) if reference_name else {"entries": []},
        }
    else:
        pools = existing["opponent_pools"]
    for pool in pools.values():
        for item in pool.get("entries", []):
            digest = adapter_hash(Path(item["path"]))
            if item.get("sha256") is not None and item["sha256"] != digest:
                raise StudyError("opponent adapter changed after its pool snapshot")
            item["sha256"] = digest
    request = {
        "schema": ROLLOUT_SCHEMA,
        "purpose": plan.purpose,
        "identity": asdict(plan.identity),
        "generation": plan.generation,
        "method": plan.cell.method.value,
        "shaping": plan.cell.shaping,
        "dpo": plan.dpo,
        "isolation": protocol.isolation.value,
        "requested_episodes": plan.requested_episodes,
        "episodes": plan.episodes,
        "group_size": plan.group_size,
        "seed": plan.identity.seed,
        "early_gate_episodes": min(204, plan.episodes),
        "output": str(plan.trace_dir),
        "policies": policies,
        "adaptive_mix": asdict(protocol.adaptive_mix),
        "opponent_manifest": str(plan.run_root / "requests" / f"gen{plan.generation}-opponents.json"),
        "opponent_pools": pools,
        "tasks": [asdict(task) for task in protocol.tasks if task.split == "train"],
        "prep_turns": plan.prep_turns,
        "optimizer_strategy": os.environ.get("ULTRON_OPTIMIZER_STRATEGY", "resume_same_objective"),
        "dpo_output": str(plan.run_root / "dpo" / f"gen{plan.generation}" / "attacker.jsonl") if plan.dpo else None,
    }
    if request["optimizer_strategy"] not in {"resume_same_objective", "reset_each_generation"}:
        raise StudyError("unknown optimizer strategy")
    # Pin the executable and relevant source/config inputs so resume cannot
    # silently reuse a different experiment under an existing run identity.
    inputs = {}
    sources = list(ROOT.glob("*.py"))
    for folder, pattern in ((ROOT / "train", "*.py"), (ROOT / "env", "*.py"), (ROOT / "cli", "*.py"), (ROOT / "eval", "*.py"), (ROOT / "harness", "*.ts"), (ROOT / "scripts", "*.sh"), (ROOT / "configs", "*.yaml"), (ROOT / "prompts", "*.md"), (ROOT / "prompts", "*.yaml")):
        sources.extend(folder.rglob(pattern))
    for path in sorted(set(sources)):
        relative = path.relative_to(ROOT)
        if "tests" not in relative.parts and not any(part.startswith(".") or part in {"node_modules", "third_party", "data"} for part in relative.parts):
            inputs[str(relative)] = file_hash(path)
    runtime_name = os.environ.get("ULTRON_ROLLOUT_RUNTIME", "")
    external = os.environ.get("ULTRON_ROLLOUT_COMMAND", "")
    request["runtime"] = {"factory": runtime_name, "command": external}
    if external:
        request["runtime"]["command_sha256"] = file_hash(Path(external))
    request["source_inputs"] = inputs
    request["training_adapters"] = {
        kind: preflight_adapter(kind).identity()
        for kind in (("grpo", "dpo") if plan.dpo else ("grpo",))
    }
    # Normalize tuple/enum task fields to their persisted JSON representation.
    request = json.loads(json.dumps(request))
    if plan.request_path.exists():
        existing = json.loads(plan.request_path.read_text())
        if existing != request:
            raise StudyError("generation inputs changed under this run identity; choose a new --run-id")
    else:
        atomic_write_json(plan.request_path, request)
    atomic_write_json(plan.plan_path, plan.to_dict())
    serving = {
        "base_model_revision": policies["attacker"]["revision"],
        "adapters": [{"id": f"{role}-{item['sha256']}", "checkpoint_id": item["checkpoint_id"], "path": item["adapter_path"], "sha256": item["sha256"], "role": role}
                     for role, item in policies.items() if item["adapter_path"] is not None],
        "base_policies": list({item["checkpoint_id"]: {"id": item["checkpoint_id"], "model": item["model"], "revision": item["revision"], "sha256": item["sha256"]}
                               for item in policies.values() if item["adapter_path"] is None}.values()),
        "role_defaults": {role: f"{role}-{item['sha256']}" if item["adapter_path"] else item["checkpoint_id"] for role, item in policies.items()},
    }
    if policies["attacker"]["revision"] != policies["defender"]["revision"]:
        raise StudyError("both roles must use the same pinned base-model revision")
    seen = {item["checkpoint_id"] for item in serving["adapters"]}
    for pool in pools.values():
        for item in pool.get("entries", []):
            if item["checkpoint_id"] in seen:
                continue
            digest = adapter_hash(Path(item["path"]))
            if item.get("sha256") is not None and item["sha256"] != digest:
                raise StudyError("opponent adapter changed after its pool snapshot")
            model_id = f"{item['role']}-{digest}"
            match = next((entry for entry in serving["adapters"] if entry["id"] == model_id), None)
            if match:
                match.setdefault("checkpoint_ids", []).append(item["checkpoint_id"])
            else:
                serving["adapters"].append({"id": model_id, "checkpoint_id": item["checkpoint_id"], "path": item["path"], "sha256": digest, "role": item["role"]})
            seen.add(item["checkpoint_id"])
    atomic_write_json(plan.run_root / "requests" / f"gen{plan.generation}-opponents.json", pools)
    atomic_write_json(plan.run_root / "requests" / f"gen{plan.generation}-serving.json", serving)
    return request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "export", "prepare", "fingerprint", "execution-check"))
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--family")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--prep-turns", type=int)
    args = parser.parse_args(argv)
    env = dict(os.environ)
    for arg, name in ((args.family, "ULTRON_MODEL_FAMILY"), (args.variant, "ULTRON_STUDY_VARIANT"), (args.seed, "ULTRON_STUDY_SEED"), (args.run_id, "ULTRON_STUDY_RUN_ID"), (args.episodes, "ULTRON_REQUESTED_EPISODES"), (args.study_root, "ULTRON_STUDY_ROOT"), (args.prep_turns, "ULTRON_PREP_TURNS")):
        if arg is not None:
            env[name] = str(arg)
    if args.pilot:
        env["ULTRON_STUDY_PURPOSE"] = "performance-pilot"
    try:
        plan = plan_from_environ(env, generation=args.generation)
        if args.command == "plan":
            print(json.dumps(plan.to_dict(), indent=2))
        elif args.command == "export":
            for name, value in plan.environ().items():
                print(f"export {name}={shlex.quote(value)}")
        elif args.command == "prepare":
            prepare(plan)
            print(plan.request_path)
        elif args.command == "execution-check":
            protocol = load_protocol()
            cap = os.environ.get("ULTRON_SPENDING_CAP_USD")
            assert_paid_execution_allowed(None if cap is None else float(cap), protocol, isolation=protocol.isolation)
            if plan.purpose != "performance-pilot" and (protocol.prep_turns is None or not protocol.prep_turns_calibrated):
                raise StudyError("record calibrated prep_turns in the study protocol before a full study run")
            if plan.purpose != "performance-pilot" and not protocol.curriculum_frozen:
                raise StudyError("freeze the calibrated training curriculum before a full study run")
            print("study execution gates passed")
        else:
            print(file_hash(plan.request_path))
    except (StudyError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
