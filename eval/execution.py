"""Checkpoint-bound evaluation and reuse of completed, versioned jobs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from ultron.train.artifacts import artifact_hash
from ultron.train.io import atomic_write_json


def load_bindings(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("version") != 1 or not isinstance(payload.get("bindings"), list):
        raise ValueError("evaluation bindings require version 1 and a bindings array")
    return payload


def resolve_binding(job: Any, bindings: dict[str, Any]) -> dict[str, Any]:
    base = None
    if job.stage.train_stage == "untrained":
        base = json.loads(job.stage.adapter_path.read_text())
        digest = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    else:
        digest = artifact_hash(job.stage.adapter_path)
    matches = [entry for entry in bindings["bindings"]
               if entry.get("role") == job.role.value and entry.get("adapter_sha256") == digest]
    if len(matches) != 1:
        raise ValueError(f"one exact checkpoint binding required for {job.cell_id}: {digest}")
    entry = matches[0]
    if not entry.get("model") or not entry.get("base_url") or not entry.get("base_model_revision"):
        raise ValueError("binding requires model, base_url and base_model_revision")
    if base is not None:
        if (entry.get("kind") != "base" or entry["base_model_revision"] != base["revision"]
                or entry.get("base_model") != base["model"]
                or entry["model"] != f"{base['model']}@{base['revision']}:base"):
            raise ValueError("untrained evaluation binding must identify the exact base model revision")
    elif digest not in entry["model"]:
        raise ValueError("served model ID must contain the full adapter SHA-256")
    return {**entry, "adapter_path": str(job.stage.adapter_path.resolve())}


def verify_binding(entry: dict[str, Any], *, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
    url = entry["base_url"].rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "ultron")})
    with opener(request, timeout=15) as response:
        payload = json.load(response)
    matches = [item for item in payload.get("data", []) if item.get("id") == entry["model"]]
    if len(matches) != 1:
        raise ValueError(f"server does not expose the requested immutable model {entry['model']}")
    if matches[0].get("base_model_revision") != entry["base_model_revision"]:
        raise ValueError("server base-model revision differs from evaluation binding")
    # The shared Ultron router publishes the digest; plain vLLM binds it in its model ID.
    published = matches[0].get("adapter_sha256", matches[0].get("policy_sha256"))
    if published is not None and published != entry["adapter_sha256"]:
        raise ValueError("server checkpoint digest differs from evaluation binding")
    if published is None and matches[0].get("root") != entry["adapter_path"]:
        raise ValueError("server must publish the checkpoint digest or its exact adapter path")


def bound_job(job: Any, entry: dict[str, Any]) -> Any:
    args = list(job.argv)
    for flag in ("--model", "--models"):
        if flag in args:
            index = args.index(flag) + 1
            prefix = "openai/" if args[index].startswith("openai/") else ""
            args[index] = prefix + entry["model"]
    env = dict(job.env)
    env.update(OPENAI_API_BASE=entry["base_url"], OPENAI_BASE_URL=entry["base_url"])
    return replace(job, argv=tuple(args), env=tuple(env.items()))


def cache_key(job: Any, entry: dict[str, Any]) -> str | None:
    from .cache_inputs import actual_cache_inputs
    context = dict(job.cache_context)
    required = ("dataset_revision", "harness_revision", "seed", "opponent_id", "inference_settings")
    if any(name not in context or context[name] is None for name in required):
        return None
    if not context["dataset_revision"] or not context["harness_revision"]:
        return None
    # Aliases that change underneath a saved result cannot establish cache identity.
    if any(str(context[name]).lower() in {"latest", "main", "master", "head"}
           for name in ("dataset_revision", "harness_revision")):
        return None
    actual_inputs = actual_cache_inputs(job)
    if actual_inputs is None:
        return None
    def normalize(value: str) -> str:
        output = str(job.workdir)
        return "<OUTPUT>" + value[len(output):] if value == output or value.startswith(output + "/") else value
    argv = [normalize(item) for item in job.argv]
    if "--job-name" in argv:
        argv[argv.index("--job-name") + 1] = "<JOB>"
    payload = {"version": 2, "checkpoint": entry["adapter_sha256"],
               "base_model_revision": entry["base_model_revision"],
               "role": job.role.value, "benchmark": job.benchmark.value,
               "metric": job.metric, "context": context, "actual_inputs": actual_inputs,
               "argv": argv, "cwd": normalize(job.cwd) if job.cwd else None,
               "env": {key: normalize(value) for key, value in job.env
                       if "KEY" not in key and "TOKEN" not in key
                       and key not in {"OPENAI_API_BASE", "OPENAI_BASE_URL"}},
               "executor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "harness_adapter_sha256": hashlib.sha256(Path(__file__).with_name("benchmarks.py").read_bytes()).hexdigest(),
               "input_resolver_sha256": hashlib.sha256(Path(__file__).with_name("cache_inputs.py").read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _attempt(job: Any) -> Any:
    path = job.workdir / "attempts" / uuid4().hex
    old, new = str(job.workdir), str(path)
    def moved(value: str) -> str:
        return new + value[len(old):] if value == old or value.startswith(old + "/") else value
    return replace(job, workdir=path, cwd=moved(job.cwd) if job.cwd else None,
                   argv=tuple(moved(item) for item in job.argv),
                   env=tuple((key, moved(value)) for key, value in job.env))


def execute_bound(job: Any, *, bindings: dict[str, Any], cache_dir: Path | None,
                  run: Callable[[Any], Any], verify: Callable[[dict[str, Any]], None] = verify_binding) -> Any:
    from .benchmarks import BenchmarkId, Role, ScoreRow, ScoreStatus
    entry = resolve_binding(job, bindings)
    bound = bound_job(job, entry)
    key = cache_key(bound, entry) if cache_dir is not None else None
    attempt = None
    def execute() -> Any:
        nonlocal attempt
        verify(entry)
        attempt = _attempt(bound)
        row = run(attempt)
        if resolve_binding(job, bindings)["adapter_sha256"] != entry["adapter_sha256"]:
            raise ValueError("checkpoint changed during evaluation")
        if key is not None and cache_key(bound, entry) != key:
            raise ValueError("dataset or harness inputs changed during evaluation")
        return row
    if key is None:
        return execute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt = cache_dir / f"{key}.json"
        if receipt.exists():
            try:
                payload = json.loads(receipt.read_text())
            except (OSError, ValueError):
                payload = {}
            row = payload.get("row", {})
            source = Path(row["source"]) if row.get("source") else None
            result_root = Path(payload["result_root"]) if payload.get("result_root") else None
            if (payload.get("key") == key and row.get("status") == "ok"
                    and source is not None and source.is_file()
                    and result_root is not None and result_root.is_dir()
                    and payload.get("result_sha256") == artifact_hash(result_root)):
                if cache_key(bound, entry) != key:
                    raise ValueError("dataset or harness inputs changed while reading the evaluation cache")
                row.update(benchmark=BenchmarkId(row["benchmark"]), role=Role(row["role"]),
                           status=ScoreStatus.OK, stage_label=job.stage.label,
                           generation=job.stage.generation, train_stage=job.stage.train_stage,
                           step=job.stage.step, detail="reused completed evaluation " + key)
                return ScoreRow(**row)
        row = execute()
        if row.status is ScoreStatus.OK and row.score is not None and math.isfinite(row.score):
            source = Path(row.source) if row.source else None
            if source is not None and source.is_file():
                atomic_write_json(receipt, {"key": key, "row": row.to_dict(),
                                           "result_root": str(attempt.workdir),
                                           "result_sha256": artifact_hash(attempt.workdir)})
        return row


def main() -> int:
    import argparse
    from .benchmarks import (BenchmarkId, BenchmarkJob, IterationStage, Role,
                             ScoreStatus, execute_jobs)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=0)
    args = parser.parse_args()
    raw = json.loads(args.job.read_text())
    stage = raw["stage"]
    job = BenchmarkJob(
        BenchmarkId(raw["benchmark"]), Role(raw["role"]),
        IterationStage(stage["generation"], Role(stage["role"]), stage["train_stage"],
                       stage["step"], Path(stage["adapter_path"])),
        tuple(raw["argv"]), tuple(raw["env"].items()), raw["cwd"],
        Path(raw["workdir"]), raw["serve_hint"], raw["metric"], raw.get("cache_context", {}),
    )
    env = dict(job.env)
    env["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "ultron")
    rows = execute_jobs([replace(job, env=tuple(env.items()))], timeout_s=args.timeout,
                        bindings=load_bindings(args.bindings), cache_dir=args.cache_dir)
    print(json.dumps(rows[0].to_dict(), indent=2))
    return 0 if rows[0].status is ScoreStatus.OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
