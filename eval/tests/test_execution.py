import json
from dataclasses import replace
from pathlib import Path

import pytest

from ultron.eval.benchmarks import (BenchmarkId, BenchmarkJob, IterationStage, ScoreRow,
                                    ScoreStatus, write_scores, load_scores)
from ultron.eval.execution import artifact_hash, cache_key, execute_bound, resolve_binding
from ultron.train.schema_v1 import Role


def fixture(tmp_path):
    import sys
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}")
    harness = tmp_path / "harness"
    harness.mkdir()
    executable = harness / "pier"
    executable.write_text(f"#!{sys.executable}\nprint('fixture harness')\n")
    executable.chmod(0o755)
    dataset = tmp_path / "tasks"
    dataset.mkdir()
    (dataset / "task.json").write_text('{"instruction":"fixture"}')
    job = BenchmarkJob(BenchmarkId.DEEP_SWE, Role.ATTACKER,
                       IterationStage(0, Role.ATTACKER, "grpo", 1, adapter),
                       (str(executable), "run", "-p", str(dataset), "--model", "attacker-lora"), (), None,
                       tmp_path / "out", "", "score",
                       {"dataset_revision": "abc123", "harness_revision": "def456",
                        "seed": 0, "opponent_id": "none", "inference_settings": {"temperature": 1},
                        "local_inputs": {"harness_root": str(harness), "dataset_root": str(dataset)}})
    digest = artifact_hash(adapter)
    entry = {"role": "attacker", "adapter_sha256": digest, "model": "attacker-" + digest,
             "base_url": "http://127.0.0.1:8001/v1", "base_model_revision": "frozen-base"}
    return job, {"version": 1, "bindings": [entry]}


def success(job):
    job.workdir.mkdir(parents=True)
    source = job.workdir / "score.json"
    source.write_text('{"score": 0.5}')
    return ScoreRow(job.benchmark, job.role, job.stage.label, job.stage.generation,
                    job.stage.train_stage, job.stage.step, 0.5, 2, ScoreStatus.OK,
                    job.metric, str(source))


def test_exact_completed_job_reused_and_changed_source_reruns(tmp_path):
    job, bindings = fixture(tmp_path)
    calls = []
    def run(bound):
        calls.append(bound)
        return success(bound)
    def execute():
        return execute_bound(job, bindings=bindings, cache_dir=tmp_path / "cache", run=run, verify=lambda _: None)
    first = execute()
    assert execute().score == first.score
    assert len(calls) == 1
    assert calls[0].argv[-1] == bindings["bindings"][0]["model"]
    Path(first.source).write_text('{"score": 1}')
    execute()
    assert len(calls) == 2
    assert calls[0].workdir != calls[1].workdir


def test_cache_requires_versions_and_changes_for_scope(tmp_path):
    job, bindings = fixture(tmp_path)
    entry = bindings["bindings"][0]
    key = cache_key(job, entry)
    assert key
    for field, value in (("dataset_revision", "other"), ("harness_revision", "other"),
                         ("seed", 1), ("opponent_id", "other"), ("inference_settings", {"temperature": 0})):
        assert cache_key(replace(job, cache_context={**job.cache_context, field: value}), entry) != key
    assert cache_key(replace(job, cache_context={}), entry) is None
    assert cache_key(replace(job, cache_context={**job.cache_context, "dataset_revision": "latest"}), entry) is None


def test_mutated_adapter_never_binds_to_previous_policy(tmp_path):
    job, bindings = fixture(tmp_path)
    (job.stage.adapter_path / "adapter_model.safetensors").write_bytes(b"new weights")
    with pytest.raises(ValueError, match="exact checkpoint binding"):
        resolve_binding(job, bindings)


def test_checkpoint_mutation_during_run_cannot_commit_score(tmp_path):
    job, bindings = fixture(tmp_path)
    def run(bound):
        (job.stage.adapter_path / "adapter_model.safetensors").write_bytes(b"new weights")
        return success(bound)
    with pytest.raises(ValueError, match="exact checkpoint binding|changed during evaluation"):
        execute_bound(job, bindings=bindings, cache_dir=tmp_path / "cache", run=run, verify=lambda _: None)
    assert not list((tmp_path / "cache").glob("*.json"))


def test_failed_new_execution_replaces_prior_success(tmp_path):
    job, _ = fixture(tmp_path)
    row = success(job)
    output = tmp_path / "scores"
    write_scores([row], output)
    write_scores([replace(row, score=None, n=None, status=ScoreStatus.FAILED)], output, replace_current=True)
    assert load_scores(output / "scores.json")[0].status is ScoreStatus.FAILED


def test_untrained_control_binds_shared_base_without_fake_adapter(tmp_path):
    import hashlib
    job, _ = fixture(tmp_path)
    base = {"model": "Qwen/Qwen3-8B", "revision": "a" * 40}
    descriptor = tmp_path / "base.json"
    descriptor.write_text(json.dumps(base))
    digest = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    entry = {"kind": "base", "role": "attacker", "adapter_sha256": digest,
             "model": f"{base['model']}@{base['revision']}:base", "base_model": base["model"],
             "base_model_revision": base["revision"], "base_url": "http://127.0.0.1:8001/v1"}
    job = replace(job, stage=replace(job.stage, train_stage="untrained", adapter_path=descriptor))
    assert resolve_binding(job, {"bindings": [entry]})["adapter_sha256"] == digest
    entry["base_model_revision"] = "b" * 40
    with pytest.raises(ValueError, match="exact base model"):
        resolve_binding(job, {"bindings": [entry]})


def test_identical_checkpoint_reuse_retains_requested_stage_identity(tmp_path):
    job, bindings = fixture(tmp_path)
    calls = []
    def run(bound):
        calls.append(bound)
        return success(bound)
    options = dict(bindings=bindings, cache_dir=tmp_path / "cache", run=run, verify=lambda _: None)
    execute_bound(job, **options)
    repeated = replace(job, workdir=tmp_path / "different-output",
                       stage=replace(job.stage, generation=2, step=10))
    row = execute_bound(repeated, **options)
    assert len(calls) == 1
    assert row.stage_label == repeated.stage.label
    assert row.generation == 2
    assert row.step == 10


def test_shared_serving_manifest_produces_usable_eval_bindings(tmp_path):
    from ultron.shared_serving import bindings_document, validate_manifest
    job, _ = fixture(tmp_path)
    (job.stage.adapter_path / "adapter_config.json").write_text('{"r":64}')
    digest = artifact_hash(job.stage.adapter_path)
    manifest = {"base_model_revision": "a" * 40,
                "adapters": [
                    {"id": role + "-" + digest, "role": role,
                     "path": str(job.stage.adapter_path), "sha256": digest}
                    for role in ("attacker", "defender")],
                "role_defaults": {role: role + "-" + digest for role in ("attacker", "defender")}}
    path = tmp_path / "serving.json"
    path.write_text(json.dumps(manifest))
    bindings = bindings_document(validate_manifest(path), 8001, 8002)
    assert resolve_binding(job, bindings)["model"] == "attacker-" + digest


def test_server_base_revision_must_match_binding(tmp_path):
    from io import BytesIO
    from ultron.eval.execution import verify_binding
    job, bindings = fixture(tmp_path)
    entry = resolve_binding(job, bindings)
    def opener(*args, **kwargs):
        return BytesIO(json.dumps({"data": [{"id": entry["model"],
                        "adapter_sha256": entry["adapter_sha256"],
                        "base_model_revision": "wrong-base"}]}).encode())
    with pytest.raises(ValueError, match="base-model revision"):
        verify_binding(entry, opener=opener)


def test_untrained_plans_keep_immutable_base_descriptors(tmp_path):
    from ultron.eval.run_benchmarks import run_suite
    from ultron.eval.benchmarks import load_suite_config
    output = tmp_path / "controls"
    run_suite(archive_root=tmp_path / "archives", output=output, untrained=True,
              base_model_revision="a" * 40)
    old_plan = json.loads((output / "plan.json").read_text())
    old_path = Path(old_plan["jobs"][0]["stage"]["adapter_path"])
    old_content = old_path.read_text()
    run_suite(archive_root=tmp_path / "archives", output=output, untrained=True,
              base_model_revision="b" * 40)
    new_plan = json.loads((output / "plan.json").read_text())
    assert new_plan["jobs"][0]["stage"]["adapter_path"] != str(old_path)
    assert old_path.read_text() == old_content
