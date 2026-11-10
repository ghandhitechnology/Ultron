import hashlib
import json
import socket
import subprocess
import sys

import pytest

from ultron.shared_serving import adapter_digest, bindings_document, load_hardware, managed_serving, validate_manifest, worker_command
from ultron.train.artifacts import artifact_hash


def make_manifest(tmp_path):
    entries = []
    for role in ("attacker", "defender"):
        path = tmp_path / role
        path.mkdir()
        (path / "adapter_config.json").write_text(json.dumps({"r": 64}))
        (path / "adapter_model.safetensors").write_bytes(role.encode())
        digest = artifact_hash(path)
        entries.append({"id": f"{role}-{digest}", "role": role, "path": str(path), "sha256": digest})
    manifest = {"adapters": entries, "role_defaults": {entry["role"]: entry["id"] for entry in entries}, "base_model_revision": "a" * 40}
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps(manifest))
    return file, manifest


def test_both_replicas_load_identical_immutable_rank64_adapters(tmp_path):
    file, expected = make_manifest(tmp_path)
    manifest = validate_manifest(file)
    commands = [worker_command(manifest, load_hardware(), "qwen-8b", port) for port in (8101, 8102)]
    for command in commands:
        assert all(f"{entry['id']}={entry['path']}" in command for entry in expected["adapters"])
        assert command[command.index("--max-lora-rank") + 1] == "64"
        assert command[command.index("--max-model-len") + 1] == "32768"
        assert command[command.index("--revision") + 1] == "a" * 40
        assert "--enable-prefix-caching" in command
        assert json.loads(command[command.index("--chat-template-kwargs") + 1])["enable_thinking"] is True
    bindings = bindings_document(manifest, 18001, 18002)["bindings"]
    assert {binding["base_url"] for binding in bindings} == {"http://127.0.0.1:18001/v1", "http://127.0.0.1:18002/v1"}


def test_manifest_rejects_mutated_adapter_before_launch(tmp_path):
    file, manifest = make_manifest(tmp_path)
    from pathlib import Path
    path = Path(manifest["adapters"][0]["path"])
    assert adapter_digest(path) == artifact_hash(path)
    (path / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed adapter"):
        validate_manifest(file)


def test_generation_zero_serves_pinned_base_without_random_lora(tmp_path):
    model, revision = "Qwen/Qwen3-8B", "a" * 40
    digest = hashlib.sha256(json.dumps({"model": model, "revision": revision}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity = f"{model}@{revision}:base"
    payload = {"base_model_revision": revision, "adapters": [], "base_policies": [
        {"id": identity, "model": model, "revision": revision, "sha256": digest}],
        "role_defaults": {"attacker": identity, "defender": identity}}
    file = tmp_path / "base.json"
    file.write_text(json.dumps(payload))
    manifest = validate_manifest(file)
    command = worker_command(manifest, load_hardware(), "qwen-8b", 8101)
    assert "--enable-lora" not in command
    assert command[command.index("--served-model-name") + 1] == identity
    assert {row["role"] for row in bindings_document(manifest, 8001, 8002)["bindings"]} == {"attacker", "defender"}


def test_managed_serving_waits_for_readiness_and_stops_on_rollout_failure(tmp_path, monkeypatch):
    # A real local process exercises startup polling and shutdown without GPUs.
    code = '''
import sys, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        body = json.dumps({'ready': True, 'instance_id': sys.argv[2]}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
'''
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    original = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        assert command[1:3] == ["-m", "ultron.shared_serving"]
        assert command[command.index("--gpu-ids") + 1] == "0,1"
        child = original([sys.executable, "-c", code, str(port), command[command.index("--instance-id") + 1]], **kwargs)
        children.append(child)
        return child

    monkeypatch.setenv("ULTRON_ROUTER_PORT", str(port))
    monkeypatch.setenv("ULTRON_ROLLOUT_GPU_IDS", "0,1")
    monkeypatch.setenv("ULTRON_SERVING_STARTUP_TIMEOUT", "5")
    monkeypatch.setattr("ultron.shared_serving.subprocess.Popen", launch)
    request = {"generation": 0, "output": str(tmp_path / "traces" / "gen0"), "identity": {"model": "qwen-8b"}}
    with pytest.raises(RuntimeError, match="guest failure"):
        with managed_serving(tmp_path / "manifest.json", request):
            assert children[0].poll() is None
            raise RuntimeError("guest failure")
    assert children[0].poll() is not None
