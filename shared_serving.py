"""Supervise two vLLM replicas and role capture endpoints on the H100 pair."""
from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Iterator, Mapping
from urllib.request import urlopen
from uuid import uuid4

import yaml

from ultron.cli.serve import _stop_process_group
from ultron.response_proxy import CaptureProxy
from ultron.serving_router import ReplicaRouter, RouterServer
from ultron.train.family import resolve
from ultron.train.artifacts import artifact_hash
from ultron.train.io import atomic_write_json
from ultron.train.schema_v1 import Role

ROOT = Path(__file__).resolve().parent


def adapter_digest(path: Path) -> str:
    """Hash the config and actual PEFT weights, independent of path/mtime."""
    names = ["adapter_config.json"]
    weights = [name for name in ("adapter_model.safetensors", "adapter_model.bin") if (path / name).is_file()]
    if not weights or not (path / names[0]).is_file():
        raise ValueError(f"adapter config or weights missing: {path}")
    return artifact_hash(path)


def validate_manifest(path: Path) -> dict:
    document = json.loads(path.read_text())
    revision = document.get("base_model_revision", "")
    if len(revision) != 40 or any(value not in "0123456789abcdef" for value in revision):
        raise ValueError("serving manifest requires a pinned 40-character base_model_revision")
    entries = document.get("adapters", [])
    bases = document.get("base_policies", [])
    if not isinstance(entries, list) or not isinstance(bases, list) or (not entries and not bases):
        raise ValueError("serving manifest requires immutable base or adapter policies")
    document["adapters"] = entries
    document["base_policies"] = bases
    ids: set[str] = set()
    for entry in bases:
        expected = hashlib.sha256(json.dumps({"model": entry["model"], "revision": revision}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (entry.get("sha256") != expected or entry.get("revision") != revision
                or entry.get("id") != f"{entry['model']}@{revision}:base"):
            raise ValueError("base policy identity differs from its pinned model/revision")
        if entry["id"] in ids:
            raise ValueError("duplicate base policy ID")
        ids.add(entry["id"])
    for entry in entries:
        role = entry.get("role")
        source = Path(entry["path"]).expanduser().resolve()
        expected = entry.get("sha256")
        if role not in {"attacker", "defender"} or adapter_digest(source) != expected:
            raise ValueError(f"invalid role or changed adapter: {source}")
        if entry.get("id") != f"{role}-{expected}":
            raise ValueError("adapter id must equal role + '-' + full content SHA-256")
        if entry["id"] in ids:
            raise ValueError("duplicate serving adapter ID")
        ids.add(entry["id"])
        config = json.loads((source / "adapter_config.json").read_text())
        if config.get("r") != 64:
            raise ValueError("H100 study serving requires rank-64 adapters")
        entry["path"] = str(source)
    defaults = document.get("role_defaults", {})
    for role in ("attacker", "defender"):
        if defaults.get(role) not in ids:
            raise ValueError(f"missing immutable {role} default")
        adapter = next((entry for entry in entries if entry["id"] == defaults[role]), None)
        if adapter and adapter["role"] != role:
            raise ValueError(f"default {role} selected the other role's adapter")
    return document


def load_hardware(path: Path = ROOT / "configs" / "h100.yaml") -> dict:
    config = yaml.safe_load(path.read_text())
    serving = config["serving"]
    if (config["hardware"]["gpu_count"] != 2 or serving["replicas"] != 2
            or serving["tensor_parallel_size"] != 1 or serving["max_model_len"] != 32768
            or serving["max_lora_rank"] != 64):
        raise ValueError("serving profile must preserve two replicas, TP1, context32768 and rank64")
    return config


def worker_command(manifest: Mapping, config: Mapping, family: str, port: int, *, prefix_caching: bool = True) -> list[str]:
    pack = resolve(family)
    serving = config["serving"]
    count = len(manifest["adapters"])
    if any(entry["model"] != pack.base_model for entry in manifest.get("base_policies", [])):
        raise ValueError("manifest base policy differs from the selected model family")
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", pack.base_model,
               "--revision", manifest["base_model_revision"], "--tokenizer-revision", manifest["base_model_revision"]]
    if manifest.get("base_policies"):
        command.append("--served-model-name")
        command.extend(entry["id"] for entry in manifest["base_policies"])
    if count:
        command.extend(["--enable-lora", "--lora-modules"])
        command.extend(f"{entry['id']}={entry['path']}" for entry in manifest["adapters"])
        command.extend(["--max-lora-rank", str(serving["max_lora_rank"]),
                        "--max-loras", str(min(count, serving["max_loras"])),
                        "--max-cpu-loras", str(max(count, serving["max_loras"]))])
    command.extend([
        "--tensor-parallel-size", "1", "--max-model-len", str(serving["max_model_len"]),
        "--max-num-seqs", str(serving["max_num_seqs"]),
        "--max-num-batched-tokens", str(serving["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(serving["gpu_memory_utilization"]),
        "--enable-chunked-prefill",
        "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
        "--host", "127.0.0.1", "--port", str(port),
    ])
    command.extend(pack.vllm_chat_template_args())
    return command


def _ports(raw: str, count: int) -> list[int]:
    values = [int(value) for value in raw.split(",")]
    if len(values) != count or len(set(values)) != count or any(not 1 <= value <= 65535 for value in values):
        raise ValueError(f"expected {count} distinct valid ports")
    return values


def _ensure_ports_free(ports: list[int]) -> None:
    reservations = []
    try:
        for port in ports:
            handle = socket.socket()
            reservations.append(handle)
            handle.bind(("127.0.0.1", port))
    finally:
        for handle in reservations:
            handle.close()


def _ensure_gpus_idle(devices: list[str]) -> None:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    identifiers = dict(tuple(part.strip() for part in line.split(",", 1)) for line in result.stdout.splitlines())
    selected = set()
    for device in devices:
        uuid = identifiers.get(device, device if device in identifiers.values() else None)
        if uuid is None:
            raise ValueError(f"GPU {device} is unavailable")
        selected.add(uuid)
    running = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15)
    if any(line.split(",", 1)[0].strip() in selected for line in running.stdout.splitlines()):
        raise ValueError("selected GPUs already have compute processes; stop training/serving before rollout")


def bindings_document(manifest: Mapping, attacker_port: int, defender_port: int) -> dict:
    bindings = [{
        "role": entry["role"], "adapter_sha256": entry["sha256"], "model": entry["id"],
        "base_url": f"http://127.0.0.1:{attacker_port if entry['role'] == 'attacker' else defender_port}/v1",
        "base_model_revision": manifest["base_model_revision"],
    } for entry in manifest["adapters"]]
    for entry in manifest.get("base_policies", []):
        for role in ("attacker", "defender"):
            bindings.append({"role": role, "adapter_sha256": entry["sha256"], "model": entry["id"],
                             "base_url": f"http://127.0.0.1:{attacker_port if role == 'attacker' else defender_port}/v1",
                             "base_model_revision": manifest["base_model_revision"], "base_model": entry["model"], "kind": "base"})
    return {"version": 1, "bindings": bindings}


def runtime_bindings(manifest: Mapping, request: Mapping) -> dict:
    """Map policy identities in the frozen plan to immutable API model IDs."""
    result = {"attacker": {}, "defender": {}}
    for role in result:
        port = os.environ.get(f"ULTRON_{role.upper()}_PORT", "8001" if role == "attacker" else "8002")
        for entry in [*manifest["adapters"], *manifest.get("base_policies", [])]:
            if entry in manifest["adapters"] and entry["role"] != role:
                continue
            descriptor = {"model": entry["id"], "sha256": entry["sha256"], "base_url": f"http://127.0.0.1:{port}/v1"}
            for identifier in (entry["id"], entry.get("checkpoint_id", entry["id"]), *entry.get("checkpoint_ids", [])):
                previous = result[role].get(identifier)
                if previous and previous != descriptor:
                    raise ValueError("serving manifest maps one checkpoint to different policies")
                result[role][identifier] = descriptor
        current = request.get("policies", {}).get(role)
        if current:
            matches = [value for value in result[role].values() if value["sha256"] == current["sha256"]]
            if not matches:
                raise ValueError(f"serving manifest does not contain the requested {role} policy")
            result[role][current["checkpoint_id"]] = matches[0]
    return result


def supervise(args, manifest: dict, config: dict) -> int:
    workers = _ports(args.worker_ports, 2)
    all_ports = workers + [args.router_port, args.attacker_port, args.defender_port]
    if len(set(all_ports)) != len(all_ports):
        raise ValueError("worker, router and role ports must all differ")
    _ensure_ports_free(all_ports)
    devices = args.gpu_ids.split(",")
    if len(devices) != 2 or len(set(devices)) != 2 or any(not value.strip() for value in devices):
        raise ValueError("exactly two distinct GPU IDs are required")
    if importlib.util.find_spec("vllm") is None:
        raise ValueError("vLLM is unavailable in this Python environment")
    from ultron.train.compute import load_hardware as load_profile, verify_hardware
    verify_hardware(replace(load_profile(args.hardware_config), devices=tuple(devices)))
    _ensure_gpus_idle(devices)
    models = [{"id": entry["id"], "object": "model", "owned_by": "ultron", "root": entry["path"],
               "adapter_sha256": entry["sha256"], "base_model_revision": manifest["base_model_revision"],
               "role": entry["role"]} for entry in manifest["adapters"]]
    models.extend({"id": entry["id"], "object": "model", "owned_by": "ultron", "root": entry["model"],
                   "adapter_sha256": entry["sha256"], "base_model_revision": manifest["base_model_revision"],
                   "kind": "base"} for entry in manifest.get("base_policies", []))
    aliases = {f"{role}-lora": model for role, model in manifest["role_defaults"].items()} if args.allow_role_aliases else {}
    router = ReplicaRouter([f"http://127.0.0.1:{port}" for port in workers], models, aliases=aliases)
    servers = []
    threads = []
    children: list[subprocess.Popen] = []
    stopped = threading.Event()
    previous = {number: signal.signal(number, lambda *_: stopped.set()) for number in (signal.SIGTERM, signal.SIGINT)}
    began = time.time()
    status = 1
    try:
        for device, port in zip(devices, workers):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
            env.pop("VLLM_ALLOW_RUNTIME_LORA_UPDATING", None)
            children.append(subprocess.Popen(worker_command(manifest, config, args.family, port, prefix_caching=not args.no_prefix_caching), env=env, start_new_session=True))
        server = RouterServer(("127.0.0.1", args.router_port), router, instance_id=args.instance_id)
        servers.append(server)
        for role, port in ((Role.ATTACKER, args.attacker_port), (Role.DEFENDER, args.defender_port)):
            servers.append(CaptureProxy(f"http://127.0.0.1:{args.router_port}", role,
                                        generation=args.generation, responses_dir=args.responses_dir,
                                        listen_host="127.0.0.1", listen_port=port))
        for server in servers:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
        while not stopped.wait(0.2):
            if any(child.poll() is not None for child in children) or any(not thread.is_alive() for thread in threads):
                break
        else:
            status = 0
    finally:
        for index, server in enumerate(servers):
            if index < len(threads):
                server.shutdown()
            server.server_close()
        for child in children:
            _stop_process_group(child)
            child.wait()
        for number, handler in previous.items():
            signal.signal(number, handler)
        if args.metrics_path:
            ended = time.time()
            atomic_write_json(args.metrics_path, {"schema": "ultron.serving-allocation.v1", "started_unix_s": began,
                                                "finished_unix_s": ended, "devices": devices,
                                                "allocated_gpu_hours": len(devices) * (ended - began) / 3600,
                                                "exit_status": status})
    return status


@contextmanager
def managed_serving(manifest: Path, request: Mapping) -> Iterator[None]:
    config = load_hardware()
    router_port = int(os.environ.get("ULTRON_ROUTER_PORT", "8200"))
    instance_id = uuid4().hex
    output = Path(request.get("output", ".")) / "metadata"
    command = [sys.executable, "-m", "ultron.shared_serving", "--manifest", str(manifest),
               "--family", request.get("identity", {}).get("model", "qwen-8b"),
               "--generation", str(request["generation"]),
               "--gpu-ids", os.environ.get("ULTRON_ROLLOUT_GPU_IDS", ",".join(map(str, config["hardware"]["devices"]))),
               "--worker-ports", os.environ.get("ULTRON_WORKER_PORTS", "8101,8102"),
               "--router-port", str(router_port),
               "--instance-id", instance_id,
               "--attacker-port", os.environ.get("ULTRON_ATTACKER_PORT", "8001"),
               "--defender-port", os.environ.get("ULTRON_DEFENDER_PORT", "8002"),
               "--responses-dir", os.environ.get("ULTRON_RESPONSES_DIR", str(Path(request["output"]).parents[1] / "responses" / f"gen{request['generation']}")),
               "--bindings-output", str(output / "serving-bindings.json"),
               "--metrics-path", str(output / f"serving-{time.time_ns()}.json")]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        deadline = time.monotonic() + float(os.environ.get("ULTRON_SERVING_STARTUP_TIMEOUT", "600"))
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("shared model workers exited before becoming ready")
            try:
                with urlopen(f"http://127.0.0.1:{router_port}/health", timeout=3) as response:
                    if response.status == 200 and json.load(response).get("instance_id") == instance_id:
                        break
            except OSError:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("shared model workers did not become ready before the startup deadline")
        yield
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired as exc:
            # The supervisor owns child groups; do not allow training after an unclean stop.
            raise RuntimeError("model workers did not stop; training must remain blocked") from exc
        if process.returncode != 0 and sys.exc_info()[0] is None:
            raise RuntimeError(f"shared model workers exited with status {process.returncode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs" / "h100.yaml")
    parser.add_argument("--family", default="qwen-8b")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--worker-ports", default="8101,8102")
    parser.add_argument("--router-port", type=int, default=8200)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--attacker-port", type=int, default=8001)
    parser.add_argument("--defender-port", type=int, default=8002)
    parser.add_argument("--responses-dir", type=Path, default=ROOT / "data" / "responses")
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--bindings-output", type=Path)
    parser.add_argument("--allow-role-aliases", action="store_true")
    parser.add_argument("--no-prefix-caching", action="store_true", help="Uncached performance baseline; preserves model and context.")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = validate_manifest(args.manifest)
        config = load_hardware(args.hardware_config)
        if args.bindings_output:
            atomic_write_json(args.bindings_output, bindings_document(manifest, args.attacker_port, args.defender_port))
        if args.preflight_only:
            commands = [worker_command(manifest, config, args.family, port, prefix_caching=not args.no_prefix_caching) for port in _ports(args.worker_ports, 2)]
            print(json.dumps({"status": "configuration_validated", "commands": commands, "vllm_installed": importlib.util.find_spec("vllm") is not None}))
            return 0
        return supervise(args, manifest, config)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"Shared serving unavailable: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
