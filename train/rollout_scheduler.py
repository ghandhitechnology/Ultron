"""Bounded episode scheduling around an explicitly supplied Pi/KVM runtime.

The runtime factory receives the immutable request document and returns an object
with preflight(), plan(), and open_worker(index). Each worker owns its guest and
implements runner_for(task) and close(). runner_for must use task.seed for model
sampling and local randomness, never process-global RNG state. It must select the
exact checkpoints in task.config. Only complete episode pairs become JSONL files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
from queue import Empty, Queue
import re
import signal
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from ultron.env.backend import IsolationBackend
from .episode_runner import EpisodeConfig, EpisodeRunner, GuestVm
from .io import atomic_write_json, atomic_write_text
from .schema_v1 import Role, TrajectoryV1


@dataclass(frozen=True)
class EpisodeTask:
    index: int
    task_id: str
    seed: int
    config: EpisodeConfig
    snapshot_sha256: str | None = None


class EpisodeWorker(Protocol):
    guest: GuestVm

    def runner_for(self, task: EpisodeTask) -> EpisodeRunner: ...

    def close(self) -> None: ...


class RolloutRuntime(Protocol):
    def preflight(self) -> Mapping[str, Any]:
        """Check real dependencies; return isolation=kvm, pi_ready/restore_verified=True."""
        ...

    def plan(self) -> Sequence[EpisodeTask]: ...

    def open_worker(self, index: int) -> EpisodeWorker: ...


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def episode_seed(seed: int, index: int) -> int:
    """Scheduling-independent seed accepted by OpenAI-compatible model APIs."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{index}".encode()).digest()[:4], "big") & 0x7FFFFFFF


def load_runtime(spec: str, request: Mapping[str, Any]) -> RolloutRuntime:
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("runtime must be module:factory for the verified Pi/KVM integration")
    try:
        factory = getattr(importlib.import_module(module_name), factory_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Pi/KVM runtime is unavailable: {spec}") from exc
    return factory(request)


def prepare(runtime: RolloutRuntime, request: Mapping[str, Any]) -> list[EpisodeTask]:
    capabilities = runtime.preflight()
    if capabilities.get("isolation") != "kvm" or not capabilities.get("pi_ready") or not capabilities.get("restore_verified"):
        raise ValueError("runtime preflight must verify KVM, Pi transport, and verified clean restores")
    tasks = list(runtime.plan())
    expected = int(request["episodes"])
    group_size = int(request.get("group_size", 6))
    if expected < 1 or group_size < 1 or expected % group_size:
        raise ValueError("episodes must contain complete groups")
    if len(tasks) != expected or sorted(task.index for task in tasks) != list(range(expected)):
        raise ValueError("runtime plan must include each requested episode index exactly once")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("runtime plan contains duplicate episode IDs")
    groups: dict[str, tuple] = {}
    counts: Counter[str] = Counter()
    for task in tasks:
        if not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,160}", task.task_id):
            raise ValueError("episode IDs must be safe study identifiers of at most 160 characters")
        cfg = task.config
        if cfg.generation != request["generation"]:
            raise ValueError("episode generation differs from request")
        if task.seed != episode_seed(int(request.get("seed", 0)), task.index):
            raise ValueError("runtime plan seed differs from deterministic episode seed")
        if cfg.episode_id != task.task_id:
            raise ValueError("EpisodeConfig.episode_id must equal task_id")
        if not all((cfg.attacker_ckpt, cfg.defender_ckpt, cfg.opponent_checkpoint_id)):
            raise ValueError("every episode requires immutable role and opponent checkpoint IDs")
        if "shaping" in request and cfg.shaping != request["shaping"]:
            raise ValueError("episode shaping differs from request")
        if request.get("tasks") and cfg.profile_id not in {entry.get("task_id", entry.get("id")) for entry in request["tasks"]}:
            raise ValueError("episode profile is outside the requested training tasks")
        policies = request.get("policies", {})
        if policies:
            if not isinstance(task.snapshot_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", task.snapshot_sha256):
                raise ValueError("study episode must pin its verified golden snapshot SHA-256")
            if cfg.learner_role not in (Role.ATTACKER, Role.DEFENDER):
                raise ValueError("study episode requires an explicit learner_role")
            learner_policy = cfg.attacker_ckpt if cfg.learner_role is Role.ATTACKER else cfg.defender_ckpt
            if learner_policy != policies[cfg.learner_role.value]["checkpoint_id"]:
                raise ValueError("learner checkpoint differs from the requested current policy")
            opponent_policy = cfg.defender_ckpt if cfg.learner_role is Role.ATTACKER else cfg.attacker_ckpt
            if cfg.opponent_checkpoint_id != opponent_policy:
                raise ValueError("opponent checkpoint differs from the frozen opposing policy")
        if request.get("prep_turns") is not None and cfg.prep_turns != request["prep_turns"]:
            raise ValueError("runtime changed defender preparation turns")
        identity = (cfg.profile_id, cfg.attacker_ckpt, cfg.defender_ckpt, cfg.opponent_checkpoint_id, cfg.shaping, cfg.prep_turns, cfg.learner_role, task.snapshot_sha256)
        if cfg.group_id in groups and groups[cfg.group_id] != identity:
            raise ValueError("profile, policies, opponent, shaping and prep must remain frozen within a group")
        groups[cfg.group_id] = identity
        counts[cfg.group_id] += 1
    if any(count != group_size for count in counts.values()):
        raise ValueError("runtime plan contains an incomplete or oversized group")
    if request.get("policies") and {task.config.learner_role for task in tasks} != {Role.ATTACKER, Role.DEFENDER}:
        raise ValueError("study plan must contain complete learner groups for both roles")
    return sorted(tasks, key=lambda task: task.index)


def _pilot_tasks(tasks: Sequence[EpisodeTask], count: int) -> list[EpisodeTask]:
    groups: dict[str, list[EpisodeTask]] = {}
    for task in tasks:
        groups.setdefault(task.config.group_id, []).append(task)
    buckets: dict[tuple[str, str], deque] = defaultdict(deque)
    for group_id, members in groups.items():
        cfg = members[0].config
        buckets[(cfg.learner_role.value if cfg.learner_role else "both", cfg.profile_id)].append(group_id)
    selected = []
    # Role first, then profile, with the original group order inside each bucket.
    roles = sorted({key[0] for key in buckets})
    profile_cursors = {role: 0 for role in roles}
    while len(selected) < count:
        progressed = False
        for role in roles:
            keys = sorted(key for key in buckets if key[0] == role and buckets[key])
            if not keys:
                continue
            position = profile_cursors[role] % len(keys)
            profile_cursors[role] += 1
            selected.extend(groups[buckets[keys[position]].popleft()])
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            raise ValueError("early gate requests more episodes than available complete groups")
    if len(selected) != count:
        raise ValueError("early gate must use complete groups")
    return sorted(selected, key=lambda task: task.index)


def _timed_runner(runner: EpisodeRunner, phases: list[dict]) -> EpisodeRunner:
    def timed(name, function):
        def invoke(*args):
            started = time.monotonic()
            event = {"phase": name}
            if name == "turn":
                event.update(role=args[1].value, turn_index=args[3])
            try:
                return function(*args)
            finally:
                event["duration_s"] = time.monotonic() - started
                phases.append(event)
        return invoke
    return EpisodeRunner(
        snapshot_sha256=runner.snapshot_sha256,
        load_profile=runner.load_profile,
        run_turn=timed("turn", runner.run_turn),
        final_probe=timed("final_probe", runner.final_probe),
        restore=timed("restore", runner.restore),
        turns_per_side=runner.turns_per_side,
        turn_probe=timed("turn_probe", runner.turn_probe) if runner.turn_probe else None,
    )


@contextmanager
def _output_lock(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".rollout.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another scheduler owns this rollout output") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _paths(output: Path, task: EpisodeTask) -> tuple[Path, Path]:
    name = f"episode-{task.index:08d}-{task.task_id}"
    return output / "metadata" / "episodes" / f"{name}.json", output / "metadata" / "commits" / f"{name}.json"


def _completed(output: Path, task: EpisodeTask, request_hash: str) -> dict | None:
    trace, metadata = _paths(output, task)
    if not metadata.exists():
        return None
    entry = json.loads(metadata.read_text())
    if entry.get("request_sha256") != request_hash or entry.get("task_sha256") != fingerprint(asdict(task)):
        raise ValueError(f"resume identity mismatch: {task.task_id}")
    if not trace.is_file() or hashlib.sha256(trace.read_bytes()).hexdigest() != entry.get("sha256"):
        raise ValueError(f"completed episode is missing or changed: {task.task_id}")
    return entry


def _commit(output: Path, task: EpisodeTask, trajectories: list[TrajectoryV1], request_hash: str, phases: list[dict], elapsed: float, guest: GuestVm, snapshot_sha256: str) -> dict:
    if len(trajectories) != 2 or {item.role for item in trajectories} != {Role.ATTACKER, Role.DEFENDER}:
        raise ValueError("episode must contain exactly one trajectory per role")
    for item in trajectories:
        expected_policy = task.config.attacker_ckpt if item.role is Role.ATTACKER else task.config.defender_ckpt
        if (item.episode_id != task.task_id or item.group_id != task.config.group_id
                or item.opponent_checkpoint_id != task.config.opponent_checkpoint_id
                or item.generation != task.config.generation or item.profile_id != task.config.profile_id
                or item.isolation_backend != IsolationBackend.KVM
                or item.behavior_policy_id != expected_policy
                or item.trainable != (task.config.learner_role is None or task.config.learner_role is item.role)):
            raise ValueError("runtime trajectory provenance differs from the frozen plan")
    if trajectories[0].terminal != trajectories[1].terminal:
        raise ValueError("role trajectories disagree on terminal outcome")
    trace, metadata = _paths(output, task)
    body = json.dumps([item.to_dict() for item in trajectories], separators=(",", ":")) + "\n"
    entry = {
        "schema": "ultron.episode-commit.v1", "task_id": task.task_id,
        "task_sha256": fingerprint(asdict(task)), "request_sha256": request_hash,
        "file": str(trace.relative_to(output)), "sha256": hashlib.sha256(body.encode()).hexdigest(),
        "index": task.index, "seed": task.seed, "group_id": task.config.group_id,
        "input_policy": {"attacker": task.config.attacker_ckpt, "defender": task.config.defender_ckpt},
        "opponent_checkpoint_id": task.config.opponent_checkpoint_id,
        "guest_id": guest.vm_id, "image_ref": guest.image_ref, "snapshot_sha256": snapshot_sha256,
        "elapsed_s": elapsed, "phases": phases,
    }
    atomic_write_text(trace, body)
    # This marker is the commit point. A trace without its marker is rerun after a crash.
    atomic_write_json(metadata, entry)
    return entry


def _materialize(source: Path, destination: Path, tasks: Sequence[EpisodeTask]) -> list[dict]:
    lines: dict[str, list[str]] = {"attacker": [], "defender": []}
    for task in sorted(tasks, key=lambda item: item.index):
        trace, _ = _paths(source, task)
        for row in json.loads(trace.read_text()):
            lines[row["role"]].append(json.dumps(row, separators=(",", ":")) + "\n")
    files = []
    for role, records in lines.items():
        body = "".join(records)
        path = destination / f"{role}.jsonl"
        atomic_write_text(path, body)
        files.append({"path": path.name, "sha256": hashlib.sha256(body.encode()).hexdigest()})
    return files


def run(runtime: RolloutRuntime, tasks: Sequence[EpisodeTask], request: Mapping[str, Any], output: Path, *, workers: int = 16, request_sha256: str | None = None, serving_factory: Callable | None = None) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    request_hash = request_sha256 or fingerprint(request)
    with _output_lock(output):
        metadata = output / "metadata"
        manifest_path = metadata / "plan.json"
        frozen = {"request_sha256": request_hash, "tasks": [asdict(task) for task in tasks]}
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != frozen:
            raise ValueError("existing rollout plan differs from this request")
        atomic_write_json(manifest_path, frozen)
        pending: Queue[EpisodeTask] = Queue()
        entries: dict[int, dict] = {}
        for task in tasks:
            complete = _completed(output, task, request_hash)
            if complete:
                entries[task.index] = complete
            else:
                pending.put(task)
        resumed = len(entries)
        stop = threading.Event()
        result_lock = threading.Lock()
        guests: set[str] = set()
        started = time.monotonic()

        def worker_loop(index: int) -> None:
            worker = None
            task = None
            phases = []
            began = time.monotonic()
            try:
                worker = runtime.open_worker(index)
                if worker.guest.isolation != IsolationBackend.KVM:
                    raise ValueError("study workers must use KVM guests")
                with result_lock:
                    if worker.guest.vm_id in guests:
                        raise ValueError("concurrent workers cannot share a guest")
                    guests.add(worker.guest.vm_id)
                while not stop.is_set():
                    try:
                        task = pending.get_nowait()
                    except Empty:
                        return
                    began = time.monotonic()
                    phases: list[dict] = []
                    runner = worker.runner_for(task)
                    if task.snapshot_sha256 is not None and runner.snapshot_sha256 != task.snapshot_sha256:
                        raise ValueError("runtime golden snapshot differs from the frozen episode plan")
                    if runner.turn_probe is None:
                        raise ValueError("production episodes require per-turn compromise/hang probes")
                    if runner.turns_per_side != int(request.get("turns_per_side", 8)):
                        raise ValueError("runtime changed the requested turn budget")
                    trajectories = _timed_runner(runner, phases).run(task.config, worker.guest)
                    entry = _commit(output, task, trajectories, request_hash, phases, time.monotonic() - began,
                                    worker.guest, runner.snapshot_sha256)
                    with result_lock:
                        entries[task.index] = entry
            except BaseException as exc:
                stop.set()
                try:
                    atomic_write_json(metadata / "attempts" / f"worker-{index}-{time.time_ns()}.json", {
                        "status": "failed", "worker": index, "request_sha256": request_hash,
                        "task_id": task.task_id if task else None, "elapsed_s": time.monotonic() - began,
                        "phases": phases, "error": f"{type(exc).__name__}: {exc}",
                    })
                except OSError:
                    pass
                raise
            finally:
                if worker is not None:
                    worker.close()

        def wave(selected: Sequence[EpisodeTask]) -> None:
            nonlocal pending
            pending = Queue()
            for task in selected:
                if task.index not in entries:
                    pending.put(task)
            count = min(workers, pending.qsize())
            if not count:
                return
            guests.clear()
            with ThreadPoolExecutor(max_workers=count, thread_name_prefix="ultron-episode") as executor:
                futures = [executor.submit(worker_loop, index) for index in range(count)]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    stop.set()
                    raise

        if not pending.empty():
            serving = serving_factory() if serving_factory else nullcontext()
            with serving:
                pilot_count = int(request.get("early_gate_episodes", 0))
                if pilot_count and pilot_count < len(tasks):
                    group_size = int(request.get("group_size", 6))
                    if pilot_count % group_size:
                        raise ValueError("early gate must use complete groups")
                    pilot = _pilot_tasks(tasks, pilot_count)
                    wave(pilot)
                    from .gates import validate_training_data
                    # Keep diagnostic copies outside the trace tree so recursive readers never double-count.
                    try:
                        with tempfile.TemporaryDirectory(prefix="ultron-pilot-", dir=output.parent) as directory:
                            _materialize(output, Path(directory), pilot)
                            report = validate_training_data(Path(directory), request=dict(request), expected_episodes=pilot_count)
                    except Exception as exc:
                        atomic_write_json(metadata / "early-gate.json", {"status": "failed", "episodes": pilot_count,
                                                                       "request_sha256": request_hash, "error": str(exc)})
                        raise
                    has_signal = all(report["roles"].get(role, {}).get("signal_groups", 0) > 0 for role in ("attacker", "defender"))
                    report.update(status="passed" if has_signal else "failed", request_sha256=request_hash)
                    atomic_write_json(metadata / "early-gate.json", report)
                    if not has_signal:
                        raise ValueError("early complete-group gate found no learning signal for a role")
                wave(tasks)
        files = _materialize(output, output, tasks)
        manifest = {
            "schema": "ultron.rollout.v1", "identity": request.get("identity", {}),
            "generation": request["generation"], "request_sha256": request_hash,
            "input_policy": request.get("policies", {}), "group_size": request.get("group_size", 6),
            "completed_episodes": len(entries), "completed_trajectories": len(entries) * 2,
            "resumed_episodes": resumed, "workers": workers,
            "elapsed_s": time.monotonic() - started,
            "files": files,
        }
        atomic_write_json(metadata / "rollout.json", manifest)
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--serving-manifest", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        request_bytes = args.request.read_bytes() if args.request else None
        request = json.loads(request_bytes) if request_bytes else {
            "generation": args.generation, "episodes": args.episodes, "seed": args.seed,
            "group_size": 6, "isolation": "kvm", "turns_per_side": 8, "output": str(args.output.resolve()),
        }
        if request["generation"] != args.generation or request["episodes"] != args.episodes:
            raise ValueError("CLI generation/episode count differs from the immutable request")
        if args.workers < 1:
            raise ValueError("workers must be positive")
        if "output" in request and Path(request["output"]).resolve() != args.output.resolve():
            raise ValueError("CLI output differs from immutable request output")
        runtime_request = dict(request)
        if args.serving_manifest:
            from ultron.shared_serving import managed_serving, runtime_bindings, validate_manifest
            manifest = validate_manifest(args.serving_manifest)
            runtime_request["serving_models"] = runtime_bindings(manifest, request)
        runtime = load_runtime(args.runtime, runtime_request)
        tasks = prepare(runtime, request)
        if args.serving_manifest:
            for task in tasks:
                if (task.config.attacker_ckpt not in runtime_request["serving_models"]["attacker"]
                        or task.config.defender_ckpt not in runtime_request["serving_models"]["defender"]):
                    raise ValueError("frozen episode plan selects a checkpoint absent from the serving manifest")
        if args.preflight_only:
            print(json.dumps({"status": "ready", "episodes": len(tasks), "runtime": args.runtime}))
            return 0
        serving_factory = (lambda: managed_serving(args.serving_manifest, request)) if args.serving_manifest else None
        def cancel(*_args):
            raise KeyboardInterrupt("rollout interrupted")
        previous = {number: signal.signal(number, cancel) for number in (signal.SIGTERM, signal.SIGINT)}
        try:
            result = run(runtime, tasks, request, args.output, workers=args.workers,
                         request_sha256=hashlib.sha256(request_bytes).hexdigest() if request_bytes else None,
                         serving_factory=serving_factory)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        print(json.dumps(result))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Rollout unavailable: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
