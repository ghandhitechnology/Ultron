"""Allocated H100 time and cost, independent of device utilization."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import yaml

DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "configs/h100.yaml"


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    gpu_model: str
    gpu_count: int
    gpu_memory_gb: int
    devices: tuple[str, ...]
    currency: str
    instance_hourly_rate: float | None
    rate_source: str | None
    rate_observed_at: str | None
    observed_devices: tuple[dict[str, str], ...] = ()


def load_hardware(path: Path = DEFAULT_PROFILE) -> HardwareProfile:
    raw = yaml.safe_load(path.read_text())
    hardware, billing = raw["hardware"], raw["billing"]
    count = hardware["gpu_count"]
    devices = tuple(str(item) for item in hardware["devices"])
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("gpu_count must be a positive integer")
    if len(set(devices)) != count:
        raise ValueError("devices must identify every allocated GPU exactly once")
    rate = billing.get("instance_hourly_rate")
    if rate is not None:
        rate = float(rate)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("instance_hourly_rate must be finite and nonnegative")
        if not billing.get("source") or not billing.get("observed_at"):
            raise ValueError("a billing rate requires source and observed_at")
    return HardwareProfile(
        str(raw["profile"]), str(hardware["gpu_model"]), count,
        int(hardware["gpu_memory_gb"]), devices, str(billing["currency"]),
        rate, billing.get("source"), billing.get("observed_at"),
    )


def verify_hardware(profile: HardwareProfile) -> list[dict[str, str]]:
    """Check selected physical devices before labeling an allocation as H100 time."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    devices = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            raise ValueError("unexpected nvidia-smi device inventory")
        devices.append(dict(zip(("index", "uuid", "name", "memory_mib"), fields)))
    selected = []
    for requested in profile.devices:
        matches = [device for device in devices if requested in (device["index"], device["uuid"])]
        if len(matches) != 1:
            raise ValueError(f"allocated GPU {requested} is missing")
        device = matches[0]
        if "H100" not in device["name"] or not 76000 <= float(device["memory_mib"]) <= 86000:
            raise ValueError(f"GPU {requested} is not an H100 80 GB: {device['name']} / {device['memory_mib']} MiB")
        selected.append(device)
    if len({device["uuid"] for device in selected}) != profile.gpu_count:
        raise ValueError("GPU indices and UUIDs must identify distinct physical devices")
    return selected


@contextmanager
def gpu_lease(profile: HardwareProfile, *, resource_id: str | None = None,
              directory: Path | None = None) -> Iterator[None]:
    """Reserve the selected devices across Ultron commands with separate run roots."""
    root = directory or Path.home() / ".cache" / "ultron" / "gpu-leases"
    root.mkdir(parents=True, exist_ok=True)
    # Billing labels may differ between runs; physical ownership must not.
    resource = os.uname().nodename
    devices = [item["uuid"] for item in profile.observed_devices] or list(profile.devices)
    locks = []
    try:
        for device in sorted(devices):
            key = hashlib.sha256(f"{resource}:{device}".encode()).hexdigest()
            handle = (root / f"{key}.lock").open("a+")
            locks.append(handle)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"GPU {device} already belongs to another Ultron allocation") from exc
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "resource": resource, "device": device}))
            handle.flush()
        yield
    finally:
        for handle in reversed(locks):
            handle.close()


class AllocationLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def record(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(json.dumps({"version": 1, "time": time.time(), **event}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def begin(self, profile: HardwareProfile, *, run_id: str, phase: str,
              scope_key: str, allocation_id: str | None = None,
              resource_id: str | None = None) -> str:
        identifier = allocation_id or uuid4().hex
        self.record({"event": "allocation_start", "allocation_id": identifier,
                     "run_id": run_id, "phase": phase, "scope_key": scope_key,
                     "resource_id": resource_id or os.uname().nodename,
                     "physical_resource_id": os.uname().nodename,
                     "profile": asdict(profile)})
        return identifier

    def end(self, allocation_id: str, *, status: int, elapsed_seconds: float | None = None) -> None:
        self.record({"event": "allocation_end", "allocation_id": allocation_id,
                     "status": status, "elapsed_seconds": elapsed_seconds})

    @contextmanager
    def interval(self, profile: HardwareProfile, *, run_id: str, phase: str,
                 scope_key: str = "", allocation_id: str | None = None,
                 resource_id: str | None = None) -> Iterator[str]:
        identifier = self.begin(profile, run_id=run_id, phase=phase, scope_key=scope_key,
                                allocation_id=allocation_id, resource_id=resource_id)
        started = time.monotonic()
        status = 1
        try:
            yield identifier
            status = 0
        finally:
            self.end(identifier, status=status, elapsed_seconds=time.monotonic() - started)


def summarize(path: Path, *, now: float | None = None, extra_cost: float = 0) -> dict[str, Any]:
    if not math.isfinite(extra_cost) or extra_cost < 0:
        raise ValueError("extra cost must be finite and nonnegative")
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    for event in events:
        kind = event.get("event")
        if kind not in {"allocation_start", "allocation_end"}:
            continue
        target = starts if kind == "allocation_start" else ends
        identifier = event["allocation_id"]
        if identifier in target:
            raise ValueError(f"duplicate {kind}: {identifier}")
        target[identifier] = event
    if set(ends) - set(starts):
        raise ValueError("allocation end has no corresponding start")
    instant = time.time() if now is None else now
    intervals, device_intervals, scopes, models, currencies = [], {}, set(), set(), set()
    declared_models, hardware_observed = set(), True
    gpu_seconds, compute_cost, unknown_rate, incomplete = 0.0, 0.0, False, []
    failed_attempts = 0
    for identifier, start in starts.items():
        end = ends.get(identifier)
        stop = end["time"] if end else instant
        duration = (end.get("elapsed_seconds") if end else None)
        duration = stop - start["time"] if duration is None else duration
        if not math.isfinite(duration) or duration < 0 or stop < start["time"]:
            raise ValueError(f"invalid allocation duration: {identifier}")
        if end is None:
            incomplete.append(identifier)
        elif end.get("status") != 0:
            failed_attempts += 1
        profile = start["profile"]
        observed = profile.get("observed_devices", [])
        hardware_observed = hardware_observed and bool(observed)
        declared_models.add(profile["gpu_model"])
        gpu_seconds += duration * profile["gpu_count"]
        rate = profile["instance_hourly_rate"]
        if rate is None:
            unknown_rate = True
        else:
            compute_cost += duration / 3600 * rate
        if observed:
            models.update(item["name"] for item in observed)
        currencies.add(profile["currency"])
        scopes.add(start["scope_key"])
        intervals.append((start["time"], stop))
        physical_devices = [item["uuid"] for item in observed] or profile["devices"]
        for device in physical_devices:
            key = (start.get("physical_resource_id", start["resource_id"]), device)
            device_intervals.setdefault(key, []).append((start["time"], stop))
    for key, spans in device_intervals.items():
        previous_stop = float("-inf")
        for begin, stop in sorted(spans):
            if begin < previous_stop:
                raise ValueError(f"overlapping allocations for {key}; record phases within one allocation")
            previous_stop = stop
    if len(currencies) > 1:
        raise ValueError("mixed billing currencies require a normalized ledger")
    phase_starts, phase_seconds = {}, {}
    for event in events:
        if event.get("event") == "phase_start":
            phase_starts[event["attempt_id"]] = event
        elif event.get("event") == "phase_end" and event["attempt_id"] in phase_starts:
            beginning = phase_starts.pop(event["attempt_id"])
            duration = event["time"] - beginning["time"]
            if duration < 0:
                raise ValueError("phase clock moved backwards")
            name = beginning["phase"]
            phase_seconds[name] = phase_seconds.get(name, 0.0) + duration
    wall = max(stop for _, stop in intervals) - min(begin for begin, _ in intervals) if intervals else 0.0
    return {"allocation_count": len(starts), "gpu_hours": gpu_seconds / 3600,
            "elapsed_hours": wall / 3600, "gpu_models": sorted(models),
            "declared_gpu_models": sorted(declared_models),
            "hardware_observed": bool(starts) and hardware_observed,
            "currency": next(iter(currencies), None),
            "compute_cost": None if unknown_rate else compute_cost,
            "extra_cost": extra_cost,
            "total_cost": None if unknown_rate else compute_cost + extra_cost,
            "scope_keys": sorted(scopes), "open_allocations": incomplete,
            "failed_attempts": failed_attempts,
            "phase_seconds": phase_seconds, "open_phases": sorted(phase_starts)}


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    problems = []
    if not baseline["hardware_observed"] or not candidate["hardware_observed"]:
        problems.append("GPU hardware must be observed for both allocations")
    if baseline["gpu_models"] != candidate["gpu_models"]:
        problems.append("GPU models differ")
    if baseline["scope_keys"] != candidate["scope_keys"] or not baseline["scope_keys"] or "" in baseline["scope_keys"]:
        problems.append("workload scope keys must match and be nonempty")
    if baseline["currency"] != candidate["currency"]:
        problems.append("billing currencies differ")
    if baseline["open_allocations"] or candidate["open_allocations"]:
        problems.append("allocation intervals are still open")
    ratios = {}
    for name in ("gpu_hours", "elapsed_hours", "total_cost"):
        left, right = baseline[name], candidate[name]
        ratios[name] = right / left if left is not None and left > 0 and right is not None else None
    if any(value is None for value in ratios.values()):
        problems.append("positive baseline measurements and complete billing rates are required")
    return {"ratios": ratios, "problems": problems,
            "meets_half_resource_target": not problems and all(value <= 0.5 for value in ratios.values()),
            "scope_verification": "Compare required study artifacts separately; scope keys label workloads."}


def run_command(ledger: AllocationLedger, profile: HardwareProfile, command: list[str],
                *, run_id: str, phase: str, scope_key: str, resource_id: str | None) -> int:
    with gpu_lease(profile, resource_id=resource_id):
        return _run_command(ledger, profile, command, run_id=run_id, phase=phase,
                            scope_key=scope_key, resource_id=resource_id)


def _run_command(ledger: AllocationLedger, profile: HardwareProfile, command: list[str],
                 *, run_id: str, phase: str, scope_key: str, resource_id: str | None) -> int:
    identifier = ledger.begin(profile, run_id=run_id, phase=phase, scope_key=scope_key, resource_id=resource_id)
    started = time.monotonic()
    status = 1
    process = None
    old_handlers = {}
    def forward(signum: int, _frame: Any) -> None:
        if process is not None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
    try:
        process = subprocess.Popen(command, start_new_session=True,
                                   env={**os.environ, "ULTRON_ALLOCATION_ID": identifier})
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, forward)
        result = process.wait()
        status = result if result >= 0 else 128 - result
        return status
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        ledger.end(identifier, status=status, elapsed_seconds=time.monotonic() - started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "begin"):
        item = sub.add_parser(name)
        item.add_argument("--ledger", type=Path, required=True)
        item.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
        item.add_argument("--run-id", required=True)
        item.add_argument("--phase", default="allocation")
        item.add_argument("--scope-key", default="")
        item.add_argument("--resource-id")
        item.add_argument("--verify-hardware", action="store_true")
        if name == "run":
            item.add_argument("argv", nargs=argparse.REMAINDER)
    end = sub.add_parser("end")
    end.add_argument("--ledger", type=Path, required=True)
    end.add_argument("--allocation-id", required=True)
    end.add_argument("--status", type=int, default=0)
    phase = sub.add_parser("phase")
    phase.add_argument("--ledger", type=Path, required=True)
    phase.add_argument("--phase", required=True)
    phase.add_argument("--state", choices=("start", "end"), required=True)
    phase.add_argument("--attempt-id", required=True)
    phase.add_argument("--allocation-id", default="")
    phase.add_argument("--status", type=int)
    report = sub.add_parser("summary")
    report.add_argument("ledger", type=Path)
    report.add_argument("--extra-cost", type=float, default=0)
    comparison = sub.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--baseline-extra-cost", type=float, default=0)
    comparison.add_argument("--candidate-extra-cost", type=float, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "summary":
            print(json.dumps(summarize(args.ledger, extra_cost=args.extra_cost), indent=2))
        elif args.command == "compare":
            print(json.dumps(compare(summarize(args.baseline, extra_cost=args.baseline_extra_cost),
                                     summarize(args.candidate, extra_cost=args.candidate_extra_cost)), indent=2))
        elif args.command == "end":
            AllocationLedger(args.ledger).end(args.allocation_id, status=args.status)
        elif args.command == "phase":
            AllocationLedger(args.ledger).record({
                "event": f"phase_{args.state}", "phase": args.phase,
                "attempt_id": args.attempt_id, "allocation_id": args.allocation_id,
                "status": args.status,
            })
        else:
            profile = load_hardware(args.profile)
            if args.verify_hardware:
                observed = verify_hardware(profile)
                profile = replace(profile, observed_devices=tuple(observed),
                                  gpu_model=" / ".join(sorted({item["name"] for item in observed})))
            ledger = AllocationLedger(args.ledger)
            kwargs = dict(run_id=args.run_id, phase=args.phase, scope_key=args.scope_key, resource_id=args.resource_id)
            if args.command == "begin":
                print(ledger.begin(profile, **kwargs))
            else:
                command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
                if not command:
                    parser.error("run requires a command after --")
                return run_command(ledger, profile, command, **kwargs)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
