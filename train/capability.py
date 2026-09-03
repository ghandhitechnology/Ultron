"""ultron.train.capability — which family packs this host can train.

Reads CPU, RAM, and nvidia-smi. Compares against the README family table.
Does not run kvm-ok, docker, or libvirt. Those stay in the bootstrap host gates.

Stdlib only so `python3 train/capability.py` works before `pip install -e .`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class CapabilityError(ValueError):
    """Unknown family name or a broken inventory read."""


class Fit(str, Enum):
    RUN = "run"
    CAUTION = "caution"
    SKIP = "skip"


@dataclass(frozen=True)
class FamilyRequirement:
    name: str
    base_model: str
    min_physical_cores: int
    min_ram_gb: int
    min_gpus: int
    min_vram_gb: int
    caution_vram_gb: int | None = None
    notes: str = ""


# Locked to the README "Server requirements" table. Keep names aligned with FamilyName.
REQUIREMENTS: tuple[FamilyRequirement, ...] = (
    FamilyRequirement(
        name="qwen-4b",
        base_model="Qwen/Qwen3.5-4B",
        min_physical_cores=32,
        min_ram_gb=128,
        min_gpus=2,
        min_vram_gb=80,
    ),
    FamilyRequirement(
        name="qwen-8b",
        base_model="Qwen/Qwen3-8B",
        min_physical_cores=32,
        min_ram_gb=192,
        min_gpus=2,
        min_vram_gb=80,
        caution_vram_gb=48,
        notes="2x L40S 48 GB only if you cut guest concurrency",
    ),
    FamilyRequirement(
        name="gemma",
        base_model="google/gemma-4-12B-it",
        min_physical_cores=32,
        min_ram_gb=192,
        min_gpus=2,
        min_vram_gb=80,
        caution_vram_gb=48,
        notes="skip 24 GB cards; 48 GB needs a smaller guest pool",
    ),
)


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    name: str
    memory_mib: int

    @property
    def memory_gb(self) -> float:
        return self.memory_mib / 1024.0


@dataclass(frozen=True)
class HostSnapshot:
    arch: str
    physical_cores: int | None
    logical_cpus: int | None
    ram_gb: float | None
    gpus: tuple[GpuSnapshot, ...]
    nvidia_error: str | None = None


@dataclass(frozen=True)
class FamilyVerdict:
    requirement: FamilyRequirement
    fit: Fit
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityReport:
    host: HostSnapshot
    verdicts: tuple[FamilyVerdict, ...]
    selected: str | None = None

    @property
    def runnable(self) -> tuple[str, ...]:
        return tuple(item.requirement.name for item in self.verdicts if item.fit is not Fit.SKIP)

    def selected_verdict(self) -> FamilyVerdict | None:
        if self.selected is None:
            return None
        for item in self.verdicts:
            if item.requirement.name == self.selected:
                return item
        return None


def parse_family_name(raw: str) -> str:
    token = raw.strip()
    names = [item.name for item in REQUIREMENTS]
    if token in names:
        return token
    expected = f"{', '.join(names[:-1])}, or {names[-1]}"
    raise CapabilityError(f"unknown model family {token!r}; expected {expected}")


def parse_physical_cores(cpuinfo: str) -> int | None:
    pairs: set[tuple[str, str]] = set()
    physical: str | None = None
    core: str | None = None
    saw_pair = False
    processors = 0
    for raw in (*cpuinfo.splitlines(), ""):
        line = raw.strip()
        if not line:
            if physical is not None and core is not None:
                pairs.add((physical, core))
                saw_pair = True
            physical = None
            core = None
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "processor":
            processors += 1
        elif key == "physical id":
            physical = value
        elif key == "core id":
            core = value
    if saw_pair:
        return len(pairs)
    return processors or None


def parse_ram_gb(meminfo: str) -> float | None:
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) < 2:
                return None
            return float(parts[1]) / 1024.0 / 1024.0
    return None


def parse_nvidia_smi(text: str) -> tuple[GpuSnapshot, ...]:
    gpus: list[GpuSnapshot] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            memory = int(float(parts[2]))
            index = int(parts[0])
        except ValueError:
            continue
        gpus.append(GpuSnapshot(index=index, name=parts[1], memory_mib=memory))
    return tuple(gpus)


def read_host(
    *,
    cpuinfo: str | None = None,
    meminfo: str | None = None,
    nvidia_csv: str | None = None,
    nvidia_error: str | None = None,
    arch: str | None = None,
    logical_cpus: int | None = None,
) -> HostSnapshot:
    cpu_text = _text_or_file(cpuinfo, Path("/proc/cpuinfo"))
    mem_text = _text_or_file(meminfo, Path("/proc/meminfo"))
    csv = nvidia_csv
    error = nvidia_error
    if csv is None and error is None:
        csv, error = _run_nvidia_smi()
    gpus = parse_nvidia_smi(csv) if csv else ()
    if csv is not None and not gpus and error is None:
        error = "nvidia-smi returned no GPUs"
    return HostSnapshot(
        arch=arch if arch is not None else platform.machine(),
        physical_cores=parse_physical_cores(cpu_text) if cpu_text else None,
        logical_cpus=os.cpu_count() if logical_cpus is None else logical_cpus,
        ram_gb=parse_ram_gb(mem_text) if mem_text else None,
        gpus=gpus,
        nvidia_error=error,
    )


def assess(host: HostSnapshot, *, family: str | None = None) -> CapabilityReport:
    selected = parse_family_name(family) if family is not None and family.strip() else None
    return CapabilityReport(
        host=host,
        verdicts=tuple(_verdict(host, item) for item in REQUIREMENTS),
        selected=selected,
    )


def report_ok(report: CapabilityReport) -> bool:
    if report.selected is None:
        return any(item.fit is not Fit.SKIP for item in report.verdicts)
    verdict = report.selected_verdict()
    return verdict is not None and verdict.fit is not Fit.SKIP


def render(report: CapabilityReport) -> str:
    host = report.host
    cores = host.physical_cores if host.physical_cores is not None else host.logical_cpus
    core_note = "physical" if host.physical_cores is not None else "logical"
    ram = f"{host.ram_gb:.1f} GiB" if host.ram_gb is not None else "RAM unknown"
    core_text = f"{cores} {core_note} cores" if cores is not None else "CPU unknown"
    lines = [
        f"Host: {host.arch}  {core_text}  {ram}  {len(host.gpus)} GPU(s)",
    ]
    if host.nvidia_error:
        lines.append(f"  nvidia-smi: {host.nvidia_error}")
    for gpu in host.gpus:
        lines.append(f"  GPU {gpu.index}  {gpu.name}  {gpu.memory_gb:.1f} GiB")
    lines.append("")
    width = max(len(item.requirement.name) for item in report.verdicts)
    model_width = max(len(item.requirement.base_model) for item in report.verdicts)
    for item in report.verdicts:
        mark = " " if report.selected is None or item.requirement.name != report.selected else "*"
        reason = f"  {'; '.join(item.reasons)}" if item.reasons else ""
        lines.append(
            f"{mark}{item.requirement.name:<{width}}  "
            f"{item.requirement.base_model:<{model_width}}  "
            f"{_fit_label(item.fit)}{reason}"
        )
    lines.append("")
    if report.runnable:
        lines.append("Runnable: " + ", ".join(report.runnable))
    else:
        lines.append("No supported family fits this host.")
    if report.selected is not None:
        verdict = report.selected_verdict()
        if verdict is None:
            lines.append(f"Selected {report.selected} is not in the requirement table.")
        elif verdict.fit is Fit.SKIP:
            lines.append(f"Selected {report.selected} cannot run on this host.")
        elif verdict.fit is Fit.CAUTION:
            lines.append(f"Selected {report.selected} can run with caveats.")
        elif verdict.fit is Fit.RUN:
            lines.append(f"Selected {report.selected} fits this host.")
        else:
            _assert_never(verdict.fit)
    return "\n".join(lines) + "\n"


def to_payload(report: CapabilityReport) -> dict[str, Any]:
    return {
        "host": {
            "arch": report.host.arch,
            "physical_cores": report.host.physical_cores,
            "logical_cpus": report.host.logical_cpus,
            "ram_gb": report.host.ram_gb,
            "gpus": [asdict(gpu) for gpu in report.host.gpus],
            "nvidia_error": report.host.nvidia_error,
        },
        "selected": report.selected,
        "runnable": list(report.runnable),
        "ok": report_ok(report),
        "verdicts": [
            {
                "family": item.requirement.name,
                "base_model": item.requirement.base_model,
                "fit": item.fit.value,
                "reasons": list(item.reasons),
            }
            for item in report.verdicts
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ultron.train.capability")
    parser.add_argument("--family", help="Check one family instead of reporting every pack.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        selected = parse_family_name(args.family) if args.family else None
    except CapabilityError as exc:
        print(exc, file=sys.stderr)
        return 2
    report = assess(read_host(), family=selected)
    if args.json:
        print(json.dumps(to_payload(report), indent=2))
    else:
        sys.stdout.write(render(report))
    return 0 if report_ok(report) else 1


def _verdict(host: HostSnapshot, req: FamilyRequirement) -> FamilyVerdict:
    reasons: list[str] = []
    fatal = False
    caution = False
    arch = host.arch.lower()
    if arch not in {"x86_64", "amd64"}:
        reasons.append(f"arch {host.arch} is not x86-64")
        fatal = True
    cores = host.physical_cores if host.physical_cores is not None else host.logical_cpus
    core_label = "physical cores" if host.physical_cores is not None else "logical CPUs"
    if cores is None:
        reasons.append("CPU count unknown")
        fatal = True
    elif cores < req.min_physical_cores:
        reasons.append(f"{cores} {core_label} < {req.min_physical_cores}")
        fatal = True
    if host.ram_gb is None:
        reasons.append("RAM unknown")
        fatal = True
    elif host.ram_gb < req.min_ram_gb * 0.95:
        reasons.append(f"{host.ram_gb:.1f} GiB RAM < {req.min_ram_gb} GiB")
        fatal = True
    if host.nvidia_error:
        reasons.append(host.nvidia_error)
        fatal = True
    elif len(host.gpus) < req.min_gpus:
        reasons.append(f"{len(host.gpus)} GPUs < {req.min_gpus}")
        fatal = True
    else:
        ranked = sorted(host.gpus, key=lambda gpu: gpu.memory_mib, reverse=True)
        weakest = ranked[req.min_gpus - 1]
        if _vram_ok(weakest.memory_mib, req.min_vram_gb):
            pass
        elif req.caution_vram_gb is not None and _vram_ok(weakest.memory_mib, req.caution_vram_gb):
            reasons.append(
                f"weakest of {req.min_gpus} GPUs is {weakest.memory_gb:.1f} GiB; "
                f"{req.min_vram_gb} GiB recommended"
            )
            if req.notes:
                reasons.append(req.notes)
            caution = True
        else:
            reasons.append(
                f"weakest of {req.min_gpus} GPUs is {weakest.memory_gb:.1f} GiB < {req.min_vram_gb} GiB"
            )
            fatal = True
    if fatal:
        fit = Fit.SKIP
    elif caution:
        fit = Fit.CAUTION
    else:
        fit = Fit.RUN
    return FamilyVerdict(requirement=req, fit=fit, reasons=tuple(reasons))


def _vram_ok(memory_mib: int, gb: int) -> bool:
    return memory_mib >= int(gb * 1024 * 0.9)


def _fit_label(fit: Fit) -> str:
    match fit:
        case Fit.RUN:
            return "RUN"
        case Fit.CAUTION:
            return "CAUTION"
        case Fit.SKIP:
            return "SKIP"
        case _:
            _assert_never(fit)


def _text_or_file(override: str | None, path: Path) -> str | None:
    if override is not None:
        return override
    try:
        return path.read_text()
    except OSError:
        return None


def _run_nvidia_smi() -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return None, "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return None, "nvidia-smi timed out"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "nvidia-smi failed").strip()
        return None, detail or "nvidia-smi failed"
    return result.stdout, None


def _assert_never(value: object) -> None:
    raise AssertionError(f"unhandled {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
