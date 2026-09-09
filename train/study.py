"""Locked final-study protocol: matrix, identities, and execution gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ultron.env.backend import IsolationBackend
from ultron.train.family import FamilyName, resolve

PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "configs" / "study" / "final.yaml"
PROTOCOL_VERSION = "final-study.v1"


class StudyError(ValueError):
    """Boundary failure for the locked study protocol."""


class ExecutionGateError(StudyError):
    """Paid or study execution is missing a required gate."""


class Method(str, Enum):
    FIXED_SINGLE = "fixed-single"
    FIXED_DIVERSE = "fixed-diverse"
    ADAPTIVE_LATEST = "adaptive-latest"
    ADAPTIVE_HISTORY = "adaptive-history"
    UNTRAINED = "untrained"


class Ablation(str, Enum):
    NONE = "none"
    SHAPING_OFF = "shaping-off"
    ADDED_DPO = "added-dpo"


class ResultSource(str, Enum):
    MEASURED = "measured"
    MOCK = "mock"


class EvalSplit(str, Enum):
    UNSEEN_INSTANCE = "unseen_instance"
    UNSEEN_COMBINATION = "unseen_combination"
    HELD_OUT_MECHANISM = "held_out_mechanism"


class TaskKind(str, Enum):
    ISOLATED = "isolated"
    COMBINATION = "combination"


@dataclass(frozen=True)
class AdaptiveMix:
    latest: float
    difficult: float
    uniform: float


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    mechanisms: tuple[str, ...]
    kind: TaskKind
    split: str
    sealed: bool
    development: bool


@dataclass(frozen=True)
class StudyCell:
    model: FamilyName
    method: Method
    ablation: Ablation
    control: bool

    @property
    def variant(self) -> str:
        if self.method is Method.UNTRAINED:
            return self.method.value
        if self.ablation is Ablation.NONE:
            return self.method.value
        return f"{self.method.value}+{self.ablation.value}"

    @property
    def shaping(self) -> bool:
        return self.ablation is not Ablation.SHAPING_OFF and self.method is not Method.UNTRAINED

    @property
    def dpo(self) -> bool:
        return self.ablation is Ablation.ADDED_DPO

    @property
    def primary(self) -> bool:
        return (
            self.model is FamilyName.QWEN_8B
            and self.method is Method.ADAPTIVE_HISTORY
            and self.ablation is Ablation.NONE
        )


@dataclass(frozen=True)
class RunIdentity:
    protocol_version: str
    model: str
    variant: str
    seed: int
    run_id: str

    def namespace(self) -> Path:
        return (
            Path("data/study")
            / self.model
            / self.variant
            / f"seed{self.seed}"
            / self.run_id
        )

    def episode_id(self, index: int) -> str:
        return (
            f"{self.protocol_version}:{self.model}:{self.variant}:"
            f"seed{self.seed}:{self.run_id}:ep{index}"
        )

    def checkpoint_id(self, role: str, step: int) -> str:
        return (
            f"{self.protocol_version}:{self.model}:{self.variant}:"
            f"seed{self.seed}:{self.run_id}:{role}:step{step}"
        )


@dataclass(frozen=True)
class ScoreIdentity:
    run: RunIdentity
    split: str
    role: str
    source: ResultSource

    def key(self) -> str:
        return "/".join(
            (
                self.run.protocol_version,
                self.run.model,
                self.run.variant,
                f"seed{self.run.seed}",
                self.run.run_id,
                self.split,
                self.role,
                self.source.value,
            )
        )


@dataclass(frozen=True)
class StudyProtocol:
    protocol_version: str
    hypothesis: str
    target_delta_pp: int
    primary_method: str
    shaping_cap: float
    terminal_win: float
    role_aware_baseline: bool
    enable_thinking: bool
    default_family: FamilyName
    secondary_family: FamilyName
    isolation: IsolationBackend
    prep_turns: int | None
    prep_turns_calibrated: bool
    curriculum_frozen: bool
    final_test_sealed: bool
    development_separate: bool
    spending_cap_required: bool
    adaptive_mix: AdaptiveMix
    splits: tuple[EvalSplit, ...]
    cells: tuple[StudyCell, ...]
    tasks: tuple[TaskSpec, ...]
    gates: tuple[str, ...]

    def cells_for(self, model: FamilyName) -> tuple[StudyCell, ...]:
        return tuple(cell for cell in self.cells if cell.model is model)


def load_protocol(
    path: Path | None = None, *, repo_root: Path | None = None
) -> StudyProtocol:
    source = path if path is not None else _protocol_path(repo_root)
    if not source.is_file():
        raise StudyError(f"missing study protocol {source}")
    payload = yaml.safe_load(source.read_text())
    if not isinstance(payload, Mapping):
        raise StudyError(f"{source} is not a mapping")
    return _parse_protocol(payload, source)


def study_matrix(protocol: StudyProtocol | None = None) -> tuple[StudyCell, ...]:
    return (protocol or load_protocol()).cells


def assert_paid_execution_allowed(
    spending_cap_usd: float | None,
    protocol: StudyProtocol | None = None,
    *,
    isolation: IsolationBackend | None = None,
) -> None:
    locked = protocol or load_protocol()
    if locked.spending_cap_required and spending_cap_usd is None:
        raise ExecutionGateError("paid execution requires an approved spending cap")
    if spending_cap_usd is not None and spending_cap_usd <= 0:
        raise ExecutionGateError("spending cap must be positive")
    if isolation is not None and isolation is not locked.isolation:
        raise ExecutionGateError(
            f"final study requires {locked.isolation.value} guests, not {isolation.value}"
        )


def assert_comparable(left: ScoreIdentity, right: ScoreIdentity) -> None:
    if left.source is not right.source:
        raise StudyError("mock results cannot be compared with measured results")
    if left.source is ResultSource.MOCK:
        raise StudyError("mock results cannot be used as measured scores")


def check_configs(*, repo_root: Path | None = None) -> list[str]:
    """Return human-readable failures. Empty means the lock holds."""
    root = Path(__file__).resolve().parent.parent if repo_root is None else repo_root
    protocol = load_protocol(repo_root=root)
    failures: list[str] = []
    pack = resolve(environ={})
    if pack.name is not protocol.default_family:
        failures.append(
            f"default family is {pack.name.value}, expected {protocol.default_family.value}"
        )
    eight = resolve(FamilyName.QWEN_8B, repo_root=root, environ={})
    thinking = (eight.chat_template_kwargs or {}).get("enable_thinking")
    if thinking is not True:
        failures.append("qwen-8b thinking is not enabled")
    for name in FamilyName:
        grpo = _read_yaml(_grpo_path(root, name))
        enabled = grpo.get("algorithm", {}).get("role_aware_baseline", {}).get("enabled")
        if enabled is not False:
            failures.append(f"{name.value} GRPO still enables the role-aware baseline")
    if protocol.protocol_version != PROTOCOL_VERSION:
        failures.append(
            f"protocol_version is {protocol.protocol_version}, expected {PROTOCOL_VERSION}"
        )
    if protocol.role_aware_baseline:
        failures.append("protocol still enables the role-aware baseline")
    if protocol.shaping_cap != 0.1:
        failures.append(f"shaping_cap is {protocol.shaping_cap}, expected 0.1")
    mix = protocol.adaptive_mix
    if (mix.latest, mix.difficult, mix.uniform) != (0.5, 0.25, 0.25):
        failures.append(f"adaptive mix is {mix}, expected 0.5/0.25/0.25")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ultron.train.study")
    parser.add_argument("command", choices=["matrix", "check"])
    args = parser.parse_args(argv)
    try:
        protocol = load_protocol()
    except StudyError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.command == "matrix":
        rows = [
            {
                "model": cell.model.value,
                "variant": cell.variant,
                "method": cell.method.value,
                "ablation": cell.ablation.value,
                "control": cell.control,
                "shaping": cell.shaping,
                "dpo": cell.dpo,
                "primary": cell.primary,
            }
            for cell in protocol.cells
        ]
        print(json.dumps(rows, indent=2))
        return 0
    failures = check_configs()
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"{protocol.protocol_version} lock holds")
    return 0


def _protocol_path(repo_root: Path | None) -> Path:
    if repo_root is None:
        return PROTOCOL_PATH
    return Path(repo_root) / "configs" / "study" / "final.yaml"


def _parse_protocol(raw: Mapping[str, Any], source: Path) -> StudyProtocol:
    try:
        mix_raw = _mapping(raw.get("adaptive_mix"), "adaptive_mix")
        mix = AdaptiveMix(
            latest=_float(mix_raw.get("latest"), "latest"),
            difficult=_float(mix_raw.get("difficult"), "difficult"),
            uniform=_float(mix_raw.get("uniform"), "uniform"),
        )
        if abs(mix.latest + mix.difficult + mix.uniform - 1.0) > 1e-9:
            raise StudyError("adaptive_mix weights must sum to 1")
        eval_raw = _mapping(raw.get("eval"), "eval")
        splits = tuple(EvalSplit(item) for item in _str_list(eval_raw.get("splits"), "splits"))
        cells = _parse_cells(_mapping(raw.get("models"), "models"))
        tasks = _parse_tasks(_mapping(raw.get("tasks"), "tasks"), raw)
        prep = raw.get("prep_turns")
        prep_turns = None if prep is None else _int(prep, "prep_turns")
        protocol = StudyProtocol(
            protocol_version=_str(raw.get("protocol_version"), "protocol_version"),
            hypothesis=_str(raw.get("hypothesis"), "hypothesis").strip(),
            target_delta_pp=_int(raw.get("target_delta_pp"), "target_delta_pp"),
            primary_method=_str(raw.get("primary_method"), "primary_method"),
            shaping_cap=_float(raw.get("shaping_cap"), "shaping_cap"),
            terminal_win=_float(raw.get("terminal_win"), "terminal_win"),
            role_aware_baseline=_bool(raw.get("role_aware_baseline"), "role_aware_baseline"),
            enable_thinking=_bool(
                _mapping(raw.get("thinking"), "thinking").get("enable_thinking"),
                "enable_thinking",
            ),
            default_family=FamilyName(_str(raw.get("default_family"), "default_family")),
            secondary_family=FamilyName(_str(raw.get("secondary_family"), "secondary_family")),
            isolation=IsolationBackend(_str(raw.get("isolation"), "isolation")),
            prep_turns=prep_turns,
            prep_turns_calibrated=_bool(
                raw.get("prep_turns_calibrated"), "prep_turns_calibrated"
            ),
            curriculum_frozen=_bool(raw.get("curriculum_frozen"), "curriculum_frozen"),
            final_test_sealed=_bool(raw.get("final_test_sealed"), "final_test_sealed"),
            development_separate=_bool(raw.get("development_separate"), "development_separate"),
            spending_cap_required=_bool(raw.get("spending_cap_required"), "spending_cap_required"),
            adaptive_mix=mix,
            splits=splits,
            cells=cells,
            tasks=tasks,
            gates=tuple(_str_list(raw.get("gates"), "gates")),
        )
    except (TypeError, ValueError) as exc:
        raise StudyError(f"{source}: {exc}") from exc
    if protocol.default_family is not FamilyName.QWEN_8B:
        raise StudyError("default_family must be qwen-8b")
    if protocol.secondary_family is not FamilyName.QWEN_4B:
        raise StudyError("secondary_family must be qwen-4b")
    return protocol


def _parse_cells(models: Mapping[str, Any]) -> tuple[StudyCell, ...]:
    cells: list[StudyCell] = []
    for model_name, spec in models.items():
        model = FamilyName(model_name)
        body = _mapping(spec, model_name)
        methods = tuple(Method(item) for item in _str_list(body.get("methods"), "methods"))
        controls = tuple(Method(item) for item in _str_list(body.get("control"), "control"))
        ablation_spec = _mapping(body.get("ablations"), "ablations")
        ablation_methods = tuple(
            Method(item) for item in _str_list(ablation_spec.get("methods"), "ablation methods")
        )
        ablation_kinds = tuple(
            Ablation(item) for item in _str_list(ablation_spec.get("kinds"), "ablation kinds")
        )
        for method in methods:
            cells.append(StudyCell(model, method, Ablation.NONE, control=False))
        for method in controls:
            cells.append(StudyCell(model, method, Ablation.NONE, control=True))
        for method in ablation_methods:
            for kind in ablation_kinds:
                if kind is Ablation.NONE:
                    raise StudyError("ablation kinds cannot include none")
                cells.append(StudyCell(model, method, kind, control=False))
    return tuple(cells)


def _parse_tasks(raw: Mapping[str, Any], protocol: Mapping[str, Any]) -> tuple[TaskSpec, ...]:
    sealed = _bool(protocol.get("final_test_sealed"), "final_test_sealed")
    tasks: list[TaskSpec] = []
    for item in _list(raw.get("train"), "train"):
        tasks.append(_task(item, split="train", sealed=False, development=False))
    for item in _list(raw.get("development"), "development"):
        tasks.append(_task(item, split="development", sealed=False, development=True))
    final = _mapping(raw.get("final_test"), "final_test")
    for split_name in ("unseen_instance", "unseen_combination", "held_out_mechanism"):
        for item in _list(final.get(split_name), split_name):
            tasks.append(_task(item, split=split_name, sealed=sealed, development=False))
    return tuple(tasks)


def _task(raw: Any, *, split: str, sealed: bool, development: bool) -> TaskSpec:
    data = _mapping(raw, "task")
    return TaskSpec(
        task_id=_str(data.get("id"), "id"),
        mechanisms=tuple(_str_list(data.get("mechanisms"), "mechanisms")),
        kind=TaskKind(_str(data.get("kind"), "kind")),
        split=split,
        sealed=sealed,
        development=development,
    )


def _grpo_path(root: Path, name: FamilyName) -> Path:
    if name is FamilyName.QWEN_4B:
        return root / "configs" / "train_grpo.yaml"
    return root / "configs" / "families" / name.value / "train_grpo.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise StudyError(f"{path} is not a mapping")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise StudyError(f"{name} must be an array")
    return value


def _str_list(value: Any, name: str) -> list[str]:
    return [_str(item, f"{name} item") for item in _list(value, name)]


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise StudyError(f"{name} must be a string")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise StudyError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise StudyError(f"{name} must be an integer")
    return value


def _float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise StudyError(f"{name} must be a number")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
