from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Mapping, TypeAlias

import yaml

from ultron import __version__
from ultron.cli.model import JobMeta
from ultron.env.backend import IsolationBackend


class ActionId(str, Enum):
    DEMO = "demo"
    GENERATION = "generation"
    ROLLOUT = "rollout"
    GRPO = "grpo"
    DPO = "dpo"
    SERVE_ATTACKER = "serve_attacker"
    SERVE_DEFENDER = "serve_defender"
    REVIEW = "review"
    ARCHIVE = "archive"
    ARCHIVE_LIST = "archive_list"
    EVAL = "eval"
    PFSP = "pfsp"
    BANDPASS = "bandpass"
    TESTS = "tests"


class ActionGroup(str, Enum):
    GYM = "gym"
    PIPELINE = "pipeline"
    TRAIN = "train"
    SERVE = "serve"
    RESULTS = "results"
    VERIFY = "verify"


class FieldKind(str, Enum):
    INT = "int"
    TEXT = "text"
    CHOICE = "choice"
    PATH = "path"
    FLAG = "flag"


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: FieldKind
    default: str
    choices: tuple[str, ...] = ()
    required: bool = True
    help: str = ""


@dataclass(frozen=True)
class ActionSpec:
    id: ActionId
    title: str
    group: ActionGroup
    summary: str
    fields: tuple[FieldSpec, ...]


@dataclass(frozen=True)
class GymPlan:
    meta: JobMeta
    delay_s: float
    kind: Literal["gym"] = "gym"


@dataclass(frozen=True)
class TmuxPlan:
    session: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    kind: Literal["tmux"] = "tmux"


@dataclass(frozen=True)
class ForegroundPlan:
    argv: tuple[str, ...]
    cwd: Path
    title: str
    kind: Literal["foreground"] = "foreground"


LaunchPlan: TypeAlias = GymPlan | TmuxPlan | ForegroundPlan


class CatalogError(ValueError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def all_actions(*, root: Path | None = None) -> tuple[ActionSpec, ...]:
    root = root or repo_root()
    profiles = _profile_ids(root)
    isolation = tuple(item.value for item in IsolationBackend)
    return (
        ActionSpec(
            ActionId.DEMO,
            "Live guest gym",
            ActionGroup.GYM,
            "Watch attacker and defender turns on a stub guest.",
            (
                _int("generation", "Generation", "0"),
                _int("episodes", "Episodes", "2", minimum_label="1"),
                _int("turns", "Turns per side", "2", minimum_label="1"),
                _choice("profile", "Profile", "web", profiles),
                _choice("isolation", "Isolation", IsolationBackend.DOCKER.value, isolation),
                FieldSpec("delay", "Turn delay seconds", FieldKind.TEXT, "0.12"),
            ),
        ),
        ActionSpec(
            ActionId.GENERATION,
            "Full generation",
            ActionGroup.PIPELINE,
            "Rollout, review, GRPO, optional DPO, archive, PFSP, and eval.",
            (
                _int("generation", "Generation", "0"),
                _int("episodes", "Episodes", "2048", minimum_label="1"),
            ),
        ),
        ActionSpec(
            ActionId.ROLLOUT,
            "Rollout only",
            ActionGroup.PIPELINE,
            "Launch the Pi/guest rollout worker for one generation.",
            (
                _int("generation", "Generation", "0"),
                _int("episodes", "Episodes", "2048", minimum_label="1"),
            ),
        ),
        ActionSpec(
            ActionId.GRPO,
            "GRPO train",
            ActionGroup.TRAIN,
            "Convert traces and train one role with GRPO.",
            (_choice("role", "Role", "attacker", ("attacker", "defender")), _int("generation", "Generation", "0")),
        ),
        ActionSpec(
            ActionId.DPO,
            "DPO train",
            ActionGroup.TRAIN,
            "Train one role with DPO. Requires generation 2 or later.",
            (_choice("role", "Role", "attacker", ("attacker", "defender")), _int("generation", "Generation", "2")),
        ),
        ActionSpec(
            ActionId.SERVE_ATTACKER,
            "Serve vLLM attacker",
            ActionGroup.SERVE,
            "Serve the attacker LoRA on 127.0.0.1:8001.",
            (_int("generation", "Generation", "0"),),
        ),
        ActionSpec(
            ActionId.SERVE_DEFENDER,
            "Serve vLLM defender",
            ActionGroup.SERVE,
            "Serve the defender LoRA on 127.0.0.1:8002.",
            (_int("generation", "Generation", "0"),),
        ),
        ActionSpec(
            ActionId.REVIEW,
            "Review traces",
            ActionGroup.RESULTS,
            "Write review.md and review.json for a generation.",
            (
                _int("generation", "Generation", "0"),
                _choice("phase", "Phase", "complete", ("rollout", "complete")),
                FieldSpec("traces", "Traces path", FieldKind.PATH, "", required=False),
                FieldSpec("include_eval", "Include eval dir", FieldKind.FLAG, "true"),
                FieldSpec("include_archive", "Include archive dir", FieldKind.FLAG, "true"),
                FieldSpec("include_pfsp", "Include PFSP pool", FieldKind.FLAG, "true"),
            ),
        ),
        ActionSpec(
            ActionId.ARCHIVE,
            "Archive weights",
            ActionGroup.RESULTS,
            "Copy LoRA checkpoints and refresh FINAL.sh.",
            (
                _choice("mode", "Mode", "generation", ("generation", "all")),
                _int("generation", "Generation", "0"),
            ),
        ),
        ActionSpec(
            ActionId.ARCHIVE_LIST,
            "List archives",
            ActionGroup.RESULTS,
            "Print the archive index.",
            (),
        ),
        ActionSpec(
            ActionId.EVAL,
            "Tier-3 eval plan",
            ActionGroup.RESULTS,
            "Write a light or full tier-3 evaluation plan.",
            (_choice("mode", "Mode", "light", ("light", "full")),),
        ),
        ActionSpec(
            ActionId.PFSP,
            "Update PFSP pool",
            ActionGroup.RESULTS,
            "Refresh the local PFSP checkpoint manifest.",
            (_int("generation", "Generation", "0"),),
        ),
        ActionSpec(
            ActionId.BANDPASS,
            "Kill-switch check",
            ActionGroup.RESULTS,
            "Fail if ASR is stuck at 0 or 1 after generation 0.",
            (
                _int("generation", "Generation", "0"),
                FieldSpec("metrics", "Metrics path", FieldKind.PATH, "", required=False),
            ),
        ),
        ActionSpec(
            ActionId.TESTS,
            "Run tests",
            ActionGroup.VERIFY,
            "Run the Python unit suites. No GPUs or guests required.",
            (_choice("suite", "Suite", "all", ("all", "train", "env", "cli")),),
        ),
    )


def spec_for(action_id: ActionId, *, root: Path | None = None) -> ActionSpec:
    for spec in all_actions(root=root):
        if spec.id is action_id:
            return spec
    raise CatalogError(f"unknown action {action_id.value}")


def parse_values(spec: ActionSpec, raw: Mapping[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for field in spec.fields:
        text = raw.get(field.key, field.default)
        if text is None:
            text = field.default
        text = str(text).strip()
        if text == "":
            if field.required:
                raise CatalogError(f"{field.label} is required")
            parsed[field.key] = None
            continue
        parsed[field.key] = _parse_field(field, text)
    return parsed


def plan(action_id: ActionId, raw: Mapping[str, str], *, root: Path | None = None) -> LaunchPlan:
    root = root or repo_root()
    spec = spec_for(action_id, root=root)
    values = parse_values(spec, raw)
    match action_id:
        case ActionId.DEMO:
            return _plan_demo(values)
        case ActionId.GENERATION:
            generation = _int_value(values, "generation")
            episodes = _int_value(values, "episodes")
            return TmuxPlan(
                session=f"ultron-gen-{generation}",
                argv=(_script(root, "run_generation.sh"), str(generation)),
                env=(("ULTRON_EPISODES", str(episodes)),),
            )
        case ActionId.ROLLOUT:
            generation = _int_value(values, "generation")
            episodes = _int_value(values, "episodes")
            return TmuxPlan(
                session=f"ultron-rollout-gen{generation}",
                argv=(
                    _script(root, "rollout_worker.sh"),
                    "--generation",
                    str(generation),
                    "--episodes",
                    str(episodes),
                ),
            )
        case ActionId.GRPO:
            role = str(values["role"])
            generation = _int_value(values, "generation")
            return TmuxPlan(
                session=f"ultron-grpo-{role}-gen{generation}",
                argv=(
                    _script(root, "train_grpo.sh"),
                    "--role",
                    role,
                    "--generation",
                    str(generation),
                ),
            )
        case ActionId.DPO:
            role = str(values["role"])
            generation = _int_value(values, "generation")
            if generation < 2:
                raise CatalogError("DPO requires generation 2 or later")
            return TmuxPlan(
                session=f"ultron-dpo-{role}-gen{generation}",
                argv=(
                    _script(root, "train_dpo.sh"),
                    "--role",
                    role,
                    "--generation",
                    str(generation),
                ),
            )
        case ActionId.SERVE_ATTACKER:
            generation = _int_value(values, "generation")
            return TmuxPlan(
                session="ultron-vllm-attacker",
                argv=(_script(root, "serve_vllm_attacker.sh"), str(generation)),
            )
        case ActionId.SERVE_DEFENDER:
            generation = _int_value(values, "generation")
            return TmuxPlan(
                session="ultron-vllm-defender",
                argv=(_script(root, "serve_vllm_defender.sh"), str(generation)),
            )
        case ActionId.REVIEW:
            return _plan_review(values, root)
        case ActionId.ARCHIVE:
            mode = str(values["mode"])
            generation = _int_value(values, "generation")
            argv = [_python(), "-m", "ultron.train.archive"]
            if mode == "all":
                argv.append("--all")
            else:
                argv.extend(["--generation", str(generation)])
            return ForegroundPlan(argv=tuple(argv), cwd=root, title="archive")
        case ActionId.ARCHIVE_LIST:
            return ForegroundPlan(
                argv=(_python(), "-m", "ultron.train.archive", "--list"),
                cwd=root,
                title="archive list",
            )
        case ActionId.EVAL:
            mode = str(values["mode"])
            return ForegroundPlan(
                argv=(_python(), "-m", "ultron.eval.run_tier3", "--mode", mode),
                cwd=root,
                title=f"eval {mode}",
            )
        case ActionId.PFSP:
            generation = _int_value(values, "generation")
            return ForegroundPlan(
                argv=(
                    _python(),
                    "-m",
                    "ultron.train.pfsp",
                    "--update-pool",
                    "--generation",
                    str(generation),
                ),
                cwd=root,
                title="pfsp",
            )
        case ActionId.BANDPASS:
            generation = _int_value(values, "generation")
            metrics = values["metrics"]
            path = (
                root / "data" / "traces" / f"gen{generation}" / "metrics.json"
                if metrics is None
                else Path(str(metrics))
            )
            return ForegroundPlan(
                argv=(
                    _python(),
                    "-m",
                    "ultron.train.bandpass",
                    "--check-kill-switch",
                    "--generation",
                    str(generation),
                    "--metrics",
                    str(path),
                ),
                cwd=root,
                title="kill-switch",
            )
        case ActionId.TESTS:
            return _plan_tests(str(values["suite"]), root)
        case _:
            _assert_never(action_id)


def _plan_demo(values: dict[str, object]) -> GymPlan:
    isolation = IsolationBackend(str(values["isolation"]))
    try:
        delay_s = float(str(values["delay"]))
    except ValueError as exc:
        raise CatalogError("Turn delay seconds must be a number") from exc
    if delay_s < 0:
        raise CatalogError("Turn delay seconds must be >= 0")
    meta = JobMeta(
        generation=_int_value(values, "generation"),
        profile_id=str(values["profile"]),
        isolation=isolation,
        episodes_planned=_int_value(values, "episodes"),
        turns_per_side=_int_value(values, "turns"),
        version=__version__,
        snapshot_sha256="demo-sha",
    )
    return GymPlan(meta=meta, delay_s=delay_s)


def _plan_review(values: dict[str, object], root: Path) -> ForegroundPlan:
    generation = _int_value(values, "generation")
    phase = str(values["phase"])
    traces = values["traces"]
    traces_path = root / "data" / "traces" / f"gen{generation}" if traces is None else Path(str(traces))
    argv = [
        _python(),
        "-m",
        "ultron.train.review",
        str(traces_path),
        "--phase",
        phase,
        "--generation",
        str(generation),
        "--output",
        str(traces_path),
    ]
    if values["include_eval"]:
        argv.extend(["--eval-dir", str(root / "data" / "eval")])
    if values["include_archive"]:
        argv.extend(["--archive-dir", str(root / "data" / "archives")])
    if values["include_pfsp"]:
        argv.extend(["--pfsp", str(root / "data" / "checkpoints" / "pfsp_pool.json")])
    return ForegroundPlan(argv=tuple(argv), cwd=root, title="review")


def _plan_tests(suite: str, root: Path) -> ForegroundPlan:
    targets = {
        "all": ("train/tests", "env/tests", "cli/tests"),
        "train": ("train/tests",),
        "env": ("env/tests",),
        "cli": ("cli/tests",),
    }
    paths = targets.get(suite)
    if paths is None:
        raise CatalogError(f"unknown test suite {suite}")
    return ForegroundPlan(
        argv=(_python(), "-m", "pytest", *paths, "-q"),
        cwd=root,
        title=f"pytest {suite}",
    )


def _parse_field(field: FieldSpec, text: str) -> object:
    match field.kind:
        case FieldKind.INT:
            try:
                value = int(text)
            except ValueError as exc:
                raise CatalogError(f"{field.label} must be an integer") from exc
            if field.key == "generation" and value < 0:
                raise CatalogError("Generation must be >= 0")
            if field.key in {"episodes", "turns"} and value < 1:
                raise CatalogError(f"{field.label} must be >= 1")
            return value
        case FieldKind.CHOICE:
            if text not in field.choices:
                raise CatalogError(f"{field.label} must be one of: {', '.join(field.choices)}")
            return text
        case FieldKind.PATH:
            return Path(text)
        case FieldKind.FLAG:
            lowered = text.lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise CatalogError(f"{field.label} must be true or false")
        case FieldKind.TEXT:
            return text
        case _:
            _assert_never(field.kind)


def _profile_ids(root: Path) -> tuple[str, ...]:
    path = root / "env" / "profiles.yaml"
    payload = yaml.safe_load(path.read_text())
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise CatalogError(f"no profiles in {path}")
    return tuple(str(name) for name in profiles)


def _int(key: str, label: str, default: str, *, minimum_label: str | None = None) -> FieldSpec:
    help_text = f"Must be >= {minimum_label}." if minimum_label else "Must be >= 0."
    return FieldSpec(key, label, FieldKind.INT, default, help=help_text)


def _choice(key: str, label: str, default: str, choices: tuple[str, ...]) -> FieldSpec:
    return FieldSpec(key, label, FieldKind.CHOICE, default, choices=choices)


def _int_value(values: dict[str, object], key: str) -> int:
    value = values[key]
    if not isinstance(value, int):
        raise CatalogError(f"{key} must be an integer")
    return value


def _script(root: Path, name: str) -> str:
    return str(root / "scripts" / name)


def _python() -> str:
    return sys.executable


def _assert_never(value: object) -> None:
    raise CatalogError(f"unhandled {type(value)!r}")
