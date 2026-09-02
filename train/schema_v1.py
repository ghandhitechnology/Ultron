from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from ultron.env.backend import IsolationBackend

SCHEMA_VERSION = "v1"
PROFILE_IDS = {
    "web",
    "db",
    "build-box",
    "jump",
    "hardened-server",
    "workstation",
    "procedural",
}


class SchemaError(ValueError):
    pass


class ReasonCode(str, Enum):
    ATTACKER_ROOT = "ATTACKER_ROOT"
    DEFENDER_HOLD = "DEFENDER_HOLD"
    TIMEOUT = "TIMEOUT"
    INFRA_FAIL = "INFRA_FAIL"
    AVAILABILITY_FAIL = "AVAILABILITY_FAIL"


class Role(str, Enum):
    ATTACKER = "attacker"
    DEFENDER = "defender"


@dataclass
class ToolEvent:
    name: str
    args: dict[str, Any]
    stdout_head: str
    stdout_tail: str
    exit_code: int | None
    duration_ms: int


@dataclass
class TrajectoryStep:
    turn_index: int
    side: Role
    prompt_token_ids: list[int]
    assistant_token_ids: list[int]
    assistant_mask: list[int]
    tool_events: list[ToolEvent]
    decision_point: bool = False
    subgoal_hits: list[str] = field(default_factory=list)
    format_valid: bool = True
    turn_reward: float = 0.0
    async_bash_pending: bool = False
    observation_hash: str = ""


@dataclass
class TerminalOutcome:
    reason_code: ReasonCode
    attacker_euid: int
    host_confirmed_root: bool
    availability_ok: bool
    attacker_reward: float
    defender_reward: float


@dataclass
class TrajectoryV1:
    schema_version: str
    episode_id: str
    generation: int
    profile_id: str
    role: Role
    adapter_id: str
    opponent_checkpoint_id: str
    group_id: str
    steps: list[TrajectoryStep]
    terminal: TerminalOutcome
    isolation_backend: IsolationBackend

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}")
        if not self.episode_id or not self.group_id or not self.opponent_checkpoint_id:
            raise SchemaError("episode_id, group_id, and opponent_checkpoint_id are required")
        if self.generation < 0:
            raise SchemaError("generation must be non-negative")
        if self.profile_id not in PROFILE_IDS:
            raise SchemaError(f"unknown profile_id: {self.profile_id}")
        try:
            IsolationBackend(self.isolation_backend)
        except ValueError as exc:
            raise SchemaError(f"unknown isolation_backend: {self.isolation_backend}") from exc
        expected_adapter = f"{self.role.value}_lora"
        if self.adapter_id != expected_adapter:
            raise SchemaError(f"{self.role.value} trajectory requires adapter_id={expected_adapter}")
        for step in self.steps:
            if len(step.assistant_token_ids) != len(step.assistant_mask):
                raise SchemaError("assistant token and mask lengths differ")
            if any(mask not in (0, 1) for mask in step.assistant_mask):
                raise SchemaError("assistant_mask values must be 0 or 1")
            if step.turn_index < 0:
                raise SchemaError("turn_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _enum_values(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TrajectoryV1:
        data = _mapping(raw, "trajectory")
        required = {
            "schema_version",
            "episode_id",
            "generation",
            "profile_id",
            "role",
            "adapter_id",
            "opponent_checkpoint_id",
            "group_id",
            "steps",
            "terminal",
            "isolation_backend",
        }
        missing = required - data.keys()
        if missing:
            raise SchemaError(f"missing fields: {sorted(missing)}")
        try:
            steps = [_step_from_dict(item) for item in _list(data["steps"], "steps")]
            terminal = _terminal_from_dict(data["terminal"])
            trajectory = cls(
                schema_version=_str(data["schema_version"], "schema_version"),
                episode_id=_str(data["episode_id"], "episode_id"),
                generation=_int(data["generation"], "generation"),
                profile_id=_str(data["profile_id"], "profile_id"),
                role=Role(data["role"]),
                adapter_id=_str(data["adapter_id"], "adapter_id"),
                opponent_checkpoint_id=_str(
                    data["opponent_checkpoint_id"], "opponent_checkpoint_id"
                ),
                group_id=_str(data["group_id"], "group_id"),
                steps=steps,
                terminal=terminal,
                isolation_backend=IsolationBackend(
                    _str(data["isolation_backend"], "isolation_backend")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise SchemaError(str(exc)) from exc
        trajectory.validate()
        return trajectory


def _step_from_dict(raw: Any) -> TrajectoryStep:
    data = _mapping(raw, "step")
    return TrajectoryStep(
        turn_index=_int(data.get("turn_index"), "turn_index"),
        side=Role(data.get("side")),
        prompt_token_ids=_int_list(data.get("prompt_token_ids"), "prompt_token_ids"),
        assistant_token_ids=_int_list(data.get("assistant_token_ids"), "assistant_token_ids"),
        assistant_mask=_int_list(data.get("assistant_mask"), "assistant_mask"),
        tool_events=[_tool_event_from_dict(item) for item in _list(data.get("tool_events"), "tool_events")],
        decision_point=_bool(data.get("decision_point", False), "decision_point"),
        subgoal_hits=[
            _str(value, "subgoal_hits item")
            for value in _list(data.get("subgoal_hits", []), "subgoal_hits")
        ],
        format_valid=_bool(data.get("format_valid", True), "format_valid"),
        turn_reward=_float(data.get("turn_reward", 0.0), "turn_reward"),
        async_bash_pending=_bool(data.get("async_bash_pending", False), "async_bash_pending"),
        observation_hash=_str(data.get("observation_hash", ""), "observation_hash"),
    )


def _tool_event_from_dict(raw: Any) -> ToolEvent:
    data = _mapping(raw, "tool event")
    exit_code = data.get("exit_code")
    if exit_code is not None:
        exit_code = _int(exit_code, "exit_code")
    args = _mapping(data.get("args"), "args")
    return ToolEvent(
        name=_str(data.get("name"), "name"),
        args=dict(args),
        stdout_head=_str(data.get("stdout_head"), "stdout_head"),
        stdout_tail=_str(data.get("stdout_tail"), "stdout_tail"),
        exit_code=exit_code,
        duration_ms=_int(data.get("duration_ms"), "duration_ms"),
    )


def _terminal_from_dict(raw: Any) -> TerminalOutcome:
    data = _mapping(raw, "terminal")
    return TerminalOutcome(
        reason_code=ReasonCode(data.get("reason_code")),
        attacker_euid=_int(data.get("attacker_euid"), "attacker_euid"),
        host_confirmed_root=_bool(data.get("host_confirmed_root", False), "host_confirmed_root"),
        availability_ok=_bool(data.get("availability_ok"), "availability_ok"),
        attacker_reward=_float(data.get("attacker_reward"), "attacker_reward"),
        defender_reward=_float(data.get("defender_reward"), "defender_reward"),
    )


def _enum_values(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_enum_values(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{name} must be an array")
    return value


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{name} must be a string")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SchemaError(f"{name} must be an integer")
    return value


def _float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SchemaError(f"{name} must be a number")
    return float(value)


def _int_list(value: Any, name: str) -> list[int]:
    return [_int(item, f"{name} item") for item in _list(value, name)]


TRAJECTORY_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "ultron/trajectory_v1.json",
    "type": "object",
    "required": [
        "schema_version",
        "episode_id",
        "generation",
        "profile_id",
        "role",
        "adapter_id",
        "opponent_checkpoint_id",
        "group_id",
        "steps",
        "terminal",
        "isolation_backend",
    ],
}
