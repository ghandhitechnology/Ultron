from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ToolRuntime(Protocol):
    def execute(self, tool_name: str, arguments: dict[str, object]) -> str: ...


@dataclass(frozen=True)
class ReactResult:
    answer: str
    tool_calls: int
    settled: bool


class ReactBaseline:
    """Integration point for a minimal non-Pi cross-harness baseline."""

    def __init__(self, runtime: ToolRuntime, max_tool_calls: int = 12) -> None:
        self.runtime = runtime
        self.max_tool_calls = max_tool_calls

    def run(self, goal: str) -> ReactResult:
        raise NotImplementedError(
            "Connect a model client and map its structured tool calls to ToolRuntime.execute"
        )
