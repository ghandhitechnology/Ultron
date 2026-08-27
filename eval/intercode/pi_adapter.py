from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InterCodeTask:
    task_id: str
    goal: str
    profile_id: str


class EpisodeBackend(Protocol):
    def run_goal(self, task: InterCodeTask) -> dict[str, object]: ...


def run_task(task: InterCodeTask, backend: EpisodeBackend) -> dict[str, object]:
    return backend.run_goal(task)
