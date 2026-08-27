from .schema_v1 import TrajectoryStep

SUBGOAL_CAP = 0.1
TERMINAL_WIN = 1.0
ATTACKER_SUBGOALS = {"suid_bin_found", "writable_path_found", "shell_spawned"}


def format_gate(steps: list[TrajectoryStep]) -> float:
    return 1.0 if all(step.format_valid for step in steps) else 0.0


def assign_gen01_attacker_turn_rewards(
    steps: list[TrajectoryStep], subgoals: set[str] | None = None
) -> None:
    allowed = ATTACKER_SUBGOALS if subgoals is None else ATTACKER_SUBGOALS & subgoals
    seen: set[str] = set()
    for step in steps:
        new_hits = {hit for hit in step.subgoal_hits if hit in allowed and hit not in seen}
        step.turn_reward = SUBGOAL_CAP * len(new_hits)
        seen.update(new_hits)


def assign_terminal_rtg(
    steps: list[TrajectoryStep], terminal_reward: float, *, generation: int = 0
) -> None:
    if not steps:
        return
    if generation >= 2:
        for step in steps:
            step.turn_reward += terminal_reward
        return
    steps[-1].turn_reward += terminal_reward


def total_gated_reward(steps: list[TrajectoryStep]) -> float:
    return sum(step.turn_reward for step in steps) * format_gate(steps)
