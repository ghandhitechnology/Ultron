from .schema_v1 import TrajectoryStep

SUBGOAL_CAP = 0.1
TERMINAL_WIN = 1.0
ATTACKER_SUBGOALS = {"suid_bin_found", "writable_path_found", "shell_spawned"}


def format_gate(steps: list[TrajectoryStep]) -> float:
    return 1.0 if all(step.format_valid for step in steps) else 0.0


def assign_verified_shaping(
    steps: list[TrajectoryStep],
    subgoals: set[str] | None = None,
    *,
    cap: float = SUBGOAL_CAP,
) -> None:
    allowed = ATTACKER_SUBGOALS if subgoals is None else ATTACKER_SUBGOALS & subgoals
    first_at: list[int] = []
    seen: set[str] = set()
    for index, step in enumerate(steps):
        new_hits = [hit for hit in step.subgoal_hits if hit in allowed and hit not in seen]
        if new_hits:
            first_at.append(index)
            seen.update(new_hits)
        step.turn_reward = 0.0
    if not first_at:
        return
    each = cap / len(first_at)
    for index in first_at:
        steps[index].turn_reward += each


def assign_gen01_attacker_turn_rewards(
    steps: list[TrajectoryStep], subgoals: set[str] | None = None
) -> None:
    assign_verified_shaping(steps, subgoals)


def assign_terminal_rtg(steps: list[TrajectoryStep], terminal_reward: float) -> None:
    if not steps:
        return
    steps[-1].turn_reward += terminal_reward


def total_gated_reward(steps: list[TrajectoryStep]) -> float:
    return sum(step.turn_reward for step in steps) * format_gate(steps)


def return_to_go(rewards: list[float]) -> list[float]:
    running = 0.0
    out: list[float] = []
    for reward in reversed(rewards):
        running += reward
        out.append(running)
    out.reverse()
    return out
