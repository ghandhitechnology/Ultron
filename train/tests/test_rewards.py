from ultron.train.rewards import (
    assign_gen01_attacker_turn_rewards,
    assign_terminal_rtg,
    format_gate,
    total_gated_reward,
)
from ultron.train.schema_v1 import Role, TrajectoryStep


def step(*, hits: list[str] | None = None, valid: bool = True) -> TrajectoryStep:
    return TrajectoryStep(
        turn_index=0,
        side=Role.ATTACKER,
        prompt_token_ids=[],
        assistant_token_ids=[],
        assistant_mask=[],
        tool_events=[],
        subgoal_hits=hits or [],
        format_valid=valid,
    )


def test_subgoals_reward_only_first_hit() -> None:
    steps = [
        step(hits=["suid_bin_found"]),
        step(hits=["suid_bin_found", "shell_spawned"]),
        step(hits=["unknown"]),
    ]
    assign_gen01_attacker_turn_rewards(steps)
    assert [item.turn_reward for item in steps] == [0.1, 0.1, 0.0]


def test_terminal_reward_and_format_gate() -> None:
    steps = [step(), step()]
    assign_terminal_rtg(steps, 1.0)
    assert [item.turn_reward for item in steps] == [0.0, 1.0]
    assert format_gate(steps) == 1.0
    steps[0].format_valid = False
    assert total_gated_reward(steps) == 0.0
