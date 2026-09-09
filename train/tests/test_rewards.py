from ultron.train.rewards import (
    assign_terminal_rtg,
    assign_verified_shaping,
    format_gate,
    return_to_go,
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


def test_subgoals_share_the_episode_cap() -> None:
    steps = [
        step(hits=["suid_bin_found"]),
        step(hits=["suid_bin_found", "shell_spawned"]),
        step(hits=["unknown"]),
    ]
    assign_verified_shaping(steps)
    assert [item.turn_reward for item in steps] == [0.05, 0.05, 0.0]
    assert sum(item.turn_reward for item in steps) == 0.1


def test_terminal_reward_and_format_gate() -> None:
    steps = [step(), step()]
    assign_terminal_rtg(steps, 1.0)
    assert [item.turn_reward for item in steps] == [0.0, 1.0]
    assert format_gate(steps) == 1.0
    steps[0].format_valid = False
    assert total_gated_reward(steps) == 0.0


def test_return_to_go_is_suffix_sum() -> None:
    assert return_to_go([0.1, 0.0, 1.0]) == [1.1, 1.0, 1.0]
