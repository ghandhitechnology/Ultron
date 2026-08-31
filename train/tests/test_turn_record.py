import pytest

from ultron.train.schema_v1 import Role
from ultron.train.turn_record import build_trajectory_step


def test_builds_valid_step() -> None:
    step = build_trajectory_step(
        turn_index=2,
        side=Role.DEFENDER,
        prompt_token_ids=[1, 2],
        assistant_token_ids=[3, 4],
        assistant_mask=[1, 0],
        tool_events=[],
        subgoal_hits=["suid_bin_found"],
    )
    assert step.turn_index == 2
    assert step.side == Role.DEFENDER
    assert step.subgoal_hits == ["suid_bin_found"]


def test_rejects_mask_length_mismatch() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        build_trajectory_step(
            turn_index=0,
            side=Role.ATTACKER,
            prompt_token_ids=[1],
            assistant_token_ids=[3, 4],
            assistant_mask=[1],
            tool_events=[],
        )


def test_rejects_non_binary_mask() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        build_trajectory_step(
            turn_index=0,
            side=Role.ATTACKER,
            prompt_token_ids=[1],
            assistant_token_ids=[3],
            assistant_mask=[2],
            tool_events=[],
        )


def test_rejects_negative_turn_index() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_trajectory_step(
            turn_index=-1,
            side=Role.ATTACKER,
            prompt_token_ids=[],
            assistant_token_ids=[],
            assistant_mask=[],
            tool_events=[],
        )
