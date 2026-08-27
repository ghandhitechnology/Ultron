import pytest

from ultron.train.dpo_pairs import extract_prefix_branch_pairs
from ultron.train.schema_v1 import (
    SCHEMA_VERSION,
    ReasonCode,
    Role,
    TerminalOutcome,
    TrajectoryStep,
    TrajectoryV1,
)


def trajectory(episode: str, reward: float, action: list[int], obs: str = "same") -> TrajectoryV1:
    return TrajectoryV1(
        schema_version=SCHEMA_VERSION,
        episode_id=episode,
        generation=2,
        profile_id="web",
        role=Role.ATTACKER,
        adapter_id="attacker_lora",
        opponent_checkpoint_id="defender-1",
        group_id="group-1",
        steps=[
            TrajectoryStep(
                turn_index=2,
                side=Role.ATTACKER,
                prompt_token_ids=[10],
                assistant_token_ids=action,
                assistant_mask=[1] * len(action),
                tool_events=[],
                decision_point=True,
                observation_hash=obs,
            )
        ],
        terminal=TerminalOutcome(
            reason_code=ReasonCode.ATTACKER_ROOT
            if reward
            else ReasonCode.DEFENDER_HOLD,
            attacker_euid=0 if reward else 1000,
            host_confirmed_root=bool(reward),
            availability_ok=True,
            attacker_reward=reward,
            defender_reward=1.0 - reward,
        ),
    )


def test_extracts_same_prefix_attacker_branch() -> None:
    pairs = extract_prefix_branch_pairs(
        trajectory("winner", 1.0, [1]),
        trajectory("loser", 0.0, [2]),
        same_group=True,
        same_profile=True,
        same_prefix_hash="same",
    )
    assert len(pairs) == 1
    assert pairs[0].chosen_token_ids == [1]
    assert pairs[0].rejected_token_ids == [2]
    assert pairs[0].branch_turn == 2


def test_rejects_different_prefix() -> None:
    pairs = extract_prefix_branch_pairs(
        trajectory("winner", 1.0, [1], "a"),
        trajectory("loser", 0.0, [2], "b"),
        same_group=True,
        same_profile=True,
        same_prefix_hash="a",
    )
    assert pairs == []


def test_rejects_unmatched_groups() -> None:
    loser = trajectory("loser", 0.0, [2])
    loser.group_id = "other"
    with pytest.raises(ValueError, match="group"):
        extract_prefix_branch_pairs(
            trajectory("winner", 1.0, [1]),
            loser,
            same_group=False,
            same_profile=True,
            same_prefix_hash="same",
        )
