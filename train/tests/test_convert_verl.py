import pytest

from ultron.train.convert_verl import trajectory_to_verl_records
from ultron.train.schema_v1 import (
    SCHEMA_VERSION,
    ReasonCode,
    Role,
    TerminalOutcome,
    TrajectoryStep,
    TrajectoryV1,
)


def make_step(turn_index: int, *, reward: float, valid: bool = True) -> TrajectoryStep:
    return TrajectoryStep(
        turn_index=turn_index,
        side=Role.ATTACKER,
        prompt_token_ids=[turn_index],
        assistant_token_ids=[turn_index, turn_index],
        assistant_mask=[1, 1],
        tool_events=[],
        format_valid=valid,
        turn_reward=reward,
    )


def make_trajectory(steps: list[TrajectoryStep]) -> TrajectoryV1:
    return TrajectoryV1(
        schema_version=SCHEMA_VERSION,
        episode_id="episode-1",
        generation=0,
        profile_id="web",
        role=Role.ATTACKER,
        adapter_id="attacker_lora",
        opponent_checkpoint_id="defender-gen0",
        group_id="group-1",
        steps=steps,
        terminal=TerminalOutcome(
            reason_code=ReasonCode.DEFENDER_HOLD,
            attacker_euid=1000,
            host_confirmed_root=False,
            availability_ok=True,
            attacker_reward=0.0,
            defender_reward=1.0,
        ),
    )


def test_one_record_per_step_with_step_reward() -> None:
    traj = make_trajectory([make_step(0, reward=0.1), make_step(1, reward=0.5)])
    records = trajectory_to_verl_records(traj, generation=1)
    assert [record["reward"] for record in records] == [0.1, 0.5]
    assert [record["extra_info"]["turn_index"] for record in records] == [0, 1]
    assert all(record["extra_info"]["episode_id"] == "episode-1" for record in records)
    assert all(record["extra_info"]["generation"] == 1 for record in records)


def test_format_gate_zeroes_all_rewards() -> None:
    traj = make_trajectory([make_step(0, reward=0.1), make_step(1, reward=0.5, valid=False)])
    records = trajectory_to_verl_records(traj, generation=1)
    assert [record["reward"] for record in records] == [0.0, 0.0]


def test_gen2_terminal_on_every_step() -> None:
    traj = make_trajectory([make_step(0, reward=1.0), make_step(1, reward=1.0)])
    records = trajectory_to_verl_records(traj, generation=2)
    assert [record["reward"] for record in records] == [1.0, 1.0]


def test_empty_trajectory_raises() -> None:
    traj = make_trajectory([])
    with pytest.raises(ValueError, match="empty trajectory"):
        trajectory_to_verl_records(traj, generation=1)
