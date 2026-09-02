import json

import pytest

from ultron.env.backend import IsolationBackend
from ultron.train.schema_v1 import (
    SCHEMA_VERSION,
    ReasonCode,
    Role,
    SchemaError,
    TerminalOutcome,
    ToolEvent,
    TrajectoryStep,
    TrajectoryV1,
)


def make_trajectory() -> TrajectoryV1:
    return TrajectoryV1(
        schema_version=SCHEMA_VERSION,
        episode_id="episode-1",
        generation=0,
        profile_id="web",
        role=Role.ATTACKER,
        adapter_id="attacker_lora",
        opponent_checkpoint_id="defender-gen0",
        group_id="group-1",
        steps=[
            TrajectoryStep(
                turn_index=0,
                side=Role.ATTACKER,
                prompt_token_ids=[1, 2],
                assistant_token_ids=[3, 4],
                assistant_mask=[1, 1],
                tool_events=[
                    ToolEvent(
                        name="bash",
                        args={"command": "id"},
                        stdout_head="1000",
                        stdout_tail="1000",
                        exit_code=0,
                        duration_ms=7,
                    )
                ],
                observation_hash="abc",
            )
        ],
        terminal=TerminalOutcome(
            reason_code=ReasonCode.DEFENDER_HOLD,
            attacker_euid=1000,
            host_confirmed_root=False,
            availability_ok=True,
            attacker_reward=0.0,
            defender_reward=1.0,
        ),
        isolation_backend=IsolationBackend.KVM,
    )


def test_schema_roundtrip_preserves_tokens() -> None:
    original = make_trajectory()
    decoded = TrajectoryV1.from_dict(json.loads(json.dumps(original.to_dict())))
    assert decoded == original
    assert decoded.steps[0].prompt_token_ids == [1, 2]
    assert decoded.steps[0].assistant_token_ids == [3, 4]


def test_schema_rejects_invalid_mask() -> None:
    trajectory = make_trajectory()
    trajectory.steps[0].assistant_mask = [1]
    with pytest.raises(SchemaError, match="lengths differ"):
        trajectory.validate()
