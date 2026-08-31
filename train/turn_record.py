from __future__ import annotations

from .schema_v1 import Role, ToolEvent, TrajectoryStep


def build_trajectory_step(
    *,
    turn_index: int,
    side: Role,
    prompt_token_ids: list[int],
    assistant_token_ids: list[int],
    assistant_mask: list[int],
    tool_events: list[ToolEvent],
    subgoal_hits: list[str] | None = None,
    decision_point: bool = False,
    format_valid: bool = True,
    async_bash_pending: bool = False,
    observation_hash: str = "",
) -> TrajectoryStep:
    if turn_index < 0:
        raise ValueError("turn_index must be non-negative")
    if len(assistant_token_ids) != len(assistant_mask):
        raise ValueError("assistant token and mask lengths differ")
    if any(mask not in (0, 1) for mask in assistant_mask):
        raise ValueError("assistant_mask values must be 0 or 1")
    return TrajectoryStep(
        turn_index=turn_index,
        side=side,
        prompt_token_ids=prompt_token_ids,
        assistant_token_ids=assistant_token_ids,
        assistant_mask=assistant_mask,
        tool_events=tool_events,
        decision_point=decision_point,
        subgoal_hits=list(subgoal_hits or []),
        format_valid=format_valid,
        async_bash_pending=async_bash_pending,
        observation_hash=observation_hash,
    )
