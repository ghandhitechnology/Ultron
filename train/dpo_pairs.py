from dataclasses import dataclass

from .schema_v1 import Role, TrajectoryV1


@dataclass(frozen=True)
class DpoPair:
    prompt_token_ids: list[int]
    chosen_token_ids: list[int]
    rejected_token_ids: list[int]
    role: Role
    episode_id: str
    branch_turn: int


def extract_prefix_branch_pairs(
    winner: TrajectoryV1,
    loser: TrajectoryV1,
    *,
    same_group: bool,
    same_profile: bool,
    same_prefix_hash: str,
) -> list[DpoPair]:
    if not same_group or winner.group_id != loser.group_id:
        raise ValueError("DPO trajectories must share a group")
    if not same_profile or winner.profile_id != loser.profile_id:
        raise ValueError("DPO trajectories must share a profile")
    if winner.opponent_checkpoint_id != loser.opponent_checkpoint_id:
        raise ValueError("DPO trajectories must share an opponent checkpoint")
    if not same_prefix_hash:
        raise ValueError("DPO trajectories require a shared prefix hash")
    if winner.terminal.attacker_reward <= loser.terminal.attacker_reward:
        raise ValueError("winner must have a higher attacker reward")

    pairs: list[DpoPair] = []
    for winner_step, loser_step in zip(winner.steps, loser.steps):
        if winner_step.observation_hash != loser_step.observation_hash:
            break
        if winner_step.side != Role.ATTACKER:
            continue
        if not winner_step.decision_point:
            continue
        if winner_step.assistant_token_ids == loser_step.assistant_token_ids:
            continue
        pairs.append(
            DpoPair(
                prompt_token_ids=list(winner_step.prompt_token_ids),
                chosen_token_ids=list(winner_step.assistant_token_ids),
                rejected_token_ids=list(loser_step.assistant_token_ids),
                role=Role.ATTACKER,
                episode_id=winner.episode_id,
                branch_turn=winner_step.turn_index,
            )
        )
    return pairs
