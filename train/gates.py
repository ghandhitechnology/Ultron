"""Study verification predicates and admission checks for recorded training data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

from ultron.train.adjudicator import ProbeResult, adjudicate
from ultron.train.schema_v1 import ReasonCode
from ultron.env.backend import IsolationBackend
from ultron.train.convert_verl import read_trajectories
from ultron.train.rewards import total_gated_reward
from ultron.train.schema_v1 import Role, TrajectoryV1
from ultron.train.study import StudyError
from ultron.train.io import atomic_write_json


class DataGateError(StudyError):
    pass


def validate_training_data(
    traces: Path, *, request: dict[str, Any], role: str | None = None,
    expected_episodes: int | None = None,
) -> dict[str, Any]:
    """Validate both transcripts, then group only current learner trajectories.

    INFRA_FAIL records remain in the journal but never enter an update. An
    infrastructure failure must not leave a five-member optimization group.
    Binary ASR is not used as a proxy for verified shaping reward variance.
    """
    expected = int(request["episodes"]) if expected_episodes is None else expected_episodes
    group_size = int(request["group_size"])
    episodes: dict[str, dict[str, TrajectoryV1]] = defaultdict(dict)
    groups: dict[tuple[str, str], list[TrajectoryV1]] = defaultdict(list)
    allowed_tasks = {item.get("task_id", item.get("id")) for item in request.get("tasks", [])}
    for name in ("attacker", "defender"):
        path = traces / f"{name}.jsonl"
        if not path.is_file():
            raise DataGateError(f"missing role trajectories: {path}")
        for trajectory in read_trajectories(path):
            if trajectory.role.value != name:
                raise DataGateError(f"{path} contains a different role")
            if name in episodes[trajectory.episode_id]:
                raise DataGateError(f"duplicate {name} trajectory for {trajectory.episode_id}")
            episodes[trajectory.episode_id][name] = trajectory
            if trajectory.generation != request["generation"]:
                raise DataGateError("trajectory generation differs from the rollout request")
            if allowed_tasks and trajectory.profile_id not in allowed_tasks:
                raise DataGateError("trajectory uses a task outside the locked training curriculum")
            if trajectory.isolation_backend is not IsolationBackend.KVM:
                raise DataGateError("study trajectories require KVM isolation")
            _validate_outcome(trajectory)
            if trajectory.trainable:
                policy = request["policies"][name]["checkpoint_id"]
                if trajectory.behavior_policy_id != policy:
                    raise DataGateError("learner trajectory was sampled by a stale or different behavior policy")
                if trajectory.terminal.reason_code is not ReasonCode.INFRA_FAIL:
                    if not trajectory.steps or not any(any(step.assistant_mask) for step in trajectory.steps):
                        raise DataGateError("learner trajectory has no assistant tokens for optimization")
                    groups[(name, trajectory.group_id)].append(trajectory)
    if len(episodes) != expected:
        raise DataGateError(f"expected {expected} complete episodes, found {len(episodes)}")
    for episode_id, roles in episodes.items():
        if set(roles) != {"attacker", "defender"}:
            raise DataGateError(f"episode {episode_id} is missing a role transcript")
        attacker, defender = roles["attacker"], roles["defender"]
        if asdict(attacker.terminal) != asdict(defender.terminal):
            raise DataGateError(f"episode {episode_id} has contradictory terminal outcomes")
        if attacker.group_id != defender.group_id or attacker.profile_id != defender.profile_id:
            raise DataGateError(f"episode {episode_id} has inconsistent task/group identity")
    counts = {name: {"groups": 0, "signal_groups": 0, "learner_episodes": 0} for name in ("attacker", "defender")}
    for (name, group_id), trajectories in groups.items():
        if len(trajectories) != group_size:
            raise DataGateError(f"{name} group {group_id} has {len(trajectories)} eligible members; expected {group_size}")
        if len({(item.profile_id, item.opponent_checkpoint_id) for item in trajectories}) != 1:
            raise DataGateError(f"{name} group {group_id} changed task or opponent")
        rewards = [total_gated_reward(item.steps) for item in trajectories]
        signal = max(rewards) - min(rewards) > 1e-12
        counts[name]["groups"] += 1
        counts[name]["signal_groups"] += int(signal)
        counts[name]["learner_episodes"] += len(trajectories)
    checked = (role,) if role is not None else ("attacker", "defender")
    for name in checked:
        if name not in counts:
            raise DataGateError("unknown training role")
        if counts[name]["groups"] == 0:
            raise DataGateError(f"{name} has no complete current-policy learner groups")
    result: dict[str, Any] = {"episodes": len(episodes), "group_size": group_size, "roles": counts}
    if role is not None:
        result.update(counts[role])
    return result


def _validate_outcome(trajectory: TrajectoryV1) -> None:
    terminal = trajectory.terminal
    values = [terminal.attacker_reward, terminal.defender_reward, *(step.turn_reward for step in trajectory.steps)]
    if any(not math.isfinite(value) for value in values):
        raise DataGateError("trajectory contains a nonfinite reward")
    for step in trajectory.steps:
        if step.side is not trajectory.role:
            raise DataGateError("trajectory contains another role's assistant step")
        if any(token < 0 for token in (*step.prompt_token_ids, *step.assistant_token_ids)):
            raise DataGateError("trajectory contains invalid token IDs")
    if terminal.reason_code is ReasonCode.ATTACKER_ROOT:
        if not terminal.host_confirmed_root and not terminal.recovered:
            raise DataGateError("attacker win lacks host-confirmed compromise evidence")
        expected = (1.0, 0.0)
    elif terminal.reason_code is ReasonCode.DEFENDER_HOLD:
        if not terminal.availability_ok or not terminal.ordinary_user_ok or terminal.unresolved:
            raise DataGateError("defender win violates availability or ordinary-user access")
        expected = (0.0, 1.0)
    else:
        expected = (0.0, 0.0)
    if (terminal.attacker_reward, terminal.defender_reward) != expected:
        raise DataGateError("terminal rewards contradict the verified outcome")
    raw_total = sum(step.turn_reward for step in trajectory.steps)
    reward = terminal.attacker_reward if trajectory.role is Role.ATTACKER else terminal.defender_reward
    cap = 0.1 if trajectory.role is Role.ATTACKER else 0.0
    if trajectory.terminal.reason_code is not ReasonCode.INFRA_FAIL and not reward - 1e-8 <= raw_total <= reward + cap + 1e-8:
        raise DataGateError("turn rewards do not preserve terminal reward and shaping cap")


def false_root_is_rejected(probe: ProbeResult) -> bool:
    reason, attacker, _ = adjudicate(probe)
    if probe.guest_attacker_euid == 0 and not probe.host_confirmed_root:
        return reason is not ReasonCode.ATTACKER_ROOT and attacker == 0.0
    return True


def timeout_is_unresolved(probe: ProbeResult) -> bool:
    if not probe.timed_out and not probe.hung:
        return True
    reason, attacker, defender = adjudicate(probe)
    if probe.guest_attacker_euid == 0 and probe.host_confirmed_root:
        return True
    unresolved = reason in {ReasonCode.TIMEOUT, ReasonCode.HANG}
    return unresolved and attacker == 0.0 and defender == 0.0


def observation_leaks_metadata(observation: str, secrets: frozenset[str]) -> bool:
    if not secrets:
        return False
    return any(secret and secret in observation for secret in secrets)


def dummy_service_is_not_availability(real_ok: bool, dummy_ok: bool) -> bool:
    return not (dummy_ok and not real_ok)


def known_solvable_holds(*, launcher_ran: bool, attacker_won: bool) -> bool:
    return launcher_ran and attacker_won


def known_patched_holds(*, launcher_ran: bool, attacker_won: bool) -> bool:
    return launcher_ran and not attacker_won


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check integrity and complete learner-group signal before training.")
    parser.add_argument("traces", type=Path)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--dpo-pairs", type=Path)
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_text())
        report = validate_training_data(args.traces, request=request, expected_episodes=args.expected_episodes)
        if args.dpo_pairs is not None:
            from .training import validate_dpo_pairs
            report["dpo_pairs"] = validate_dpo_pairs(args.dpo_pairs, generation=request["generation"])
        if args.output:
            atomic_write_json(args.output, report)
        print(json.dumps(report, indent=2))
        for role, count in report["roles"].items():
            if count["signal_groups"] == 0:
                raise DataGateError(f"{role} has no reward variance in complete learner groups; inspect the pilot before training")
    except (StudyError, OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
