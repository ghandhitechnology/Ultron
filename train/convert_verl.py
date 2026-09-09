from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .rewards import format_gate, return_to_go
from .io import atomic_write_text
from .schema_v1 import ReasonCode, TrajectoryV1


def trajectory_to_verl_records(
    traj: TrajectoryV1, generation: int, *, behavior_policy_id: str = ""
) -> list[dict[str, Any]]:
    traj.validate()
    if traj.terminal.reason_code is ReasonCode.INFRA_FAIL:
        return []
    if not traj.steps:
        raise ValueError("cannot convert an empty trajectory")
    gate = format_gate(traj.steps)
    rewards = [step.turn_reward * gate for step in traj.steps]
    returns = return_to_go(rewards)
    policy = behavior_policy_id or traj.adapter_id
    return [
        {
            "prompt": step.prompt_token_ids,
            "response": step.assistant_token_ids,
            "assistant_mask": step.assistant_mask,
            "reward": reward,
            "data_source": "ultron",
            "extra_info": {
                "turn_index": step.turn_index,
                "episode_id": traj.episode_id,
                "role": traj.role.value,
                "group_id": traj.group_id,
                "adapter_id": traj.adapter_id,
                "opponent_checkpoint_id": traj.opponent_checkpoint_id,
                "generation": generation,
                "return_to_go": ret,
                "behavior_policy_id": policy,
            },
        }
        for step, reward, ret in zip(traj.steps, rewards, returns, strict=True)
    ]


def read_trajectories(path: Path) -> Iterable[TrajectoryV1]:
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield TrajectoryV1.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def convert_jsonl(source: Path, destination: Path, generation: int) -> None:
    records = [
        record
        for traj in read_trajectories(source)
        for record in trajectory_to_verl_records(traj, generation)
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("install ultron[parquet] to write parquet") from exc
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            pq.write_table(pa.Table.from_pylist(records), temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return
    atomic_write_text(destination, "".join(json.dumps(record) + "\n" for record in records))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Convert trajectory schema v1 to veRL rows.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--generation", type=int, required=True)
    args = parser.parse_args()
    convert_jsonl(args.source, args.destination, args.generation)


if __name__ == "__main__":
    _main()
