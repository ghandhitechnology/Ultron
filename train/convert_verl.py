from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .rewards import total_gated_reward
from .schema_v1 import TrajectoryV1


def trajectory_to_verl_record(traj: TrajectoryV1, generation: int) -> dict[str, Any]:
    traj.validate()
    if not traj.steps:
        raise ValueError("cannot convert an empty trajectory")
    return {
        "prompt": traj.steps[-1].prompt_token_ids,
        "response": traj.steps[-1].assistant_token_ids,
        "reward": total_gated_reward(traj.steps),
        "data_source": "ultron",
        "extra_info": {
            "group_id": traj.group_id,
            "role": traj.role.value,
            "adapter_id": traj.adapter_id,
            "opponent_checkpoint_id": traj.opponent_checkpoint_id,
            "generation": generation,
        },
    }


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
    records = [trajectory_to_verl_record(traj, generation) for traj in read_trajectories(source)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("install ultron[parquet] to write parquet") from exc
        pq.write_table(pa.Table.from_pylist(records), destination)
        return
    with destination.open("w") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Convert trajectory schema v1 to veRL rows.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--generation", type=int, required=True)
    args = parser.parse_args()
    convert_jsonl(args.source, args.destination, args.generation)


if __name__ == "__main__":
    _main()
