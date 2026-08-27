from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalPlan:
    held_out_episodes_per_profile: int
    procedural_templates: int
    intercode_scope: str
    debian12_episodes: int
    react_tasks: int
    run_ablations: bool


PLANS = {
    "light": EvalPlan(50, 20, "smoke-10", 0, 20, False),
    "full": EvalPlan(200, 50, "full", 100, 100, True),
}


def build_plan(mode: str) -> EvalPlan:
    try:
        return PLANS[mode]
    except KeyError as exc:
        raise ValueError(f"unknown tier-3 mode: {mode}") from exc


def _main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run Ultron tier-3 evaluation.")
    parser.add_argument("--mode", choices=sorted(PLANS), required=True)
    parser.add_argument("--output", type=Path, default=Path("data/eval"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for the KVM-backed evaluator once its adapters are configured.",
    )
    args = parser.parse_args()
    plan = build_plan(args.mode)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"tier3_{args.mode}_plan.json"
    destination.write_text(json.dumps(asdict(plan), indent=2) + "\n")
    print(destination)
    if args.execute:
        raise SystemExit("KVM evaluation adapter is not configured in this research scaffold")


if __name__ == "__main__":
    _main()
