"""Plan, execute, ingest, and plot held-out public benchmark evals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ultron.eval.benchmarks import (
    BenchmarkError,
    EvalMode,
    SuiteConfig,
    build_jobs,
    discover_iteration_stages,
    execute_jobs,
    ingest_scores,
    incomplete_arms,
    load_scores,
    load_suite_config,
    merge_scores,
    mode_for_generation,
    mock_scores,
    parse_eval_mode,
    parse_role,
    planned_scores,
    write_plan,
    write_scores,
)
from ultron.eval.plot import write_plots
from ultron.train.family import resolve
from ultron.train.schema_v1 import Role


def eval_root_for_archive(archive_root: Path) -> Path:
    return archive_root.parent / "eval"


def run_suite(
    *,
    generation: int | None = None,
    mode: EvalMode | None = None,
    roles: tuple[Role, ...] = (Role.ATTACKER, Role.DEFENDER),
    archive_root: Path | None = None,
    checkpoint_root: Path | None = None,
    output: Path | None = None,
    config: SuiteConfig | None = None,
    execute: bool = False,
    mock: bool = False,
    ingest: tuple[Path, ...] = (),
    timeout_s: int = 0,
    strict: bool = False,
) -> dict[str, object]:
    suite = config or load_suite_config()
    if archive_root is None:
        pack = resolve()
        archive_root = pack.archive_root
    destination = output or (eval_root_for_archive(archive_root) / "benchmarks")
    resolved_mode = mode or mode_for_generation(generation, suite)
    stages = discover_iteration_stages(
        archive_root=archive_root,
        checkpoint_root=checkpoint_root,
        generation=generation,
        roles=roles,
    )
    jobs = build_jobs(stages, config=suite, mode=resolved_mode, output=destination)
    plan_path = write_plan(jobs, output=destination, mode=resolved_mode, generation=generation)
    rows = planned_scores(jobs)
    if mock:
        rows = mock_scores(jobs)
    ingested = []
    for path in ingest:
        ingested.extend(ingest_scores(path))
    if execute:
        rows = merge_scores(rows, execute_jobs(jobs, timeout_s=timeout_s))
    rows = merge_scores(rows, ingested)
    scores_path = write_scores(rows, destination)
    plots = write_plots(load_scores(scores_path), destination)
    missing = incomplete_arms(load_scores(scores_path))
    payload = {
        "mode": resolved_mode.value,
        "generation": generation,
        "plan": str(plan_path),
        "scores": str(scores_path),
        "svg": str(plots["svg"]),
        "html": str(plots["html"]),
        "markdown": str(plots["markdown"]),
        "stages": [stage.label for stage in stages],
        "job_count": len(jobs),
        "incomplete_arms": list(missing),
    }
    if strict and missing:
        raise BenchmarkError("incomplete benchmark arms: " + ", ".join(missing))
    return payload


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Score archived attacker and defender adapters on ExploitBench, DeepSWE, and Terminal-Bench."
    )
    parser.add_argument("--generation", type=int)
    parser.add_argument("--all", action="store_true", help="Scan every archived generation.")
    parser.add_argument("--mode", choices=[item.value for item in EvalMode])
    parser.add_argument("--role", choices=["both", "attacker", "defender"], default="both")
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--execute", action="store_true", help="Invoke installed harness CLIs when present.")
    parser.add_argument("--mock", action="store_true", help="Write deterministic placeholder scores for plumbing checks.")
    parser.add_argument("--ingest", type=Path, action="append", default=[])
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.generation is not None and args.generation < 0:
        parser.error("--generation must be >= 0")
    generation = None if args.all else args.generation
    roles: tuple[Role, ...]
    if args.role == "both":
        roles = (Role.ATTACKER, Role.DEFENDER)
    else:
        roles = (parse_role(args.role),)
    execute = args.execute or os.environ.get("ULTRON_BENCHMARK_EXECUTE") == "1"
    try:
        payload = run_suite(
            generation=generation,
            mode=parse_eval_mode(args.mode) if args.mode else None,
            roles=roles,
            archive_root=args.archive_dir,
            checkpoint_root=args.checkpoint_root,
            output=args.output,
            config=load_suite_config(args.config) if args.config else None,
            execute=execute,
            mock=args.mock,
            ingest=tuple(args.ingest),
            timeout_s=args.timeout,
            strict=args.strict,
        )
    except BenchmarkError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, indent=2))
    print(payload["svg"])


if __name__ == "__main__":
    _main()
