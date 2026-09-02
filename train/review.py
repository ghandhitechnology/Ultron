from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ultron.env.backend import IsolationBackend

from .bandpass import kill_switch_reason, select_profiles
from .pfsp import load_pool
from .schema_v1 import ReasonCode, Role, TrajectoryV1

ATTACKER_FILE = "attacker.jsonl"
DEFENDER_FILE = "defender.jsonl"
METRICS_FILE = "metrics.json"
REVIEW_JSON = "review.json"
REVIEW_MARKDOWN = "review.md"
PFSP_LIMIT = 8
WINDOW_COUNT = 10
INFRA_WARN = 0.05
AVAIL_WARN = 0.05
FORMAT_WARN = 0.80
ASR_SHIFT = 0.15
MIN_SHIFT_EPISODES = 20
TOOL_ERROR_WARN = 0.50
MIN_TOOL_CALLS = 20
METRICS_ASR_TOL = 0.02
EXAMPLE_LIMIT = 8
UNPAIRED_LIMIT = 20
EXPECTED_FILES = (ATTACKER_FILE, DEFENDER_FILE, METRICS_FILE)
_BLOCKS = "▁▂▃▄▅▆▇█"


class Phase(str, Enum):
    ROLLOUT = "rollout"
    COMPLETE = "complete"


class Verdict(str, Enum):
    USABLE = "usable"
    CAUTION = "caution"
    UNUSABLE = "unusable"


class FindingSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: FindingSeverity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FileReport:
    name: str
    path: str
    present: bool
    records: int
    parse_errors: int


@dataclass(frozen=True)
class ParseIssue:
    path: str
    line: int
    error: str


@dataclass(frozen=True)
class WindowStats:
    index: int
    start: int
    end: int
    episode_count: int
    asr: float | None
    reason_counts: dict[str, int]
    format_valid_rate: float | None
    infra_fail_rate: float | None
    availability_fail_rate: float | None
    mean_turns: float | None
    tool_error_rate: float | None


@dataclass(frozen=True)
class HalfStats:
    episode_count: int
    asr: float | None
    format_valid_rate: float | None
    infra_fail_rate: float | None


@dataclass(frozen=True)
class FlowReport:
    windows: tuple[WindowStats, ...]
    first_half: HalfStats
    second_half: HalfStats
    asr_delta: float | None
    sparkline: str


@dataclass(frozen=True)
class NamedSlice:
    name: str
    episode_count: int
    asr: float | None
    reason_counts: dict[str, int]


@dataclass(frozen=True)
class Outcomes:
    episode_count: int
    role_trajectories: int
    asr: float | None
    asr_excluding_infra: float | None
    reason_counts: dict[str, int]
    by_profile: tuple[NamedSlice, ...]
    by_opponent: tuple[NamedSlice, ...]


@dataclass(frozen=True)
class RewardReport:
    format_valid_count: int
    format_valid_rate: float | None
    mean_attacker_reward: float | None
    mean_defender_reward: float | None
    mean_attacker_turns: float | None
    mean_defender_turns: float | None
    subgoal_hit_counts: dict[str, int]
    decision_point_count: int
    episodes_with_no_tools: int


@dataclass(frozen=True)
class ToolNameStats:
    name: str
    calls: int
    failures: int
    unknown_exit: int
    duration_ms: int
    error_rate: float | None


@dataclass(frozen=True)
class ToolReport:
    calls: int
    failures: int
    unknown_exit: int
    duration_ms: int
    error_rate: float | None
    by_name: tuple[ToolNameStats, ...]


@dataclass(frozen=True)
class InfraReport:
    infra_fail_count: int
    availability_fail_count: int
    timeout_count: int
    infra_fail_rate: float | None
    availability_fail_rate: float | None
    timeout_rate: float | None
    availability_defender_wins: tuple[str, ...]
    unconfirmed_root_wins: tuple[str, ...]
    infra_nonzero_rewards: tuple[str, ...]


@dataclass(frozen=True)
class GateReport:
    reported_asr: float | None
    computed_asr: float | None
    kill_switch: str | None
    bandpass_profiles: tuple[str, ...]
    win_rates: dict[str, float]


@dataclass(frozen=True)
class EvalReport:
    expected_mode: str | None
    plan_path: str | None
    results_path: str | None
    metrics_path: str | None
    incomplete_arms: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveReport:
    manifest_path: str | None
    roles: tuple[str, ...]
    checkpoint_count: int | None


@dataclass(frozen=True)
class PfspReport:
    path: str | None
    present: bool
    generation: int | None
    size: int
    entries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Artifacts:
    metrics_path: str | None
    metrics_error: str | None
    metrics: dict[str, Any]
    eval: EvalReport
    archive: ArchiveReport
    pfsp: PfspReport
    eval_requested: bool
    archive_requested: bool
    pfsp_requested: bool


@dataclass(frozen=True)
class Completeness:
    files: tuple[FileReport, ...]
    parse_issues: tuple[ParseIssue, ...]
    unpaired_episode_ids: tuple[str, ...]
    unpaired_count: int
    missing_expected: tuple[str, ...]
    terminal_mismatch_ids: tuple[str, ...]
    drifted_groups: tuple[str, ...]
    generations: tuple[int, ...]


@dataclass(frozen=True)
class Identity:
    generation: int | None
    phase: Phase
    traces_dir: str
    episode_count: int
    role_trajectory_count: int
    roles: tuple[str, ...]
    profiles: tuple[str, ...]
    groups: int
    isolation_backends: tuple[str, ...]
    adapters: tuple[str, ...]
    opponents: tuple[str, ...]


@dataclass(frozen=True)
class JobReview:
    identity: Identity
    completeness: Completeness
    outcomes: Outcomes
    flow: FlowReport
    rewards: RewardReport
    tools: ToolReport
    infra: InfraReport
    gates: GateReport
    artifacts: Artifacts
    findings: tuple[Finding, ...]
    verdict: Verdict

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class _Episode:
    episode_id: str
    order: int
    generation: int
    profile_id: str
    group_id: str
    opponent_checkpoint_id: str
    isolation_backend: str
    reason_code: str
    attacker_euid: int
    host_confirmed_root: bool
    availability_ok: bool
    attacker_reward: float
    defender_reward: float
    roles: tuple[str, ...]
    terminal_mismatch: bool
    opponent_ids: tuple[str, ...]
    attacker_steps: int
    defender_steps: int
    format_valid: bool
    subgoal_hits: tuple[str, ...]
    decision_points: int
    tool_calls: int
    tool_failures: int
    tool_unknown_exit: int
    tool_duration_ms: int


@dataclass
class _ToolAcc:
    calls: int = 0
    failures: int = 0
    unknown_exit: int = 0
    duration_ms: int = 0


def review_traces(
    traces: Path,
    *,
    phase: Phase = Phase.COMPLETE,
    generation: int | None = None,
    window_count: int = WINDOW_COUNT,
    eval_dir: Path | None = None,
    archive_dir: Path | None = None,
    pfsp_path: Path | None = None,
) -> JobReview:
    traces = traces.resolve()
    files, parse_issues, trajectories = _load_traces(traces)
    episodes, unpaired = _pair_episodes(trajectories)
    inferred = _infer_generation(traces, episodes, generation)
    tools = _tool_report(trajectories)
    outcomes = _outcomes(episodes, len(trajectories))
    flow = _flow_report(episodes, window_count)
    rewards = _reward_report(episodes)
    infra = _infra_report(episodes)
    artifacts = _artifacts(
        traces,
        generation=inferred,
        eval_dir=eval_dir,
        archive_dir=archive_dir,
        pfsp_path=pfsp_path,
    )
    gates = _gate_report(outcomes, artifacts.metrics, inferred)
    identity = Identity(
        generation=inferred,
        phase=phase,
        traces_dir=str(traces),
        episode_count=len(episodes),
        role_trajectory_count=len(trajectories),
        roles=_unique(traj.role.value for traj in trajectories),
        profiles=_unique(ep.profile_id for ep in episodes),
        groups=len({ep.group_id for ep in episodes}),
        isolation_backends=_unique(ep.isolation_backend for ep in episodes),
        adapters=_unique(traj.adapter_id for traj in trajectories),
        opponents=_unique(ep.opponent_checkpoint_id for ep in episodes),
    )
    expected_missing = _missing_expected(traces, files)
    completeness = Completeness(
        files=tuple(files),
        parse_issues=tuple(parse_issues[:EXAMPLE_LIMIT]),
        unpaired_episode_ids=tuple(unpaired[:UNPAIRED_LIMIT]),
        unpaired_count=len(unpaired),
        missing_expected=expected_missing,
        terminal_mismatch_ids=tuple(
            item.episode_id for item in episodes if item.terminal_mismatch
        )[:EXAMPLE_LIMIT],
        drifted_groups=_drifted_groups(episodes),
        generations=_unique_ints(item.generation for item in episodes),
    )
    draft = JobReview(
        identity=identity,
        completeness=completeness,
        outcomes=outcomes,
        flow=flow,
        rewards=rewards,
        tools=tools,
        infra=infra,
        gates=gates,
        artifacts=artifacts,
        findings=(),
        verdict=Verdict.USABLE,
    )
    findings = tuple(finding for rule in RULES for finding in rule(draft))
    return replace(draft, findings=findings, verdict=verdict_from(findings))


def verdict_from(findings: Iterable[Finding]) -> Verdict:
    severities = {item.severity for item in findings}
    if FindingSeverity.BLOCK in severities:
        return Verdict.UNUSABLE
    if FindingSeverity.WARN in severities:
        return Verdict.CAUTION
    return Verdict.USABLE


def write_review(review: JobReview, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REVIEW_JSON
    markdown_path = output_dir / REVIEW_MARKDOWN
    json_path.write_text(json.dumps(review.to_dict(), indent=2) + "\n")
    markdown_path.write_text(render_markdown(review))
    return json_path, markdown_path


def render_markdown(review: JobReview) -> str:
    ident = review.identity
    generation = "unknown" if ident.generation is None else str(ident.generation)
    lines = [
        f"# Ultron job review gen {generation} ({ident.phase.value})",
        "",
        f"**Verdict.** `{review.verdict.value}`",
        "",
        f"**Episodes.** {ident.episode_count} unique. {ident.role_trajectory_count} role trajectories.",
        f"**ASR.** {_fmt_rate(review.outcomes.asr)} overall. {_fmt_rate(review.outcomes.asr_excluding_infra)} excluding infra-fail.",
        f"**Groups.** {ident.groups}. **Profiles.** {_join(ident.profiles) or 'none'}.",
        "",
        "## How the work went",
        "",
        f"ASR flow `{review.flow.sparkline}`",
        "",
        f"First half ASR {_fmt_rate(review.flow.first_half.asr)} across {review.flow.first_half.episode_count} episodes.",
        f"Second half ASR {_fmt_rate(review.flow.second_half.asr)} across {review.flow.second_half.episode_count} episodes.",
        f"ASR delta (second minus first) {_fmt_signed(review.flow.asr_delta)}.",
        "",
        "| window | n | ASR | hold | root | timeout | infra | avail | format |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window in review.flow.windows:
        reasons = window.reason_counts
        lines.append(
            "| {index} | {n} | {asr} `{bar}` | {hold} | {root} | {timeout} | {infra} | {avail} | {fmt} |".format(
                index=window.index,
                n=window.episode_count,
                asr=_fmt_rate(window.asr),
                bar=_bar(window.asr, 8),
                hold=reasons.get(ReasonCode.DEFENDER_HOLD.value, 0),
                root=reasons.get(ReasonCode.ATTACKER_ROOT.value, 0),
                timeout=reasons.get(ReasonCode.TIMEOUT.value, 0),
                infra=reasons.get(ReasonCode.INFRA_FAIL.value, 0),
                avail=reasons.get(ReasonCode.AVAILABILITY_FAIL.value, 0),
                fmt=_fmt_rate(window.format_valid_rate),
            )
        )
    lines.extend(
        [
            "",
            "## Outcomes",
            "",
            "| reason | count |",
            "| --- | ---: |",
        ]
    )
    for reason, count in review.outcomes.reason_counts.items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "### By profile", "", "| profile | n | ASR |", "| --- | ---: | ---: |"])
    for item in review.outcomes.by_profile:
        lines.append(f"| `{item.name}` | {item.episode_count} | {_fmt_rate(item.asr)} |")
    lines.extend(["", "### By opponent", "", "| opponent | n | ASR |", "| --- | ---: | ---: |"])
    for item in review.outcomes.by_opponent:
        lines.append(f"| `{item.name}` | {item.episode_count} | {_fmt_rate(item.asr)} |")
    lines.extend(
        [
            "",
            "## Reward health",
            "",
            f"Format-valid rate {_fmt_rate(review.rewards.format_valid_rate)} ({review.rewards.format_valid_count}/{review.identity.episode_count}).",
            f"Mean attacker reward {_fmt_rate(review.rewards.mean_attacker_reward)}. Mean defender reward {_fmt_rate(review.rewards.mean_defender_reward)}.",
            f"Mean attacker turns {_fmt_mean(review.rewards.mean_attacker_turns)}. Mean defender turns {_fmt_mean(review.rewards.mean_defender_turns)}.",
            f"Decision points {review.rewards.decision_point_count}. Episodes with no tool calls {review.rewards.episodes_with_no_tools}.",
            "",
            "| subgoal | first-hit episodes |",
            "| --- | ---: |",
        ]
    )
    if review.rewards.subgoal_hit_counts:
        for name, count in review.rewards.subgoal_hit_counts.items():
            lines.append(f"| `{name}` | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            "## Tool flow",
            "",
            f"Calls {review.tools.calls}. Failures {review.tools.failures}. Unknown exits {review.tools.unknown_exit}.",
            f"Error rate {_fmt_rate(review.tools.error_rate)}. Duration ms {review.tools.duration_ms}.",
            "",
            "| tool | calls | failures | unknown | error rate | duration ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if review.tools.by_name:
        for item in review.tools.by_name:
            lines.append(
                f"| `{item.name}` | {item.calls} | {item.failures} | {item.unknown_exit} | {_fmt_rate(item.error_rate)} | {item.duration_ms} |"
            )
    else:
        lines.append("| none | 0 | 0 | 0 | n/a | 0 |")
    lines.extend(
        [
            "",
            "## Infra",
            "",
            f"Infra-fail {_fmt_rate(review.infra.infra_fail_rate)} ({review.infra.infra_fail_count}).",
            f"Availability-fail {_fmt_rate(review.infra.availability_fail_rate)} ({review.infra.availability_fail_count}).",
            f"Timeout {_fmt_rate(review.infra.timeout_rate)} ({review.infra.timeout_count}).",
            "",
            "## Completeness",
            "",
            "| file | present | records | parse errors |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for item in review.completeness.files:
        lines.append(
            f"| `{item.name}` | {str(item.present).lower()} | {item.records} | {item.parse_errors} |"
        )
    missing = _join(f"`{name}`" for name in review.completeness.missing_expected) or "none"
    unpaired = review.completeness.unpaired_count
    lines.extend(
        [
            "",
            f"Missing expected files {missing}.",
            f"Unpaired episodes {unpaired}.",
            f"Parse issues shown {len(review.completeness.parse_issues)} of all collected.",
            "",
            "## Gates and artifacts",
            "",
            f"Reported ASR {_fmt_rate(review.gates.reported_asr)}. Computed ASR {_fmt_rate(review.gates.computed_asr)}.",
            f"Kill switch {review.gates.kill_switch or 'clear'}.",
            f"Bandpass profiles {_join(review.gates.bandpass_profiles) or 'none'}.",
            f"Eval mode {review.artifacts.eval.expected_mode or 'none'}. Plan {review.artifacts.eval.plan_path or 'missing'}. Results {review.artifacts.eval.results_path or 'missing'}.",
            f"Archive manifest {review.artifacts.archive.manifest_path or 'missing'}.",
            f"PFSP entries {review.artifacts.pfsp.size}. Present {str(review.artifacts.pfsp.present).lower()}.",
            "",
            "## Findings",
            "",
        ]
    )
    if not review.findings:
        lines.append("No findings.")
    else:
        lines.append("| severity | code | message |")
        lines.append("| --- | --- | --- |")
        for item in review.findings:
            message = item.message.replace("|", "/")
            lines.append(f"| `{item.severity.value}` | `{item.code}` | {message} |")
    lines.extend(
        [
            "",
            "## Next",
            "",
            f"Machine-readable copy is `{REVIEW_JSON}` next to this file.",
            "Re-run with `--strict` to exit nonzero on BLOCK findings.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_traces(
    traces: Path,
) -> tuple[list[FileReport], list[ParseIssue], list[TrajectoryV1]]:
    files: list[FileReport] = []
    parse_issues: list[ParseIssue] = []
    trajectories: list[TrajectoryV1] = []
    jsonl_paths = _jsonl_paths(traces)
    seen: set[str] = set()
    for path in jsonl_paths:
        records, issues = _read_jsonl(path)
        files.append(
            FileReport(
                name=path.name,
                path=str(path),
                present=True,
                records=len(records),
                parse_errors=len(issues),
            )
        )
        parse_issues.extend(issues)
        trajectories.extend(records)
        seen.add(path.name)
    if traces.is_dir():
        for name in EXPECTED_FILES:
            if name in seen:
                continue
            path = traces / name
            if name == METRICS_FILE:
                files.append(
                    FileReport(
                        name=name,
                        path=str(path),
                        present=path.is_file(),
                        records=0,
                        parse_errors=0,
                    )
                )
                continue
            files.append(
                FileReport(name=name, path=str(path), present=False, records=0, parse_errors=0)
            )
    return files, parse_issues, trajectories


def _jsonl_paths(traces: Path) -> list[Path]:
    if traces.is_file():
        return [traces]
    if not traces.is_dir():
        return []
    return sorted(path for path in traces.glob("*.jsonl") if path.is_file())


def _read_jsonl(path: Path) -> tuple[list[TrajectoryV1], list[ParseIssue]]:
    records: list[TrajectoryV1] = []
    issues: list[ParseIssue] = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(TrajectoryV1.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                issues.append(ParseIssue(path=str(path), line=line_number, error=str(exc)))
    return records, issues


def _pair_episodes(
    trajectories: list[TrajectoryV1],
) -> tuple[list[_Episode], list[str]]:
    grouped: dict[str, dict[str, TrajectoryV1]] = {}
    order: list[str] = []
    for traj in trajectories:
        bucket = grouped.setdefault(traj.episode_id, {})
        if traj.episode_id not in order:
            order.append(traj.episode_id)
        bucket[traj.role.value] = traj
    episodes = [
        _episode_from_roles(index, episode_id, grouped[episode_id])
        for index, episode_id in enumerate(order)
    ]
    unpaired = [
        episode_id
        for episode_id in order
        if set(grouped[episode_id]) != {Role.ATTACKER.value, Role.DEFENDER.value}
    ]
    return episodes, unpaired


def _episode_from_roles(order: int, episode_id: str, roles: Mapping[str, TrajectoryV1]) -> _Episode:
    primary = roles.get(Role.ATTACKER.value) or next(iter(roles.values()))
    terminals = [item.terminal for item in roles.values()]
    terminal_mismatch = any(
        (
            item.reason_code,
            item.attacker_reward,
            item.defender_reward,
            item.host_confirmed_root,
            item.availability_ok,
            item.attacker_euid,
        )
        != (
            primary.terminal.reason_code,
            primary.terminal.attacker_reward,
            primary.terminal.defender_reward,
            primary.terminal.host_confirmed_root,
            primary.terminal.availability_ok,
            primary.terminal.attacker_euid,
        )
        for item in terminals
    )
    attacker = roles.get(Role.ATTACKER.value)
    defender = roles.get(Role.DEFENDER.value)
    steps = [step for traj in roles.values() for step in traj.steps]
    tool_calls = 0
    tool_failures = 0
    tool_unknown = 0
    tool_duration = 0
    for step in steps:
        for event in step.tool_events:
            tool_calls += 1
            tool_duration += event.duration_ms
            if event.exit_code is None:
                tool_unknown += 1
            elif event.exit_code != 0:
                tool_failures += 1
    hits: list[str] = []
    seen_hits: set[str] = set()
    source_steps = attacker.steps if attacker is not None else []
    for step in source_steps:
        for hit in step.subgoal_hits:
            if hit in seen_hits:
                continue
            seen_hits.add(hit)
            hits.append(hit)
    return _Episode(
        episode_id=episode_id,
        order=order,
        generation=primary.generation,
        profile_id=primary.profile_id,
        group_id=primary.group_id,
        opponent_checkpoint_id=primary.opponent_checkpoint_id,
        isolation_backend=primary.isolation_backend.value
        if isinstance(primary.isolation_backend, IsolationBackend)
        else str(primary.isolation_backend),
        reason_code=primary.terminal.reason_code.value
        if isinstance(primary.terminal.reason_code, ReasonCode)
        else str(primary.terminal.reason_code),
        attacker_euid=primary.terminal.attacker_euid,
        host_confirmed_root=primary.terminal.host_confirmed_root,
        availability_ok=primary.terminal.availability_ok,
        attacker_reward=primary.terminal.attacker_reward,
        defender_reward=primary.terminal.defender_reward,
        roles=tuple(sorted(roles)),
        terminal_mismatch=terminal_mismatch,
        opponent_ids=_unique(item.opponent_checkpoint_id for item in roles.values()),
        attacker_steps=len(attacker.steps) if attacker is not None else 0,
        defender_steps=len(defender.steps) if defender is not None else 0,
        format_valid=all(step.format_valid for step in steps) if steps else True,
        subgoal_hits=tuple(hits),
        decision_points=sum(1 for step in source_steps if step.decision_point),
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        tool_unknown_exit=tool_unknown,
        tool_duration_ms=tool_duration,
    )


def _outcomes(episodes: list[_Episode], role_trajectories: int) -> Outcomes:
    return Outcomes(
        episode_count=len(episodes),
        role_trajectories=role_trajectories,
        asr=_asr(episodes),
        asr_excluding_infra=_asr(episodes, exclude_infra=True),
        reason_counts=_reason_counts(episodes),
        by_profile=_named_slices(episodes, lambda item: item.profile_id),
        by_opponent=_named_slices(episodes, lambda item: item.opponent_checkpoint_id),
    )


def _flow_report(episodes: list[_Episode], window_count: int) -> FlowReport:
    windows = tuple(
        WindowStats(
            index=index,
            start=chunk[0].order if chunk else 0,
            end=(chunk[-1].order + 1) if chunk else 0,
            episode_count=len(chunk),
            asr=_asr(chunk),
            reason_counts=_reason_counts(chunk),
            format_valid_rate=_rate(sum(1 for item in chunk if item.format_valid), len(chunk)),
            infra_fail_rate=_rate(_count_reason(chunk, ReasonCode.INFRA_FAIL), len(chunk)),
            availability_fail_rate=_rate(
                _count_reason(chunk, ReasonCode.AVAILABILITY_FAIL), len(chunk)
            ),
            mean_turns=_mean([float(item.attacker_steps) for item in chunk]),
            tool_error_rate=_tool_error_rate(chunk),
        )
        for index, chunk in enumerate(_split_even(episodes, window_count), start=1)
    )
    mid = len(episodes) // 2
    first = episodes[:mid]
    second = episodes[mid:]
    first_half = _half(first)
    second_half = _half(second)
    delta = None
    if first_half.asr is not None and second_half.asr is not None:
        delta = second_half.asr - first_half.asr
    return FlowReport(
        windows=windows,
        first_half=first_half,
        second_half=second_half,
        asr_delta=delta,
        sparkline=_sparkline([item.asr for item in windows]),
    )


def _half(episodes: list[_Episode]) -> HalfStats:
    total = len(episodes)
    return HalfStats(
        episode_count=total,
        asr=_asr(episodes),
        format_valid_rate=_rate(sum(1 for item in episodes if item.format_valid), total),
        infra_fail_rate=_rate(_count_reason(episodes, ReasonCode.INFRA_FAIL), total),
    )


def _reward_report(episodes: list[_Episode]) -> RewardReport:
    subgoals: dict[str, int] = {}
    for item in episodes:
        for hit in item.subgoal_hits:
            subgoals[hit] = subgoals.get(hit, 0) + 1
    return RewardReport(
        format_valid_count=sum(1 for item in episodes if item.format_valid),
        format_valid_rate=_rate(sum(1 for item in episodes if item.format_valid), len(episodes)),
        mean_attacker_reward=_mean([item.attacker_reward for item in episodes]),
        mean_defender_reward=_mean([item.defender_reward for item in episodes]),
        mean_attacker_turns=_mean([float(item.attacker_steps) for item in episodes]),
        mean_defender_turns=_mean([float(item.defender_steps) for item in episodes]),
        subgoal_hit_counts=dict(sorted(subgoals.items())),
        decision_point_count=sum(item.decision_points for item in episodes),
        episodes_with_no_tools=sum(1 for item in episodes if item.tool_calls == 0),
    )


def _tool_report(trajectories: list[TrajectoryV1]) -> ToolReport:
    by_name: dict[str, _ToolAcc] = defaultdict(_ToolAcc)
    total = _ToolAcc()
    for traj in trajectories:
        for step in traj.steps:
            for event in step.tool_events:
                acc = by_name[event.name]
                acc.calls += 1
                acc.duration_ms += event.duration_ms
                total.calls += 1
                total.duration_ms += event.duration_ms
                if event.exit_code is None:
                    acc.unknown_exit += 1
                    total.unknown_exit += 1
                elif event.exit_code != 0:
                    acc.failures += 1
                    total.failures += 1
    names = tuple(
        ToolNameStats(
            name=name,
            calls=acc.calls,
            failures=acc.failures,
            unknown_exit=acc.unknown_exit,
            duration_ms=acc.duration_ms,
            error_rate=_rate(acc.failures, acc.calls),
        )
        for name, acc in sorted(by_name.items(), key=lambda item: (-item[1].calls, item[0]))
    )
    return ToolReport(
        calls=total.calls,
        failures=total.failures,
        unknown_exit=total.unknown_exit,
        duration_ms=total.duration_ms,
        error_rate=_rate(total.failures, total.calls),
        by_name=names,
    )


def _infra_report(episodes: list[_Episode]) -> InfraReport:
    total = len(episodes)
    avail_wins = tuple(
        item.episode_id
        for item in episodes
        if item.reason_code == ReasonCode.AVAILABILITY_FAIL.value and item.defender_reward > 0
    )
    unconfirmed = tuple(
        item.episode_id
        for item in episodes
        if item.reason_code == ReasonCode.ATTACKER_ROOT.value and not item.host_confirmed_root
    )
    infra_reward = tuple(
        item.episode_id
        for item in episodes
        if item.reason_code == ReasonCode.INFRA_FAIL.value
        and (item.attacker_reward != 0 or item.defender_reward != 0)
    )
    infra_count = _count_reason(episodes, ReasonCode.INFRA_FAIL)
    avail_count = _count_reason(episodes, ReasonCode.AVAILABILITY_FAIL)
    timeout_count = _count_reason(episodes, ReasonCode.TIMEOUT)
    return InfraReport(
        infra_fail_count=infra_count,
        availability_fail_count=avail_count,
        timeout_count=timeout_count,
        infra_fail_rate=_rate(infra_count, total),
        availability_fail_rate=_rate(avail_count, total),
        timeout_rate=_rate(timeout_count, total),
        availability_defender_wins=avail_wins[:EXAMPLE_LIMIT],
        unconfirmed_root_wins=unconfirmed[:EXAMPLE_LIMIT],
        infra_nonzero_rewards=infra_reward[:EXAMPLE_LIMIT],
    )


def _artifacts(
    traces: Path,
    *,
    generation: int | None,
    eval_dir: Path | None,
    archive_dir: Path | None,
    pfsp_path: Path | None,
) -> Artifacts:
    metrics_path = traces / METRICS_FILE if traces.is_dir() else traces.parent / METRICS_FILE
    metrics, metrics_error = _load_metrics(metrics_path)
    eval_report = _eval_report(eval_dir, generation) if eval_dir is not None else _empty_eval()
    archive = (
        _archive_report(archive_dir, generation)
        if archive_dir is not None
        else ArchiveReport(None, (), None)
    )
    pfsp = _pfsp_report(pfsp_path) if pfsp_path is not None else PfspReport(None, False, None, 0, ())
    return Artifacts(
        metrics_path=str(metrics_path) if metrics_path.is_file() else None,
        metrics_error=metrics_error if not metrics_path.is_file() or metrics is None else None,
        metrics=metrics or {},
        eval=eval_report,
        archive=archive,
        pfsp=pfsp,
        eval_requested=eval_dir is not None,
        archive_requested=archive_dir is not None,
        pfsp_requested=pfsp_path is not None,
    )


def _gate_report(outcomes: Outcomes, metrics: Mapping[str, Any], generation: int | None) -> GateReport:
    reported = None
    if "asr" in metrics:
        try:
            reported = float(metrics["asr"])
        except (TypeError, ValueError):
            reported = None
    computed = outcomes.asr
    kill = None
    asr_for_gate = reported if reported is not None else computed
    if asr_for_gate is not None and generation is not None:
        kill = kill_switch_reason(asr_for_gate, generation)
    win_rates = {item.name: item.asr for item in outcomes.by_profile if item.asr is not None}
    profiles = (
        tuple(select_profiles(win_rates, generation=generation))
        if generation is not None and win_rates
        else tuple(win_rates)
    )
    return GateReport(
        reported_asr=reported,
        computed_asr=computed,
        kill_switch=kill,
        bandpass_profiles=profiles,
        win_rates=win_rates,
    )


def _eval_report(eval_dir: Path, generation: int | None) -> EvalReport:
    mode = _eval_mode(generation)
    plan = eval_dir / f"tier3_{mode}_plan.json" if mode else None
    results = eval_dir / f"tier3_{mode}_results.json" if mode else None
    metrics = eval_dir / f"tier3_{mode}_metrics.json" if mode else None
    incomplete: tuple[str, ...] = ()
    if results is not None and results.is_file():
        try:
            payload = json.loads(results.read_text())
        except json.JSONDecodeError:
            payload = None
        incomplete = _incomplete_arms(payload)
    return EvalReport(
        expected_mode=mode,
        plan_path=str(plan) if plan is not None and plan.is_file() else None,
        results_path=str(results) if results is not None and results.is_file() else None,
        metrics_path=str(metrics) if metrics is not None and metrics.is_file() else None,
        incomplete_arms=incomplete,
    )


def _empty_eval() -> EvalReport:
    return EvalReport(None, None, None, None, ())


def _archive_report(archive_dir: Path, generation: int | None) -> ArchiveReport:
    if generation is None:
        return ArchiveReport(None, (), None)
    manifest_path = archive_dir / f"gen{generation}" / "manifest.json"
    if not manifest_path.is_file():
        return ArchiveReport(None, (), None)
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return ArchiveReport(str(manifest_path), (), None)
    roles = tuple(sorted(payload.get("roles", {}))) if isinstance(payload, dict) else ()
    checkpoints = payload.get("checkpoints") if isinstance(payload, dict) else None
    count = len(checkpoints) if isinstance(checkpoints, list) else None
    return ArchiveReport(str(manifest_path), roles, count)


def _pfsp_report(path: Path) -> PfspReport:
    if not path.is_file():
        return PfspReport(str(path), False, None, 0, ())
    payload = json.loads(path.read_text()) if path.is_file() else {}
    entries = load_pool(path)
    generation = payload.get("generation") if isinstance(payload, dict) else None
    return PfspReport(
        path=str(path),
        present=True,
        generation=generation if isinstance(generation, int) else None,
        size=len(entries),
        entries=tuple(entry.to_dict() for entry in entries),
    )


def _load_metrics(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "metrics.json must be an object"
    return payload, None


def _missing_expected(traces: Path, files: list[FileReport]) -> tuple[str, ...]:
    if not traces.is_dir():
        return ()
    present = {item.name for item in files if item.present}
    return tuple(name for name in EXPECTED_FILES if name not in present)


def _infer_generation(
    traces: Path, episodes: list[_Episode], requested: int | None
) -> int | None:
    if requested is not None:
        return requested
    if episodes:
        counts: dict[int, int] = {}
        for item in episodes:
            counts[item.generation] = counts.get(item.generation, 0) + 1
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    match = re.search(r"gen(\d+)(?:/|$)", traces.as_posix())
    if match:
        return int(match.group(1))
    return None


def _eval_mode(generation: int | None) -> str | None:
    if generation == 2:
        return "light"
    if generation == 4:
        return "full"
    return None


def _incomplete_arms(payload: Any) -> tuple[str, ...]:
    if isinstance(payload, Mapping):
        arms = payload.get("incomplete_arms", [])
        if isinstance(arms, list):
            return tuple(str(item) for item in arms)
    return ()


def _asr(episodes: list[_Episode], *, exclude_infra: bool = False) -> float | None:
    pool = [
        item
        for item in episodes
        if not (exclude_infra and item.reason_code == ReasonCode.INFRA_FAIL.value)
    ]
    if not pool:
        return None
    wins = sum(1 for item in pool if item.reason_code == ReasonCode.ATTACKER_ROOT.value)
    return wins / len(pool)


def _reason_counts(episodes: list[_Episode]) -> dict[str, int]:
    counts = {code.value: 0 for code in ReasonCode}
    for item in episodes:
        counts[item.reason_code] = counts.get(item.reason_code, 0) + 1
    return counts


def _named_slices(
    episodes: list[_Episode], key: Callable[[_Episode], str]
) -> tuple[NamedSlice, ...]:
    grouped: dict[str, list[_Episode]] = {}
    for item in episodes:
        grouped.setdefault(key(item), []).append(item)
    return tuple(
        NamedSlice(
            name=name,
            episode_count=len(items),
            asr=_asr(items),
            reason_counts=_reason_counts(items),
        )
        for name, items in sorted(grouped.items())
    )


def _split_even(items: list[_Episode], window_count: int) -> list[list[_Episode]]:
    if not items:
        return []
    count = min(max(window_count, 1), len(items))
    base, extra = divmod(len(items), count)
    windows: list[list[_Episode]] = []
    index = 0
    for window_index in range(count):
        take = base + (1 if window_index < extra else 0)
        windows.append(items[index : index + take])
        index += take
    return windows


def _count_reason(episodes: list[_Episode], reason: ReasonCode) -> int:
    return sum(1 for item in episodes if item.reason_code == reason.value)


def _tool_error_rate(episodes: list[_Episode]) -> float | None:
    calls = sum(item.tool_calls for item in episodes)
    failures = sum(item.tool_failures for item in episodes)
    return _rate(failures, calls)


def _rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return count / total


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _unique_ints(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(values))


def _drifted_groups(episodes: list[_Episode]) -> tuple[str, ...]:
    grouped: dict[str, set[str]] = {}
    for item in episodes:
        grouped.setdefault(item.group_id, set()).update(item.opponent_ids)
        grouped[item.group_id].add(item.opponent_checkpoint_id)
    return tuple(sorted(group_id for group_id, opponents in grouped.items() if len(opponents) > 1))


def _join(values: Iterable[str]) -> str:
    return ", ".join(values)


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_mean(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _fmt_signed(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _bar(value: float | None, width: int = 12) -> str:
    if value is None:
        return "·" * width
    filled = round(max(0.0, min(1.0, value)) * width)
    return "█" * filled + "░" * (width - filled)


def _sparkline(values: list[float | None]) -> str:
    chars: list[str] = []
    for value in values:
        if value is None:
            chars.append("·")
            continue
        bounded = max(0.0, min(1.0, value))
        index = min(len(_BLOCKS) - 1, int(bounded * len(_BLOCKS)))
        chars.append(_BLOCKS[index])
    return "".join(chars) if chars else "·"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _finding(
    code: str,
    severity: FindingSeverity,
    message: str,
    **evidence: Any,
) -> Finding:
    return Finding(code=code, severity=severity, message=message, evidence=evidence)


def _rule_empty(review: JobReview) -> list[Finding]:
    if review.identity.episode_count:
        return []
    return [
        _finding(
            "empty_traces",
            FindingSeverity.BLOCK,
            "No valid episodes were parsed from the trace directory.",
            files=[item.name for item in review.completeness.files],
        )
    ]


def _rule_parse_errors(review: JobReview) -> list[Finding]:
    total = sum(item.parse_errors for item in review.completeness.files)
    if not total:
        return []
    severity = (
        FindingSeverity.BLOCK if review.identity.episode_count == 0 else FindingSeverity.WARN
    )
    return [
        _finding(
            "parse_errors",
            severity,
            f"{total} trajectory line(s) failed to parse.",
            count=total,
            examples=[issue.__dict__ for issue in review.completeness.parse_issues],
        )
    ]


def _rule_missing_expected(review: JobReview) -> list[Finding]:
    missing = review.completeness.missing_expected
    if not missing:
        return []
    jsonl_missing = tuple(name for name in missing if name.endswith(".jsonl"))
    findings: list[Finding] = []
    if jsonl_missing:
        findings.append(
            _finding(
                "missing_role_file",
                FindingSeverity.WARN,
                "Expected role JSONL file(s) are missing.",
                missing=list(jsonl_missing),
            )
        )
    if METRICS_FILE in missing:
        severity = (
            FindingSeverity.INFO
            if review.identity.phase is Phase.ROLLOUT
            else FindingSeverity.WARN
        )
        findings.append(
            _finding(
                "missing_metrics",
                severity,
                "metrics.json is missing. ASR is computed from trajectories only.",
            )
        )
    return findings


def _rule_unpaired(review: JobReview) -> list[Finding]:
    if not review.completeness.unpaired_count:
        return []
    return [
        _finding(
            "unpaired_episodes",
            FindingSeverity.WARN,
            f"{review.completeness.unpaired_count} episode(s) lack a paired attacker and defender trajectory.",
            count=review.completeness.unpaired_count,
            examples=list(review.completeness.unpaired_episode_ids),
        )
    ]


def _rule_terminal_mismatch(review: JobReview) -> list[Finding]:
    ids = review.completeness.terminal_mismatch_ids
    if not ids:
        return []
    return [
        _finding(
            "paired_terminal_mismatch",
            FindingSeverity.BLOCK,
            "Paired attacker and defender trajectories disagree on the terminal outcome.",
            examples=list(ids),
        )
    ]


def _rule_integrity(review: JobReview) -> list[Finding]:
    findings: list[Finding] = []
    if review.infra.availability_defender_wins:
        findings.append(
            _finding(
                "availability_defender_win",
                FindingSeverity.BLOCK,
                "Availability failures awarded a defender reward.",
                examples=list(review.infra.availability_defender_wins),
            )
        )
    if review.infra.infra_nonzero_rewards:
        findings.append(
            _finding(
                "infra_nonzero_reward",
                FindingSeverity.BLOCK,
                "Infra-fail episodes have a nonzero role reward.",
                examples=list(review.infra.infra_nonzero_rewards),
            )
        )
    if review.infra.unconfirmed_root_wins:
        findings.append(
            _finding(
                "unconfirmed_root_win",
                FindingSeverity.BLOCK,
                "ATTACKER_ROOT appears without host confirmation.",
                examples=list(review.infra.unconfirmed_root_wins),
            )
        )
    return findings


def _rule_group_opponent(review: JobReview) -> list[Finding]:
    drifted = review.completeness.drifted_groups
    if not drifted:
        return []
    return [
        _finding(
            "group_opponent_drift",
            FindingSeverity.BLOCK,
            "A GRPO group used more than one opponent checkpoint.",
            groups=list(drifted),
        )
    ]


def _rule_asr_stuck(review: JobReview) -> list[Finding]:
    asr = review.gates.computed_asr
    generation = review.identity.generation
    if asr is None or generation is None:
        return []
    reason = kill_switch_reason(asr, generation)
    if reason:
        return [_finding("asr_stuck", FindingSeverity.BLOCK, reason, asr=asr, generation=generation)]
    if generation == 0 and asr in (0.0, 1.0) and review.identity.episode_count:
        return [
            _finding(
                "asr_extreme_gen0",
                FindingSeverity.INFO,
                f"Generation 0 ASR is {asr:.1f}. Confirm this is a plumbing sample rather than a research run.",
                asr=asr,
            )
        ]
    return []


def _rule_metrics_mismatch(review: JobReview) -> list[Finding]:
    reported = review.gates.reported_asr
    computed = review.gates.computed_asr
    if reported is None or computed is None:
        return []
    if abs(reported - computed) <= METRICS_ASR_TOL:
        return []
    return [
        _finding(
            "metrics_asr_mismatch",
            FindingSeverity.WARN,
            "metrics.json ASR disagrees with trajectory ASR.",
            reported=reported,
            computed=computed,
        )
    ]


def _rule_rates(review: JobReview) -> list[Finding]:
    findings: list[Finding] = []
    if review.infra.infra_fail_rate is not None and review.infra.infra_fail_rate >= INFRA_WARN:
        findings.append(
            _finding(
                "high_infra_fail",
                FindingSeverity.WARN,
                f"Infra-fail rate {review.infra.infra_fail_rate:.3f} is at least {INFRA_WARN:.2f}.",
                rate=review.infra.infra_fail_rate,
                count=review.infra.infra_fail_count,
            )
        )
    if (
        review.infra.availability_fail_rate is not None
        and review.infra.availability_fail_rate >= AVAIL_WARN
    ):
        findings.append(
            _finding(
                "high_availability_fail",
                FindingSeverity.WARN,
                f"Availability-fail rate {review.infra.availability_fail_rate:.3f} is at least {AVAIL_WARN:.2f}.",
                rate=review.infra.availability_fail_rate,
                count=review.infra.availability_fail_count,
            )
        )
    if (
        review.rewards.format_valid_rate is not None
        and review.rewards.format_valid_rate < FORMAT_WARN
    ):
        findings.append(
            _finding(
                "format_collapse",
                FindingSeverity.WARN,
                f"Format-valid rate {review.rewards.format_valid_rate:.3f} is below {FORMAT_WARN:.2f}.",
                rate=review.rewards.format_valid_rate,
            )
        )
    return findings


def _rule_flow_shift(review: JobReview) -> list[Finding]:
    delta = review.flow.asr_delta
    n = review.identity.episode_count
    if delta is None or n < MIN_SHIFT_EPISODES:
        return []
    if delta <= -ASR_SHIFT:
        return [
            _finding(
                "late_asr_drop",
                FindingSeverity.WARN,
                f"ASR fell by {abs(delta):.3f} from the first half to the second half.",
                delta=delta,
                first=review.flow.first_half.asr,
                second=review.flow.second_half.asr,
            )
        ]
    if delta >= ASR_SHIFT:
        return [
            _finding(
                "late_asr_rise",
                FindingSeverity.INFO,
                f"ASR rose by {delta:.3f} from the first half to the second half.",
                delta=delta,
                first=review.flow.first_half.asr,
                second=review.flow.second_half.asr,
            )
        ]
    return []


def _rule_tools(review: JobReview) -> list[Finding]:
    findings: list[Finding] = []
    usable = review.identity.episode_count - review.infra.infra_fail_count
    if usable > 0 and review.rewards.episodes_with_no_tools >= max(1, usable // 2):
        findings.append(
            _finding(
                "no_tool_events",
                FindingSeverity.WARN,
                "Many non-empty episodes have no tool calls.",
                episodes_with_no_tools=review.rewards.episodes_with_no_tools,
                episodes=review.identity.episode_count,
            )
        )
    if (
        review.tools.calls >= MIN_TOOL_CALLS
        and review.tools.error_rate is not None
        and review.tools.error_rate >= TOOL_ERROR_WARN
    ):
        findings.append(
            _finding(
                "high_tool_error_rate",
                FindingSeverity.WARN,
                f"Tool error rate {review.tools.error_rate:.3f} is at least {TOOL_ERROR_WARN:.2f}.",
                rate=review.tools.error_rate,
                calls=review.tools.calls,
            )
        )
    return findings


def _rule_eval(review: JobReview) -> list[Finding]:
    if review.identity.phase is Phase.ROLLOUT:
        return []
    findings: list[Finding] = []
    mode = review.artifacts.eval.expected_mode
    if mode and review.artifacts.eval.plan_path is None:
        findings.append(
            _finding(
                "eval_plan_missing",
                FindingSeverity.WARN,
                f"Generation {review.identity.generation} expects a {mode} eval plan and none is present.",
                mode=mode,
            )
        )
    if review.artifacts.eval.plan_path and review.artifacts.eval.results_path is None:
        findings.append(
            _finding(
                "eval_results_missing",
                FindingSeverity.WARN,
                "An eval plan exists without result rows. Incomplete arms must be recorded, not dropped.",
                plan=review.artifacts.eval.plan_path,
            )
        )
    if review.artifacts.eval.incomplete_arms:
        findings.append(
            _finding(
                "eval_arm_incomplete",
                FindingSeverity.WARN,
                "Eval results list incomplete arms.",
                arms=list(review.artifacts.eval.incomplete_arms),
            )
        )
    return findings


def _rule_generation_mismatch(review: JobReview) -> list[Finding]:
    generations = review.completeness.generations
    inferred = review.identity.generation
    if len(generations) <= 1 and (not generations or inferred in generations or inferred is None):
        return []
    return [
        _finding(
            "generation_mismatch",
            FindingSeverity.WARN,
            "Trajectories do not share one generation id.",
            generations=list(generations),
            inferred=inferred,
        )
    ]


def _rule_archive_pfsp(review: JobReview) -> list[Finding]:
    if review.identity.phase is Phase.ROLLOUT:
        return []
    findings: list[Finding] = []
    if review.artifacts.pfsp_requested and not review.artifacts.pfsp.present:
        findings.append(
            _finding(
                "pfsp_missing",
                FindingSeverity.WARN,
                "PFSP manifest path was supplied but the file is missing.",
                path=review.artifacts.pfsp.path,
            )
        )
    if review.artifacts.pfsp.present and review.artifacts.pfsp.size == 0:
        findings.append(
            _finding(
                "pfsp_empty",
                FindingSeverity.WARN,
                "PFSP pool is present and empty.",
                path=review.artifacts.pfsp.path,
            )
        )
    if review.artifacts.pfsp.size > PFSP_LIMIT:
        findings.append(
            _finding(
                "pfsp_over_limit",
                FindingSeverity.WARN,
                f"PFSP pool has {review.artifacts.pfsp.size} entries. Limit is {PFSP_LIMIT}.",
                size=review.artifacts.pfsp.size,
            )
        )
    return findings


def _rule_archive_missing(review: JobReview) -> list[Finding]:
    if review.identity.phase is Phase.ROLLOUT:
        return []
    if not review.artifacts.archive_requested:
        return []
    if review.artifacts.archive.manifest_path is not None:
        return []
    if review.identity.generation is None:
        return []
    return [
        _finding(
            "archive_missing",
            FindingSeverity.WARN,
            "Complete-phase review found no archive manifest for this generation.",
            generation=review.identity.generation,
        )
    ]


RULES: tuple[Callable[[JobReview], list[Finding]], ...] = (
    _rule_empty,
    _rule_parse_errors,
    _rule_missing_expected,
    _rule_unpaired,
    _rule_terminal_mismatch,
    _rule_integrity,
    _rule_group_opponent,
    _rule_asr_stuck,
    _rule_metrics_mismatch,
    _rule_rates,
    _rule_flow_shift,
    _rule_tools,
    _rule_generation_mismatch,
    _rule_eval,
    _rule_archive_pfsp,
    _rule_archive_missing,
)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a researcher-facing how-the-work-went review for an Ultron job."
    )
    parser.add_argument("traces", type=Path, help="Trace directory or a single trajectory JSONL file.")
    parser.add_argument("--phase", choices=[item.value for item in Phase], default=Phase.COMPLETE.value)
    parser.add_argument("--generation", type=int)
    parser.add_argument("--windows", type=int, default=WINDOW_COUNT)
    parser.add_argument("--output", type=Path, help="Directory for review.md and review.json.")
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--pfsp", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 on unusable, 1 on caution.",
    )
    args = parser.parse_args()
    if args.windows < 1:
        parser.error("--windows must be >= 1")
    traces: Path = args.traces
    if not traces.exists():
        raise SystemExit(f"trace path does not exist: {traces}")
    output = args.output or (traces if traces.is_dir() else traces.parent)
    review = review_traces(
        traces,
        phase=Phase(args.phase),
        generation=args.generation,
        window_count=args.windows,
        eval_dir=args.eval_dir,
        archive_dir=args.archive_dir,
        pfsp_path=args.pfsp,
    )
    json_path, markdown_path = write_review(review, output)
    print(
        json.dumps(
            {
                "verdict": review.verdict.value,
                "generation": review.identity.generation,
                "phase": review.identity.phase.value,
                "episodes": review.identity.episode_count,
                "asr": review.outcomes.asr,
                "markdown": str(markdown_path),
                "json": str(json_path),
                "findings": [
                    {
                        "code": item.code,
                        "severity": item.severity.value,
                        "message": item.message,
                    }
                    for item in review.findings
                ],
            },
            indent=2,
        )
    )
    if args.strict:
        if review.verdict is Verdict.UNUSABLE:
            raise SystemExit(2)
        if review.verdict is Verdict.CAUTION:
            raise SystemExit(1)


if __name__ == "__main__":
    _main()
