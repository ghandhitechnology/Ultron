from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ultron.cli.catalog import repo_root
from ultron.train.review import JobReview, Phase, review_traces, write_review


class ResultsError(Exception):
    pass


@dataclass(frozen=True)
class FindingSummary:
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class ReviewSummary:
    generation: int | None
    phase: str
    verdict: str
    episodes: int
    asr: float | None
    traces: Path
    review_json: Path | None
    review_md: Path | None
    findings: tuple[FindingSummary, ...]


@dataclass(frozen=True)
class GenerationArtifacts:
    generation: int
    traces_dir: Path
    review: ReviewSummary | None
    metrics_path: Path | None
    archive_dir: Path | None


def traces_root(*, root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "traces"


def archive_root(*, root: Path | None = None) -> Path:
    return (root or repo_root()) / "data" / "archives"


def discover_generations(*, root: Path | None = None) -> tuple[GenerationArtifacts, ...]:
    root = root or repo_root()
    found: dict[int, GenerationArtifacts] = {}
    traces = traces_root(root=root)
    if traces.is_dir():
        for path in sorted(traces.iterdir()):
            generation = _gen_suffix(path.name)
            if generation is None or not path.is_dir():
                continue
            found[generation] = _artifacts(generation, root=root)
    archives = archive_root(root=root)
    if archives.is_dir():
        for path in sorted(archives.iterdir()):
            generation = _gen_suffix(path.name)
            if generation is None or not path.is_dir():
                continue
            if generation not in found:
                found[generation] = _artifacts(generation, root=root)
    return tuple(found[key] for key in sorted(found))


def load_review(traces: Path) -> ReviewSummary | None:
    if traces.is_file():
        directory = traces.parent
    else:
        directory = traces
    payload_path = directory / "review.json"
    if not payload_path.is_file():
        return None
    try:
        payload = json.loads(payload_path.read_text())
    except json.JSONDecodeError as exc:
        raise ResultsError(f"invalid review.json: {payload_path}: {exc}") from exc
    return _summary_from_payload(payload, traces=directory, review_json=payload_path)


def fetch_review(
    traces: Path,
    *,
    generation: int | None = None,
    phase: str = "complete",
    eval_dir: Path | None = None,
    archive_dir: Path | None = None,
    pfsp_path: Path | None = None,
) -> ReviewSummary:
    if not traces.exists():
        raise ResultsError(f"trace path does not exist: {traces}")
    try:
        review_phase = Phase(phase)
    except ValueError as exc:
        raise ResultsError(f"unknown review phase: {phase}") from exc
    review = review_traces(
        traces,
        phase=review_phase,
        generation=generation,
        eval_dir=eval_dir,
        archive_dir=archive_dir,
        pfsp_path=pfsp_path,
    )
    output = traces if traces.is_dir() else traces.parent
    json_path, markdown_path = write_review(review, output)
    return summarize_review(review, traces=output, review_json=json_path, review_md=markdown_path)


def summarize_review(
    review: JobReview,
    *,
    traces: Path,
    review_json: Path | None = None,
    review_md: Path | None = None,
) -> ReviewSummary:
    return ReviewSummary(
        generation=review.identity.generation,
        phase=review.identity.phase.value,
        verdict=review.verdict.value,
        episodes=review.identity.episode_count,
        asr=review.outcomes.asr,
        traces=traces,
        review_json=review_json,
        review_md=review_md,
        findings=tuple(
            FindingSummary(code=item.code, severity=item.severity.value, message=item.message)
            for item in review.findings
        ),
    )


def read_markdown(summary: ReviewSummary) -> str:
    if summary.review_md is None or not summary.review_md.is_file():
        return "No review.md yet. Fetch results to write one."
    return summary.review_md.read_text()


def _artifacts(generation: int, *, root: Path) -> GenerationArtifacts:
    traces_dir = traces_root(root=root) / f"gen{generation}"
    archive_dir = archive_root(root=root) / f"gen{generation}"
    metrics = traces_dir / "metrics.json"
    return GenerationArtifacts(
        generation=generation,
        traces_dir=traces_dir,
        review=load_review(traces_dir) if traces_dir.exists() else None,
        metrics_path=metrics if metrics.is_file() else None,
        archive_dir=archive_dir if archive_dir.is_dir() else None,
    )


def _summary_from_payload(
    payload: object,
    *,
    traces: Path,
    review_json: Path,
) -> ReviewSummary:
    if not isinstance(payload, dict):
        raise ResultsError(f"review.json must be an object: {review_json}")
    identity = payload.get("identity") if isinstance(payload.get("identity"), dict) else {}
    outcomes = payload.get("outcomes") if isinstance(payload.get("outcomes"), dict) else {}
    findings_raw = payload.get("findings")
    findings: list[FindingSummary] = []
    if isinstance(findings_raw, list):
        for item in findings_raw:
            if not isinstance(item, dict):
                continue
            findings.append(
                FindingSummary(
                    code=str(item.get("code", "")),
                    severity=str(item.get("severity", "")),
                    message=str(item.get("message", "")),
                )
            )
    markdown = traces / "review.md"
    asr = outcomes.get("asr")
    return ReviewSummary(
        generation=_optional_int(identity.get("generation")),
        phase=str(identity.get("phase") or payload.get("phase") or "complete"),
        verdict=str(payload.get("verdict", "unknown")),
        episodes=_optional_int(identity.get("episode_count")) or 0,
        asr=None if asr is None else float(asr),
        traces=traces,
        review_json=review_json,
        review_md=markdown if markdown.is_file() else None,
        findings=tuple(findings),
    )


def _gen_suffix(name: str) -> int | None:
    if not name.startswith("gen"):
        return None
    suffix = name.removeprefix("gen")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ResultsError(f"expected an integer, got {value!r}") from exc
