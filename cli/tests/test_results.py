import json
from pathlib import Path

import pytest

from ultron.cli.results import (
    ResultsError,
    discover_generations,
    fetch_review,
    load_review,
)


def test_discover_and_load_review(tmp_path: Path) -> None:
    traces = tmp_path / "data" / "traces" / "gen2"
    traces.mkdir(parents=True)
    payload = {
        "verdict": "usable",
        "identity": {"generation": 2, "phase": "complete", "episode_count": 8},
        "outcomes": {"asr": 0.25},
        "findings": [{"code": "ok", "severity": "info", "message": "fine"}],
    }
    (traces / "review.json").write_text(json.dumps(payload))
    (traces / "review.md").write_text("# review\n")
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.25}))
    found = discover_generations(root=tmp_path)
    assert len(found) == 1
    assert found[0].generation == 2
    family_archives = tmp_path / "data" / "families" / "gemma" / "archives" / "gen4"
    family_archives.mkdir(parents=True)
    family_only = discover_generations(root=tmp_path, archive_dir=family_archives.parent)
    assert {item.generation for item in family_only} == {2, 4}
    assert found[0].metrics_path is not None
    review = load_review(traces)
    assert review is not None
    assert review.verdict == "usable"
    assert review.episodes == 8
    assert review.asr == 0.25
    assert review.findings[0].code == "ok"


def test_fetch_review_writes_artifacts(tmp_path: Path) -> None:
    traces = tmp_path / "gen1"
    traces.mkdir()
    summary = fetch_review(traces, generation=1, phase="complete")
    assert summary.generation == 1
    assert (traces / "review.json").is_file()
    assert (traces / "review.md").is_file()
    assert summary.verdict in {"usable", "caution", "unusable"}


def test_fetch_missing_traces_is_typed_error(tmp_path: Path) -> None:
    with pytest.raises(ResultsError, match="does not exist"):
        fetch_review(tmp_path / "missing")
