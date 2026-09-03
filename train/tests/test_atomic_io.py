from __future__ import annotations

import json
from pathlib import Path

import pytest

from ultron.train.io import atomic_write_json, atomic_write_text


def test_atomic_write_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text("old")

    atomic_write_json(target, {"status": "complete"})

    assert json.loads(target.read_text()) == {"status": "complete"}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_keeps_previous_file_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.txt"
    target.write_text("stable")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr("ultron.train.io.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_text(target, "partial")

    assert target.read_text() == "stable"
    assert list(tmp_path.iterdir()) == [target]
