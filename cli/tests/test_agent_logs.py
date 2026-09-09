from __future__ import annotations

import os
from pathlib import Path

import pytest

from ultron.cli.agent_logs import read_agent_logs
from ultron.cli.jobs import JobsError


def _log(tmp_path: Path, session: str = "agent") -> Path:
    path = tmp_path / "data" / "logs" / f"{session}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_initial_tail_then_reads_every_new_byte_in_order(tmp_path: Path) -> None:
    path = _log(tmp_path)
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    initial = read_agent_logs("agent", root=tmp_path, tail=2, max_bytes=64)
    assert initial["text"] == "two\nthree\n"
    assert initial["skipped_bytes"] == len("one\n")
    assert initial["has_more"] is False

    with path.open("a", encoding="utf-8") as handle:
        handle.write("four\nfive\n")
    first = read_agent_logs("agent", root=tmp_path, cursor=initial["next_cursor"], max_bytes=5)
    second = read_agent_logs("agent", root=tmp_path, cursor=first["next_cursor"], max_bytes=5)
    third = read_agent_logs("agent", root=tmp_path, cursor=second["next_cursor"], max_bytes=5)

    assert first["text"] + second["text"] + third["text"] == "four\nfive\n"
    assert first["has_more"] is True
    assert second["has_more"] is False
    assert third["has_more"] is False
    assert all(part["skipped_bytes"] == 0 for part in (first, second, third))


def test_rotation_resets_and_reads_new_file_from_start(tmp_path: Path) -> None:
    path = _log(tmp_path)
    path.write_text("old\n", encoding="utf-8")
    initial = read_agent_logs("agent", root=tmp_path)
    rotated = path.with_suffix(".old")
    path.rename(rotated)
    path.write_text("new file\n", encoding="utf-8")

    update = read_agent_logs("agent", root=tmp_path, cursor=initial["next_cursor"])

    assert update["reset"] is True
    assert update["text"] == "new file\n"


def test_truncation_resets_same_file_and_reads_from_start(tmp_path: Path) -> None:
    path = _log(tmp_path)
    path.write_text("long old output\n", encoding="utf-8")
    initial = read_agent_logs("agent", root=tmp_path)
    with path.open("r+b") as handle:
        handle.truncate(0)
        handle.write(b"new\n")

    update = read_agent_logs("agent", root=tmp_path, cursor=initial["next_cursor"])

    assert update["reset"] is True
    assert update["text"] == "new\n"


def test_copytruncate_regrowth_resets_even_when_new_file_is_longer(tmp_path: Path) -> None:
    path = _log(tmp_path)
    path.write_text("old output\n", encoding="utf-8")
    initial = read_agent_logs("agent", root=tmp_path)
    original_inode = path.stat().st_ino
    with path.open("r+b") as handle:
        handle.truncate(0)
        handle.write(b"replacement output is longer\n")
    assert path.stat().st_ino == original_inode

    update = read_agent_logs("agent", root=tmp_path, cursor=initial["next_cursor"])

    assert update["reset"] is True
    assert update["text"] == "replacement output is longer\n"


def test_initial_tail_is_bounded_for_a_giant_line(tmp_path: Path) -> None:
    path = _log(tmp_path)
    path.write_bytes(b"x" * 100_000)

    result = read_agent_logs("agent", root=tmp_path, max_bytes=127, tail=50)

    assert result["text"] == "x" * 127
    assert result["skipped_bytes"] == 100_000 - 127
    assert result["has_more"] is False


def test_split_unicode_and_ansi_sequences_resume_without_artifacts(tmp_path: Path) -> None:
    path = _log(tmp_path)
    path.write_bytes(b"")
    cursor = read_agent_logs("agent", root=tmp_path, tail=0)["next_cursor"]
    path.write_bytes("A🙂B\x1b[31mred\x1b[0m\n".encode("utf-8"))

    chunks: list[str] = []
    for _ in range(20):
        result = read_agent_logs("agent", root=tmp_path, cursor=cursor, max_bytes=3)
        chunks.append(result["text"])
        cursor = result["next_cursor"]
        if not result["has_more"]:
            break

    assert "".join(chunks) == "A🙂Bred\n"
    assert "�" not in "".join(chunks)
    assert "\x1b" not in "".join(chunks)


def test_control_characters_are_removed_and_carriage_returns_are_lines(tmp_path: Path) -> None:
    path = _log(tmp_path)
    path.write_bytes(b"start\x00\x08\rprogress\r\ndone\tvalue\n")

    result = read_agent_logs("agent", root=tmp_path)

    assert result["text"] == "start\nprogress\ndone\tvalue\n"


def test_tail_zero_returns_no_text_and_starts_at_current_end(tmp_path: Path) -> None:
    path = _log(tmp_path)
    path.write_text("old\n", encoding="utf-8")
    initial = read_agent_logs("agent", root=tmp_path, tail=0)
    assert initial["text"] == ""
    assert initial["skipped_bytes"] == path.stat().st_size

    with path.open("a", encoding="utf-8") as handle:
        handle.write("new\n")
    update = read_agent_logs("agent", root=tmp_path, cursor=initial["next_cursor"])
    assert update["text"] == "new\n"


@pytest.mark.parametrize("session", ["", "../agent", "agent.name", "x" * 41])
def test_rejects_unsafe_session_names(tmp_path: Path, session: str) -> None:
    with pytest.raises(ValueError, match="session"):
        read_agent_logs(session, root=tmp_path)


@pytest.mark.parametrize("max_bytes", [0, 262_145, True])
def test_rejects_invalid_byte_limits(tmp_path: Path, max_bytes: object) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        read_agent_logs("agent", root=tmp_path, max_bytes=max_bytes)  # type: ignore[arg-type]


def test_rejects_tampered_or_cross_session_cursor(tmp_path: Path) -> None:
    path = _log(tmp_path)
    path.write_text("output\n", encoding="utf-8")
    cursor = read_agent_logs("agent", root=tmp_path)["next_cursor"]
    assert isinstance(cursor, str)
    replacement = "A" if cursor[-1] != "A" else "B"

    with pytest.raises(ValueError, match="cursor"):
        read_agent_logs("agent", root=tmp_path, cursor=cursor[:-1] + replacement)
    _log(tmp_path, "other").write_text("output\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cursor"):
        read_agent_logs("other", root=tmp_path, cursor=cursor)


def test_missing_log_is_a_jobs_error(tmp_path: Path) -> None:
    with pytest.raises(JobsError, match="no log for agent"):
        read_agent_logs("agent", root=tmp_path)


def test_fifo_log_is_rejected_without_opening_it_as_a_stream(tmp_path: Path) -> None:
    path = _log(tmp_path)
    os.mkfifo(path)

    with pytest.raises(JobsError, match="not a regular file"):
        read_agent_logs("agent", root=tmp_path)
