from __future__ import annotations

import base64
import codecs
import hashlib
import hmac
import json
import os
import re
import stat as stat_module
from pathlib import Path
from typing import Any

from ultron.cli.jobs import JobsError, log_path

SCHEMA_VERSION = 1
MAX_READ_BYTES = 256 * 1024
MAX_CURSOR_LENGTH = 4096

_SESSION_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,40}\Z")
_CURSOR_PREFIX = "v1"
_CURSOR_DOMAIN = b"ultron-agent-log-cursor-v1\0"
_CURSOR_KEYS = frozenset(
    {"v", "session", "path", "dev", "ino", "offset", "anchor", "utf8", "terminal"}
)
_ANCHOR_BYTES = 64
_TERMINAL_STATES = frozenset(
    {"text", "cr", "esc", "esc_intermediate", "csi", "osc", "osc_esc", "string", "string_esc"}
)


def read_agent_logs(
    session: str,
    *,
    cursor: str | None = None,
    max_bytes: int = 16_384,
    tail: int = 50,
    root: Path | None = None,
) -> dict[str, object]:
    """Read a bounded initial tail or the next sequential bytes of an agent log."""
    _validate_request(session, cursor=cursor, max_bytes=max_bytes, tail=tail)
    path = log_path(session, root=root)
    bound_path = os.path.abspath(os.fspath(path))
    saved = _decode_cursor(cursor, session=session, path=bound_path) if cursor is not None else None

    handle = _open_regular_log(path, session=session)

    try:
        with handle:
            file_stat = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(file_stat.st_mode):
                raise JobsError(f"log for {session} is not a regular file: {path}")
            if saved is None:
                raw, offset, skipped = _read_initial(
                    handle, size=file_stat.st_size, max_bytes=max_bytes, tail=tail
                )
                pending = b""
                terminal_state = "text"
                reset = False
            else:
                same_file = saved["dev"] == file_stat.st_dev and saved["ino"] == file_stat.st_ino
                offset_in_range = saved["offset"] <= file_stat.st_size
                anchor_matches = (
                    same_file
                    and offset_in_range
                    and hmac.compare_digest(saved["anchor"], _anchor_digest(handle, saved["offset"]))
                )
                reset = not same_file or not offset_in_range or not anchor_matches
                offset = 0 if reset else saved["offset"]
                pending = b"" if reset else saved["utf8"]
                terminal_state = "text" if reset else saved["terminal"]
                handle.seek(offset)
                raw = handle.read(min(max_bytes, max(0, file_stat.st_size - offset)))
                offset += len(raw)
                skipped = 0

            anchor = _anchor_digest(handle, offset)
    except OSError as exc:
        raise JobsError(f"cannot read log for {session}: {exc}") from exc

    decoded, pending = _decode_utf8(raw, pending)
    text, terminal_state = _strip_terminal(decoded, terminal_state)
    next_cursor = _encode_cursor(
        session=session,
        path=bound_path,
        dev=file_stat.st_dev,
        ino=file_stat.st_ino,
        offset=offset,
        anchor=anchor,
        utf8=pending,
        terminal=terminal_state,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "session": session,
        "log_path": str(path),
        "text": text,
        "next_cursor": next_cursor,
        "has_more": offset < file_stat.st_size,
        "reset": reset,
        "skipped_bytes": skipped,
    }


def _open_regular_log(path: Path, *, session: str) -> Any:
    try:
        before_open = path.stat()
        if not stat_module.S_ISREG(before_open.st_mode):
            raise JobsError(f"log for {session} is not a regular file: {path}")
        flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise JobsError(f"no log for {session}: {path}") from exc
    except JobsError:
        raise
    except OSError as exc:
        raise JobsError(f"cannot read log for {session}: {exc}") from exc

    try:
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _validate_request(session: object, *, cursor: object, max_bytes: object, tail: object) -> None:
    if not isinstance(session, str) or _SESSION_PATTERN.fullmatch(session) is None:
        raise ValueError("session must be 1-40 characters using only letters, digits, '_' or '-'")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_READ_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {MAX_READ_BYTES}")
    if type(tail) is not int or tail < 0:
        raise ValueError("tail must be >= 0")
    if cursor is not None and not isinstance(cursor, str):
        raise ValueError("cursor must be a string or None")


def _read_initial(handle: Any, *, size: int, max_bytes: int, tail: int) -> tuple[bytes, int, int]:
    if tail == 0:
        return b"", size, size

    start = max(0, size - max_bytes)
    handle.seek(start)
    raw = handle.read(min(max_bytes, size - start))
    offset = start + len(raw)

    lines = raw.splitlines(keepends=True)
    if len(lines) > tail:
        dropped = sum(map(len, lines[:-tail]))
        start += dropped
        raw = raw[dropped:]

    while raw and raw[0] & 0xC0 == 0x80:
        start += 1
        raw = raw[1:]
    return raw, offset, start


def _decode_utf8(raw: bytes, pending: bytes) -> tuple[str, bytes]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        decoder.setstate((pending, 0))
        text = decoder.decode(raw, final=False)
        remaining, _ = decoder.getstate()
    except (UnicodeError, ValueError):
        text = (pending + raw).decode("utf-8", errors="replace")
        remaining = b""
    return text, remaining


def _anchor_digest(handle: Any, offset: int) -> bytes:
    start = max(0, offset - _ANCHOR_BYTES)
    handle.seek(start)
    anchor = handle.read(offset - start)
    return hashlib.sha256(anchor).digest()[:16]


def _strip_terminal(text: str, state: str) -> tuple[str, str]:
    output: list[str] = []
    for character in text:
        code = ord(character)

        if state == "cr":
            state = "text"
            if character == "\n":
                continue

        if state == "text":
            if character == "\x1b":
                state = "esc"
            elif character == "\x9b":
                state = "csi"
            elif character == "\x9d":
                state = "osc"
            elif character in {"\x90", "\x98", "\x9e", "\x9f"}:
                state = "string"
            elif character == "\r":
                output.append("\n")
                state = "cr"
            elif character in {"\n", "\t"} or code >= 0x20 and code != 0x7F and not 0x80 <= code <= 0x9F:
                output.append(character)
        elif state == "esc":
            if character == "[":
                state = "csi"
            elif character == "]":
                state = "osc"
            elif character in {"P", "X", "^", "_"}:
                state = "string"
            elif 0x20 <= code <= 0x2F:
                state = "esc_intermediate"
            else:
                state = "text"
        elif state == "esc_intermediate":
            if not 0x20 <= code <= 0x2F:
                state = "text"
        elif state == "csi":
            if 0x40 <= code <= 0x7E:
                state = "text"
        elif state == "osc":
            if character == "\x07":
                state = "text"
            elif character == "\x1b":
                state = "osc_esc"
        elif state == "osc_esc":
            state = "text" if character == "\\" else "osc"
        elif state == "string":
            if character == "\x1b":
                state = "string_esc"
        elif state == "string_esc":
            state = "text" if character == "\\" else "string"
    return "".join(output), state


def _encode_cursor(
    *,
    session: str,
    path: str,
    dev: int,
    ino: int,
    offset: int,
    anchor: bytes,
    utf8: bytes,
    terminal: str,
) -> str:
    payload = {
        "v": SCHEMA_VERSION,
        "session": session,
        "path": path,
        "dev": dev,
        "ino": ino,
        "offset": offset,
        "anchor": _base64_encode(anchor),
        "utf8": _base64_encode(utf8),
        "terminal": terminal,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(_CURSOR_DOMAIN + encoded).digest()[:16]
    return f"{_CURSOR_PREFIX}.{_base64_encode(encoded)}.{_base64_encode(checksum)}"


def _decode_cursor(cursor: str, *, session: str, path: str) -> dict[str, Any]:
    try:
        if len(cursor) > MAX_CURSOR_LENGTH:
            raise ValueError
        prefix, encoded_payload, encoded_checksum = cursor.split(".")
        if prefix != _CURSOR_PREFIX:
            raise ValueError
        payload_bytes = _base64_decode(encoded_payload)
        checksum = _base64_decode(encoded_checksum)
        expected = hashlib.sha256(_CURSOR_DOMAIN + payload_bytes).digest()[:16]
        if len(checksum) != len(expected) or not hmac.compare_digest(checksum, expected):
            raise ValueError
        payload = json.loads(payload_bytes)
        if not isinstance(payload, dict) or set(payload) != _CURSOR_KEYS:
            raise ValueError
        if payload["v"] != SCHEMA_VERSION or payload["session"] != session or payload["path"] != path:
            raise ValueError
        if any(type(payload[key]) is not int or payload[key] < 0 for key in ("dev", "ino", "offset")):
            raise ValueError
        if not isinstance(payload["terminal"], str) or payload["terminal"] not in _TERMINAL_STATES:
            raise ValueError
        if not isinstance(payload["anchor"], str) or not isinstance(payload["utf8"], str):
            raise ValueError
        anchor = _base64_decode(payload["anchor"])
        pending = _base64_decode(payload["utf8"])
        if len(anchor) != 16 or len(pending) > 3:
            raise ValueError
    except (KeyError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid log cursor") from exc

    payload["anchor"] = anchor
    payload["utf8"] = pending
    return payload


def _base64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64_decode(value: str) -> bytes:
    if not isinstance(value, str) or len(value) % 4 == 1:
        raise ValueError
    padding = "=" * (-len(value) % 4)
    decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    if _base64_encode(decoded) != value:
        raise ValueError
    return decoded
