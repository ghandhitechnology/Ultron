from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Mapping, Sequence, TypeAlias

from ultron.cli.catalog import repo_root


class SessionState(str, Enum):
    RUNNING = "running"
    DEAD = "dead"
    MISSING = "missing"


@dataclass(frozen=True)
class SessionInfo:
    name: str
    state: SessionState
    pid: int | None
    command: str
    log_path: Path


@dataclass(frozen=True)
class Started:
    session: str
    kind: Literal["started"] = "started"


@dataclass(frozen=True)
class AlreadyRunning:
    session: str
    info: SessionInfo
    kind: Literal["already_running"] = "already_running"


StartResult: TypeAlias = Started | AlreadyRunning


class JobsError(Exception):
    pass


def tmux_job_script(*, root: Path | None = None) -> Path:
    return (root or repo_root()) / "scripts" / "tmux_job.sh"


def log_dir(*, root: Path | None = None, env: Mapping[str, str] | None = None) -> Path:
    environ = env if env is not None else os.environ
    override = environ.get("ULTRON_TMUX_LOG_DIR")
    if override:
        return Path(override)
    return (root or repo_root()) / "data" / "logs"


def log_path(session: str, *, root: Path | None = None, env: Mapping[str, str] | None = None) -> Path:
    return log_dir(root=root, env=env) / f"{session}.log"


def list_sessions(*, root: Path | None = None, env: Mapping[str, str] | None = None) -> tuple[SessionInfo, ...]:
    result = _run(["list"], root=root, env=env)
    if result.returncode != 0:
        raise JobsError(result.stderr.strip() or "failed to list tmux jobs")
    return parse_status_output(result.stdout, root=root, env=env)


def running_sessions(*, root: Path | None = None, env: Mapping[str, str] | None = None) -> tuple[SessionInfo, ...]:
    return tuple(item for item in list_sessions(root=root, env=env) if item.state is SessionState.RUNNING)


def session_status(session: str, *, root: Path | None = None, env: Mapping[str, str] | None = None) -> SessionInfo:
    result = _run(["status", session], root=root, env=env)
    parsed = parse_status_output(result.stdout, root=root, env=env)
    if parsed:
        return parsed[0]
    if result.returncode != 0:
        return SessionInfo(
            name=session,
            state=SessionState.MISSING,
            pid=None,
            command="",
            log_path=log_path(session, root=root, env=env),
        )
    raise JobsError(result.stderr.strip() or f"no status for {session}")


def start_session(
    session: str,
    argv: Sequence[str],
    *,
    extra_env: Mapping[str, str] | None = None,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> StartResult:
    if not argv:
        raise JobsError("start requires a command")
    merged = _environ(root=root, env=env)
    if extra_env:
        merged.update(extra_env)
    result = _run(["start", session, "--", *argv], root=root, env=merged)
    if result.returncode == 0:
        return Started(session=session)
    text = f"{result.stderr}\n{result.stdout}"
    if "already running" in text:
        return AlreadyRunning(session=session, info=session_status(session, root=root, env=merged))
    raise JobsError(result.stderr.strip() or result.stdout.strip() or f"failed to start {session}")


def stop_session(session: str, *, root: Path | None = None, env: Mapping[str, str] | None = None) -> None:
    result = _run(["stop", session], root=root, env=env)
    if result.returncode != 0:
        raise JobsError(result.stderr.strip() or f"failed to stop {session}")


def read_logs(
    session: str,
    *,
    tail: int = 200,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    path = log_path(session, root=root, env=env)
    if not path.is_file():
        raise JobsError(f"no log for {session}: {path}")
    lines = path.read_text(errors="replace").splitlines()
    if tail > 0:
        lines = lines[-tail:]
    return "\n".join(lines)


def parse_status_output(
    text: str,
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[SessionInfo, ...]:
    items: list[SessionInfo] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "No Ultron tmux sessions.":
            continue
        parts = line.split("\t")
        name = parts[0]
        path = log_path(name, root=root, env=env)
        if len(parts) >= 2 and parts[1] == "missing":
            items.append(
                SessionInfo(name=name, state=SessionState.MISSING, pid=None, command="", log_path=path)
            )
            continue
        if len(parts) < 4:
            raise JobsError(f"unreadable job status: {line}")
        try:
            pid = int(parts[1])
        except ValueError as exc:
            raise JobsError(f"unreadable job pid: {line}") from exc
        dead = parts[2] == "1"
        items.append(
            SessionInfo(
                name=name,
                state=SessionState.DEAD if dead else SessionState.RUNNING,
                pid=pid,
                command=parts[3],
                log_path=path,
            )
        )
    return tuple(items)


def _run(
    args: Sequence[str],
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    script = tmux_job_script(root=root)
    if not script.is_file():
        raise JobsError(f"missing {script}")
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=str(root or repo_root()),
        env=_environ(root=root, env=env),
        text=True,
        capture_output=True,
    )


def _environ(*, root: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, str]:
    merged = dict(os.environ if env is None else env)
    merged.setdefault("ULTRON_TMUX_DETACH", "1")
    if "ULTRON_TMUX_LOG_DIR" not in merged:
        merged["ULTRON_TMUX_LOG_DIR"] = str(log_dir(root=root, env=merged))
    return merged
