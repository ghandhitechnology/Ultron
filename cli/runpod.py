"""Monitor an Ultron run on RunPod from a local computer."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


RUNPOD_TIMEOUT_SECONDS = 15
REMOTE_TIMEOUT_SECONDS = 20
MAX_COMMAND_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_PROVIDER_LOG_LINES = 100
_POD_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_SESSION = re.compile(r"[A-Za-z0-9_-]{1,40}")
_SAFE_POD_FIELDS = (
    "id",
    "name",
    "desiredStatus",
    "runtimeStatus",
    "runtimeStatusReason",
    "lastStatusChange",
    "uptimeSeconds",
    "imageName",
    "gpuTypeId",
    "gpuCount",
    "vcpuCount",
    "memoryInGb",
    "containerDiskInGb",
    "volumeInGb",
    "volumeMountPath",
    "costPerHr",
    "ports",
)


class RunPodMonitorError(Exception):
    pass


@dataclass(frozen=True)
class SSHConnection:
    ip: str
    port: int
    user: str
    identity: Path | None

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.ip}"


@dataclass(frozen=True)
class RemoteResult:
    document: dict[str, Any] | None
    returncode: int
    stderr: str
    stdout_prefix: str


def add_runpod_commands(sub: argparse._SubParsersAction) -> None:
    runpod = sub.add_parser(
        "runpod",
        help="Monitor an Ultron job on RunPod from this computer.",
    )
    actions = runpod.add_subparsers(dest="runpod_action", required=True)

    status = actions.add_parser("status", help="Combine RunPod and remote Ultron status.")
    _add_connection_arguments(status)
    _add_status_arguments(status, include_session=True)

    watch = actions.add_parser("watch", help="Poll a remote job and retain observations locally.")
    _add_connection_arguments(watch)
    _add_status_arguments(watch, include_session=False)
    watch.add_argument("--session", required=True, help="Remote tmux session to follow.")
    watch.add_argument("--timeout", type=float, default=300, help="Polling duration in seconds, default 300.")
    watch.add_argument("--interval", type=float, default=10, help="Seconds between observations, default 10.")
    watch.add_argument(
        "--max-misses",
        type=int,
        default=3,
        help="Consecutive incomplete observations allowed before stopping, default 3.",
    )
    watch.add_argument(
        "--record",
        type=Path,
        help="JSONL observation path; default data/monitoring/<pod-id>/observations.jsonl.",
    )
    watch.add_argument("--no-record", action="store_true", help="Do not retain observations locally.")

    logs = actions.add_parser("logs", help="Read bounded remote logs with an optional local cursor file.")
    _add_connection_arguments(logs)
    logs.add_argument("--session", required=True)
    logs.add_argument("--cursor")
    logs.add_argument("--cursor-file", type=Path, help="Read and update the cursor in this local file.")
    logs.add_argument("--max-bytes", type=int, default=16384)
    logs.add_argument("--tail", type=int, default=50)
    logs.add_argument("--json", action="store_true")


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pod_id", nargs="?", help="RunPod pod ID; defaults to ULTRON_RUNPOD_ID.")
    parser.add_argument(
        "--repo",
        default=os.environ.get("ULTRON_RUNPOD_REPO", "/workspace/Ultron"),
        help="Ultron checkout path on the pod, default /workspace/Ultron.",
    )
    parser.add_argument("--ssh-user", default=os.environ.get("ULTRON_RUNPOD_SSH_USER", "root"))
    parser.add_argument("--identity", type=Path, help="Override the SSH identity reported by runpodctl.")
    parser.add_argument("--remote-python", help="Python executable on the pod; auto-detected when omitted.")
    parser.add_argument("--runpodctl", default=os.environ.get("ULTRON_RUNPODCTL", "runpodctl"))
    parser.add_argument("--connect-timeout", type=int, default=10)


def _add_status_arguments(parser: argparse.ArgumentParser, *, include_session: bool) -> None:
    if include_session:
        parser.add_argument("--session", help="Scope Ultron status to one remote tmux session.")
    parser.add_argument("--details", action="store_true", help="Include bounded log and response previews.")
    parser.add_argument("--response-limit", type=int, default=4)
    parser.add_argument("--state-dir", help="Pipeline state directory on the pod.")
    parser.add_argument("--responses-dir", help="Response archive directory on the pod.")
    parser.add_argument("--stale-seconds", type=float, default=300)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument(
        "--system-log-tail",
        type=int,
        default=20,
        help="RunPod system log lines to collect when SSH or Ultron status is unavailable, default 20.",
    )
    parser.add_argument("--json", action="store_true")


def run_runpod_command(args: argparse.Namespace) -> int:
    try:
        pod_id = args.pod_id or os.environ.get("ULTRON_RUNPOD_ID")
        _validate_common(args, pod_id)
        assert pod_id is not None
        if args.runpod_action == "logs":
            return _run_logs(args, pod_id)
        if args.runpod_action == "watch":
            return _run_watch(args, pod_id)
        snapshot = collect_remote_status(args, pod_id)
        _emit_status(snapshot, as_json=args.json)
        return int(snapshot["exit_code"])
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RunPodMonitorError, ValueError) as exc:
        document = {
            "schema_version": 1,
            "observed_at": _now(),
            "error": str(exc),
            "exit_code": 2,
        }
        _emit_status(document, as_json=getattr(args, "json", False))
        return 2


def _validate_common(args: argparse.Namespace, pod_id: str | None) -> None:
    if not pod_id or not _POD_ID.fullmatch(pod_id):
        raise ValueError("pod ID must contain 1-128 letters, digits, underscores, or hyphens")
    if not args.repo.startswith("/") or _has_control(args.repo):
        raise ValueError("--repo must be an absolute path without control characters")
    if _has_control(args.ssh_user) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.ssh_user):
        raise ValueError("--ssh-user contains unsupported characters")
    if _has_control(args.runpodctl) or not args.runpodctl:
        raise ValueError("--runpodctl contains unsupported characters")
    if not 1 <= args.connect_timeout <= 60:
        raise ValueError("--connect-timeout must be between 1 and 60 seconds")
    session = getattr(args, "session", None)
    if session and not _SESSION.fullmatch(session):
        raise ValueError("session must contain 1-40 letters, digits, underscores, or hyphens")
    if args.remote_python and _has_control(args.remote_python):
        raise ValueError("--remote-python contains control characters")
    if getattr(args, "runpod_action", None) in {"status", "watch"}:
        if not math.isfinite(args.stale_seconds) or args.stale_seconds < 0:
            raise ValueError("--stale-seconds must be finite and >= 0")
        if not 0 <= args.response_limit <= 20:
            raise ValueError("--response-limit must be between 0 and 20")
        if not 0 <= args.system_log_tail <= MAX_PROVIDER_LOG_LINES:
            raise ValueError(f"--system-log-tail must be between 0 and {MAX_PROVIDER_LOG_LINES}")
    if getattr(args, "runpod_action", None) == "watch":
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 86400:
            raise ValueError("--timeout must be > 0 and <= 86400 seconds")
        if not math.isfinite(args.interval) or not 0.5 <= args.interval <= 300:
            raise ValueError("--interval must be between 0.5 and 300 seconds")
        if not 1 <= args.max_misses <= 20:
            raise ValueError("--max-misses must be between 1 and 20")
        if args.no_record and args.record:
            raise ValueError("--record and --no-record cannot be used together")
    if getattr(args, "runpod_action", None) == "logs":
        if not 1 <= args.max_bytes <= 262144:
            raise ValueError("--max-bytes must be between 1 and 262144")
        if args.tail < 0:
            raise ValueError("--tail must be >= 0")
        if args.cursor and (len(args.cursor) > 4096 or _has_control(args.cursor)):
            raise ValueError("--cursor is too long or contains control characters")


def collect_remote_status(args: argparse.Namespace, pod_id: str) -> dict[str, Any]:
    raw_pod = _get_pod(args.runpodctl, pod_id)
    pod = _safe_pod(raw_pod)
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "observed_at": _now(),
        "pod": pod,
        "transport": {"state": "unavailable"},
        "ultron": None,
        "provider_logs": {"available": False, "entries": []},
        "issues": [],
        "exit_code": 2,
    }
    connection, reason = _connection(raw_pod, args)
    if connection is None:
        snapshot["transport"]["reason"] = reason
        snapshot["issues"].append({"code": "ssh_unavailable", "message": reason})
    else:
        snapshot["transport"].update(
            {"state": "connecting", "destination": connection.destination, "port": connection.port}
        )
        remote = _remote_json(
            connection,
            _remote_status_args(args),
            repo=args.repo,
            remote_python=args.remote_python,
            connect_timeout=args.connect_timeout,
        )
        snapshot["transport"].update(
            {
                "state": "unavailable" if remote.returncode == 255 else "connected",
                "returncode": remote.returncode,
            }
        )
        if remote.stderr:
            snapshot["transport"]["stderr"] = remote.stderr
        if remote.stdout_prefix:
            snapshot["transport"]["stdout_prefix"] = remote.stdout_prefix
        if _is_status_document(remote.document):
            snapshot["ultron"] = remote.document
        else:
            code = "ssh_failed" if remote.returncode == 255 else "ultron_status_unavailable"
            remote_error = remote.document.get("error") if isinstance(remote.document, dict) else None
            message = str(remote_error or remote.stderr or "remote command returned no valid JSON status")
            snapshot["issues"].append({"code": code, "message": message})

    if snapshot["ultron"] is None and args.system_log_tail:
        snapshot["provider_logs"] = _provider_logs(args.runpodctl, pod_id, args.system_log_tail)
    snapshot["exit_code"] = _combined_exit_code(snapshot)
    return snapshot


def _get_pod(runpodctl: str, pod_id: str) -> dict[str, Any]:
    result = _command([runpodctl, "pod", "get", pod_id, "--output", "json"], RUNPOD_TIMEOUT_SECONDS)
    if result.returncode != 0:
        raise RunPodMonitorError(_command_error("runpodctl pod get", result))
    document = _single_json(result.stdout)
    if not isinstance(document, dict):
        raise RunPodMonitorError("runpodctl pod get returned an unexpected JSON value")
    if str(document.get("id", "")) != pod_id:
        raise RunPodMonitorError("runpodctl pod get returned a different pod")
    return document


def _safe_pod(document: dict[str, Any]) -> dict[str, Any]:
    return {field: document[field] for field in _SAFE_POD_FIELDS if field in document}


def _connection(raw_pod: dict[str, Any], args: argparse.Namespace) -> tuple[SSHConnection | None, str]:
    ssh = raw_pod.get("ssh")
    if not isinstance(ssh, dict):
        return None, "runpodctl did not report direct SSH connection details"
    if ssh.get("error"):
        return None, _bounded(str(ssh["error"]))
    ip = str(ssh.get("ip", ""))
    try:
        port = int(ssh.get("port", 0))
    except (TypeError, ValueError):
        port = 0
    if not ip or _has_control(ip) or not 1 <= port <= 65535:
        return None, "runpodctl returned invalid SSH connection details"
    identity = args.identity.expanduser() if args.identity else None
    if identity is None:
        key = ssh.get("ssh_key")
        if isinstance(key, dict):
            if key.get("exists") is False:
                return None, "RunPod SSH key is not configured; run runpodctl doctor"
            if key.get("path"):
                identity = Path(str(key["path"])).expanduser()
    if identity is not None and not identity.is_file():
        return None, f"SSH identity does not exist: {identity}"
    return SSHConnection(ip=ip, port=port, user=args.ssh_user, identity=identity), ""


def _remote_status_args(args: argparse.Namespace) -> list[str]:
    command = ["status", "--json", "--stale-seconds", str(args.stale_seconds)]
    if args.session:
        command.extend(("--session", args.session))
    if args.details:
        command.append("--details")
    command.extend(("--response-limit", str(args.response_limit)))
    if args.state_dir:
        command.extend(("--state-dir", args.state_dir))
    if args.responses_dir:
        command.extend(("--responses-dir", args.responses_dir))
    if args.no_gpu:
        command.append("--no-gpu")
    return command


def _remote_json(
    connection: SSHConnection,
    remote_args: Sequence[str],
    *,
    repo: str,
    remote_python: str | None,
    connect_timeout: int,
) -> RemoteResult:
    remote = _remote_command(repo, remote_python, remote_args)
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
        "-p",
        str(connection.port),
    ]
    if connection.identity is not None:
        command.extend(("-o", "IdentitiesOnly=yes", "-i", str(connection.identity)))
    command.extend((connection.destination, remote))
    try:
        result = _command(command, REMOTE_TIMEOUT_SECONDS + connect_timeout)
    except RunPodMonitorError as exc:
        return RemoteResult(document=None, returncode=255, stderr=str(exc), stdout_prefix="")
    document, prefix = _last_json_line(result.stdout)
    return RemoteResult(
        document=document,
        returncode=result.returncode,
        stderr=_bounded(result.stderr.strip()),
        stdout_prefix=_bounded(prefix),
    )


def _remote_command(repo: str, remote_python: str | None, remote_args: Sequence[str]) -> str:
    module_args = ["-m", "ultron.cli.main", *remote_args]
    if remote_python:
        launch = shlex.join([remote_python, *module_args])
    else:
        venv = shlex.join(["./.venv/bin/python", *module_args])
        system = shlex.join(["python3", *module_args])
        launch = f"if [ -x ./.venv/bin/python ]; then exec {venv}; else exec {system}; fi"
    script = f"cd -- {shlex.quote(repo)} && {launch}"
    return f"bash -lc {shlex.quote(script)}"


def _provider_logs(runpodctl: str, pod_id: str, tail: int) -> dict[str, Any]:
    try:
        result = _command(
            [runpodctl, "pod", "logs", pod_id, "--tail", str(tail), "--source", "system", "--output", "json"],
            RUNPOD_TIMEOUT_SECONDS,
        )
    except RunPodMonitorError as exc:
        return {"available": False, "entries": [], "reason": str(exc)}
    if result.returncode != 0:
        return {"available": False, "entries": [], "reason": _command_error("runpodctl pod logs", result)}
    entries: list[dict[str, Any]] = []
    try:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=_reject_json_constant)
            if isinstance(value, dict):
                entries.append({key: value[key] for key in ("source", "line", "ts") if key in value})
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        entries.append({key: item[key] for key in ("source", "line", "ts") if key in item})
    except json.JSONDecodeError:
        return {"available": False, "entries": [], "reason": "runpodctl pod logs returned invalid JSON"}
    return {"available": True, "entries": entries[-tail:]}


def _combined_exit_code(snapshot: dict[str, Any]) -> int:
    runtime = str(snapshot.get("pod", {}).get("runtimeStatus", "unknown")).lower()
    if runtime in {"stopped", "terminated"}:
        return 1
    ultron = snapshot.get("ultron")
    if isinstance(ultron, dict):
        code = ultron.get("exit_code")
        if isinstance(code, int) and code in {0, 1, 2}:
            return code
    return 2


def _is_status_document(document: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(document, dict)
        and document.get("schema_version") == 1
        and isinstance(document.get("jobs"), list)
        and isinstance(document.get("exit_code"), int)
        and document["exit_code"] in {0, 1, 2}
    )


def _run_watch(args: argparse.Namespace, pod_id: str) -> int:
    started = time.monotonic()
    misses = 0
    record = None if args.no_record else args.record or Path("data") / "monitoring" / pod_id / "observations.jsonl"
    while True:
        try:
            snapshot = collect_remote_status(args, pod_id)
        except (OSError, RunPodMonitorError) as exc:
            snapshot = {
                "schema_version": 1,
                "observed_at": _now(),
                "pod": {"id": pod_id, "runtimeStatus": "unknown"},
                "transport": {"state": "unavailable", "reason": str(exc)},
                "ultron": None,
                "provider_logs": {"available": False, "entries": []},
                "issues": [{"code": "runpod_status_unavailable", "message": str(exc)}],
                "exit_code": 2,
            }
        elapsed = time.monotonic() - started
        reason, terminal_code = _watch_outcome(snapshot, args.session)
        if snapshot["exit_code"] == 2 and terminal_code is None:
            misses += 1
        else:
            misses = 0
        if misses >= args.max_misses:
            reason, terminal_code = "monitoring_unavailable", 2
        elif terminal_code is None and elapsed >= args.timeout:
            reason, terminal_code = "timeout", 124
        snapshot["watch"] = {
            "reason": reason,
            "elapsed_seconds": round(elapsed, 3),
            "timeout_seconds": args.timeout,
            "consecutive_misses": misses,
            "max_misses": args.max_misses,
            "exit_code": terminal_code,
        }
        if record is not None:
            snapshot["record_path"] = str(record)
            _append_observation(record, snapshot)
        _emit_status(snapshot, as_json=args.json)
        if terminal_code is not None:
            return terminal_code
        time.sleep(min(args.interval, max(0, args.timeout - elapsed)))


def _watch_outcome(snapshot: dict[str, Any], session: str) -> tuple[str, int | None]:
    runtime = str(snapshot.get("pod", {}).get("runtimeStatus", "unknown")).lower()
    if runtime in {"stopped", "terminated"}:
        return "pod_stopped", 1
    ultron = snapshot.get("ultron")
    if not isinstance(ultron, dict):
        return "poll", None
    jobs = ultron.get("jobs")
    if not isinstance(jobs, list):
        return "poll", None
    selected = [job for job in jobs if isinstance(job, dict) and job.get("name", job.get("session")) == session]
    if not selected:
        return "poll", None
    remote_code = ultron.get("exit_code")
    if remote_code == 1:
        return "failed", 1
    if remote_code == 2:
        return "poll", None
    state = str(selected[0].get("state", "unknown"))
    if state == "succeeded":
        return "completed", 0
    if state in {"failed", "missing"}:
        return "failed", 1
    if state == "exited":
        return "unknown_exit", 2
    return "poll", None


def _append_observation(path: Path, snapshot: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        handle.flush()


def _run_logs(args: argparse.Namespace, pod_id: str) -> int:
    raw_pod = _get_pod(args.runpodctl, pod_id)
    connection, reason = _connection(raw_pod, args)
    if connection is None:
        raise RunPodMonitorError(reason)
    cursor = args.cursor
    if cursor is None and args.cursor_file and args.cursor_file.expanduser().is_file():
        cursor = _read_cursor(args.cursor_file, pod_id, args.session)
    command = ["logs", "--session", args.session, "--json", "--max-bytes", str(args.max_bytes)]
    if cursor:
        command.extend(("--cursor", cursor))
    else:
        command.extend(("--tail", str(args.tail)))
    remote = _remote_json(
        connection,
        command,
        repo=args.repo,
        remote_python=args.remote_python,
        connect_timeout=args.connect_timeout,
    )
    if remote.returncode != 0:
        raise RunPodMonitorError(remote.stderr or f"remote log command failed with exit {remote.returncode}")
    if remote.document is None:
        raise RunPodMonitorError(remote.stderr or "remote log command returned no JSON")
    if "error" in remote.document:
        raise RunPodMonitorError(str(remote.document["error"]))
    if (
        not isinstance(remote.document.get("text"), str)
        or not isinstance(remote.document.get("next_cursor"), str)
        or not isinstance(remote.document.get("has_more"), bool)
    ):
        raise RunPodMonitorError("remote log command returned an unexpected JSON object")
    result = dict(remote.document)
    result["pod_id"] = pod_id
    result["transport"] = {
        "state": "connected",
        "destination": connection.destination,
        "port": connection.port,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")), flush=True)
    else:
        text = str(result.get("text", ""))
        if text:
            print(text, end="" if text.endswith("\n") else "\n", flush=True)
        print(
            f"next_cursor={result.get('next_cursor', '')} has_more={str(result.get('has_more', False)).lower()}",
            file=sys.stderr,
            flush=True,
        )
    if args.cursor_file:
        try:
            _write_cursor(args.cursor_file, pod_id, args.session, result["next_cursor"])
        except OSError as exc:
            print(f"cannot update cursor file: {exc}", file=sys.stderr, flush=True)
            return 2
    return 0


def _read_cursor(path: Path, pod_id: str, session: str) -> str:
    source = path.expanduser()
    try:
        if source.stat().st_size > 16384:
            raise RunPodMonitorError("cursor file exceeds 16384 bytes")
        document = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except RunPodMonitorError:
        raise
    except (OSError, ValueError) as exc:
        raise RunPodMonitorError(f"cannot read cursor file: {exc}") from exc
    if not isinstance(document, dict):
        raise RunPodMonitorError("cursor file must contain a JSON object")
    if document.get("pod_id") != pod_id or document.get("session") != session:
        raise RunPodMonitorError("cursor file belongs to another pod or session")
    cursor = document.get("cursor")
    if not isinstance(cursor, str) or not cursor or len(cursor) > 4096 or _has_control(cursor):
        raise RunPodMonitorError("cursor file has no usable cursor")
    return cursor


def _write_cursor(path: Path, pod_id: str, session: str, cursor: str) -> None:
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": 1, "pod_id": pod_id, "session": session, "cursor": cursor}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
    try:
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _emit_status(document: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")), flush=True)
        return
    if document.get("error"):
        print(document["error"], file=sys.stderr, flush=True)
        return
    pod = document.get("pod", {})
    print(
        f"RunPod {pod.get('name', pod.get('id', '?'))}: {pod.get('runtimeStatus', 'unknown')} "
        f"desired={pod.get('desiredStatus', 'unknown')}",
        flush=True,
    )
    transport = document.get("transport", {})
    destination = transport.get("destination", "")
    address = f" {destination}:{transport.get('port')}" if destination else ""
    print(f"  SSH: {transport.get('state', 'unknown')}{address}", flush=True)
    ultron = document.get("ultron")
    if isinstance(ultron, dict):
        for job in ultron.get("jobs", []):
            print(f"  {job.get('name', job.get('session', '?'))}: {job.get('state', 'unknown')}", flush=True)
            tail = job.get("log", {}).get("tail")
            if tail:
                print(str(tail), flush=True)
        for pipeline in ultron.get("pipelines", []):
            stages = ", ".join(
                f"{stage.get('name', '?')}={stage.get('state', '?')}"
                for stage in pipeline.get("stages", [])
            )
            print(f"  pipeline {pipeline.get('name', '?')}: {stages or 'no stages'}", flush=True)
        gpu = ultron.get("resources", {}).get("gpu", {})
        for device in gpu.get("devices", []):
            print(
                f"  GPU {device.get('index')}: {device.get('utilization_percent')}% "
                f"{device.get('memory_used_mib')}/{device.get('memory_total_mib')} MiB",
                flush=True,
            )
        for disk in ultron.get("resources", {}).get("disks", []):
            print(f"  disk {disk.get('path')}: {disk.get('free_percent')}% free", flush=True)
        for run in ultron.get("responses", {}).get("runs", []):
            print(f"  responses {run.get('run_id', '?')}: {run.get('status', 'unknown')}", flush=True)
            for preview in run.get("previews", []):
                print(f"    {preview.get('role', '?')}: {preview.get('text', '')}", flush=True)
        for issue in ultron.get("issues", []):
            print(f"  ultron {issue.get('code', '')}: {issue.get('message', '')}", flush=True)
        for error in ultron.get("errors", []):
            if isinstance(error, dict):
                print(f"  ultron {error.get('source', 'error')}: {error.get('message', '')}", flush=True)
            else:
                print(f"  ultron error: {error}", flush=True)
    for issue in document.get("issues", []):
        print(f"  {issue.get('code')}: {issue.get('message')}", flush=True)
    provider = document.get("provider_logs", {})
    for entry in provider.get("entries", []):
        print(f"  runpod: {entry.get('line', '')}", flush=True)
    if provider.get("reason"):
        print(f"  runpod logs: {provider['reason']}", flush=True)
    if document.get("watch"):
        print(f"  watch: {document['watch']['reason']}", flush=True)


def _command(argv: Sequence[str], timeout: int | float) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(argv),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RunPodMonitorError(f"command not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RunPodMonitorError(f"command timed out: {argv[0]}") from exc
    if len(result.stdout.encode()) > MAX_COMMAND_OUTPUT_BYTES or len(result.stderr.encode()) > MAX_COMMAND_OUTPUT_BYTES:
        raise RunPodMonitorError(f"command output exceeded {MAX_COMMAND_OUTPUT_BYTES} bytes: {argv[0]}")
    return result


def _single_json(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise RunPodMonitorError("command returned invalid JSON") from exc


def _last_json_line(text: str) -> tuple[dict[str, Any] | None, str]:
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].strip()
        if not line:
            continue
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value, "\n".join(lines[:index]).strip()
    return None, text.strip()


def _command_error(label: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
    return f"{label} failed: {_bounded(detail)}"


def _bounded(value: str, limit: int = 1200) -> str:
    return value if len(value) <= limit else value[:limit] + "..."


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON number: {value}")


def _has_control(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
