"""Bounded, read-only snapshots for agents supervising local Ultron runs."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ultron.cli import jobs
from ultron.cli.catalog import repo_root

MAX_DIRECTORY_ENTRIES = 512
MAX_JOBS = 128
MAX_STATE_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_RESPONSE_RUNS = 20
MAX_PREVIEW_CHARS = 600
MAX_LOG_BYTES = 8192
GPU_TIMEOUT_SECONDS = 3
_TERMINAL_ESCAPE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _plain(text: str) -> str:
    return _TERMINAL_ESCAPE.sub("", text)


def _issue(snapshot: dict, code: str, severity: str, message: str, **context: object) -> None:
    snapshot["issues"].append({"code": code, "severity": severity, "message": message, **context})


def _error(snapshot: dict, source: str, exc: Exception) -> None:
    snapshot["errors"].append({"source": source, "message": str(exc)[:MAX_PREVIEW_CHARS]})


def _read(path: Path, limit: int, *, tail: bool = False) -> tuple[bytes, bool]:
    # Refuse pipes/devices: opening those can block even with a byte limit.
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"not a regular file: {path}")
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if tail:
            handle.seek(max(0, size - limit))
        data = handle.read(limit + (0 if tail else 1))
    return data[:limit], size > limit or len(data) > limit


def _entries(directory: Path, budget: list[int]) -> tuple[list[Path], bool]:
    result = []
    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if budget[0] <= 0:
                    return result, True
                budget[0] -= 1
                # Skip symlinks so scans cannot escape their configured directory.
                if not entry.is_symlink():
                    result.append(Path(entry.path))
    except FileNotFoundError:
        return [], False
    return result, False


def _pid_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _integer(value: object) -> int | None:
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return None


def _job_snapshot(info: jobs.SessionInfo, snapshot: dict, now: float, stale_seconds: float) -> dict:
    state = info.state.value
    if state == "dead":
        state = "succeeded" if info.exit_code == 0 else "failed" if info.exit_code is not None else "exited"
    result = {
        "session": info.name, "state": state, "pid": info.pid,
        "command": info.command[:MAX_PREVIEW_CHARS], "exit_code": info.exit_code,
        "log": {"path": str(info.log_path), "available": False},
    }
    if state in {"failed", "missing"}:
        _issue(snapshot, f"job_{state}", "error", f"Session {info.name} is {state}.", session=info.name)
    elif state == "exited":
        _issue(snapshot, "job_exit_unknown", "warning", f"Session {info.name} exited without a recorded exit code.", session=info.name)
    try:
        info_stat = info.log_path.stat()
        data, truncated = _read(info.log_path, MAX_LOG_BYTES, tail=True)
        age = max(0.0, now - info_stat.st_mtime)
        result["log"].update(available=True, size_bytes=info_stat.st_size, age_seconds=round(age, 1),
                             quiet=state == "running" and age >= stale_seconds,
                             tail=_plain(data.decode("utf-8", errors="replace")), truncated=truncated)
        if result["log"]["quiet"]:
            _issue(snapshot, "log_quiet", "info", f"No log update for {age:.0f} seconds.", session=info.name)
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        _error(snapshot, f"log:{info.name}", exc)
    return result


def _state_fields(path: Path) -> dict[str, str]:
    data, truncated = _read(path, MAX_STATE_BYTES)
    if truncated:
        raise ValueError(f"stage file exceeds {MAX_STATE_BYTES} bytes: {path}")
    fields = {}
    for line in data.decode("utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in fields:
            raise ValueError(f"invalid stage state: {path}")
        fields[key] = value
    return fields


def _mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return -1


def _stage(path: Path, snapshot: dict, now: float, session: str | None,
           manifest: dict | None = None) -> dict | None:
    try:
        fields = _state_fields(path)
        if not fields or not any(key in fields for key in ("attempt", "status", "pid", "state")):
            raise ValueError(f"missing stage metadata: {path}")
        pid = _integer(fields.get("pid"))
        if "pid" in fields and (pid is None or pid <= 0):
            raise ValueError(f"invalid stage pid: {path}")
        for key in ("attempt", "max_attempts", "status"):
            if key in fields and (_integer(fields[key]) is None or _integer(fields[key]) < (0 if key == "status" else 1)):
                raise ValueError(f"invalid stage {key}: {path}")
        filename_state = {".done": "succeeded", ".failed": "failed", ".running": "running", ".retrying": "retrying"}[path.suffix]
        state = fields.get("state", filename_state)
        state = {"done": "succeeded", "complete": "succeeded"}.get(state, state)
        if state not in {"succeeded", "failed", "running", "retrying"}:
            raise ValueError(f"invalid stage state {state!r}: {path}")
        owner_alive = _pid_alive(pid) if state in {"running", "retrying"} else None
        if owner_alive is False:
            state = "orphaned"
        historical = bool(manifest and manifest.get("run_id") and fields.get("run_id") != manifest["run_id"])
        scoped = not historical and (session is None or fields.get("session") == session)
        return {
            "name": path.stem, "path": str(path), "pipeline": fields.get("pipeline", path.parent.name),
            "state": state, "recorded_state": filename_state, "session": fields.get("session") or None,
            "in_scope": scoped, "pid": pid, "owner_alive": owner_alive,
            "run_id": fields.get("run_id"), "historical": historical,
            "attempt": _integer(fields.get("attempt")), "max_attempts": _integer(fields.get("max_attempts")),
            "exit_code": _integer(fields.get("status")), "started_at": fields.get("started_at"),
            "retry_at": _integer(fields.get("retry_at")),
            "updated_at": fields.get("updated_at"), "age_seconds": round(max(0.0, now - path.stat().st_mtime), 1),
        }
    except FileNotFoundError:
        # Writers remove the previous state after atomically publishing the next.
        return None
    except (OSError, UnicodeError, ValueError) as exc:
        _error(snapshot, str(path), exc)
        return None


def _pipelines(directory: Path, snapshot: dict, now: float, session: str | None) -> list[dict]:
    budget = [MAX_DIRECTORY_ENTRIES]
    try:
        entries, truncated = _entries(directory, budget)
        directories = [directory]
        candidates = [entry for entry in entries if entry.is_file()]
        for entry in entries:
            if entry.is_dir():
                directories.append(entry)
                children, limited = _entries(entry, budget)
                truncated |= limited
                candidates.extend(child for child in children if child.is_file())
        # Atomic replacements can briefly leave multiple state files. Prefer the newest.
        selected: dict[tuple[Path, str], Path] = {}
        for path in candidates:
            if path.suffix not in {".done", ".failed", ".running", ".retrying"}:
                continue
            key = (path.parent, path.stem)
            previous = selected.get(key)
            if previous is None or _mtime(path) > _mtime(previous):
                selected[key] = path
        groups: dict[Path, dict] = {}
        manifests = {}
        for parent in directories:
            try:
                manifest = _state_fields(parent / ".pipeline")
                if not manifest.get("run_id"):
                    raise ValueError(f"pipeline manifest has no run_id: {parent / '.pipeline'}")
                manifests[parent] = manifest
                groups[parent] = {"name": manifest.get("pipeline", parent.name), "path": str(parent),
                                  "run_id": manifest["run_id"], "session": manifest.get("session") or None,
                                  "pid": _integer(manifest.get("pid")), "started_at": manifest.get("started_at"), "stages": []}
            except FileNotFoundError:
                pass
            except (OSError, UnicodeError, ValueError) as exc:
                _error(snapshot, str(parent / ".pipeline"), exc)
        for path in sorted(selected.values()):
            stage = _stage(path, snapshot, now, session, manifests.get(path.parent))
            if stage is None:
                continue
            group = groups.setdefault(path.parent, {"name": stage["pipeline"], "path": str(path.parent), "stages": []})
            group["stages"].append(stage)
            if stage["in_scope"] and stage["state"] in {"failed", "orphaned"}:
                _issue(snapshot, f"stage_{stage['state']}", "error", f"Stage {stage['name']} is {stage['state']}.",
                       pipeline=stage["pipeline"], stage=stage["name"], session=stage["session"])
            elif stage["in_scope"] and stage["state"] == "retrying":
                _issue(snapshot, "stage_retrying", "info", f"Stage {stage['name']} is waiting to retry.", pipeline=stage["pipeline"], stage=stage["name"])
        if truncated:
            _issue(snapshot, "state_scan_truncated", "warning", f"Stage scan reached its {MAX_DIRECTORY_ENTRIES} entry limit.")
        snapshot["pipeline_scan"] = {"directory": str(directory), "truncated": truncated}
        return list(groups.values())
    except OSError as exc:
        _error(snapshot, "pipelines", exc)
        return []


def _latest_responses(directory: Path, snapshot: dict) -> dict[Path, str]:
    paths = {}
    for role in ("attacker", "defender"):
        pointer = directory / f".latest-{role}.json"
        try:
            data, truncated = _read(pointer, MAX_STATE_BYTES)
            if truncated:
                raise ValueError(f"response pointer exceeds read limit: {pointer}")
            document = json.loads(data)
            if not isinstance(document, dict) or not isinstance(document.get("path"), str):
                raise ValueError(f"invalid response pointer: {pointer}")
            target = (directory / document["path"]).resolve()
            if not target.is_relative_to(directory.resolve()) or target.name != "responses.json":
                raise ValueError(f"response pointer escapes archive directory: {pointer}")
            paths[target] = role
        except FileNotFoundError:
            continue
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            _error(snapshot, str(pointer), exc)
    return paths


def _responses(directory: Path, snapshot: dict, now: float) -> dict:
    result: dict = {"directory": str(directory), "runs": [], "truncated": False, "scope": "archive_history"}
    try:
        latest = _latest_responses(directory, snapshot)
        entries, result["truncated"] = _entries(directory, [MAX_DIRECTORY_ENTRIES])
        paths = [entry / "responses.json" for entry in entries if entry.is_dir()]
        paths += [directory / "responses.json", *latest]
        paths = [path for path in paths if path.is_file() and not path.is_symlink()]
        paths = list(dict.fromkeys(path.resolve() for path in paths))
        paths.sort(key=lambda path: (path in latest, _mtime(path)), reverse=True)
        result["truncated"] |= len(paths) > MAX_RESPONSE_RUNS
        for path in paths[:MAX_RESPONSE_RUNS]:
            run = {"path": str(path), "run_id": path.parent.name, "available": False,
                   "latest_for_role": latest.get(path)}
            result["runs"].append(run)
            try:
                data, truncated = _read(path, MAX_RESPONSE_BYTES)
                run["age_seconds"] = round(max(0.0, now - path.stat().st_mtime), 1)
                if truncated:
                    run.update(truncated=True, reason="snapshot_exceeds_read_limit")
                    result["truncated"] = True
                    continue
                document = json.loads(data)
                if not isinstance(document, dict) or not isinstance(document.get("responses"), list):
                    raise ValueError(f"invalid response archive: {path}")
                records = document["responses"]
                if any(not isinstance(record, dict) for record in records):
                    raise ValueError(f"invalid response record: {path}")
                counts = Counter(str(record.get("status", "unknown")) for record in records)
                meta = document.get("meta")
                run.update(available=True, status=str(document.get("status", "unknown"))[:80],
                           generation=_integer(meta.get("generation")) if isinstance(meta, dict) else None,
                           created_at=str(document.get("created_at", ""))[:80],
                           error=str(document["error"])[:MAX_PREVIEW_CHARS] if document.get("error") else None,
                           response_count=len(records), counts=dict(counts), previews=[{
                               "response_id": str(record.get("response_id", ""))[:MAX_PREVIEW_CHARS],
                               "role": str(record.get("role", ""))[:80], "status": str(record.get("status", "unknown"))[:80],
                               "text": _plain(str(record.get("text", ""))[-MAX_PREVIEW_CHARS:]),
                               "truncated": len(str(record.get("text", ""))) > MAX_PREVIEW_CHARS,
                           } for record in records[-3:]])
                # Snapshot summaries deliberately do not replay potentially unbounded journals.
                journal = path.with_name("response-events.jsonl")
                run["journal_replayed"] = False
                run["journal_present"] = journal.is_file()
            except FileNotFoundError:
                run.update(reason="snapshot_disappeared")
            except (OSError, ValueError, UnicodeError, RecursionError) as exc:
                _error(snapshot, str(path), exc)
    except OSError as exc:
        _error(snapshot, "responses", exc)
    return result


def _gpu() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi_not_found", "devices": []}
    command = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        # Redirect output to files so even broken drivers cannot fill Python memory.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=GPU_TIMEOUT_SECONDS, check=False)
            stdout.seek(0)
            output = stdout.read(16 * 1024 + 1)
            if completed.returncode != 0:
                stderr.seek(0)
                return {"available": False, "reason": "query_failed", "detail": stderr.read(600).decode(errors="replace"), "devices": []}
        if len(output) > 16 * 1024:
            return {"available": False, "reason": "query_output_too_large", "devices": []}
        devices = []
        for values in csv.reader(output.decode("utf-8").splitlines(), skipinitialspace=True):
            if len(values) != 6:
                raise ValueError("unexpected nvidia-smi columns")
            device = {"index": _integer(values[0]), "name": values[1][:200]}
            for key, value in zip(("utilization_percent", "memory_used_mib", "memory_total_mib", "temperature_c"), values[2:]):
                try:
                    number = float(value)
                    device[key] = number if math.isfinite(number) else None
                except ValueError:
                    device[key] = None
            devices.append(device)
        return {"available": bool(devices), "reason": None if devices else "no_devices", "devices": devices}
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "query_timeout", "devices": []}
    except (OSError, ValueError, UnicodeError) as exc:
        return {"available": False, "reason": "query_error", "detail": str(exc)[:600], "devices": []}


def _disk(path: Path, snapshot: dict) -> dict:
    measured = path.absolute()
    try:
        while not measured.exists() and measured != measured.parent:
            measured = measured.parent
        usage = shutil.disk_usage(measured)
        return {"available": True, "path": str(path), "measured_path": str(measured),
                "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free,
                "free_percent": round(usage.free / usage.total * 100, 2) if usage.total else None}
    except OSError as exc:
        _error(snapshot, f"disk:{path}", exc)
        return {"available": False, "path": str(path), "reason": str(exc)[:600]}


def collect_status(*, session: str | None = None, root: Path | None = None,
                   state_dir: Path | None = None, responses_dir: Path | None = None,
                   stale_seconds: float = 300, include_gpu: bool = True) -> dict:
    """Collect local evidence without loading models, replaying journals, or changing jobs.

    ``session`` scopes failure issues to that tmux job and explicitly associated
    stages. Response runs are historical snapshot summaries, never job liveness.
    """
    if not math.isfinite(stale_seconds) or stale_seconds < 0:
        raise ValueError("stale_seconds must be finite and non-negative")
    root = Path(root) if root is not None else repo_root()
    state_dir = Path(state_dir or os.environ.get("ULTRON_PIPELINE_STATE_DIR") or root / "data" / "job-state").expanduser()
    responses_dir = Path(responses_dir or os.environ.get("ULTRON_RESPONSES_DIR") or root / "data" / "responses").expanduser()
    now = time.time()
    snapshot: dict = {
        "schema_version": 1, "observed_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "session": session, "jobs": [], "pipelines": [], "responses": {}, "resources": {}, "issues": [], "errors": [],
        "limits": {"directory_entries": MAX_DIRECTORY_ENTRIES, "jobs": MAX_JOBS, "state_bytes": MAX_STATE_BYTES,
                   "response_bytes": MAX_RESPONSE_BYTES, "response_runs": MAX_RESPONSE_RUNS,
                   "preview_chars": MAX_PREVIEW_CHARS, "log_bytes": MAX_LOG_BYTES},
    }
    try:
        sessions = (jobs.session_status(session, root=root),) if session else jobs.list_sessions(root=root)
        snapshot["jobs"] = [_job_snapshot(info, snapshot, now, stale_seconds) for info in sessions[:MAX_JOBS]]
        if not sessions:
            _issue(snapshot, "no_jobs", "info", "No Ultron tmux jobs found.")
        if len(sessions) > MAX_JOBS:
            _issue(snapshot, "job_scan_truncated", "warning", f"Only {MAX_JOBS} jobs included.")
    except (jobs.JobsError, OSError) as exc:
        _error(snapshot, "jobs", exc)
    snapshot["pipelines"] = _pipelines(state_dir, snapshot, now, session)
    snapshot["responses"] = _responses(responses_dir, snapshot, now)
    disks = [_disk(path, snapshot) for path in dict.fromkeys((root, state_dir, responses_dir))]
    snapshot["resources"]["disk"] = disks[0]
    snapshot["resources"]["disks"] = disks
    for disk in disks:
        if disk.get("free_percent") is not None and disk["free_percent"] < 5:
            _issue(snapshot, "disk_low", "warning", "Less than 5% disk space remains.", path=disk["path"])
    snapshot["resources"]["gpu"] = _gpu() if include_gpu else {"available": False, "reason": "disabled", "devices": []}
    snapshot["exit_code"] = status_exit_code(snapshot)
    return snapshot


def status_exit_code(snapshot: dict) -> int:
    """0: no observed failure; 1: job/stage failure; 2: incomplete monitoring.

    Quiet logs, retries, historical response errors and unavailable GPU telemetry
    do not establish a failed run. A recorded failure takes priority over errors.
    """
    if any(issue.get("severity") == "error" for issue in snapshot.get("issues", [])):
        return 1
    if any(job.get("state") == "exited" for job in snapshot.get("jobs", [])):
        return 2
    return 2 if snapshot.get("errors") else 0
