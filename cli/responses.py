"""Per-run response JSON with a durable journal for in-flight chunks."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition, RLock, Thread
from typing import IO
from uuid import uuid4

from ultron.train.io import atomic_write_text

from ultron.cli.model import (
    EpisodeEnded, JobEnded, JobError, JobEvent, JobMeta, ModelResponseDelta,
    ModelResponseFinished, ModelResponseStarted, ProbeFinished, ProbeStarted,
    RestoreFinished, RestoreStarted, TurnEnded, TurnStarted,
)

_ARCHIVE_EVENTS = (
    ModelResponseStarted, ModelResponseDelta, ModelResponseFinished,
    RestoreStarted, RestoreFinished, TurnStarted, TurnEnded,
    ProbeStarted, ProbeFinished, EpisodeEnded, JobEnded, JobError,
)
_DURABILITY_BARRIERS = (
    ModelResponseFinished, RestoreFinished, TurnEnded, ProbeFinished,
    EpisodeEnded, JobEnded, JobError,
)
_TIMING_KINDS = {
    "restore_started", "restore_finished", "turn_started", "turn_ended",
    "probe_started", "probe_finished", "episode_ended", "job_ended", "error",
}


class _GroupCommitScheduler:
    """Run every timed group commit on one lazily started daemon thread."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._scheduled: dict[DurableGroupWriter, tuple[float, int]] = {}
        self._thread: Thread | None = None

    def schedule(self, writer: DurableGroupWriter, delay_s: float, token: int) -> None:
        with self._condition:
            self._scheduled[writer] = (time.monotonic() + delay_s, token)
            if self._thread is None or not self._thread.is_alive():
                self._thread = Thread(
                    target=self._run, name="response-group-commit", daemon=True,
                )
                self._thread.start()
            self._condition.notify()

    def cancel(self, writer: DurableGroupWriter, token: int) -> None:
        with self._condition:
            current = self._scheduled.get(writer)
            if current is not None and current[1] == token:
                self._scheduled.pop(writer, None)
                self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._scheduled:
                    self._condition.wait()
                now = time.monotonic()
                deadline = min(value[0] for value in self._scheduled.values())
                if deadline > now:
                    self._condition.wait(deadline - now)
                    continue
                due = [
                    (writer, token)
                    for writer, (scheduled_at, token) in self._scheduled.items()
                    if scheduled_at <= now
                ]
                for writer, _ in due:
                    self._scheduled.pop(writer, None)
            for writer, token in due:
                writer._timed_sync(token)


_GROUP_COMMIT_SCHEDULER = _GroupCommitScheduler()


class DurableGroupWriter:
    """Append immediately and group fsync calls behind a target durability window."""

    def __init__(
        self,
        path: Path,
        mode: str,
        *,
        sync_interval_s: float = 0.05,
        sync_bytes: int = 256 * 1024,
    ) -> None:
        if mode not in {"a", "w", "ab", "wb"}:
            raise ValueError("mode must be append or write, in text or binary form")
        if sync_interval_s < 0 or sync_bytes <= 0:
            raise ValueError("group commit interval must be nonnegative and byte threshold positive")
        self.path = Path(path)
        self._handle: IO = self.path.open(mode, encoding=None if "b" in mode else "utf-8")
        self._sync_interval_s = sync_interval_s
        self._sync_bytes = sync_bytes
        self._lock = RLock()
        self._pending_bytes = 0
        self._schedule_token: int | None = None
        self._next_token = 0
        self._error: OSError | None = None
        self._closed = False

    def write(self, data: str | bytes) -> None:
        with self._lock:
            self._raise_if_unwritable()
            try:
                self._handle.write(data)
                # Make new records visible to crash-recovery readers without forcing
                # stable storage for every token.
                self._handle.flush()
            except OSError as exc:
                self._error = exc
                raise
            size = len(data) if isinstance(data, bytes) else len(data.encode("utf-8"))
            self._pending_bytes += size
            if self._sync_interval_s == 0 or self._pending_bytes >= self._sync_bytes:
                self._cancel_sync_locked()
                self._sync_locked()
            elif self._schedule_token is None:
                self._next_token += 1
                self._schedule_token = self._next_token
                _GROUP_COMMIT_SCHEDULER.schedule(
                    self, self._sync_interval_s, self._schedule_token,
                )

    def barrier(self) -> None:
        with self._lock:
            self._raise_if_unwritable()
            self._cancel_sync_locked()
            self._sync_locked()

    def close(self, *, durable: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._cancel_sync_locked()
            error = self._error
            if durable and error is None:
                try:
                    self._sync_locked()
                except OSError as exc:
                    error = exc
            self._closed = True
            try:
                self._handle.close()
            except OSError as exc:
                error = error or exc
            if error is not None and durable:
                raise error

    def _timed_sync(self, token: int) -> None:
        with self._lock:
            if self._schedule_token != token:
                return
            self._schedule_token = None
            if self._closed or self._error is not None:
                return
            try:
                self._sync_locked()
            except OSError:
                # The next write or explicit completion barrier reports the error
                # to the request thread, which then fails closed.
                pass

    def _sync_locked(self) -> None:
        if not self._pending_bytes:
            return
        try:
            os.fsync(self._handle.fileno())
        except OSError as exc:
            self._error = exc
            raise
        self._pending_bytes = 0

    def _cancel_sync_locked(self) -> None:
        if self._schedule_token is not None:
            token, self._schedule_token = self._schedule_token, None
            _GROUP_COMMIT_SCHEDULER.cancel(self, token)

    def _raise_if_unwritable(self) -> None:
        if self._error is not None:
            raise self._error
        if self._closed:
            raise OSError(f"durable writer is closed: {self.path}")


def default_response_directory() -> Path:
    override = os.environ.get("ULTRON_RESPONSES_DIR")
    return Path(override).expanduser() if override else Path(__file__).resolve().parents[1] / "data" / "responses"


def _apply_record(document: dict, records: dict[str, dict], event: dict) -> None:
    kind = event["kind"]
    if kind in _TIMING_KINDS:
        document.setdefault("events", []).append(event)
    if kind == "model_response_started":
        record = {
            "response_id": event["response_id"],
            "episode_index": event["episode_index"],
            "turn_index": event["turn_index"],
            "role": event["role"],
            "model": event["model"],
            "text": "",
            "status": "streaming",
            "error": None,
            "started_at_s": event["at_s"],
            "finished_at_s": None,
        }
        records[event["response_id"]] = record
        document["responses"].append(record)
    elif kind == "model_response_delta":
        records[event["response_id"]]["text"] += event["text"]
    elif kind == "model_response_finished":
        records[event["response_id"]].update(
            status=event["status"], error=event["error"], finished_at_s=event["at_s"],
        )
    elif kind in ("job_ended", "error"):
        status = "complete" if kind == "job_ended" else "cancelled" if event["operation"] == "cancel" else "error"
        error = event.get("message")
        document.update(status=status, error=error)
        for record in records.values():
            if record["status"] == "streaming":
                record.update(status="cancelled" if status == "cancelled" else "error", error=error or "Response did not finish")


def load_response_archive(path: Path | str) -> dict:
    """Read a snapshot plus complete journal records written after that snapshot."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    journal = path.with_name("response-events.jsonl")
    if not journal.exists():
        return document
    records = {record["response_id"]: record for record in document["responses"]}
    applied = document.get("journal_events", 0)
    with journal.open("rb") as handle:
        for line in handle:
            # An abrupt process exit can leave the final journal write incomplete.
            if not line.endswith(b"\n"):
                break
            entry = json.loads(line)
            if entry["sequence"] > applied:
                _apply_record(document, records, entry["event"])
                document["journal_events"] = entry["sequence"]
    return document


class ResponseArchive:
    def __init__(
        self,
        meta: JobMeta,
        directory: Path | None = None,
        *,
        sync_interval_s: float = 0.05,
        sync_bytes: int = 256 * 1024,
    ) -> None:
        self._lock = RLock()
        self.run_id = uuid4().hex
        self.path = (Path(directory).expanduser() if directory is not None else default_response_directory()) / self.run_id / "responses.json"
        self.path.parent.mkdir(parents=True, exist_ok=False)
        self.journal_path = self.path.with_name("response-events.jsonl")
        self.document = {
            "schema_version": 1,
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "meta": asdict(meta),
            "status": "running",
            "error": None,
            "journal_events": 0,
            "responses": [],
            "events": [],
        }
        self._records: dict[str, dict] = {}
        self._first_delta: set[str] = set()
        self._written_roles: set[str] = set()
        self._journal: DurableGroupWriter | None = None
        self.flush()
        self._journal = DurableGroupWriter(
            self.journal_path, "a", sync_interval_s=sync_interval_s, sync_bytes=sync_bytes,
        )

    def record(self, event: JobEvent) -> None:
        self.record_many((event,))

    def record_many(self, events: Iterable[JobEvent]) -> None:
        accepted = [event for event in events if isinstance(event, _ARCHIVE_EVENTS)]
        if not accepted:
            return
        with self._lock:
            if self.document["status"] != "running":
                return
            snapshot = False
            durable = False
            terminal = False
            try:
                for event in accepted:
                    if self.document["status"] != "running":
                        break
                    self._record(event)
                    if isinstance(event, ModelResponseDelta) and event.response_id not in self._first_delta:
                        self._first_delta.add(event.response_id)
                        snapshot = True
                    snapshot = snapshot or isinstance(
                        event, (ModelResponseStarted, ModelResponseFinished, JobEnded, JobError)
                    )
                    durable = durable or isinstance(event, _DURABILITY_BARRIERS)
                terminal = self.document["status"] != "running"
                if durable:
                    self._journal_barrier()
                if snapshot:
                    self.flush()
            except OSError as exc:
                terminal = True
                self._record_storage_failure(exc)
                raise
            finally:
                if terminal:
                    # Every terminal path has already attempted an explicit barrier.
                    # Closing must not turn a confirmed commit into a false failure.
                    self._close_journal(durable=False)

    def _record(self, event: JobEvent) -> None:
        sequence = self.document["journal_events"] + 1
        payload = asdict(event)
        if self._journal is None:
            raise OSError("response journal is unavailable")
        self._journal.write(
            json.dumps({"sequence": sequence, "event": payload}, ensure_ascii=False) + "\n"
        )
        _apply_record(self.document, self._records, payload)
        self.document["journal_events"] = sequence

    def finish(self, status: str, error: str | None = None) -> None:
        self.record(JobError(
            error or "Response recording stopped",
            "cancel" if status == "cancelled" else "drive",
            0,
        ))

    def flush(self) -> None:
        with self._lock:
            self._journal_barrier()
            atomic_write_text(
                self.path,
                json.dumps(self.document, ensure_ascii=False, indent=2) + "\n",
            )
            # Constant-size pointers keep current activity discoverable after many runs.
            roles = {getattr(record["role"], "value", record["role"]) for record in self.document["responses"]}
            for role in (roles & {"attacker", "defender"}) - self._written_roles:
                atomic_write_text(
                    self.path.parent.parent / f".latest-{role}.json",
                    json.dumps({"schema_version": 1, "run_id": self.run_id, "role": role,
                                "path": f"{self.run_id}/responses.json",
                                "updated_at": datetime.now(timezone.utc).isoformat()}) + "\n",
                )
                self._written_roles.add(role)

    def _journal_barrier(self) -> None:
        if self._journal is not None:
            self._journal.barrier()

    def _record_storage_failure(self, exc: OSError) -> None:
        event = JobError(f"Response archive write failed: {exc}", "drive", 0)
        payload = asdict(event)
        try:
            self._record(event)
            self._journal_barrier()
        except OSError:
            if self.document["status"] != "error":
                _apply_record(self.document, self._records, payload)
        try:
            # The journal and snapshot are independent recovery paths. Avoid the
            # failed journal barrier while persisting the overriding error view.
            atomic_write_text(
                self.path,
                json.dumps(self.document, ensure_ascii=False, indent=2) + "\n",
            )
        except OSError:
            pass

    def _close_journal(self, *, durable: bool = True) -> None:
        if self._journal is not None:
            journal, self._journal = self._journal, None
            journal.close(durable=durable)
