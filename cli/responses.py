"""Per-run response JSON with a durable journal for in-flight chunks."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from ultron.train.io import atomic_write_text

from ultron.cli.model import (
    JobEnded, JobError, JobEvent, JobMeta, ModelResponseDelta,
    ModelResponseFinished, ModelResponseStarted,
)

_RESPONSE_EVENTS = (ModelResponseStarted, ModelResponseDelta, ModelResponseFinished, JobEnded, JobError)


def default_response_directory() -> Path:
    override = os.environ.get("ULTRON_RESPONSES_DIR")
    return Path(override).expanduser() if override else Path(__file__).resolve().parents[1] / "data" / "responses"


def _apply_record(document: dict, records: dict[str, dict], event: dict) -> None:
    kind = event["kind"]
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
    """Read a snapshot plus all durably journaled chunks, including after a crash."""
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
    def __init__(self, meta: JobMeta, directory: Path | None = None) -> None:
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
        }
        self._records: dict[str, dict] = {}
        self._first_delta: set[str] = set()
        self._last_snapshot = 0.0
        self.flush()

    def record(self, event: JobEvent) -> None:
        if not isinstance(event, _RESPONSE_EVENTS):
            return
        with self._lock:
            if self.document["status"] != "running":
                return
            self._record(event)

    def _record(self, event: JobEvent) -> None:
        sequence = self.document["journal_events"] + 1
        payload = asdict(event)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"sequence": sequence, "event": payload}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _apply_record(self.document, self._records, payload)
        self.document["journal_events"] = sequence
        if isinstance(event, ModelResponseDelta):
            first = event.response_id not in self._first_delta
            self._first_delta.add(event.response_id)
            if not first and time.monotonic() - self._last_snapshot < 1.0:
                return
        self.flush()

    def finish(self, status: str, error: str | None = None) -> None:
        with self._lock:
            if self.document["status"] != "running":
                return
            self._record(JobError(error or "Response recording stopped", "cancel" if status == "cancelled" else "drive", 0))

    def flush(self) -> None:
        atomic_write_text(
            self.path,
            json.dumps(self.document, ensure_ascii=False, indent=2) + "\n",
        )
        # Constant-size pointers keep current activity discoverable after many runs.
        roles = {getattr(record["role"], "value", record["role"]) for record in self.document["responses"]}
        for role in roles & {"attacker", "defender"}:
            atomic_write_text(
                self.path.parent.parent / f".latest-{role}.json",
                json.dumps({"schema_version": 1, "run_id": self.run_id, "role": role,
                            "path": f"{self.run_id}/responses.json",
                            "updated_at": datetime.now(timezone.utc).isoformat()}) + "\n",
            )
        self._last_snapshot = time.monotonic()
