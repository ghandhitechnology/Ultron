"""Model text observation, independent of rollout and UI implementations."""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import Any, Literal
from urllib.request import Request, urlopen
from uuid import uuid4

ResponseStatus = Literal["streaming", "complete", "error", "cancelled"]


@dataclass(frozen=True)
class ResponseUpdate:
    kind: Literal["started", "delta", "finished"]
    response_id: str
    model: str = ""
    text: str = ""
    status: ResponseStatus = "streaming"
    error: str | None = None


_observer: ContextVar[Callable[[ResponseUpdate], None] | None] = ContextVar(
    "model_response_observer", default=None
)


@contextmanager
def observe_responses(callback: Callable[[ResponseUpdate], None]) -> Iterator[None]:
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


class ModelResponse:
    """Wrap generation and publish each received text chunk before returning."""

    def __init__(self, model: str) -> None:
        self.response_id = uuid4().hex
        self.model = model
        self.text = ""
        self._emit = _observer.get()

    def _publish(self, kind, **kwargs) -> None:
        if self._emit is not None:
            self._emit(ResponseUpdate(kind, self.response_id, model=self.model, **kwargs))

    def __enter__(self) -> ModelResponse:
        self._publish("started")
        return self

    def write(self, text: str) -> None:
        if text:
            self.text += text
            self._publish("delta", text=text)

    def __exit__(self, exc_type, exc, traceback) -> None:
        status = "complete" if exc is None else "error"
        if isinstance(exc, CancelledError) or (exc is not None and not isinstance(exc, Exception)):
            status = "cancelled"
        self._publish("finished", status=status, error=None if exc is None else str(exc))


def stream_chat_completion(
    endpoint: str,
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    api_key: str | None = None,
    timeout_s: float = 60,
) -> str:
    """Read a chat-completions SSE endpoint and publish provider deltas live.

    Pass the full endpoint, for example http://localhost:8000/v1/chat/completions.
    A run_turn implementation can call this within drive_job to populate both
    its battle view and response archive. The returned text is unmodified.
    """
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        endpoint,
        data=json.dumps({"model": model, "messages": list(messages), "stream": True}).encode(),
        headers=headers,
        method="POST",
    )
    with ModelResponse(model) as response:
        with urlopen(request, timeout=timeout_s) as stream:
            data: list[str] = []
            for raw in stream:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                elif not line and data:
                    payload = "\n".join(data)
                    data.clear()
                    if payload == "[DONE]":
                        return response.text
                    event = json.loads(payload)
                    if "error" in event:
                        raise RuntimeError(f"model stream error: {event['error']}")
                    for choice in event.get("choices", []):
                        if choice.get("index", 0) == 0:
                            text = choice.get("delta", {}).get("content")
                            if isinstance(text, str):
                                response.write(text)
            raise RuntimeError("model stream ended before [DONE]")
