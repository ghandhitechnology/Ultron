from __future__ import annotations

import queue
import threading
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from ultron.cli.app import SimApp
from ultron.cli.demo import origin_clock
from ultron.cli.model import JobError, JobEvent, JobMeta, JobSnapshot, Phase, apply
from ultron.cli.observe import EpisodeCase, drive_job
from ultron.cli.responses import ResponseArchive
from ultron.train.episode_runner import EpisodeRunner

SENTINEL = object()


def run_live_job(
    meta: JobMeta,
    runner: EpisodeRunner,
    cases: Iterable[EpisodeCase],
    *,
    clock=None,
    screenshot: Path | None = None,
    sim: bool = True,
    responses_dir: Path | None = None,
) -> JobSnapshot:
    events: queue.Queue[JobEvent | object] = queue.Queue()
    cancel = threading.Event()
    archive = ResponseArchive(meta, responses_dir)
    clock = clock or origin_clock()

    def emit(event: JobEvent) -> None:
        events.put(event)

    def worker() -> None:
        try:
            drive_job(
                meta, runner, cases, emit=emit, clock=clock,
                responses_dir=responses_dir, cancel=cancel, archive=archive,
            )
        except Exception:
            pass
        finally:
            events.put(SENTINEL)

    thread = threading.Thread(target=worker, name="ultron-sim-driver", daemon=True)
    thread.start()
    app = SimApp(meta, events, sentinel=SENTINEL, sim=sim)
    app.snapshot = replace(app.snapshot, responses_path=str(archive.path.resolve()))
    try:
        if screenshot is not None:
            _export_screenshot(app, screenshot)
        else:
            app.run()
    finally:
        cancel.set()
        thread.join(timeout=2)
        if thread.is_alive():
            archive.finish("cancelled", "Battle view closed")
    # Preserve worker results that arrived after the last UI refresh.
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            break
        if event is not SENTINEL:
            app.snapshot = apply(app.snapshot, event)
    if app.snapshot.phase not in (Phase.COMPLETE, Phase.FAILED):
        app.snapshot = apply(app.snapshot, JobError("Battle view closed", "cancel", clock()))
    return app.snapshot


def _export_screenshot(app, path: Path) -> None:
    import asyncio

    async def capture() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(500):
                await pilot.pause(0.05)
                if app.snapshot.phase in (Phase.COMPLETE, Phase.FAILED):
                    await pilot.pause(0.2)
                    break
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(app.export_screenshot())
            await pilot.click("#sandbox")
            await pilot.pause(0.1)
            expanded = path.with_name(f"{path.stem}_sandbox{path.suffix}")
            expanded.write_text(app.export_screenshot())

    asyncio.run(capture())
