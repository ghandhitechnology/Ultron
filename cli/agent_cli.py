"""Noninteractive commands for agents supervising detached training jobs."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path


def add_agent_commands(sub: argparse._SubParsersAction) -> None:
    for name in ("status", "watch"):
        parser = sub.add_parser(name, help="Inspect jobs, stages, response capture, and resources." if name == "status" else "Poll job health for a bounded time without attaching.")
        parser.add_argument("--session", help="Scope job health to one tmux session.")
        parser.add_argument("--json", action="store_true", help="Emit JSON; watch emits one JSON object per line.")
        parser.add_argument("--details", action="store_true", help="Include bounded log tails and response text previews.")
        parser.add_argument("--response-limit", type=int, default=4, help="Response archives to include, 0–20; default 4.")
        parser.add_argument("--state-dir", type=Path, help="Pipeline state directory or parent of pipeline directories.")
        parser.add_argument("--responses-dir", type=Path, help="Response archive directory.")
        parser.add_argument("--stale-seconds", type=float, default=300, help="Report quiet logs after this age; quiet does not mean failed.")
        parser.add_argument("--no-gpu", action="store_true", help="Skip the nvidia-smi probe.")
        if name == "watch":
            parser.add_argument("--timeout", type=float, default=60, help="Polling duration in seconds, default 60; exit 124 on timeout.")
            parser.add_argument("--interval", type=float, default=5, help="Seconds between polls, default 5.")
    logs = sub.add_parser("logs", help="Read bounded logs and return a cursor for the next read.")
    logs.add_argument("--session", required=True)
    logs.add_argument("--cursor", help="The next_cursor returned by the previous read.")
    logs.add_argument("--max-bytes", type=int, default=16384)
    logs.add_argument("--tail", type=int, default=50, help="Initial tail lines; ignored when a cursor is provided.")
    logs.add_argument("--json", action="store_true")
    job = sub.add_parser("job", help="Start, stop, or restart a detached tmux job.")
    actions = job.add_subparsers(dest="job_action", required=True)
    for name in ("start", "stop", "restart"):
        action = actions.add_parser(name)
        action.add_argument("--json", action="store_true")
        action.add_argument("session")
        if name == "start":
            action.add_argument("command", nargs=argparse.REMAINDER, help="Command argv after --; executed from the repository root.")


def _session(value: str | None) -> None:
    if value is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value):
        raise ValueError("session must contain 1–40 letters, digits, underscores, or hyphens")


def _emit(document: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(document, ensure_ascii=False, separators=(",", ":")), flush=True)
        return
    if "error" in document:
        print(document["error"], file=sys.stderr, flush=True)
        return
    if "action" in document:
        print(f"{document['session']}: {document['result']}", flush=True)
        return
    if "next_cursor" in document:
        text = document.get("text", "")
        if text:
            print(text, end="" if text.endswith("\n") else "\n", flush=True)
        print(f"next_cursor={document['next_cursor']} has_more={str(document.get('has_more', False)).lower()}", file=sys.stderr, flush=True)
        return
    print(f"Ultron status {document.get('observed_at', '')}", flush=True)
    for item in document.get("jobs", []):
        print(f"  {item.get('name', item.get('session', '?'))}: {item.get('state', 'unknown')} exit={item.get('exit_code')}")
        if item.get("log", {}).get("tail"):
            print(item["log"]["tail"])
    for pipeline in document.get("pipelines", []):
        labels = []
        for stage in pipeline.get("stages", []):
            suffix = " [history]" if stage.get("historical") else " [other session]" if stage.get("in_scope") is False else ""
            labels.append(f"{stage.get('name', '?')}={stage.get('state', '?')}{suffix}")
        stages = ", ".join(labels)
        print(f"  pipeline {pipeline.get('name', '?')}: {stages}")
    resources = document.get("resources", {})
    for disk in resources.get("disks", []):
        print(f"  disk {disk.get('path')}: {disk.get('free_percent')}% free")
    gpu = resources.get("gpu", {})
    for device in gpu.get("devices", []):
        print(f"  GPU {device.get('index')}: {device.get('utilization_percent')}% utilization, {device.get('memory_used_mib')}/{device.get('memory_total_mib')} MiB")
    for run in document.get("responses", {}).get("runs", []):
        print(f"  responses {run.get('run_id')}: {run.get('status', 'unavailable')} {run.get('path')}")
        for preview in run.get("previews", []):
            print(f"    {preview.get('role')}: {preview.get('text', '')}")
    for issue in document.get("issues", []):
        print(f"  {issue.get('severity', 'info')} {issue.get('code', '')}: {issue.get('message', '')}")
    for error in document.get("errors", []):
        print(f"  monitor error: {error}")
    if not document.get("jobs"):
        print("  no jobs found")
    if "watch" in document:
        print(f"  watch: {document['watch']['reason']}")
    sys.stdout.flush()


def run_agent_command(args: argparse.Namespace) -> int:
    from ultron.cli.jobs import JobsError

    try:
        _session(getattr(args, "session", None))
        if args.cmd == "logs":
            from ultron.cli.agent_logs import read_agent_logs

            result = read_agent_logs(args.session, cursor=args.cursor, max_bytes=args.max_bytes, tail=args.tail)
            _emit(result, as_json=args.json)
            return 0
        if args.cmd == "job":
            return _job(args)
        if not math.isfinite(args.stale_seconds) or args.stale_seconds < 0:
            raise ValueError("--stale-seconds must be finite and >= 0")
        if not 0 <= args.response_limit <= 20:
            raise ValueError("--response-limit must be between 0 and 20")
        if args.cmd == "watch":
            if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
                raise ValueError("--timeout must be > 0 and <= 3600 seconds")
            if not math.isfinite(args.interval) or not 0.1 <= args.interval <= 60:
                raise ValueError("--interval must be between 0.1 and 60 seconds")
        return _monitor(args)
    except BrokenPipeError:
        return 0
    except (JobsError, OSError, ValueError) as exc:
        _emit({"schema_version": 1, "error": str(exc), "command": args.cmd}, as_json=args.json)
        return 2
    except KeyboardInterrupt:
        return 130


def _monitor(args: argparse.Namespace) -> int:
    from ultron.cli.monitor import collect_status, status_exit_code

    started = time.monotonic()
    snapshot = None
    while True:
        if snapshot is not None:
            elapsed = time.monotonic() - started
            if elapsed >= args.timeout:
                snapshot["watch"] = {"reason": "timeout", "elapsed_seconds": round(elapsed, 3),
                                     "timeout_seconds": args.timeout, "exit_code": 124}
                _emit(snapshot, as_json=args.json)
                return 124
        snapshot = collect_status(
            session=args.session, state_dir=args.state_dir,
            responses_dir=args.responses_dir, stale_seconds=args.stale_seconds,
            include_gpu=not args.no_gpu,
        )
        if not args.details:
            for job in snapshot.get("jobs", []):
                job.get("log", {}).pop("tail", None)
            for run in snapshot.get("responses", {}).get("runs", []):
                run.pop("previews", None)
        responses = snapshot.get("responses", {})
        runs = responses.get("runs", [])
        responses["sampled_runs"] = len(runs)
        responses["runs"] = runs[:args.response_limit]
        responses["truncated"] = responses.get("truncated", False) or len(runs) > args.response_limit
        code = status_exit_code(snapshot)
        if args.cmd == "status":
            _emit(snapshot, as_json=args.json)
            return code
        elapsed = time.monotonic() - started
        targeted = [job for job in snapshot.get("jobs", []) if job.get("name", job.get("session")) == args.session]
        complete = bool(args.session and targeted and all(job.get("state") == "succeeded" for job in targeted))
        unknown_exit = any(job.get("state") == "exited" for job in targeted)
        if code:
            reason, exit_code = ("monitor_error" if code == 2 else "failed"), code
        elif complete:
            reason, exit_code = "completed", 0
        elif unknown_exit:
            reason, exit_code = "unknown_exit", 2
        elif elapsed >= args.timeout:
            reason, exit_code = "timeout", 124
        else:
            reason, exit_code = "poll", None
        snapshot["watch"] = {"reason": reason, "elapsed_seconds": round(elapsed, 3),
                             "timeout_seconds": args.timeout, "exit_code": exit_code}
        _emit(snapshot, as_json=args.json)
        if exit_code is not None:
            return exit_code
        time.sleep(min(args.interval, args.timeout - elapsed))


def _job(args: argparse.Namespace) -> int:
    from ultron.cli.jobs import AlreadyRunning, log_path, restart_session, start_session, stop_session

    if args.job_action == "start":
        argv = args.command
        if argv and argv[0] == "--":
            argv = argv[1:]
        if not argv:
            raise ValueError("job start requires a command after --")
        result = start_session(args.session, argv, extra_env={"PYTHONUNBUFFERED": "1"})
        state = "already_running" if isinstance(result, AlreadyRunning) else "started"
    elif args.job_action == "restart":
        restart_session(args.session)
        state = "restarted"
    else:
        stop_session(args.session)
        state = "stopped"
    _emit({"schema_version": 1, "action": args.job_action, "session": args.session,
           "result": state, "log_path": str(log_path(args.session))}, as_json=args.json)
    return 0
