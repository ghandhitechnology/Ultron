# Monitor a RunPod job from a local computer

Use the local `ultron runpod` commands to observe the RunPod container and the Ultron job in one record. The command asks `runpodctl` for the current pod and direct SSH details, then runs the bounded Ultron collector in the checkout on the pod. It does not expose a web port.

Install the current [RunPod CLI](https://docs.runpod.io/runpodctl/overview) on your computer and configure its API key and SSH key:

```bash
brew install runpod/runpodctl/runpodctl
runpodctl doctor
runpodctl pod get <pod-id>
```

The pod needs a public mapping for `22/tcp`. RunPod can change the external IP or port when a pod resets, so each observation reads fresh connection details. Install Ultron in the checkout on the pod with `python -m pip install -e .`. The local checkout also needs the base package so it can provide the `ultron runpod` command.

Set the two values that stay constant for a run:

```bash
export ULTRON_RUNPOD_ID=<pod-id>
export ULTRON_RUNPOD_REPO=/workspace/Ultron
```

You can pass the pod ID as a positional argument instead. `--identity` overrides the key reported by `runpodctl`, and `--remote-python` selects a particular interpreter. Without the latter, the command uses `<repo>/.venv/bin/python` when present and falls back to `python3`.

## Observe from the local computer

```bash
# One combined snapshot.
ultron runpod status --session ultron-gen-0 --json

# Poll through short SSH outages and retain every observation locally.
ultron runpod watch --session ultron-gen-0 --json --timeout 3600

# Read only new remote log bytes on each invocation.
ultron runpod logs --session ultron-gen-0 --json \
  --cursor-file data/monitoring/<pod-id>/ultron-gen-0.cursor.json
```

The combined status keeps four evidence sources separate:

- `pod` is the filtered RunPod lifecycle and allocation record. Environment variables from the provider response are never copied into the output.
- `transport` reports direct SSH connection state and the remote command result.
- `ultron` contains tmux jobs, pipeline stages, response summaries, disks, and GPU telemetry from the pod.
- `provider_logs` contains a bounded RunPod system-log tail when SSH or the remote collector is unavailable.

`watch` allows three consecutive incomplete observations by default. This prevents one dropped SSH connection from ending a monitoring loop. It writes JSONL to `data/monitoring/<pod-id>/observations.jsonl` unless you pass `--no-record` or choose another path with `--record`. A successful observation resets the miss count.

RunPod `runtimeStatus` and Ultron job state answer different questions. A running pod can still have a failed training job. A stopped or terminated pod is terminal for the watch. An initializing pod or a lost SSH connection is incomplete monitoring, and the watch retries it within `--max-misses`.

## Launch on the pod

The commands below run in the Ultron checkout on the pod. They need the base Python package and tmux.

Keep the checkout, logs, response archives, and checkpoints on the pod's persistent storage. Use the same `TMUX_TMPDIR`, `ULTRON_TMUX_LOG_DIR`, `ULTRON_PIPELINE_STATE_DIR`, and `ULTRON_RESPONSES_DIR` values when launching and monitoring. Defaults put artifacts under the checkout's `data/` directory.

## Launch and observe

```bash
# Start detached. Python output is unbuffered; SSH can disconnect.
ultron job start --json ultron-gen-0 -- ./scripts/run_generation.sh 0

# Read one snapshot without opening the TUI.
ultron status --session ultron-gen-0 --json

# Watch for completion or failure for up to 60 seconds.
ultron watch --session ultron-gen-0 --json --interval 10 --timeout 60

# Inspect bounded log tails and response previews when needed.
ultron status --session ultron-gen-0 --json --details
```

`job start` returns `started` or `already_running`. A duplicate launch leaves the running job alone. An exited session requires `job restart`. Commands after `--` are passed as arguments and run from the repository root. To use pipes or shell operators, explicitly pass `bash -lc '...'`.

`status` reports job exit codes, pipeline stages and attempts, response archive summaries, disk space, and GPU utilization, memory, and temperature. GPU telemetry uses a short `nvidia-smi` probe. Use `--no-gpu` to skip it. Missing GPU telemetry appears as `available: false` with a reason.

`watch --json` writes one complete JSON object per line and flushes each observation. It returns when the selected job completes, a failure is observed, monitoring fails, or the timeout expires. Its last object contains `watch.reason` and the command's `watch.exit_code`. The top-level `exit_code` describes the last status observation. Ending the watch leaves the job running. Timeout bounds polling; a probe already in progress can extend the command by its own timeout.

## Read only new logs

```bash
ultron logs --session ultron-gen-0 --json --tail 40 --max-bytes 16384
```

Save `next_cursor` from the result and pass it into the next read:

```bash
ultron logs --session ultron-gen-0 --json --cursor "$cursor" --max-bytes 16384
```

The cursor read advances sequentially. When `has_more` is true, read again with the returned cursor to drain the remaining bytes. `reset` marks rotation or truncation. `skipped_bytes` records bytes omitted from an initial tail or during reset. Terminal escape sequences are removed. Use `--tail 0` on the first read to start at the end without replaying history. Store cursors separately for each session.

## Decide what to do next

| Evidence | Next action |
| --- | --- |
| Job `running`, stage `running` | Continue watching; read new logs as needed. |
| Stage `retrying` | Check `attempt`, `max_attempts`, and `retry_at`; let the scheduled retry run. |
| `log_quiet` | Compare GPU activity, stage age, and new response activity; gather evidence before restarting. |
| Job `failed` or current stage `failed` | Read the final log chunk and recorded exit code, fix the cause, then restart. |
| Current stage `orphaned` | Its recorded owner PID is gone. Inspect the job and logs before resuming. |
| Job `succeeded` | Inspect saved review findings and output artifacts before launching the next generation. |
| `disk_low` | Check the reported filesystem and artifact growth before another long stage. |
| `errors` contains entries | Monitoring is incomplete. Fix the reported read or tmux error and collect another snapshot. |

Quiet logs alone do not establish a hung job. Response errors are archive history and do not establish the selected job's current state. Stage records from earlier pipeline invocations remain visible as history; current failures are scoped to the latest invocation. No command automatically restarts or stops a job.

```bash
# Restart an exited session with the original command.
ultron job restart --json ultron-gen-0

# Stop the session and its process group.
ultron job stop --json ultron-gen-0
```

Generation stages reuse completed work when command and input fingerprints match. If you replace an input artifact at the same path, change `ULTRON_PIPELINE_INPUT_KEY` for the next launch. To change a job's command or environment, stop its session and start it again with the new values. `job restart` reuses the original command and environment.

## Output contract

All JSON objects include `schema_version: 1`. Pod-local status objects contain `observed_at`, `session`, `jobs`, `pipelines`, `responses`, `resources`, `issues`, `errors`, `limits`, and `exit_code`. Local RunPod status wraps that object under `ultron` and adds `pod`, `transport`, and `provider_logs`. Each job uses `session` as its identifier. Stage records include `in_scope` so an agent can distinguish the selected job from other pipelines. Text previews require `--details`; full model output stays in the referenced `responses.json` files and journals. Output includes up to four response archives by default; `--response-limit 0` omits them and `--response-limit 20` includes a larger sample.

| Exit code | Meaning |
| --- | --- |
| `0` | Status collected with no observed failure; for a targeted watch, the job succeeded. |
| `1` | Confirmed job or current stage failure, orphaned stage, or missing selected session. |
| `2` | Invalid input, job-control error, or incomplete monitoring. |
| `124` | Watch timed out; the job may still be running. |
| `130` | Watch interrupted from the terminal. |

For `ultron runpod`, exit `1` can come from a terminal RunPod lifecycle state or the remote Ultron status. Exit `2` means the combined observation is incomplete, including RunPod API, SSH, remote command, or JSON failures. The `watch` command only returns exit `2` after the configured consecutive-miss limit, or when a selected job exits without a recorded code.

Recorded failures take precedence over monitoring errors. A status exit of `0` with an empty `jobs` array means no jobs were found. It does not mean a generation completed. For automation, select an expected session and inspect its state.

Scans and reads have explicit limits. Truncation is reported in the snapshot. Latest attacker and defender response pointers remain available as the archive grows, while history scans cover a bounded sample. Response summaries read periodic JSON snapshots; their journals can contain newer chunks. Use `--state-dir` or `--responses-dir` to inspect a specific pipeline or archive directory.

An agent loop should persist the session and log cursor, run a bounded watch, inspect the final JSON object and exit code, then read new logs only when needed. On timeout, schedule another watch. On success, inspect outputs. On failure, diagnose before an explicit restart.
