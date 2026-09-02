# Ultron

[Korean version](README.ko.md)

Ultron is a research trainer for environment-grounded asymmetric self-play. Separate attacker and defender LoRAs act through Pi in isolated Ubuntu 18.04 guests. Guests are Docker or KVM, selected by `guest_backend`. GRPO, DPO, and vLLM run on cloud GPU VMs. A guest probe plus an independent host-side check adjudicates uid 0. Availability probes prevent the defender from scoring by breaking required services.

<p align="center">
  <img src="docs/screenshots/demo_gym.png" alt="Live guest gym after a stub episode, with attacker, sandbox, and defender panes, a process log, and episode progress bars" width="920" />
</p>

`ultron-sim demo` is the live guest gym. The header keeps generation, episode, profile, and ETA in view. The three panes are the attacker LoRA, the guest, and the defender LoRA. Watch the log if you want the turn clock. The bars track episode and turn progress. This shot is SIM MODE: a real `EpisodeRunner` running against stub guests, so you can learn the layout with no GPUs and no VMs.

This repository is research-ready scaffolding. Pure Python contracts, reward logic, PFSP sampling, DPO pair extraction, configuration, launch scripts, and tests run without GPUs or guests. vLLM serving and GRPO or DPO training run on any NVIDIA host. Isolated rollouts need Docker (`scripts/bootstrap_cloud.sh`) or native KVM. See [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md).

## Server requirements

Python unit tests, reward logic, and `ultron-sim demo` need no GPU. Isolated rollouts, vLLM, GRPO, and DPO do.

Supported bases are Qwen 4B, Qwen 8B, and Gemma 12B. The checked-in default is `Qwen/Qwen3.5-4B`. Point `configs/model.yaml`, `ULTRON_BASE_MODEL`, `configs/train_grpo.yaml`, and `configs/train_dpo.yaml` at `Qwen/Qwen3-8B` or `google/gemma-4-12B-it` (or use `--family` / `ULTRON_MODEL_FAMILY`) only on a host that meets that row. Qwen 27B, 35B, and larger Qwen MoE checkpoints are not supported.

Figures assume two vLLM processes (attacker and defender), BF16 weights, `max_model_len` 32768, LoRA rank 64, 16 CPU-only guests at 2 vCPU / 4 GiB each, and GRPO or DPO after both servers stop. Guests never take a GPU. Pin one GPU per role. Do not overlap vLLM and FSDP.

| Variant | CPU | Host RAM | Recommended GPU setup |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` (locked default) | 32 physical cores; 64 safer | 128 GB | 2× A100 80 GB, or 2× H100 80 GB. Attacker on GPU 0, defender on GPU 1 (`ULTRON_ATTACKER_GPU` / `ULTRON_DEFENDER_GPU`). |
| `Qwen/Qwen3-8B` | 32 physical cores; 64 safer | 192 GB | 2× H100 80 GB or 2× A100 80 GB with the same one-GPU-per-role pin. 2× L40S 48 GB only if you cut guest concurrency. |
| `google/gemma-4-12B-it` | 32 physical cores; 64 safer | 192 GB | 2× H100 80 GB preferred; 2× A100 80 GB is the fallback. Skip 24 GB cards. 48 GB cards need a smaller guest pool. |

Every row also needs x86-64, Ubuntu 22.04 or 24.04, an NVIDIA datacenter driver, and at least 1 TB local NVMe for the base image, overlays, Hugging Face cache, traces, and checkpoints. Docker guests use `scripts/bootstrap_cloud.sh` and do not need `/dev/kvm`. Native KVM guests need `/dev/kvm` and the 32-core floor if you keep 16 VMs.

## Quick start

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest train/tests/ -q
```

Or with [uv](https://docs.astral.sh/uv/) and the committed lockfile:

```bash
uv sync --extra dev
uv run pytest train/tests/ -q
```

The gym and the experiment console need the TUI extra:

```bash
python -m pip install -e '.[tui]'
ultron-sim demo
ultron-sim
```

Initialize upstream code when you are on the training server:

```bash
git submodule update --init --recursive
npm install
npm run check
```

Do not place guest images, traces, model weights, credentials, or checkpoints in Git. The ignore rules cover the standard paths.

## Repository map

- `train/` owns trajectory schema v1 (`schema_v1.py`), episode orchestration (`episode_runner.py`), turn records (`turn_record.py`), dual-probe adjudication (`adjudicator.py`), credit assignment and reward shaping (`rewards.py`), role-aware baseline RAE (`rae.py`), opponent pool PFSP-8 (`pfsp.py`), prefix-branch DPO pair extraction (`dpo_pairs.py`), veRL parquet/jsonl dataset conversion (`convert_verl.py`), bandpass win-rate filters and kill-switch checks (`bandpass.py`), model family packs (`family.py`), checkpoint archiving and `FINAL.sh` management (`archive.py`), and post-run job reviews (`review.py`).
- `env/` owns guest isolation abstractions (`backend.py`), Docker guest backend (`docker_backend.py`), libvirt/KVM templates (`libvirt/`), vsock guest RPC (`guest_agent_client.py`), in-guest agent daemon (`guest-agent/`), host proc probes (`probes.py`), service TCP availability probes (`availability.py`), backing image hash verification (`snapshot.py`), VM pool quarantine management (`vm_pool.py`), and cloud-init guest environment profiles (`cloud-init/`, `profiles.yaml`).
- `harness/` defines the Pi-facing TypeScript execution environment and turn interfaces (`execution_env.ts`), turn alternation clock (`turn_clock.ts`), agent session factory (`session_factory.ts`), model endpoints configuration (`models.json`), and JSONL event stream export (`export_jsonl.ts`).
- `eval/` defines tier-3 evaluation plans and runner (`run_tier3.py`), procedural template generators (`procedural/`), InterCode evaluation adapters (`intercode/`), Debian 12 zero-shot build scripts (`debian12/`), and ReAct baseline scaffolding (`react_baseline.py`).
- `configs/` records locked model (`model.yaml`), host/VM topology (`bm-gpu.yaml`), generation loops (`generation.yaml`), training algorithms (`train_grpo.yaml`, `train_dpo.yaml`), evaluation plans (`eval_tier3.yaml`), and selectable model family packs (`families/qwen-8b/`, `families/gemma/`).
- `scripts/` contains host environment bootstrap gates (`bootstrap_bm.sh`, `bootstrap_cloud.sh`), tmux lifecycle management (`tmux_job.sh`, `lib_tmux.sh`), model family environment loader (`lib_family.sh`), vLLM role servers (`serve_vllm_attacker.sh`, `serve_vllm_defender.sh`), rollout worker (`rollout_worker.sh`), adapter resolution (`resolve_adapter.sh`), training entry points (`train_grpo.sh`, `train_dpo.sh`), full generation loop orchestrator (`run_generation.sh`), and weight archiver (`archive_weights.sh`).
- `cli/` is the experiment console (`ultron-sim` / `ultron-sim console`) and live guest-gym dashboard (`ultron-sim demo`). Built with Textual, it provides interactive job launching across actions (demo, generation, rollout, GRPO, DPO, serve, review, archive, eval, tests), real-time tmux monitoring, review report viewing, and live simulation of agent-sandbox interactions.
- `prompts/` contains attacker and defender system instructions (`attacker_system.md`, `defender_system.md`) and per-profile research goals (`goals/profiles.yaml`).

## Long-running jobs

Generation, rollout, GRPO, DPO, and vLLM launchers start in named tmux sessions so they keep running after SSH disconnects, hangup, or a closed terminal. Nested calls from `run_generation.sh` stay in the parent session.

```bash
./scripts/run_generation.sh 0
./scripts/serve_vllm_attacker.sh
./scripts/tmux_job.sh list
./scripts/tmux_job.sh attach ultron-gen-0
./scripts/tmux_job.sh logs ultron-gen-0
./scripts/tmux_job.sh stop ultron-vllm-attacker
```

Set `ULTRON_NO_TMUX=1` to run in the current shell. See [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md).

## Model families

A job pins one base-model family. The unset family is `qwen-4b` (`Qwen/Qwen3.5-4B`), which uses the top-level `configs/` files (`configs/model.yaml`, `configs/train_grpo.yaml`, `configs/train_dpo.yaml`) and stores checkpoints and archives under `data/checkpoints` and `data/archives`. Supported alternative families are `qwen-8b` (`Qwen/Qwen3-8B`) and `gemma` (`google/gemma-4-12B-it`), which live under `configs/families/<name>/` and write outputs under `data/families/<name>/checkpoints` and `data/families/<name>/archives`.

<p align="center">
  <img src="docs/screenshots/console_family_gemma.png" alt="Experiment console with the Gemma family pin in the header" width="900" />
</p>

The pin in the header is the same selector you set with `--family` or `ULTRON_MODEL_FAMILY`. Here it reads `gemma`.

<p align="center">
  <img src="docs/screenshots/console_family_dropdown.png" alt="Model family dropdown listing Gemma and the two Qwen packs" width="900" />
</p>

Press `m` to jump to the selector. You get three names and only three: `qwen-4b`, `qwen-8b`, and `gemma`.

<p align="center">
  <img src="docs/screenshots/console_family_qwen8b.png" alt="Experiment console with the Qwen 8B family pin" width="900" />
</p>

Pick `qwen-8b` and everything writes under `data/families/qwen-8b/`. The default `qwen-4b` pack stays where it is, on `data/checkpoints` and `data/archives`.

```bash
# Default (qwen-4b)
./scripts/run_generation.sh 0

# Select family via CLI flag or environment variable
./scripts/run_generation.sh --family qwen-8b 0
ULTRON_MODEL_FAMILY=gemma ./scripts/serve_vllm_attacker.sh
```

`--family` and `ULTRON_MODEL_FAMILY` are the authoritative selector. `ULTRON_BASE_MODEL` is not an override: if set, it must agree with the chosen pack or the job will abort. Gemma automatically omits vLLM `--chat-template-kwargs`, while Qwen packs disable thinking (`enable_thinking: false`).

The module `ultron.train.family` exports the active family configuration into environment variables (`ULTRON_MODEL_FAMILY`, `ULTRON_PACK_BASE_MODEL`, `ULTRON_MODEL_CONFIG`, `ULTRON_CHECKPOINT_ROOT`, `ULTRON_ARCHIVE_ROOT`, `ULTRON_PFSP_MANIFEST`, etc.) for seamless integration across shell scripts and Python workers.

## Experiment console

`ultron-sim` opens a full-screen TUI for the complete research loop: pick a generation, launch rollouts or training, monitor tmux jobs, run unit tests, and view `review.md` findings and metrics.

```bash
python -m pip install -e '.[tui]'
ultron-sim
# or explicitly:
ultron-sim console
```

<p align="center">
  <img src="docs/screenshots/console_catalog.png" alt="Experiment console catalog with Full generation selected, generation 0 and 2048 episodes" width="900" />
</p>

The list on the left is the catalog, grouped into gym, pipeline, train, serve, results, and verify. Pick an action and the right pane shows it with its fields. Full generation chains the whole loop: rollout, review, GRPO, an optional DPO step, archive, PFSP, and eval.

Key bindings:
- `enter`: run the selected action
- `m`: focus the model-family selector dropdown (`qwen-4b`, `qwen-8b`, `gemma`)
- `a`: view catalog of runnable actions
- `j`: list running and finished tmux jobs with live log inspection
- `r`: view generation results, metrics, and `review.md` reports
- `t`: jump directly to the unit test runner action
- `s`: stop the currently selected tmux job
- `escape`: back / dismiss
- `q`: quit

The header selector pins `qwen-4b`, `qwen-8b`, or `gemma` for every launch. You can also pass `--family` on startup: `ultron-sim --family gemma`.

Available console action groups:
- **Gym**: `demo` (run simulated episodes in the live gym UI).
- **Pipeline**: `generation` (run full generation pipeline), `rollout` (launch rollout worker).
- **Train**: `grpo` (train GRPO for attacker or defender), `dpo` (train prefix-branch DPO for attacker).
- **Serve**: `serve_attacker` (serve attacker LoRA on vLLM), `serve_defender` (serve defender LoRA on vLLM).
- **Results**: `review` (generate or view rollout/complete reviews), `archive` (archive generation checkpoints), `archive_list` (list archived weights and `FINAL.sh` paths).
- **Verify**: `eval` (plan tier-3 evaluation), `pfsp` (inspect or update PFSP opponent pool), `bandpass` (run kill-switch checks), `tests` (run pytest test suites for `train`, `env`, or `cli`).

<p align="center">
  <img src="docs/screenshots/console_jobs.png" alt="Experiment console tmux jobs view with session, state, pid, and command columns" width="900" />
</p>

`j` opens the tmux job table. From there, `enter` opens logs, `s` stops the session you have selected, and `g` refreshes the list. Long jobs still live in the named sessions that `scripts/tmux_job.sh` starts, not in the console.

<p align="center">
  <img src="docs/screenshots/console_results_with_gen.png" alt="Generation results table showing gen 3, usable verdict, 12 episodes, and ASR 0.420" width="900" />
</p>

`r` lists the generations it finds under `data/traces` and `data/archives`. Hit `enter` on a row and the console pulls in that generation's `review.md`. Verdict, episode count, and ASR all come straight from that file. This is a reader, not a replacement for `train/review.py`.

<p align="center">
  <img src="docs/screenshots/console_run_archive_list.png" alt="Foreground run view showing archive list JSON and exit 0" width="900" />
</p>

Foreground actions like list archives, tests, and the kill-switch check stream right into this run view. Anything launched through tmux leaves the console and keeps running in its own session.

`ultron-sim demo` runs the live guest-gym view: attacker and defender turns, sandbox identity, a scrolling process log, progress, and ETA:

```bash
ultron-sim demo --episodes 2 --turns-per-side 2
```

Click a pane (or press `a` / `s` / `d` / `t`) to expand detail. The demo drives a real `EpisodeRunner` with stub guests so you can watch the layout without GPUs or VMs. In production, the same injected `restore` / `run_turn` / `final_probe` callables wrap real guest instances while keeping `EpisodeRunner.run` unchanged. The console can also launch this gym via the "Live guest gym" action.

## Guest isolation backends

Ultron supports two guest execution backends, selected by `guest_backend` in `configs/bm-gpu.yaml`:

1. **Docker (`docker`)**: Default for cloud GPU VMs and local development. Container guests are provisioned on an internal bridge network (`ultron-isolated`) using `scripts/bootstrap_cloud.sh`. Adjudication uses host `/proc` inspection (`/proc/<pid>/root/etc/passwd` and process status tables) rather than `docker exec`, avoiding privilege escalation inside the test harness.
2. **KVM (`kvm`)**: Used on bare-metal hosts with `/dev/kvm`. Guests run as full hardware-virtualized VMs on an isolated libvirt network (`virbr-ultron`). Host-to-guest communication uses virtio vsock (`GuestAgentClient`), and snapshots are verified by SHA-256 before restoration.

Both backends implement the `GuestBackend` protocol in `ultron.env.backend` (`restore`, `stop`, `exec_as_user`, `confirm_root`, `verify_image`).

## Safety boundary

Ultron targets disposable guests on a default-deny isolated libvirt network. The repository contains misconfiguration identifiers, not exploit payloads, CVE procedures, or host escape material. Run only on systems you own or have explicit permission to test.

## Current implementation boundary

The Python unit tests do not require hardware. The Pi-to-guest bridge and veRL launch commands are explicit integration points:
- `rollout_worker.sh` delegates to `ULTRON_ROLLOUT_COMMAND` (the binary or script joining Pi sessions, `EpisodeRunner`, guest backend, and profiles).
- `train_dpo.sh` delegates to `ULTRON_DPO_COMMAND` (the pinned veRL or TRL DPO training script).
- `run_tier3.py` generates light and full evaluation plans; passing `--execute` validates adapter connectivity against target environments.

Set the integration commands and hardware gates only after the corresponding milestone gate passes. See [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md) for gate requirements M0 through M7.
