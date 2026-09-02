# Ultron

[Korean version](README.ko.md)

Ultron is a research trainer for environment-grounded asymmetric self-play. Separate attacker and defender LoRAs act through Pi in isolated Ubuntu 18.04 guests. Guests are Docker or KVM, selected by `guest_backend`. GRPO, DPO, and vLLM run on cloud GPU VMs. A guest probe plus an independent host-side check adjudicates uid 0. Availability probes prevent the defender from scoring by breaking required services.

This repository is research-ready scaffolding. Pure Python contracts, reward logic, PFSP sampling, DPO pair extraction, configuration, launch scripts, and tests run without GPUs or guests. vLLM serving and GRPO or DPO training run on any NVIDIA host. Isolated rollouts need Docker (`scripts/bootstrap_cloud.sh`) or native KVM. See [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md).

## Server requirements

Python unit tests, reward logic, and `ultron-sim demo` need no GPU. Isolated rollouts, vLLM, GRPO, and DPO do.

The locked base model is `Qwen/Qwen3.5-4B`. Other Qwen3.5 sizes are a config swap (`configs/model.yaml`, `ULTRON_BASE_MODEL`, `configs/train_grpo.yaml`, `configs/train_dpo.yaml`) only if the host meets that row.

Figures assume the checked-in trainer shape: two vLLM processes (attacker and defender), BF16 weights, `max_model_len` 32768, LoRA rank 64, 16 CPU-only guests at 2 vCPU / 4 GiB each, and GRPO or DPO after both servers stop. Guests never take a GPU. Rollout uses one GPU per vLLM process. Training reuses those GPUs; do not overlap vLLM and FSDP.

| Variant | GPU | CPU | Host RAM |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-0.8B` | 2× 24 GB | 16 physical cores (8 guests) or 32 (16 guests) | 64 GB with 8 guests; 96 GB with 16 guests |
| `Qwen/Qwen3.5-2B` | 2× 40 GB (24 GB if you cut guest concurrency) | 32 physical cores | 96 GB |
| `Qwen/Qwen3.5-4B` (locked) | 2× 80 GB (H100 or A100) | 32 physical cores; 64 or more is safer | 128 GB |
| `Qwen/Qwen3.5-9B` | 2× 80 GB | 32 physical cores; 64 safer | 192 GB |
| `Qwen/Qwen3.5-27B` | 4× 80 GB | 64 physical cores | 256 GB |
| `Qwen/Qwen3.5-35B-A3B` | 4× 80 GB | 64 physical cores | 256 GB |

`Qwen3.5-27B` weights are about 54 GB BF16 and `Qwen3.5-35B-A3B` keeps about 70 GB of experts resident, so one 80 GB card cannot hold a full 32k, 16-guest vLLM replica. Use four GPUs (two per role) or raise `tensor_model_parallel_size` and serve one role at a time. `Qwen3.5-122B-A10B` and `Qwen3.5-397B-A17B` need more GPUs than `configs/train_grpo.yaml` (`tensor_model_parallel_size: 1`, `n_gpus_per_node: 2`). Do not point `base_model` at them without rewriting serve and train configs.

Every GPU row also needs x86-64, Ubuntu 22.04 or 24.04, an NVIDIA datacenter driver, and at least 1 TB local NVMe for the base image, overlays, Hugging Face cache, traces, and checkpoints. Docker guests use `scripts/bootstrap_cloud.sh` and do not need `/dev/kvm`. Native KVM guests need `/dev/kvm` and the 32-core floor if you keep 16 VMs.

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

Initialize upstream code when you are on the training server:

```bash
git submodule update --init --recursive
npm install
npm run check
```

Do not place guest images, traces, model weights, credentials, or checkpoints in Git. The ignore rules cover the standard paths.

## Repository map

- `train/` owns trajectory schema v1, adjudication, rewards, RAE, PFSP, DPO pairing, conversion, curriculum, post-run job review, and orchestration interfaces.
- `env/` owns guest RPC, host probes, availability, snapshots, the VM pool, cloud-init profiles, and libvirt templates.
- `harness/` defines the Pi-facing TypeScript execution and turn interfaces.
- `eval/` defines tier-3 plans, procedural selection, InterCode integration points, and the ReAct baseline interface.
- `configs/` records the locked model, host, generation, training, and evaluation parameters.
- `scripts/` contains host checks, vLLM launchers, rollout launch, and generation training entry points.
- `cli/` is the experiment console (`ultron-sim`) and live guest-gym dashboard (`ultron-sim demo`). The console launches generation, training, review, tests, and tmux jobs. It does not replace `train/review.py`.

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

## Experiment console

`ultron-sim` opens a full-screen TUI for the research loop: pick a generation, launch rollout or training, watch tmux jobs, run unit tests, and fetch `review.md` results.

```bash
python -m pip install -e '.[tui]'
ultron-sim
```

Keys: `enter` runs the selected action, `j` lists jobs, `r` lists generation results, `t` jumps to tests, `s` stops a job, `q` quits.

`ultron-sim demo` is the live guest-gym view: attacker and defender turns, sandbox identity, a scrolling process log, progress, and ETA. Click a pane (or press `a` / `s` / `d` / `t`) to expand detail. The demo drives a real `EpisodeRunner` with stub guests so you can watch the layout without GPUs or VMs. Production attach wraps the same injected `restore` / `run_turn` / `final_probe` callables and leaves `EpisodeRunner.run` unchanged. The console can open that gym as the "Live guest gym" action.

## Safety boundary

Ultron targets disposable guests on a default-deny isolated libvirt network. The repository contains misconfiguration identifiers, not exploit payloads, CVE procedures, or host escape material. Run only on systems you own or have explicit permission to test.

## Current implementation boundary

The Python unit tests do not require hardware. The Pi-to-KVM bridge and veRL launch commands are explicit integration points. `rollout_worker.sh` requires `ULTRON_ROLLOUT_COMMAND`; `train_dpo.sh` requires `ULTRON_DPO_COMMAND`. Set them only after the corresponding milestone gate passes.
