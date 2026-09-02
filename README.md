# Ultron

[Korean version](README.ko.md)

Ultron is a research trainer for environment-grounded asymmetric self-play. Separate attacker and defender LoRAs act through Pi in isolated Ubuntu 18.04 guests. Guests are Docker or KVM, selected by `guest_backend`. GRPO, DPO, and vLLM run on cloud GPU VMs. A guest probe plus an independent host-side check adjudicates uid 0. Availability probes prevent the defender from scoring by breaking required services.

This repository is research-ready scaffolding. Pure Python contracts, reward logic, PFSP sampling, DPO pair extraction, configuration, launch scripts, and tests run without GPUs or guests. vLLM serving and GRPO or DPO training run on any NVIDIA host. Isolated rollouts need Docker (`scripts/bootstrap_cloud.sh`) or native KVM. See [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md).

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

## Model families

A job pins one base-model family. Unset family is `qwen-4b` (`Qwen/Qwen3.5-4B`) and keeps the original `configs/` files plus `data/checkpoints` and `data/archives`. `qwen-8b` is `Qwen/Qwen3-8B`. `gemma` is `google/gemma-2-9b-it`. Those two live under `configs/families/<name>/` and write under `data/families/<name>/`.

```bash
./scripts/run_generation.sh 0
./scripts/run_generation.sh --family qwen-8b 0
ULTRON_MODEL_FAMILY=gemma ./scripts/serve_vllm_attacker.sh
```

`--family` and `ULTRON_MODEL_FAMILY` are the selector. `ULTRON_BASE_MODEL` is not. If that variable is set and disagrees with the pack, the job exits. Gemma omits vLLM `--chat-template-kwargs`. Qwen packs pass thinking off.

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
