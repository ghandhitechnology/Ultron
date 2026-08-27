# Ultron

Ultron is a research trainer for environment-grounded asymmetric self-play. Separate attacker and defender LoRAs act through Pi in isolated Ubuntu 18.04 KVM guests. A guest probe plus an independent host-side check adjudicates uid 0. Availability probes prevent the defender from scoring by breaking required services.

This repository is research-ready scaffolding. Pure Python contracts, reward logic, PFSP sampling, DPO pair extraction, configuration, launch scripts, and tests run without GPUs or KVM. VM execution, Pi integration, vLLM serving, and training require the bare-metal setup in [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md).

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

- `train/` owns trajectory schema v1, adjudication, rewards, RAE, PFSP, DPO pairing, conversion, curriculum, and orchestration interfaces.
- `env/` owns guest RPC, host probes, availability, snapshots, the VM pool, cloud-init profiles, and libvirt templates.
- `harness/` defines the Pi-facing TypeScript execution and turn interfaces.
- `eval/` defines tier-3 plans, procedural selection, InterCode integration points, and the ReAct baseline interface.
- `configs/` records the locked model, host, generation, training, and evaluation parameters.
- `scripts/` contains host checks, vLLM launchers, rollout launch, and generation training entry points.

## Safety boundary

Ultron targets disposable guests on a default-deny isolated libvirt network. The repository contains misconfiguration identifiers, not exploit payloads, CVE procedures, or host escape material. Run only on systems you own or have explicit permission to test.

## Current implementation boundary

The Python unit tests do not require hardware. The Pi-to-KVM bridge and veRL launch commands are explicit integration points. `rollout_worker.sh` requires `ULTRON_ROLLOUT_COMMAND`; `train_dpo.sh` requires `ULTRON_DPO_COMMAND`. Set them only after the corresponding milestone gate passes.
