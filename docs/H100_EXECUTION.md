# H100 study execution

Use the same two H100 80 GB GPUs for the baseline and optimized runs. `configs/h100.yaml` specifies two serving replicas, 16 episode workers, context length 32768, rank-64 LoRAs, and packing settings for training. The default Qwen3-8B model keeps thinking enabled; Qwen3.5-4B remains the secondary comparison. The complete study matrix, reward definitions, turn limits, KVM isolation, and evaluation splits remain in `configs/study/final.yaml`.

## Plan a study cell

Planning runs on a CPU machine and starts no services:

```bash
python -m ultron.train.study check
python -m ultron.train.study matrix
python -m ultron.train.pipeline plan \
  --family qwen-8b --variant adaptive-history \
  --seed 0 --run-id h100-pilot --generation 0 --episodes 240 \
  --pilot --prep-turns 0
```

Every cell has its own `data/study/<model>/<variant>/seed<seed>/<run-id>/` namespace for traces, converted training data, checkpoints, archives, evaluation, and stage state. Use a new run ID when changing frozen inputs. A seed and run ID identify a particular execution; the final study's seed count still needs to be set through its research protocol.

`--pilot` labels a performance fixture separately from the final study. The zero preparation turns above are an explicit fixture setting. Choose and record preparation turns for the pilot; full-study execution requires the protocol's calibrated preparation count and frozen curriculum. Actual execution retains the existing approved spending-cap gate through `ULTRON_SPENDING_CAP_USD`.

Four generations means 0 through 3. DPO runs only in the `added-dpo` variants and only from generation 2 onward. The full evaluation plan follows generation 3. Executing that plan requires the production tier-3 evaluator. The nominal 2,048-game budget rounds upward to 2,052 games so six-member groups are complete. Both role transcripts are retained. The runtime explicitly identifies learner trajectories, and the training gate reports the actual learner groups for each role. Historical opponents' actions are retained as observations.

## Runtime and training integration

The generation launcher preflights an installed Pi/KVM runtime and a recorded-trajectory trainer before rollout. This checkout provides the scheduler and validated input/output contracts. A production Pi/KVM adapter and numerically validated GRPO/DPO consumer remain deployment dependencies.

`ultron.train.rollout_scheduler` loads a `module:factory` runtime. The runtime supplies a frozen episode plan and creates independent workers with their own guest and model sessions. Each production `EpisodeTask` pins `snapshot_sha256`. The request's `serving_models` maps each role and checkpoint ID to the immutable API model, content hash, and endpoint. The scheduler immediately assigns the next episode to a free worker, preserves the order of actions within each episode, records restore/turn/probe timings, and commits each complete episode atomically. Resume verifies the frozen request and stored trajectory hashes. Per-turn probes continue to observe transient compromise and recovery.

The trainer is configured with `ULTRON_GRPO_COMMAND` and `ULTRON_GRPO_ADAPTER_MANIFEST`. Added-DPO cells also need `ULTRON_DPO_COMMAND` and `ULTRON_DPO_ADAPTER_MANIFEST`. The manifest pins the executable and its revision, declares recorded-token input, and forbids replacement rollouts. See `train/training.py` for the request and result schemas. The output must include a serveable adapter, optimizer state, input hashes, and a receipt identifying its parent policy. Export model files into an immutable adapter subdirectory, with optimizer files and the receipt outside it.

Generation zero uses a pinned base-model revision from `ULTRON_BASE_MODEL_REVISION`. Later generations use the preceding final policy for that role. Weight provenance and optimizer provenance are tracked separately. The default resumes the preceding optimizer for the same objective. `ULTRON_OPTIMIZER_STRATEGY=reset_each_generation` explicitly selects a different, fingerprinted convention.

H100 training requests enable dynamic token batches and removal of padding. The family configuration still defines the effective batch, learning rate, epochs, KL objective, and group size. The adapter must compute group advantages before packing and preserve token IDs, assistant masks, and loss normalization. A manifest establishes the interface; a real update comparison establishes numerical parity.

## Serve both roles on both GPUs

`python -m ultron.shared_serving --help` describes the serving supervisor. Its immutable manifest binds each adapter to a content hash and pins the base model and tokenizer revision. Each replica exposes the same policies. The router balances active requests and keeps conversation prefixes on a preferred worker when load permits. Sampled completions are always generated anew.

The default ports are worker ports 8101 and 8102, router port 8200, and role capture ports 8001 and 8002. Set distinct ports and device assignments for separate allocations. The supervisor verifies the H100 inventory and releases both worker processes before training. The legacy role servers remain available for comparisons.

Use `--bindings-output <file.json>` to export evaluation bindings. `--preflight-only` validates the manifest and prints commands without starting vLLM. `--no-prefix-caching` provides an uncached serving comparison with the same model and context.

Response capture groups durable writes on a 50 ms timer or at 256 KiB instead of synchronizing every chunk. Completion waits for the canonical raw body, provider events, and response journal to reach stable storage before forwarding the final SSE marker. Restore, turn, probe, episode, and job events also reach the journal. The strict per-write baseline is available through `sync_interval_s=0` on `CaptureProxy` and `ResponseArchive`.

## Reuse completed evaluation

Every real benchmark job requires an immutable checkpoint binding and verifies the server's model identity before execution. Separate attempt directories prevent stale verifier outputs from supplying a new result. A failed rerun replaces its prior score for that cell.

```bash
python -m ultron.eval.run_benchmarks \
  --archive-dir /path/to/run/archives --all --mode full \
  --output /path/to/run/eval/benchmarks \
  --bindings /path/to/eval-bindings.json --workers 2 --execute --strict
```

Caching requires `cache_identity` in each benchmark's configuration. Supply the resolved `dataset_revision`, `harness_revision`, `seed`, `opponent_id`, and `inference_settings`, plus `local_inputs.harness_root` and `local_inputs.dataset_root`. The executor checks that the command uses those local inputs and hashes their actual contents and executable. Python console scripts must resolve to the declared installed harness source. Unresolved sources and remote dataset aliases run without reuse.

For Terminal-Bench, set `tasks_path` to a frozen local task directory. Ultron then uses Harbor's local-dataset `-p` argument. Published dataset names continue through `-d`. See [Harbor's dataset interface](https://www.harborframework.com/docs/datasets).

The cache key also includes the checkpoint contents, base-model revision, role, mode, command, configuration, and Ultron's evaluation code. Inputs are checked again after execution and before reuse. A saved result is reused only while its output artifacts still match their recorded hash. `--no-resume` forces fresh execution. All requested checkpoints and benchmark jobs remain in the plan.

Evaluate untrained controls with `--untrained --family qwen-8b --base-model-revision <commit>`, then repeat for `qwen-4b`. Use a separate output directory for each family. This plans both roles against the pinned base policy and requires matching base-policy serving bindings when executed.

## Count allocated time and cost

Enter the actual booked instance rate, currency, source, and observation date in a copy of `configs/h100.yaml`. The default rate is unset. Rates apply to the entire allocation, including every GPU billed by the provider. If an eight-GPU node is booked, account for eight GPUs even while only two perform this job. Include CPU hosting, storage, tax, and other charges using the summary's `--extra-cost` option in the same currency.

Wrap the entire booked work interval, including warm-up, guest work, training, evaluation, and retries:

```bash
ULTRON_NO_TMUX=1 python -m ultron.train.compute run \
  --profile /path/to/h100-billing.yaml \
  --ledger /path/to/allocation.jsonl --run-id h100-pilot \
  --scope-key paired-240-episode-pilot --verify-hardware \
  -- /path/to/pilot-command
python -m ultron.train.compute summary /path/to/allocation.jsonl --extra-cost 0
```

The wrapped command must stay in the foreground until its work finishes. Set `ULTRON_NO_TMUX=1` for launchers that would otherwise detach. `compute run` takes a per-device lease, so separate study run directories cannot start overlapping Ultron allocations on the same GPUs. If the rental starts earlier or ends later, use `compute begin` at allocation start and `compute end` at release instead. Those commands record billing intervals; they do not reserve devices. Record phases within that allocation; overlapping billing intervals for the same host/device pair are rejected to prevent double counting. An allocation left open after a crash continues accruing time in summaries until it is closed.

```bash
python -m ultron.train.compute compare baseline.jsonl optimized.jsonl \
  --baseline-extra-cost 0 --candidate-extra-cost 0
```

Comparison reports elapsed-time, GPU-hour, and cost ratios. Scope keys must match, hardware models and currencies must agree, and every allocation must be closed. Check study artifacts separately: matching scope labels do not establish experimental equivalence.

## Measure the improvement

Use 240 episodes, or forty complete groups, for the performance fixture. Keep the initial policies, task mix, opponent schedule, seeds, context, thinking, tool limits, and reward settings fixed. Compare one episode worker with sixteen workers; separately compare uncached and cached serving. Include long turns, guest resets, and infrastructure failures. Report learner groups per role, completion rate, timeouts, rewards, and compromise/recovery observations alongside throughput.

Replay identical recorded trajectories through baseline and packed training to compare masks, group advantages, losses, optimizer steps, and exported policy behavior. Pair live runs to check whether changed batching alters outcomes under the wall-clock turn limit. Include opponent preparation and final evaluation in the full-study accounting.

The acceptance target is at most half the elapsed time, half the allocated GPU-hours, and half the total cost for the same completed study. The local tests exercise scheduling, identities, capture, reuse, and accounting. H100 throughput and training parity require the production GPU pilot.
