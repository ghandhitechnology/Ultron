# About the locked final study

The final study asks whether adaptive self-play improves both roles on unseen security mechanisms at matched GPU cost. The target is a five-point gain per role over the fixed diverse pool.

`configs/study/final.yaml` is the lock. `python -m ultron.train.study check` fails if the default family, thinking flag, shaping cap, adaptive mix, or GRPO role-aware baseline drift. `python -m ultron.train.study matrix` prints the cells.

## What you run

The default job is Qwen3-8B with thinking on. Train four 8B methods plus an untrained control: fixed single, fixed diverse, adaptive latest, and adaptive history. Compare shaping-off and added-DPO only on fixed-diverse and adaptive-history. The main method is GRPO with verified shaping. Restrict 4B to fixed-diverse, adaptive-history, and its untrained control.

Paid GPU execution still needs an approved spending cap. Implementation of this protocol does not spend that budget.

## How a trial is scored

KVM guests are required for study runs. A protected host verifier, not the guest, confirms root. Ground-truth weakness metadata stays outside participant observations.

The first independently verified attacker-controlled root is compromise. Recovery is a separate flag. Defender success needs the hold plus zero failures of declared legitimate workflows, including ordinary user access. A timeout or hang stays unresolved with reward 0/0. It is never a defender win.

Verified shaping is capped at 0.1 total per episode. A terminal win is 1.0. veRL rows keep `reward` and `return_to_go` as separate fields and drop `INFRA_FAIL` samples.

## Opponent sampling

Diverse-pool methods start from the same independently trained reference pool. Adaptive history draws 50% latest, 25% difficult historical, and 25% uniform historical using learner-relative results.

## Evaluation splits

Report unseen instances, unseen combinations, and held-out mechanisms separately. Freeze the training curriculum after the pilot. Keep development tasks out of the sealed final-test set.

Score each role against an independently trained, frozen opponent panel. Use Pi for extra cross-runtime evaluation. Treat public capability benchmarks as secondary.

## Gates before scale

Known-solvable and known-patched tasks must pass through the real launcher. Also test false root proofs, verifier tampering, metadata leakage, dummy services, and lost ordinary-user access. Prove one real end-to-end policy update before scaling. Match total GPU-hours across variants, including opponent preparation.
