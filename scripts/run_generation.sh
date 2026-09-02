#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_tmux.sh
source "${ROOT}/scripts/lib_tmux.sh"
cd "${ROOT}"
GEN="${1:-0}"
ultron_maybe_tmux "ultron-gen-${GEN}"
EPISODES="${ULTRON_EPISODES:-2048}"

echo "=== Ultron generation ${GEN} ==="
"${ROOT}/scripts/rollout_worker.sh" --generation "${GEN}" --episodes "${EPISODES}"
python -m ultron.train.review \
  "data/traces/gen${GEN}" \
  --phase rollout \
  --generation "${GEN}" \
  --output "data/traces/gen${GEN}"
"${ROOT}/scripts/train_grpo.sh" --role attacker --generation "${GEN}"
"${ROOT}/scripts/train_grpo.sh" --role defender --generation "${GEN}"

if [[ "${GEN}" -ge 2 ]]; then
  "${ROOT}/scripts/train_dpo.sh" --role attacker --generation "${GEN}"
fi

python -m ultron.train.archive --generation "${GEN}"
python -m ultron.train.pfsp --update-pool --generation "${GEN}"
if [[ "${GEN}" -eq 2 ]]; then
  python -m ultron.eval.run_tier3 --mode light
elif [[ "${GEN}" -eq 4 ]]; then
  python -m ultron.eval.run_tier3 --mode full
fi

python -m ultron.train.review \
  "data/traces/gen${GEN}" \
  --phase complete \
  --generation "${GEN}" \
  --output "data/traces/gen${GEN}" \
  --eval-dir data/eval \
  --archive-dir data/archives \
  --pfsp data/checkpoints/pfsp_pool.json

METRICS="data/traces/gen${GEN}/metrics.json"
if [[ -f "${METRICS}" ]]; then
  python -m ultron.train.bandpass \
    --check-kill-switch \
    --generation "${GEN}" \
    --metrics "${METRICS}"
fi
