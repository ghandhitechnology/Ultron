#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_tmux.sh
source "${ROOT}/scripts/lib_tmux.sh"
# shellcheck source=lib_family.sh
source "${ROOT}/scripts/lib_family.sh"
cd "${ROOT}"

LAUNCH_ARGS=("$@")
GEN=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --family) export ULTRON_MODEL_FAMILY="$2"; shift 2 ;;
    --) shift; break ;;
    -*)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
    *)
      if [[ -n "${GEN}" ]]; then
        echo "Unknown argument: $1" >&2
        exit 2
      fi
      GEN="$1"
      shift
      ;;
  esac
done
GEN="${GEN:-0}"
ultron_load_family
ultron_maybe_tmux "ultron-gen-${GEN}" "${LAUNCH_ARGS[@]}"
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
python -m ultron.train.pfsp --update-pool --generation "${GEN}" --manifest "${ULTRON_PFSP_MANIFEST}"
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
  --archive-dir "${ULTRON_ARCHIVE_ROOT}" \
  --pfsp "${ULTRON_PFSP_MANIFEST}"

METRICS="data/traces/gen${GEN}/metrics.json"
if [[ -f "${METRICS}" ]]; then
  python -m ultron.train.bandpass \
    --check-kill-switch \
    --generation "${GEN}" \
    --metrics "${METRICS}"
fi
