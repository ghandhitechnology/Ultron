#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_tmux.sh
source "${ROOT}/scripts/lib_tmux.sh"
# shellcheck source=lib_family.sh
source "${ROOT}/scripts/lib_family.sh"
# shellcheck source=lib_pipeline.sh
source "${ROOT}/scripts/lib_pipeline.sh"
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
if [[ ! "${GEN}" =~ ^[0-9]+$ ]]; then
  echo "Generation must be a non-negative integer." >&2
  exit 2
fi
ultron_load_family
ultron_maybe_tmux "ultron-gen-${GEN}" "${LAUNCH_ARGS[@]}"
EPISODES="${ULTRON_EPISODES:-2048}"
if [[ ! "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ULTRON_EPISODES must be a positive integer." >&2
  exit 2
fi
ultron_pipeline_init "${ULTRON_MODEL_FAMILY}-gen${GEN}"

echo "=== Ultron generation ${GEN} ==="
ultron_run_stage rollout \
  "${ROOT}/scripts/rollout_worker.sh" --generation "${GEN}" --episodes "${EPISODES}"
ultron_run_stage review-rollout "${ULTRON_PYTHON}" -m ultron.train.review \
  "data/traces/gen${GEN}" \
  --phase rollout \
  --generation "${GEN}" \
  --output "data/traces/gen${GEN}"
ultron_run_stage grpo-attacker \
  "${ROOT}/scripts/train_grpo.sh" --role attacker --generation "${GEN}"
ultron_run_stage grpo-defender \
  "${ROOT}/scripts/train_grpo.sh" --role defender --generation "${GEN}"

if [[ "${GEN}" -ge 2 ]]; then
  ultron_run_stage dpo-attacker \
    "${ROOT}/scripts/train_dpo.sh" --role attacker --generation "${GEN}"
fi

ultron_run_stage archive "${ULTRON_PYTHON}" -m ultron.train.archive --generation "${GEN}"
ultron_run_stage pfsp "${ULTRON_PYTHON}" -m ultron.train.pfsp \
  --update-pool --generation "${GEN}" --manifest "${ULTRON_PFSP_MANIFEST}"
if [[ "${GEN}" -eq 2 ]]; then
  ultron_run_stage tier3-light "${ULTRON_PYTHON}" -m ultron.eval.run_tier3 --mode light
elif [[ "${GEN}" -eq 4 ]]; then
  ultron_run_stage tier3-full "${ULTRON_PYTHON}" -m ultron.eval.run_tier3 --mode full
fi

ultron_run_stage review-complete "${ULTRON_PYTHON}" -m ultron.train.review \
  "data/traces/gen${GEN}" \
  --phase complete \
  --generation "${GEN}" \
  --output "data/traces/gen${GEN}" \
  --eval-dir data/eval \
  --archive-dir "${ULTRON_ARCHIVE_ROOT}" \
  --pfsp "${ULTRON_PFSP_MANIFEST}"

METRICS="data/traces/gen${GEN}/metrics.json"
if [[ -f "${METRICS}" ]]; then
  ultron_run_stage bandpass "${ULTRON_PYTHON}" -m ultron.train.bandpass \
    --check-kill-switch \
    --generation "${GEN}" \
    --metrics "${METRICS}"
fi
