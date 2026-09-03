#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_tmux.sh
source "${ROOT}/scripts/lib_tmux.sh"
# shellcheck source=lib_family.sh
source "${ROOT}/scripts/lib_family.sh"

LAUNCH_ARGS=("$@")
ROLE=""
GENERATION=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --generation) GENERATION="$2"; shift 2 ;;
    --family) export ULTRON_MODEL_FAMILY="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "${ROLE}" != "attacker" && "${ROLE}" != "defender" ]]; then
  echo "--role must be attacker or defender" >&2
  exit 2
fi
if [[ -z "${GENERATION}" || ! "${GENERATION}" =~ ^[0-9]+$ ]]; then
  echo "--generation must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -d third_party/verl/verl ]]; then
  echo "Initialize third_party/verl before training." >&2
  exit 2
fi
ultron_load_family
ultron_maybe_tmux "ultron-grpo-${ROLE}-gen${GENERATION}" "${LAUNCH_ARGS[@]}"

INPUT="data/traces/gen${GENERATION}/${ROLE}.jsonl"
OUTPUT="data/verl/gen${GENERATION}/${ROLE}.jsonl"
"${ULTRON_PYTHON}" -m ultron.train.convert_verl "${INPUT}" "${OUTPUT}" --generation "${GENERATION}"

PYTHONPATH="third_party/verl:${PYTHONPATH:-}" \
"${ULTRON_PYTHON}" -m verl.trainer.main_ppo \
  --config-path "${ULTRON_GRPO_CONFIG_PATH}" \
  --config-name "${ULTRON_GRPO_CONFIG_NAME}" \
  "data.train_files=${OUTPUT}" \
  "trainer.default_local_dir=${ULTRON_CHECKPOINT_ROOT}/gen${GENERATION}/${ROLE}_lora"
