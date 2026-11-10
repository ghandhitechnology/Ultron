#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
GEN="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_family.sh
source "${ROOT}/scripts/lib_family.sh"
ultron_load_family
ARCHIVE_ROOT="${ULTRON_ARCHIVE_ROOT:-${ROOT}/data/archives}"
CHECKPOINT_ROOT="${ULTRON_CHECKPOINT_ROOT:-${ROOT}/data/checkpoints}"

if [[ "${ROLE}" != "attacker" && "${ROLE}" != "defender" ]]; then
  echo "usage: resolve_adapter.sh <attacker|defender> [generation]" >&2
  exit 2
fi
if [[ -n "${ULTRON_RUN_ROOT:-}" ]]; then
  exec "${ULTRON_PYTHON}" -m ultron.train.training resolve --role "${ROLE}" --generation "${GEN}"
fi

if [[ "${ROLE}" == "attacker" && -n "${ULTRON_ATTACKER_ADAPTER:-}" ]]; then
  printf '%s\n' "${ULTRON_ATTACKER_ADAPTER}"
  exit 0
fi
if [[ "${ROLE}" == "defender" && -n "${ULTRON_DEFENDER_ADAPTER:-}" ]]; then
  printf '%s\n' "${ULTRON_DEFENDER_ADAPTER}"
  exit 0
fi

for candidate in \
  "${ARCHIVE_ROOT}/gen${GEN}/${ROLE}_lora" \
  "${CHECKPOINT_ROOT}/gen${GEN}/${ROLE}_lora/adapter" \
  "${CHECKPOINT_ROOT}/gen${GEN}/${ROLE}_lora"
do
  if [[ -f "${candidate}/adapter_config.json" ]]; then
    printf '%s\n' "${candidate}"
    exit 0
  fi
done

printf '%s\n' "${ARCHIVE_ROOT}/gen${GEN}/${ROLE}_lora"
