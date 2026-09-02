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

if [[ "${ROLE}" == "attacker" && -n "${ULTRON_ATTACKER_ADAPTER:-}" ]]; then
  printf '%s\n' "${ULTRON_ATTACKER_ADAPTER}"
  exit 0
fi
if [[ "${ROLE}" == "defender" && -n "${ULTRON_DEFENDER_ADAPTER:-}" ]]; then
  printf '%s\n' "${ULTRON_DEFENDER_ADAPTER}"
  exit 0
fi

if [[ -x "${ARCHIVE_ROOT}/FINAL.sh" ]]; then
  resolved="$("${ARCHIVE_ROOT}/FINAL.sh" "${ROLE}" 2>/dev/null || true)"
  if [[ -n "${resolved}" && -f "${resolved}/adapter_config.json" ]]; then
    printf '%s\n' "${resolved}"
    exit 0
  fi
fi

for candidate in \
  "${ARCHIVE_ROOT}/final/${ROLE}_lora" \
  "${ARCHIVE_ROOT}/gen${GEN}/${ROLE}_lora" \
  "${ARCHIVE_ROOT}/latest/${ROLE}_lora" \
  "${CHECKPOINT_ROOT}/gen${GEN}/${ROLE}_lora"
do
  if [[ -f "${candidate}/adapter_config.json" ]]; then
    printf '%s\n' "${candidate}"
    exit 0
  fi
done

printf '%s\n' "${ARCHIVE_ROOT}/gen${GEN}/${ROLE}_lora"
