#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
GEN="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

if [[ -x "${ROOT}/data/archives/FINAL.sh" ]]; then
  resolved="$("${ROOT}/data/archives/FINAL.sh" "${ROLE}" 2>/dev/null || true)"
  if [[ -n "${resolved}" && -f "${resolved}/adapter_config.json" ]]; then
    printf '%s\n' "${resolved}"
    exit 0
  fi
fi

for candidate in \
  "${ROOT}/data/archives/final/${ROLE}_lora" \
  "${ROOT}/data/archives/gen${GEN}/${ROLE}_lora" \
  "${ROOT}/data/archives/latest/${ROLE}_lora" \
  "${ROOT}/data/checkpoints/gen${GEN}/${ROLE}_lora"
do
  if [[ -f "${candidate}/adapter_config.json" ]]; then
    printf '%s\n' "${candidate}"
    exit 0
  fi
done

printf '%s\n' "${ROOT}/data/archives/gen${GEN}/${ROLE}_lora"
