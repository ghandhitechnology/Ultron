#!/usr/bin/env bash
set -euo pipefail

ROLE=""
GENERATION=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --generation) GENERATION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "${ROLE}" != "attacker" && "${ROLE}" != "defender" ]]; then
  echo "--role must be attacker or defender" >&2
  exit 2
fi
if [[ -z "${GENERATION}" || "${GENERATION}" -lt 2 ]]; then
  echo "DPO requires --generation 2 or later." >&2
  exit 2
fi
if [[ -z "${ULTRON_DPO_COMMAND:-}" ]]; then
  echo "Set ULTRON_DPO_COMMAND after pinning a veRL or TRL DPO launcher." >&2
  exit 2
fi

PAIRS="data/dpo/gen${GENERATION}/${ROLE}.jsonl"
exec "${ULTRON_DPO_COMMAND}" \
  --config "configs/train_dpo.yaml" \
  --pairs "${PAIRS}" \
  --output "data/checkpoints/gen${GENERATION}/${ROLE}_lora_dpo"
