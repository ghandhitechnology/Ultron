#!/usr/bin/env bash
set -euo pipefail

GENERATION=""
EPISODES=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --generation) GENERATION="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${GENERATION}" || -z "${EPISODES}" ]]; then
  echo "Usage: $0 --generation N --episodes N" >&2
  exit 2
fi
if [[ -z "${ULTRON_ROLLOUT_COMMAND:-}" ]]; then
  echo "Set ULTRON_ROLLOUT_COMMAND to the M3-verified Pi/KVM rollout launcher." >&2
  exit 2
fi

mkdir -p "data/traces/gen${GENERATION}"
exec "${ULTRON_ROLLOUT_COMMAND}" \
  --generation "${GENERATION}" \
  --episodes "${EPISODES}" \
  --output "data/traces/gen${GENERATION}"
