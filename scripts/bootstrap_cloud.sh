#!/usr/bin/env bash
set -euo pipefail

failures=0

if ! command -v nvidia-smi >/dev/null; then
  echo "MISSING: nvidia-smi" >&2
  failures=$((failures + 1))
elif ! nvidia-smi; then
  echo "FAILED: nvidia-smi" >&2
  failures=$((failures + 1))
fi

if ! command -v docker >/dev/null; then
  echo "MISSING: docker" >&2
  failures=$((failures + 1))
elif ! docker info >/dev/null; then
  echo "FAILED: docker info" >&2
  failures=$((failures + 1))
else
  if ! docker network inspect ultron-isolated >/dev/null 2>&1; then
    if ! docker network create --internal ultron-isolated; then
      echo "FAILED: docker network create --internal ultron-isolated" >&2
      failures=$((failures + 1))
    fi
  fi
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "Cloud host gates failed with ${failures} error(s)." >&2
  exit 1
fi

echo "Cloud host gates passed."
