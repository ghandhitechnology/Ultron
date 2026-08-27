#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM="${INTERCODE_ROOT:-${ROOT}/third_party/intercode}"

if [[ ! -d "${UPSTREAM}" ]]; then
  echo "Set INTERCODE_ROOT to an upstream InterCode checkout." >&2
  exit 2
fi

echo "InterCode stock runner is an integration stub. Follow upstream's pinned evaluator command." >&2
exit 2
