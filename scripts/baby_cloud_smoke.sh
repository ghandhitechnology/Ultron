#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! python3 -c "import ultron.env.baby_smoke" >/dev/null 2>&1; then
  echo "Install Ultron first: python -m pip install -e '.[dev]'" >&2
  exit 2
fi

exec python3 -m ultron.env.baby_smoke "$@"
