#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_runtime.sh
source "${ROOT}/scripts/lib_runtime.sh"
ultron_load_runtime "${ROOT}"
cd "${ROOT}"
exec "${ULTRON_PYTHON}" -m ultron.train.archive "$@"
