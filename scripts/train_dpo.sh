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
if [[ -z "${GENERATION}" || ! "${GENERATION}" =~ ^[0-9]+$ || "${GENERATION}" -lt 2 ]]; then
  echo "DPO requires --generation 2 or later." >&2
  exit 2
fi
ultron_load_family
cd "${ROOT}"
"${ULTRON_PYTHON}" -m ultron.train.training dpo --preflight-only
ultron_maybe_tmux "${ULTRON_STUDY_SESSION:-ultron}-dpo-${ROLE}-gen${GENERATION}" "${LAUNCH_ARGS[@]}"
exec "${ULTRON_PYTHON}" -m ultron.train.training dpo --role "${ROLE}" --generation "${GENERATION}"
