#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_tmux.sh
source "${ROOT}/scripts/lib_tmux.sh"
cd "${ROOT}"

LAUNCH_ARGS=("$@")
GENERATION=""
EPISODES=""
PREFLIGHT=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --generation) GENERATION="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --preflight-only) PREFLIGHT=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${GENERATION}" || -z "${EPISODES}" ]]; then
  echo "Usage: $0 --generation N --episodes N" >&2
  exit 2
fi
if [[ ! "${GENERATION}" =~ ^[0-9]+$ || ! "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Generation must be non-negative and episodes must be positive integers." >&2
  exit 2
fi
if [[ -z "${ULTRON_ROLLOUT_COMMAND:-}" && -z "${ULTRON_ROLLOUT_RUNTIME:-}" ]]; then
  echo "Set ULTRON_ROLLOUT_RUNTIME to a KVM/Pi runtime factory or ULTRON_ROLLOUT_COMMAND to its installed launcher." >&2
  exit 2
fi
if [[ "${PREFLIGHT}" == "0" ]]; then
  ultron_maybe_tmux "${ULTRON_STUDY_SESSION:-ultron}-rollout-gen${GENERATION}" "${LAUNCH_ARGS[@]}"
fi
OUTPUT="${ULTRON_TRACE_ROOT:-data/traces}/gen${GENERATION}"
EXTRA=()
if [[ -n "${ULTRON_ROLLOUT_REQUEST:-}" ]]; then
  EXTRA+=(--request "${ULTRON_ROLLOUT_REQUEST}")
fi
if [[ "${PREFLIGHT}" == "1" ]]; then
  EXTRA+=(--preflight-only)
fi
if [[ -n "${ULTRON_ROLLOUT_RUNTIME:-}" ]]; then
  source "${ROOT}/scripts/lib_runtime.sh"
  ultron_load_runtime "${ROOT}"
  if [[ -n "${ULTRON_SERVING_MANIFEST:-}" ]]; then
    EXTRA+=(--serving-manifest "${ULTRON_SERVING_MANIFEST}")
  fi
  exec "${ULTRON_PYTHON}" -m ultron.train.rollout_scheduler \
    --generation "${GENERATION}" --episodes "${EPISODES}" --output "${OUTPUT}" \
    --runtime "${ULTRON_ROLLOUT_RUNTIME}" --workers "${ULTRON_ROLLOUT_WORKERS:-16}" \
    --seed "${ULTRON_ROLLOUT_SEED:-0}" ${EXTRA[@]+"${EXTRA[@]}"}
fi

if [[ "${PREFLIGHT}" == "0" ]]; then
  mkdir -p "${OUTPUT}"
fi
exec "${ULTRON_ROLLOUT_COMMAND}" \
  --generation "${GENERATION}" \
  --episodes "${EPISODES}" \
  --output "${OUTPUT}" ${EXTRA[@]+"${EXTRA[@]}"}
