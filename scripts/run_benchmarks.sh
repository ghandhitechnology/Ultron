#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_tmux.sh
source "${ROOT}/scripts/lib_tmux.sh"
# shellcheck source=lib_family.sh
source "${ROOT}/scripts/lib_family.sh"
cd "${ROOT}"

LAUNCH_ARGS=("$@")
GEN=""
ALL=0
MODE=""
ROLE="both"
EXECUTE=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --family) export ULTRON_MODEL_FAMILY="$2"; shift 2 ;;
    --generation) GEN="$2"; shift 2 ;;
    --all) ALL=1; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --role) ROLE="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --) shift; break ;;
    -*)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

ultron_load_family
SESSION="ultron-bench"
if [[ "${ALL}" -eq 1 ]]; then
  SESSION="ultron-bench-all"
elif [[ -n "${GEN}" ]]; then
  SESSION="ultron-bench-gen${GEN}"
fi
ultron_maybe_tmux "${SESSION}" "${LAUNCH_ARGS[@]}"

EVAL_ROOT="$(dirname "${ULTRON_ARCHIVE_ROOT}")/eval"
ARGS=(
  --archive-dir "${ULTRON_ARCHIVE_ROOT}"
  --output "${EVAL_ROOT}/benchmarks"
  --role "${ROLE}"
)
if [[ "${ALL}" -eq 1 || -z "${GEN}" ]]; then
  ARGS+=(--all)
else
  ARGS+=(--generation "${GEN}")
fi
if [[ -n "${MODE}" ]]; then
  ARGS+=(--mode "${MODE}")
fi
if [[ "${EXECUTE}" -eq 1 || "${ULTRON_BENCHMARK_EXECUTE:-}" == "1" ]]; then
  ARGS+=(--execute)
fi

echo "=== Ultron archived-weight benchmarks ==="
python -m ultron.eval.run_benchmarks "${ARGS[@]}"
