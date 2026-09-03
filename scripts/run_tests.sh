#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_family.sh
source "${ROOT}/scripts/lib_family.sh"
cd "${ROOT}"

SUITE="${1:-all}"
ultron_load_family

case "${SUITE}" in
  all) paths=(train/tests env/tests cli/tests eval/tests) ;;
  train) paths=(train/tests) ;;
  env) paths=(env/tests) ;;
  cli) paths=(cli/tests) ;;
  eval) paths=(eval/tests) ;;
  *)
    echo "Unknown test suite: ${SUITE}" >&2
    exit 2
    ;;
esac

echo "=== Ultron tests (${SUITE}) ==="
python -m pytest "${paths[@]}" -q

echo "=== Post-test public benchmarks on archived weights ==="
EVAL_ROOT="$(dirname "${ULTRON_ARCHIVE_ROOT}")/eval"
python -m ultron.eval.run_benchmarks \
  --all \
  --archive-dir "${ULTRON_ARCHIVE_ROOT}" \
  --output "${EVAL_ROOT}/benchmarks"
