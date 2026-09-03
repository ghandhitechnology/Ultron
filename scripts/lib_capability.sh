# shellcheck shell=bash

ultron_check_model_capability() {
  local here root py extra=()
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root="$(cd "${here}/.." && pwd)"
  if [[ "${ULTRON_SKIP_MODEL_CAPABILITY:-}" == "1" ]]; then
    echo "Skipping model capability check (ULTRON_SKIP_MODEL_CAPABILITY=1)."
    return 0
  fi
  if [[ -x "${root}/.venv/bin/python" ]]; then
    py="${root}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    py="python3"
  else
    echo "WARNING: python3 is missing; skipping model capability check." >&2
    return 0
  fi
  if [[ -n "${ULTRON_MODEL_FAMILY:-}" ]]; then
    extra+=(--family "${ULTRON_MODEL_FAMILY}")
  fi
  echo "=== Model capability ==="
  if "${py}" -c 'import ultron.train.capability' >/dev/null 2>&1; then
    "${py}" -m ultron.train.capability "${extra[@]}"
  else
    "${py}" "${root}/train/capability.py" "${extra[@]}"
  fi
}
