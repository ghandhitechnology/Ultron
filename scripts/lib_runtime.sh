# shellcheck shell=bash

ultron_python_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    >/dev/null 2>&1
}

ultron_load_runtime() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then
    local here
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    root="$(cd "${here}/.." && pwd)"
  fi

  if [[ -n "${ULTRON_PYTHON:-}" ]]; then
    if { [[ -x "${ULTRON_PYTHON}" ]] || command -v "${ULTRON_PYTHON}" >/dev/null 2>&1; } \
      && ultron_python_supported "${ULTRON_PYTHON}"; then
      export ULTRON_PYTHON
      return 0
    fi
    echo "Configured ULTRON_PYTHON must be executable Python 3.10 or newer: ${ULTRON_PYTHON}" >&2
    return 2
  fi

  if [[ -x "${root}/.venv/bin/python" ]]; then
    ULTRON_PYTHON="${root}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    ULTRON_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    ULTRON_PYTHON="$(command -v python)"
  else
    echo "Python 3 is required. Create .venv or install python3." >&2
    return 2
  fi
  if ! ultron_python_supported "${ULTRON_PYTHON}"; then
    echo "Python 3.10 or newer is required: ${ULTRON_PYTHON}" >&2
    return 2
  fi
  export ULTRON_PYTHON
}
