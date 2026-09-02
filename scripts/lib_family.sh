# shellcheck shell=bash
# Source from a launch script, then call: ultron_load_family
# Optional --family is already exported as ULTRON_MODEL_FAMILY.

ultron_load_family() {
  local here root py
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root="$(cd "${here}/.." && pwd)"
  if [[ -x "${root}/.venv/bin/python" ]]; then
    py="${root}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    py="python"
  else
    py="python3"
  fi
  # declare -x inside a function is local. -g keeps the job pin in the caller.
  eval "$(cd "${root}" && "${py}" -m ultron.train.family export | sed 's/^declare -x /declare -gx /')"
}
