# shellcheck shell=bash

ultron_load_family() {
  local here root exported
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  root="$(cd "${here}/.." && pwd)"
  # shellcheck source=lib_runtime.sh
  source "${here}/lib_runtime.sh"
  ultron_load_runtime "${root}" || return 2
  exported="$(cd "${root}" && "${ULTRON_PYTHON}" -m ultron.train.family export)" || return 2
  # `export` persists outside this function and works on Bash 3 through Bash 5.
  eval "$(printf '%s\n' "${exported}" | sed 's/^declare -x /export /')"
}
