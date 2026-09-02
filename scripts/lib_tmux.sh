# shellcheck shell=bash
# Source from a launch script, then call: ultron_maybe_tmux <session-name>
# Re-execs the caller inside an isolated tmux session unless already wrapped.

ultron_maybe_tmux() {
  local session="${1:-}"
  if [[ -z "${session}" ]]; then
    echo "ultron_maybe_tmux: session name required" >&2
    return 2
  fi
  if [[ "${ULTRON_NO_TMUX:-}" == "1" || -n "${ULTRON_TMUX_SESSION:-}" ]]; then
    return 0
  fi
  local here caller
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  caller="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)/$(basename "${BASH_SOURCE[1]}")"
  exec "${here}/tmux_job.sh" wrap "${session}" "${caller}" "$@"
}
