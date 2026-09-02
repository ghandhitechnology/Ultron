#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="${ROOT}/scripts/tmux.conf"
LOG_DIR="${ULTRON_TMUX_LOG_DIR:-${ROOT}/data/logs}"
SOCKET_PREFIX="${ULTRON_TMUX_SOCKET_PREFIX:-ultron}"

usage() {
  cat <<'EOF'
Usage: tmux_job.sh <command> [args]

Run Ultron jobs in isolated tmux sessions so they survive SSH disconnects,
closed terminals, and hangup.

Commands:
  wrap <session> <command> [args]   Start a session; attach when a TTY is present
  start <session> [--] <command>... Start detached; fail if the session exists
  attach <session>                  Attach to a running session
  status [session]                  Show session state
  stop <session>                    Kill a session
  logs <session>                    Print the session log
  list                              List Ultron tmux sessions

Standard session names:
  ultron-gen-N
  ultron-rollout-genN
  ultron-grpo-<role>-genN
  ultron-dpo-<role>-genN
  ultron-vllm-attacker
  ultron-vllm-defender

Environment:
  ULTRON_NO_TMUX=1          Skip auto-wrap in launch scripts
  ULTRON_TMUX_DETACH=1      Never attach after wrap
  ULTRON_TMUX_LOG_DIR       Log directory (default: <repo>/data/logs)
  ULTRON_TMUX_SOCKET_PREFIX Socket prefix (default: ultron)
  ULTRON_TMUX_SESSION       Set inside a wrapped job; nested launchers stay put
EOF
}

die() {
  echo "$*" >&2
  exit 2
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    die "tmux is required so jobs survive disconnects. Install tmux or set ULTRON_NO_TMUX=1."
  fi
}

validate_session() {
  local session="$1"
  if [[ ! "${session}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    die "Invalid session name: ${session}"
  fi
}

socket_name() {
  printf '%s\n' "$1"
}

socket_path() {
  printf '%s/%s\n' "$(sockdir)" "$(socket_name "$1")"
}

tmux_cmd() {
  local session="$1"
  shift
  tmux -L "$(socket_name "${session}")" -f "${CONF}" "$@"
}

log_path() {
  printf '%s/%s.log\n' "${LOG_DIR}" "$1"
}

sockdir() {
  printf '%s/tmux-%s\n' "${TMUX_TMPDIR:-/tmp}" "$(id -u)"
}

has_session() {
  local session="$1"
  local sock
  sock="$(socket_path "${session}")"
  if [[ ! -e "${sock}" ]]; then
    return 1
  fi
  tmux_cmd "${session}" has-session -t "=${session}" 2>/dev/null
}

should_attach() {
  [[ -t 0 && -t 1 && "${ULTRON_TMUX_DETACH:-}" != "1" && -z "${TMUX:-}" ]]
}

start_session() {
  local session="$1"
  shift
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  if [[ "$#" -lt 1 ]]; then
    die "start requires a command"
  fi
  validate_session "${session}"
  require_tmux
  mkdir -p "${LOG_DIR}"
  local log
  log="$(log_path "${session}")"
  if has_session "${session}"; then
    echo "Session ${session} is already running." >&2
    echo "Attach: ${ROOT}/scripts/tmux_job.sh attach ${session}" >&2
    echo "Logs:   ${log}" >&2
    return 1
  fi
  tmux_cmd "${session}" new-session -d -s "${session}" -c "${ROOT}" -- \
    bash -c 'trap "" HUP; export ULTRON_TMUX_SESSION="$1"; exec > >(tee -a "$2") 2>&1; shift 2; printf "[%s] start %s\n" "$(date -Is)" "$*"; exec "$@"' \
    bash "${session}" "${log}" "$@"
  echo "Started ${session}"
  echo "  attach: ${ROOT}/scripts/tmux_job.sh attach ${session}"
  echo "  logs:   ${log}"
  echo "  stop:   ${ROOT}/scripts/tmux_job.sh stop ${session}"
}

cmd_wrap() {
  local session="${1:-}"
  shift || true
  if [[ -z "${session}" || "$#" -lt 1 ]]; then
    die "Usage: tmux_job.sh wrap <session> <command> [args]"
  fi
  if has_session "${session}"; then
    echo "Session ${session} is already running." >&2
    echo "Attach: ${ROOT}/scripts/tmux_job.sh attach ${session}" >&2
    if should_attach; then
      exec_attach "${session}"
    fi
    exit 1
  fi
  start_session "${session}" "$@"
  if should_attach; then
    exec_attach "${session}"
  fi
}

exec_attach() {
  local session="$1"
  validate_session "${session}"
  require_tmux
  if ! has_session "${session}"; then
    die "Session not found: ${session}"
  fi
  exec tmux -L "$(socket_name "${session}")" -f "${CONF}" attach-session -t "=${session}"
}

cmd_stop() {
  local session="${1:-}"
  if [[ -z "${session}" ]]; then
    die "Usage: tmux_job.sh stop <session>"
  fi
  validate_session "${session}"
  require_tmux
  if ! has_session "${session}"; then
    die "Session not found: ${session}"
  fi
  tmux_cmd "${session}" kill-server
  echo "Stopped ${session}"
}

cmd_status_one() {
  local session="$1"
  if ! has_session "${session}"; then
    printf '%s\tmissing\n' "${session}"
    return 1
  fi
  tmux_cmd "${session}" list-panes -t "=${session}" \
    -F "${session}	#{pane_pid}	#{pane_dead}	#{pane_current_command}"
}

cmd_status() {
  require_tmux
  if [[ -n "${1:-}" ]]; then
    cmd_status_one "$1"
    return
  fi
  local dir found=0 sock session
  dir="$(sockdir)"
  shopt -s nullglob
  for sock in "${dir}/${SOCKET_PREFIX}"*; do
    session="${sock##*/}"
    if has_session "${session}"; then
      cmd_status_one "${session}" || true
      found=1
    fi
  done
  if [[ "${found}" -eq 0 ]]; then
    echo "No Ultron tmux sessions."
  fi
}

cmd_logs() {
  local session="${1:-}"
  if [[ -z "${session}" ]]; then
    die "Usage: tmux_job.sh logs <session>"
  fi
  validate_session "${session}"
  local log
  log="$(log_path "${session}")"
  if [[ ! -f "${log}" ]]; then
    die "No log for ${session}: ${log}"
  fi
  cat "${log}"
}

cmd_list() {
  cmd_status
}

main() {
  local command="${1:-}"
  if [[ -z "${command}" || "${command}" == "-h" || "${command}" == "--help" ]]; then
    usage
    exit 0
  fi
  shift
  case "${command}" in
    wrap) cmd_wrap "$@" ;;
    start)
      if [[ "$#" -lt 2 ]]; then
        die "Usage: tmux_job.sh start <session> [--] <command>..."
      fi
      start_session "$@"
      ;;
    attach)
      exec_attach "${1:-}"
      ;;
    status) cmd_status "${1:-}" ;;
    stop) cmd_stop "${1:-}" ;;
    logs) cmd_logs "${1:-}" ;;
    list) cmd_list ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
