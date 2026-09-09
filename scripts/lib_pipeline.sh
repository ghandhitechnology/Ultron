# shellcheck shell=bash

ultron_pipeline_init() {
  local pipeline="${1:-}"
  if [[ -z "${pipeline}" || ! "${pipeline}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ultron_pipeline_init: safe pipeline name required" >&2
    return 2
  fi
  if [[ -z "${ULTRON_PIPELINE_STATE_DIR:-}" ]]; then
    local here root
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    root="$(cd "${here}/.." && pwd)"
    ULTRON_PIPELINE_STATE_DIR="${root}/data/job-state/${pipeline}"
  fi
  mkdir -p "${ULTRON_PIPELINE_STATE_DIR}"
  ULTRON_PIPELINE_NAME="${pipeline}"
  ULTRON_PIPELINE_SESSION="${ULTRON_TMUX_SESSION:-}"
  ULTRON_PIPELINE_STARTED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  ULTRON_PIPELINE_RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')-$$-${RANDOM}-${RANDOM}"
  ULTRON_PIPELINE_CHAIN_FINGERPRINT="$(
    printf 'pipeline\0%s\0input\0%s\0' \
      "${pipeline}" "${ULTRON_PIPELINE_INPUT_KEY:-}" | cksum | awk '{print $1 "-" $2}'
  )"
  ultron_write_stage_state \
    "${ULTRON_PIPELINE_STATE_DIR}/.pipeline" \
    "pipeline=${ULTRON_PIPELINE_NAME}" \
    "run_id=${ULTRON_PIPELINE_RUN_ID}" \
    "session=${ULTRON_PIPELINE_SESSION}" \
    "pid=$$" \
    "started_at=${ULTRON_PIPELINE_STARTED_AT}"
  export \
    ULTRON_PIPELINE_NAME \
    ULTRON_PIPELINE_RUN_ID \
    ULTRON_PIPELINE_SESSION \
    ULTRON_PIPELINE_STARTED_AT \
    ULTRON_PIPELINE_STATE_DIR
}

ultron_write_stage_state() {
  local path="$1"
  shift
  local temporary="${path}.tmp.$$"
  {
    printf 'updated_at=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '%s\n' "$@"
  } > "${temporary}"
  mv -f "${temporary}" "${path}"
}

ultron_stage_fingerprint() {
  local command_kind="" command_path=""
  command_kind="$(type -t -- "${1:-}" 2>/dev/null || true)"
  if [[ "${command_kind}" == "file" ]]; then
    command_path="$(type -P -- "$1")"
  fi

  {
    printf 'pipeline_chain\0%s\0' "${ULTRON_PIPELINE_CHAIN_FINGERPRINT:-}"
    printf 'cwd\0%s\0' "${PWD}"
    printf 'argv\0%s\0' "$@"
    printf 'command_kind\0%s\0' "${command_kind}"
    if [[ -n "${command_path}" && -r "${command_path}" ]]; then
      printf 'command_path\0%s\0' "${command_path}"
      cksum < "${command_path}"
    elif [[ "${command_kind}" == "function" ]]; then
      declare -f -- "$1"
    fi
  } | cksum | awk '{print $1 "-" $2}'
}

ultron_advance_pipeline_chain() {
  local stage="$1" fingerprint="$2" completion_id="$3"
  ULTRON_PIPELINE_CHAIN_FINGERPRINT="$(
    printf 'previous\0%s\0stage\0%s\0fingerprint\0%s\0completion\0%s\0' \
      "${ULTRON_PIPELINE_CHAIN_FINGERPRINT}" "${stage}" "${fingerprint}" "${completion_id}" \
      | cksum | awk '{print $1 "-" $2}'
  )"
}

ultron_stage_state_value() {
  local path="$1" key="$2"
  [[ -f "${path}" ]] || return 1
  awk -F= -v key="${key}" '$1 == key { print substr($0, length(key) + 2); exit }' "${path}"
}

ultron_acquire_stage_lock() {
  local lock_path="$1"
  local reclaim_path="${lock_path}.reclaim"
  local owner="" confirmed_owner=""
  [[ -d "${reclaim_path}" ]] && return 75
  if mkdir "${lock_path}" 2>/dev/null; then
    printf '%s\n' "$$" > "${lock_path}/pid"
    return 0
  fi

  owner="$(sed -n '1p' "${lock_path}/pid" 2>/dev/null || true)"
  [[ "${owner}" =~ ^[1-9][0-9]*$ ]] || return 76
  kill -0 "${owner}" 2>/dev/null && return 75
  mkdir "${reclaim_path}" 2>/dev/null || return 75

  confirmed_owner="$(sed -n '1p' "${lock_path}/pid" 2>/dev/null || true)"
  if [[ "${confirmed_owner}" != "${owner}" ]] || kill -0 "${confirmed_owner}" 2>/dev/null; then
    rmdir "${reclaim_path}" 2>/dev/null || true
    return 75
  fi
  rm -f "${lock_path}/pid"
  if ! rmdir "${lock_path}" 2>/dev/null || ! mkdir "${lock_path}" 2>/dev/null; then
    rmdir "${reclaim_path}" 2>/dev/null || true
    return 75
  fi
  printf '%s\n' "$$" > "${lock_path}/pid"
  rmdir "${reclaim_path}" 2>/dev/null || true
}

ultron_release_stage_lock() {
  local lock_path="$1" owner=""
  owner="$(sed -n '1p' "${lock_path}/pid" 2>/dev/null || true)"
  if [[ "${owner}" == "$$" ]]; then
    rm -f "${lock_path}/pid"
    rmdir "${lock_path}" 2>/dev/null || true
  fi
}

ultron_restore_stage_traps() {
  local int_trap="$1" term_trap="$2"
  trap - INT TERM
  [[ -n "${int_trap}" ]] && eval "${int_trap}"
  [[ -n "${term_trap}" ]] && eval "${term_trap}"
  return 0
}

ultron_write_active_stage_state() {
  local path="$1" state="$2"
  shift 2
  ultron_write_stage_state \
    "${path}" \
    "state=${state}" \
    "pipeline=${ULTRON_PIPELINE_NAME}" \
    "run_id=${ULTRON_ACTIVE_STAGE_RUN_ID}" \
    "stage=${ULTRON_ACTIVE_STAGE_NAME}" \
    "session=${ULTRON_ACTIVE_STAGE_SESSION}" \
    "pid=$$" \
    "started_at=${ULTRON_ACTIVE_STAGE_STARTED_AT}" \
    "attempt=${ULTRON_ACTIVE_STAGE_ATTEMPT}" \
    "max_attempts=${ULTRON_ACTIVE_STAGE_MAX_ATTEMPTS}" \
    "fingerprint=${ULTRON_ACTIVE_STAGE_FINGERPRINT}" \
    "$@"
}

ultron_interrupt_stage() {
  local signal="$1" status="$2"
  trap - INT TERM
  if [[ -n "${ULTRON_ACTIVE_STAGE_RUNNING_PATH:-}" ]]; then
    ultron_write_active_stage_state \
      "${ULTRON_ACTIVE_STAGE_FAILED_PATH}" \
      failed \
      "status=${status}" \
      "signal=${signal}"
    rm -f "${ULTRON_ACTIVE_STAGE_RUNNING_PATH}"
    ultron_release_stage_lock "${ULTRON_ACTIVE_STAGE_LOCK_PATH}"
  fi
  exit "${status}"
}

ultron_run_stage() {
  local stage="${1:-}"
  shift || true
  if [[ -z "${ULTRON_PIPELINE_STATE_DIR:-}" ]]; then
    echo "ultron_run_stage: call ultron_pipeline_init first" >&2
    return 2
  fi
  if [[ -z "${stage}" || ! "${stage}" =~ ^[A-Za-z0-9_.-]+$ || "$#" -eq 0 ]]; then
    echo "ultron_run_stage: safe stage name and command required" >&2
    return 2
  fi

  local attempts="${ULTRON_STAGE_MAX_ATTEMPTS:-2}"
  local delay="${ULTRON_STAGE_RETRY_DELAY_SECONDS:-5}"
  if [[ ! "${attempts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ULTRON_STAGE_MAX_ATTEMPTS must be a positive integer" >&2
    return 2
  fi
  if [[ ! "${delay}" =~ ^[0-9]+$ ]]; then
    echo "ULTRON_STAGE_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
    return 2
  fi

  local done_path="${ULTRON_PIPELINE_STATE_DIR}/${stage}.done"
  local running_path="${ULTRON_PIPELINE_STATE_DIR}/${stage}.running"
  local failed_path="${ULTRON_PIPELINE_STATE_DIR}/${stage}.failed"
  local lock_path="${ULTRON_PIPELINE_STATE_DIR}/${stage}.lock"
  local fingerprint
  fingerprint="$(ultron_stage_fingerprint "$@")"

  local lock_status=0
  ultron_acquire_stage_lock "${lock_path}" || lock_status=$?
  if [[ "${lock_status}" -ne 0 ]]; then
    if [[ "${lock_status}" -eq 76 ]]; then
      echo "Stage ${stage} has a lock with no valid owner; remove ${lock_path} after checking for a live job." >&2
      return 75
    fi
    echo "Stage ${stage} is already running in another process." >&2
    return 75
  fi

  local running_pid=""
  running_pid="$(ultron_stage_state_value "${running_path}" pid 2>/dev/null || true)"
  if [[ "${running_pid}" =~ ^[1-9][0-9]*$ && "${running_pid}" != "$$" ]] \
    && kill -0 "${running_pid}" 2>/dev/null; then
    ultron_release_stage_lock "${lock_path}"
    echo "Stage ${stage} is already running in process ${running_pid}." >&2
    return 75
  fi

  if [[ "${ULTRON_PIPELINE_RESUME:-1}" != "0" && -f "${done_path}" ]]; then
    if grep -q "^fingerprint=${fingerprint}$" "${done_path}"; then
      local completion_id
      completion_id="$(ultron_stage_state_value "${done_path}" completion_id 2>/dev/null || true)"
      completion_id="${completion_id:-legacy-${fingerprint}}"
      rm -f "${running_path}" "${failed_path}"
      ultron_release_stage_lock "${lock_path}"
      ultron_advance_pipeline_chain "${stage}" "${fingerprint}" "${completion_id}"
      echo "=== Stage ${stage}: already complete ==="
      return 0
    fi
    echo "=== Stage ${stage}: inputs changed; running again ==="
  fi
  if [[ -f "${running_path}" || -f "${failed_path}" ]]; then
    echo "=== Recovering unfinished stage ${stage} ==="
  fi
  rm -f "${done_path}"

  local previous_int_trap previous_term_trap
  previous_int_trap="$(trap -p INT)"
  previous_term_trap="$(trap -p TERM)"
  ULTRON_ACTIVE_STAGE_RUNNING_PATH="${running_path}"
  ULTRON_ACTIVE_STAGE_FAILED_PATH="${failed_path}"
  ULTRON_ACTIVE_STAGE_LOCK_PATH="${lock_path}"
  ULTRON_ACTIVE_STAGE_RUN_ID="${ULTRON_PIPELINE_RUN_ID}"
  ULTRON_ACTIVE_STAGE_NAME="${stage}"
  ULTRON_ACTIVE_STAGE_SESSION="${ULTRON_PIPELINE_SESSION}"
  ULTRON_ACTIVE_STAGE_STARTED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  ULTRON_ACTIVE_STAGE_MAX_ATTEMPTS="${attempts}"
  ULTRON_ACTIVE_STAGE_FINGERPRINT="${fingerprint}"
  ULTRON_ACTIVE_STAGE_ATTEMPT=1
  trap 'ultron_interrupt_stage INT 130' INT
  trap 'ultron_interrupt_stage TERM 143' TERM

  local attempt=1 status
  while [[ "${attempt}" -le "${attempts}" ]]; do
    ULTRON_ACTIVE_STAGE_ATTEMPT="${attempt}"
    echo "=== Stage ${stage}: attempt ${attempt}/${attempts} ==="
    rm -f "${failed_path}"
    ultron_write_active_stage_state "${running_path}" running
    if "$@"; then
      local completion_id
      completion_id="$(date -u '+%Y%m%dT%H%M%SZ')-$$-${RANDOM}"
      ultron_write_active_stage_state \
        "${done_path}" done "status=0" "completion_id=${completion_id}"
      rm -f "${running_path}" "${failed_path}"
      ultron_release_stage_lock "${lock_path}"
      ultron_restore_stage_traps "${previous_int_trap}" "${previous_term_trap}"
      ultron_advance_pipeline_chain "${stage}" "${fingerprint}" "${completion_id}"
      echo "=== Stage ${stage}: complete ==="
      return 0
    else
      status=$?
    fi
    case "${status}" in
      2|126|127)
        ultron_write_active_stage_state "${failed_path}" failed "status=${status}"
        rm -f "${running_path}"
        ultron_release_stage_lock "${lock_path}"
        ultron_restore_stage_traps "${previous_int_trap}" "${previous_term_trap}"
        echo "Stage ${stage} cannot start with status ${status}; retry disabled." >&2
        return "${status}"
        ;;
    esac
    if [[ "${status}" -ge 128 && "${status}" -le 192 ]]; then
      ultron_write_active_stage_state "${failed_path}" failed "status=${status}"
      rm -f "${running_path}"
      ultron_release_stage_lock "${lock_path}"
      ultron_restore_stage_traps "${previous_int_trap}" "${previous_term_trap}"
      echo "Stage ${stage} was interrupted with status ${status}; retry disabled." >&2
      return "${status}"
    fi
    if [[ "${attempt}" -ge "${attempts}" ]]; then
      ultron_write_active_stage_state "${failed_path}" failed "status=${status}"
      rm -f "${running_path}"
      ultron_release_stage_lock "${lock_path}"
      ultron_restore_stage_traps "${previous_int_trap}" "${previous_term_trap}"
      echo "Stage ${stage} failed with status ${status} after ${attempt} attempt(s)." >&2
      return "${status}"
    fi
    echo "Stage ${stage} failed with status ${status}; retrying in ${delay}s." >&2
    local retry_at
    retry_at=$(($(date -u '+%s') + delay))
    ultron_write_active_stage_state \
      "${running_path}" retrying "status=${status}" "retry_at=${retry_at}"
    if [[ "${delay}" -gt 0 ]]; then
      sleep "${delay}"
    fi
    attempt=$((attempt + 1))
  done
}
