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
  export ULTRON_PIPELINE_STATE_DIR
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
  printf '%s\0' "$@" | cksum | awk '{print $1 "-" $2}'
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
  local fingerprint
  fingerprint="$(ultron_stage_fingerprint "$@")"
  if [[ "${ULTRON_PIPELINE_RESUME:-1}" != "0" && -f "${done_path}" ]]; then
    if grep -q "^fingerprint=${fingerprint}$" "${done_path}"; then
      echo "=== Stage ${stage}: already complete ==="
      return 0
    fi
    echo "=== Stage ${stage}: inputs changed; running again ==="
  fi
  if [[ -f "${running_path}" || -f "${failed_path}" ]]; then
    echo "=== Recovering unfinished stage ${stage} ==="
  fi

  local attempt=1 status
  while [[ "${attempt}" -le "${attempts}" ]]; do
    echo "=== Stage ${stage}: attempt ${attempt}/${attempts} ==="
    ultron_write_stage_state \
      "${running_path}" "attempt=${attempt}" "pid=$$" "fingerprint=${fingerprint}"
    if "$@"; then
      ultron_write_stage_state \
        "${done_path}" "attempt=${attempt}" "status=0" "fingerprint=${fingerprint}"
      rm -f "${running_path}" "${failed_path}"
      echo "=== Stage ${stage}: complete ==="
      return 0
    else
      status=$?
    fi
    ultron_write_stage_state "${failed_path}" "attempt=${attempt}" "status=${status}"
    rm -f "${running_path}"
    case "${status}" in
      2|126|127)
        echo "Stage ${stage} cannot start with status ${status}; retry disabled." >&2
        return "${status}"
        ;;
    esac
    if [[ "${attempt}" -ge "${attempts}" ]]; then
      echo "Stage ${stage} failed with status ${status} after ${attempt} attempt(s)." >&2
      return "${status}"
    fi
    echo "Stage ${stage} failed with status ${status}; retrying in ${delay}s." >&2
    if [[ "${delay}" -gt 0 ]]; then
      sleep "${delay}"
    fi
    attempt=$((attempt + 1))
  done
}
