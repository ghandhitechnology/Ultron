#!/usr/bin/env bash
set -euo pipefail

GEN="${GEN:-${1:-0}}"
if [[ ! "${GEN}" =~ ^[0-9]+$ ]]; then
  echo "Generation must be a non-negative integer." >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_tmux.sh
source "${ROOT}/scripts/lib_tmux.sh"
# shellcheck source=lib_family.sh
source "${ROOT}/scripts/lib_family.sh"
ultron_load_family
ADAPTER="${ULTRON_DEFENDER_ADAPTER:-$("${ROOT}/scripts/resolve_adapter.sh" defender "${GEN}")}"

if [[ ! -d "${ADAPTER}" ]]; then
  echo "Defender adapter not found: ${ADAPTER}" >&2
  exit 2
fi
ultron_maybe_tmux "ultron-vllm-defender" "$@"

chat_kwargs=()
if [[ -n "${ULTRON_VLLM_CHAT_TEMPLATE_KWARGS}" ]]; then
  chat_kwargs+=(--chat-template-kwargs "${ULTRON_VLLM_CHAT_TEMPLATE_KWARGS}")
fi
LISTEN_PORT="${ULTRON_DEFENDER_PORT:-8002}"
UPSTREAM_PORT="${ULTRON_DEFENDER_UPSTREAM_PORT:-8102}"
RESPONSES_DIR="${ULTRON_RESPONSES_DIR:-${ROOT}/data/responses}"

exec env CUDA_VISIBLE_DEVICES="${ULTRON_DEFENDER_GPU:-1}" \
"${ULTRON_PYTHON}" -m ultron.cli.serve \
  --role defender \
  --generation "${GEN}" \
  --listen-port "${LISTEN_PORT}" \
  --upstream-port "${UPSTREAM_PORT}" \
  --responses-dir "${RESPONSES_DIR}" \
  -- \
  "${ULTRON_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${ULTRON_PACK_BASE_MODEL}" \
  --enable-lora \
  --lora-modules "defender-lora=${ADAPTER}" \
  "${chat_kwargs[@]}" \
  --max-model-len "${ULTRON_VLLM_MAX_MODEL_LEN}" \
  --host 127.0.0.1 \
  --port "${UPSTREAM_PORT}" \
  --gpu-memory-utilization "${ULTRON_VLLM_GPU_MEMORY_UTILIZATION}"
