#!/usr/bin/env bash
set -euo pipefail

GEN="${GEN:-${1:-0}}"
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

CUDA_VISIBLE_DEVICES="${ULTRON_DEFENDER_GPU:-1}" \
python -m vllm.entrypoints.openai.api_server \
  --model "${ULTRON_BASE_MODEL}" \
  --enable-lora \
  --lora-modules "defender-lora=${ADAPTER}" \
  "${chat_kwargs[@]}" \
  --max-model-len "${ULTRON_VLLM_MAX_MODEL_LEN}" \
  --host 127.0.0.1 \
  --port 8002 \
  --gpu-memory-utilization "${ULTRON_VLLM_GPU_MEMORY_UTILIZATION}"
