#!/usr/bin/env bash
set -euo pipefail

GEN="${GEN:-${1:-0}}"
BASE="${ULTRON_BASE_MODEL:-Qwen/Qwen3.5-4B}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_tmux.sh
source "${ROOT}/scripts/lib_tmux.sh"
ADAPTER="${ULTRON_ATTACKER_ADAPTER:-$("${ROOT}/scripts/resolve_adapter.sh" attacker "${GEN}")}"

if [[ ! -d "${ADAPTER}" ]]; then
  echo "Attacker adapter not found: ${ADAPTER}" >&2
  exit 2
fi
ultron_maybe_tmux "ultron-vllm-attacker"

CUDA_VISIBLE_DEVICES="${ULTRON_ATTACKER_GPU:-0}" \
python -m vllm.entrypoints.openai.api_server \
  --model "${BASE}" \
  --enable-lora \
  --lora-modules "attacker-lora=${ADAPTER}" \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --max-model-len 32768 \
  --host 127.0.0.1 \
  --port 8001 \
  --gpu-memory-utilization 0.85
