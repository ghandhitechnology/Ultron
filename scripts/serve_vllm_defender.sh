#!/usr/bin/env bash
set -euo pipefail

GEN="${GEN:-${1:-0}}"
BASE="${ULTRON_BASE_MODEL:-Qwen/Qwen3.5-4B}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER="${ULTRON_DEFENDER_ADAPTER:-$("${ROOT}/scripts/resolve_adapter.sh" defender "${GEN}")}"

if [[ ! -d "${ADAPTER}" ]]; then
  echo "Defender adapter not found: ${ADAPTER}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${ULTRON_DEFENDER_GPU:-1}" \
python -m vllm.entrypoints.openai.api_server \
  --model "${BASE}" \
  --enable-lora \
  --lora-modules "defender-lora=${ADAPTER}" \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --max-model-len 32768 \
  --host 127.0.0.1 \
  --port 8002 \
  --gpu-memory-utilization 0.85
