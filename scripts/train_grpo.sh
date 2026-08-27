#!/usr/bin/env bash
set -euo pipefail

ROLE=""
GENERATION=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --generation) GENERATION="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "${ROLE}" != "attacker" && "${ROLE}" != "defender" ]]; then
  echo "--role must be attacker or defender" >&2
  exit 2
fi
if [[ -z "${GENERATION}" ]]; then
  echo "--generation is required" >&2
  exit 2
fi
if [[ ! -d third_party/verl/verl ]]; then
  echo "Initialize third_party/verl before training." >&2
  exit 2
fi

INPUT="data/traces/gen${GENERATION}/${ROLE}.jsonl"
OUTPUT="data/verl/gen${GENERATION}/${ROLE}.jsonl"
python -m ultron.train.convert_verl "${INPUT}" "${OUTPUT}" --generation "${GENERATION}"

PYTHONPATH="third_party/verl:${PYTHONPATH:-}" \
python -m verl.trainer.main_ppo \
  --config-path "$(pwd)/configs" \
  --config-name train_grpo \
  "data.train_files=${OUTPUT}" \
  "trainer.default_local_dir=data/checkpoints/gen${GENERATION}/${ROLE}_lora"
