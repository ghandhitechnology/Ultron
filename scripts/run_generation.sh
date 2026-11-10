#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/lib_tmux.sh"
source "${ROOT}/scripts/lib_family.sh"
source "${ROOT}/scripts/lib_pipeline.sh"
cd "${ROOT}"

LAUNCH_ARGS=("$@")
GEN=""
PLAN_ONLY=0
PLAN_ARGS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --family|--variant|--seed|--run-id|--episodes|--study-root|--prep-turns)
      [[ "$#" -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      PLAN_ARGS+=("$1" "$2"); shift 2 ;;
    --plan) PLAN_ONLY=1; shift ;;
    --pilot) PLAN_ARGS+=(--pilot); shift ;;
    --) shift; break ;;
    -*) echo "Unknown argument: $1" >&2; exit 2 ;;
    *)
      [[ -z "${GEN}" ]] || { echo "Unknown argument: $1" >&2; exit 2; }
      GEN="$1"; shift ;;
  esac
done
GEN="${GEN:-0}"
[[ "${GEN}" =~ ^[0-9]+$ ]] || { echo "Generation must be a non-negative integer." >&2; exit 2; }
source "${ROOT}/scripts/lib_runtime.sh"
ultron_load_runtime "${ROOT}"
if [[ "${PLAN_ONLY}" == "1" ]]; then
  exec "${ULTRON_PYTHON}" -m ultron.train.pipeline plan --generation "${GEN}" ${PLAN_ARGS[@]+"${PLAN_ARGS[@]}"}
fi
exported="$("${ULTRON_PYTHON}" -m ultron.train.pipeline export --generation "${GEN}" ${PLAN_ARGS[@]+"${PLAN_ARGS[@]}"})"
eval "${exported}"
ultron_load_family
ultron_maybe_tmux "${ULTRON_STUDY_SESSION}-gen${GEN}" ${LAUNCH_ARGS[@]+"${LAUNCH_ARGS[@]}"}
echo "=== Ultron generation ${GEN}: ${ULTRON_MODEL_FAMILY}/${ULTRON_STUDY_VARIANT}/seed${ULTRON_STUDY_SEED}/${ULTRON_STUDY_RUN_ID} ==="
echo "Requested ${ULTRON_REQUESTED_EPISODES} games; running ${ULTRON_EPISODES} in complete six-member groups."

# Resolve all policy/trainer/runtime dependencies before allocating rollout work.
"${ULTRON_PYTHON}" -m ultron.train.pipeline prepare
"${ULTRON_PYTHON}" -m ultron.train.pipeline execution-check
"${ROOT}/scripts/rollout_worker.sh" --generation "${GEN}" --episodes "${ULTRON_EPISODES}" --preflight-only
export ULTRON_PIPELINE_INPUT_KEY
ULTRON_PIPELINE_INPUT_KEY="$("${ULTRON_PYTHON}" -m ultron.train.pipeline fingerprint)"
if [[ -z "${ULTRON_ALLOCATION_ID:-}" ]]; then
  export ULTRON_NO_TMUX=1
  exec "${ULTRON_PYTHON}" -m ultron.train.compute run \
    --ledger "${ULTRON_COMPUTE_LEDGER}" --phase "generation${GEN}" \
    --run-id "${ULTRON_STUDY_SESSION}" --scope-key "${ULTRON_COMPUTE_SCOPE}" \
    --profile "${ROOT}/configs/h100.yaml" --verify-hardware \
    -- "${ROOT}/scripts/run_generation.sh" ${LAUNCH_ARGS[@]+"${LAUNCH_ARGS[@]}"}
fi
ultron_pipeline_init "${ULTRON_STUDY_SESSION}-gen${GEN}"
TRACE_DIR="${ULTRON_TRACE_ROOT}/gen${GEN}"

ULTRON_STAGE_OUTPUT_FILES="${TRACE_DIR}/attacker.jsonl
${TRACE_DIR}/defender.jsonl
${TRACE_DIR}/metadata/rollout.json" ultron_run_stage rollout \
  "${ROOT}/scripts/rollout_worker.sh" --generation "${GEN}" --episodes "${ULTRON_EPISODES}"
ultron_run_stage review-rollout "${ULTRON_PYTHON}" -m ultron.train.review \
  "${TRACE_DIR}" --phase rollout --generation "${GEN}" --output "${TRACE_DIR}"
# Check actual learner-group variance; binary ASR alone misses shaping signal.
DPO_GATE_ARGS=()
if [[ "${ULTRON_STUDY_DPO}" == "1" ]]; then
  DPO_GATE_ARGS+=(--dpo-pairs "${ULTRON_DPO_ROOT}/gen${GEN}/attacker.jsonl")
fi
ULTRON_STAGE_OUTPUT_FILES="${TRACE_DIR}/training-gate.json" ultron_run_stage training-gate \
  "${ULTRON_PYTHON}" -m ultron.train.gates "${TRACE_DIR}" \
  --request "${ULTRON_ROLLOUT_REQUEST}" --output "${TRACE_DIR}/training-gate.json" ${DPO_GATE_ARGS[@]+"${DPO_GATE_ARGS[@]}"}
for ROLE in attacker defender; do
  ULTRON_STAGE_OUTPUT_FILES="${ULTRON_CHECKPOINT_ROOT}/gen${GEN}/${ROLE}_lora/training-receipt.json" \
    ultron_run_stage "grpo-${ROLE}" "${ROOT}/scripts/train_grpo.sh" --role "${ROLE}" --generation "${GEN}"
done
if [[ "${ULTRON_STUDY_DPO}" == "1" ]]; then
  ULTRON_STAGE_OUTPUT_FILES="${ULTRON_CHECKPOINT_ROOT}/gen${GEN}/attacker_lora_dpo/training-receipt.json" \
    ultron_run_stage dpo-attacker "${ROOT}/scripts/train_dpo.sh" --role attacker --generation "${GEN}"
fi
ULTRON_STAGE_OUTPUT_FILES="${ULTRON_ARCHIVE_ROOT}/gen${GEN}/manifest.json" ultron_run_stage archive \
  "${ULTRON_PYTHON}" -m ultron.train.archive --generation "${GEN}" \
  --checkpoint-root "${ULTRON_CHECKPOINT_ROOT}" --archive-root "${ULTRON_ARCHIVE_ROOT}" \
  --pfsp-manifest "${ULTRON_PFSP_MANIFEST}" --model-config "${ULTRON_MODEL_CONFIG}"
if [[ -n "${ULTRON_STUDY_EVAL_MODE}" ]]; then
  ultron_run_stage "tier3-${ULTRON_STUDY_EVAL_MODE}-plan" "${ULTRON_PYTHON}" -m ultron.eval.run_tier3 \
    --mode "${ULTRON_STUDY_EVAL_MODE}" --output "${ULTRON_EVAL_ROOT}/gen${GEN}"
fi
ultron_run_stage review-complete "${ULTRON_PYTHON}" -m ultron.train.review \
  "${TRACE_DIR}" --phase complete --generation "${GEN}" --output "${TRACE_DIR}" \
  --eval-dir "${ULTRON_EVAL_ROOT}" --archive-dir "${ULTRON_ARCHIVE_ROOT}" --pfsp "${ULTRON_PFSP_MANIFEST}"
