#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRT_MODEL="${SRT_MODEL:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_joint/merged}"
INPUT_JSONL="${INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_train.jsonl}"
VALID_JSONL="${VALID_JSONL:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_valid.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/sql_distill_phase2_4gpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/adapter}"
MERGED_OUT="${MERGED_OUT:-${OUTPUT_ROOT}/merged}"
DEBUG_MANIFEST="${DEBUG_MANIFEST:-${PROJECT_ROOT}/data/distill/sql_distill_phase2_4gpu_debug_manifest.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/sql_distill_phase2_4gpu_checkpoints}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-256}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

${ACCELERATE_BIN} \
  --num_processes 4 \
  --mixed_precision bf16 \
  "${PROJECT_ROOT}/src/phase2_distill/train_distill_kl.py" \
  --student-model "${SRT_MODEL}" \
  --teacher-model "${SRT_MODEL}" \
  --input-jsonl "${INPUT_JSONL}" \
  --valid-jsonl "${VALID_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --debug-manifest "${DEBUG_MANIFEST}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --eval-max-samples "${EVAL_MAX_SAMPLES}" \
  --save-every-steps "${SAVE_EVERY_STEPS}" \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --backend hf \
  --batch-size 1 \
  --rollout-batch-size 1 \
  --num-train-epochs 1 \
  --learning-rate 5e-6 \
  --max-length 8192 \
  --max-new-tokens 256 \
  --temperature 1.0 \
  --bf16 \
  --gradient-checkpointing \
  --save-debug-manifest \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/srt/merge_lora_adapter.py" \
  --base-model "${SRT_MODEL}" \
  --adapter-path "${OUTPUT_DIR}" \
  --output-dir "${MERGED_OUT}" \
  --torch-dtype bfloat16 \
  --device-map auto

echo "Phase2 adapter: ${OUTPUT_DIR}"
echo "Phase2 standalone model: ${MERGED_OUT}"
