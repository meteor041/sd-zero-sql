#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql"
PHASE1_STAGE2_MODEL="${PHASE1_STAGE2_MODEL:-${PROJECT_ROOT}/outputs/qwen3_4b_phase1_1k_tp4_full/stage2}"
INPUT_JSONL="${INPUT_JSONL:-${PROJECT_ROOT}/data/ches_train_sft_train_4k.jsonl}"
VALID_JSONL="${VALID_JSONL:-${PROJECT_ROOT}/data/ches_train_sft_valid_4k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sql_distill_phase2_4gpu}"
DEBUG_MANIFEST="${DEBUG_MANIFEST:-${PROJECT_ROOT}/data/distill/sql_distill_phase2_4gpu_debug_manifest.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/outputs/sql_distill_phase2_4gpu_checkpoints}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-16}"
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
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

${ACCELERATE_BIN} \
  --num_processes 4 \
  --mixed_precision bf16 \
  "${PROJECT_ROOT}/src/phase2_distill/train_distill_kl.py" \
  --student-model "${PHASE1_STAGE2_MODEL}" \
  --teacher-model "${PHASE1_STAGE2_MODEL}" \
  --input-jsonl "${INPUT_JSONL}" \
  --valid-jsonl "${VALID_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --debug-manifest "${DEBUG_MANIFEST}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --eval-max-samples "${EVAL_MAX_SAMPLES}" \
  --save-every-steps "${SAVE_EVERY_STEPS}" \
  --eval-every-steps "${EVAL_EVERY_STEPS}" \
  --backend hf \
  --batch-size 1 \
  --rollout-batch-size 1 \
  --num-train-epochs 1 \
  --learning-rate 1e-5 \
  --max-length 4096 \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --bf16 \
  --gradient-checkpointing \
  --save-debug-manifest \
  "${EXTRA_ARGS[@]}"
