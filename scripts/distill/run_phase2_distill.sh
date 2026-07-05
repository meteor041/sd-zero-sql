#!/usr/bin/env bash
set -euo pipefail

STUDENT_MODEL="${STUDENT_MODEL:-/data/model/Qwen3-4B-Instruct-2507}"
TEACHER_MODEL="${TEACHER_MODEL:-/data/model/Qwen3-4B-Instruct-2507}"
INPUT_JSONL="${INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/data/ches_train_sft_train_4k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/outputs/sql_distill_smoke}"
DEBUG_MANIFEST="${DEBUG_MANIFEST:-/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/data/distill/sql_distill_debug_manifest.jsonl}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}

${PYTHON_BIN} /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/src/phase2_distill/train_distill_kl.py \
  --student-model "${STUDENT_MODEL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --debug-manifest "${DEBUG_MANIFEST}" \
  --max-samples 8 \
  --backend hf \
  --batch-size 2 \
  --rollout-batch-size 2 \
  --num-train-epochs 1 \
  --learning-rate 1e-5 \
  --max-length 4096 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --bf16 \
  --save-debug-manifest
