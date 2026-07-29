#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/ches_qwen3_4b_sft_from_phase1_4init_correct_sql_dbsplit_train.jsonl}"
VALID_FILE="${VALID_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/ches_qwen3_4b_sft_from_phase1_4init_correct_sql_dbsplit_valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_paper_full_32k}"
PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
GPU_SET="${GPU_SET:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

if [[ "${NUM_PROCESSES}" != "4" ]]; then
  echo "Paper alignment requires NUM_PROCESSES=4 for effective global batch size 4." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing base model directory: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing training JSONL: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VALID_FILE}" ]]; then
  echo "Missing validation JSONL: ${VALID_FILE}" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c "import liger_kernel" >/dev/null 2>&1; then
  echo "liger-kernel is required for the paper-aligned run." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_DIR}"

${ACCELERATE_BIN} \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train_sft.py" \
  --model-path "${MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --valid-file "${VALID_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length 32768 \
  --overlength-policy error \
  --num-train-epochs 3 \
  --learning-rate 5e-6 \
  --weight-decay 1e-4 \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 10 \
  --save-steps 200 \
  --eval-steps 200 \
  --save-total-limit 3 \
  --seed 42 \
  --full-finetune \
  --gradient-checkpointing \
  --use-liger-kernel \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer \
  --bf16 \
  --report-to none

echo "Paper-aligned full SFT model: ${OUTPUT_DIR}"
