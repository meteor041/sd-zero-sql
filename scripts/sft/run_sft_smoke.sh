#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/data/model/Qwen3-4B-Instruct-2507"
TRAIN_FILE="/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_train.jsonl"
VALID_FILE="/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_valid.jsonl"
OUTPUT_DIR="/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_smoke"

# 默认只用当前空闲的 3 张卡，避免与他人任务冲突
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${OUTPUT_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}

${ACCELERATE_BIN} \
  --num_processes 3 \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train_sft.py" \
  --model-path "${MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --valid-file "${VALID_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length 8192 \
  --overlength-policy error \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --logging-steps 1 \
  --save-steps 20 \
  --eval-steps 20 \
  --save-total-limit 2 \
  --seed 42 \
  --gradient-checkpointing \
  --bf16 \
  --report-to none \
  --max-train-samples 128 \
  --max-eval-samples 32
