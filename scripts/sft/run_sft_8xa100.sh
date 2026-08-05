#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_train.jsonl}"
VALID_FILE="${VALID_FILE:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_valid.jsonl}"
MODEL_OUT="${MODEL_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_full_8k}"
PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
GPU_SET="${GPU_SET:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${MODEL_OUT}"

${ACCELERATE_BIN} \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train_sft.py" \
  --model-path "${MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --valid-file "${VALID_FILE}" \
  --output-dir "${MODEL_OUT}" \
  --max-length 8192 \
  --overlength-policy error \
  --tuning-mode full \
  --num-train-epochs 3 \
  --learning-rate 5e-6 \
  --weight-decay 1e-4 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --optim adamw_torch \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --sync-each-batch \
  --logging-steps 10 \
  --save-steps 200 \
  --eval-steps 200 \
  --save-total-limit 3 \
  --seed 42 \
  --gradient-checkpointing \
  --bf16 \
  --use-liger-kernel \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer \
  --report-to none

echo "SQL-SFT full model: ${MODEL_OUT}"
