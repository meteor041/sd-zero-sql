#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/data/model/Qwen3-4B-Instruct-2507"
TRAIN_FILE="/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/ches_train_sft_train_4k.jsonl"
VALID_FILE="/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/ches_train_sft_valid_4k.jsonl"
OUTPUT_DIR="/data/huwenp/emb/lxy/ches_sql_sft/outputs/qwen3_4b_sft_lora_4k"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${OUTPUT_DIR}"
mkdir -p /home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/logs

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}

${ACCELERATE_BIN} \
  --num_processes 8 \
  --mixed_precision bf16 \
  /home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/scripts/sft/train_sft.py \
  --model-path "${MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --valid-file "${VALID_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length 4096 \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --logging-steps 10 \
  --save-steps 200 \
  --eval-steps 200 \
  --save-total-limit 3 \
  --seed 42 \
  --gradient-checkpointing \
  --bf16 \
  --report-to none
