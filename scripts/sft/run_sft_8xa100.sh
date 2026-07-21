#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_train.jsonl}"
VALID_FILE="${VALID_FILE:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_valid.jsonl}"
ADAPTER_OUT="${ADAPTER_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_lora_8k}"
MERGED_OUT="${MERGED_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_merged_8k}"
PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
GPU_SET="${GPU_SET:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${ADAPTER_OUT}" "${MERGED_OUT}"

${ACCELERATE_BIN} \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train_sft.py" \
  --model-path "${MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --valid-file "${VALID_FILE}" \
  --output-dir "${ADAPTER_OUT}" \
  --max-length 8192 \
  --overlength-policy error \
  --num-train-epochs 3 \
  --learning-rate 1e-4 \
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
  --gradient-checkpointing \
  --bf16 \
  --report-to none

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/srt/merge_lora_adapter.py" \
  --base-model "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_OUT}" \
  --output-dir "${MERGED_OUT}" \
  --torch-dtype bfloat16 \
  --device-map auto

echo "SQL-SFT adapter: ${ADAPTER_OUT}"
echo "SQL-SFT standalone model: ${MERGED_OUT}"
