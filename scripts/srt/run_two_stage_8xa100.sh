#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="/data/model/Qwen3-4B-Instruct-2507"
SFT_ADAPTER="/data/huwenp/emb/lxy/ches_sql_sft/outputs/qwen3_4b_sft_lora_4k"
TRAIN_DATA_DIR="/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/srt"
DATA_PREFIX="ches_qwen3_4b_srt"
STAGE1_FILE="${TRAIN_DATA_DIR}/${DATA_PREFIX}_stage1.jsonl"
STAGE2_FILE="${TRAIN_DATA_DIR}/${DATA_PREFIX}_stage2.jsonl"
CKPT_ROOT="/data/huwenp/emb/lxy/ches_sql_sft/outputs/qwen3_4b_srt_two_stage"
STAGE1_OUT="${CKPT_ROOT}/stage1"
STAGE2_OUT="${CKPT_ROOT}/stage2"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${STAGE1_OUT}" "${STAGE2_OUT}"
mkdir -p /home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/logs

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}

# Stage 1: generation loss style data
${ACCELERATE_BIN} \
  --num_processes 8 \
  --mixed_precision bf16 \
  /home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/scripts/srt/train_srt_stage.py \
  --model-path "${BASE_MODEL}" \
  --adapter-path "${SFT_ADAPTER}" \
  --train-file "${STAGE1_FILE}" \
  --valid-file "${STAGE1_FILE}" \
  --output-dir "${STAGE1_OUT}" \
  --max-length 4096 \
  --num-train-epochs 2 \
  --learning-rate 1e-4 \
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

# Stage 2: revision loss style data, initialized from stage1 output
${ACCELERATE_BIN} \
  --num_processes 8 \
  --mixed_precision bf16 \
  /home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/scripts/srt/train_srt_stage.py \
  --model-path "${BASE_MODEL}" \
  --adapter-path "${STAGE1_OUT}" \
  --train-file "${STAGE2_FILE}" \
  --valid-file "${STAGE2_FILE}" \
  --output-dir "${STAGE2_OUT}" \
  --max-length 4096 \
  --num-train-epochs 2 \
  --learning-rate 1e-4 \
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
