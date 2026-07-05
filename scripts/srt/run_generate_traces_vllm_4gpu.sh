#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/data/model/Qwen3-4B-Instruct-2507"
INPUT_JSONL="/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/ches_train_sft_train_4k.jsonl"
OUTPUT_JSONL="/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/srt/traces_train_1k_stratified_vllm.jsonl"
SUMMARY_JSON="/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/srt/traces_train_1k_stratified_vllm_summary.json"

# Default to 4 GPUs when they are free. Override externally if needed.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}

# Recommended starting point for 4xA100-40G.
${PYTHON_BIN} /home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/scripts/srt/generate_phase1_traces.py \
  --model-path "${MODEL_PATH}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --max-samples 1000 \
  --sampling-mode stratified \
  --min-per-db 5 \
  --seed 42 \
  --backend vllm \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --batch-size 32 \
  --max-new-tokens 128 \
  --temperature 0.0
