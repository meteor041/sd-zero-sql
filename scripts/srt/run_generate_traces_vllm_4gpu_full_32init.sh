#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
INPUT_JSONL="${INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_32init_feedbackfix.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_32init_feedbackfix_summary.json}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/envs/vllm310/bin/python}

${PYTHON_BIN} /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/srt/generate_phase1_traces.py \
  --model-path "${MODEL_PATH}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --sampling-mode stratified \
  --min-per-db 5 \
  --seed 42 \
  --backend vllm \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --batch-size 8 \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --num-inits 32 \
  --sample-chunk-size 8 \
  --verifier-workers 16
