#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_merged_8k}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_JSONL="/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl"
OUTPUT_JSONL="${OUTPUT_JSONL:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_1init_3revision.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_1init_3revision_summary.json}"

# Default to 4 GPUs when they are free. Override externally if needed.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "MODEL_PATH must be a standalone SQL-SFT model: ${MODEL_PATH}" >&2
  exit 1
fi

# Recommended full-data starting point for 4xA100-40G with num_inits=32.
${PYTHON_BIN} "${SCRIPT_DIR}/generate_phase1_traces.py" \
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
  --temperature 0.7 \
  --top-p 1.0 \
  --num-inits 1 \
  --num-revisions 3 \
  --max-model-len 8192 \
  --verifier-workers 16
