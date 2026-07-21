#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_merged_8k}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_JSONL="${INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_shard_vllm.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_shard_vllm_summary.json}"

MAX_SAMPLES="${MAX_SAMPLES:-100}"
NUM_INITS="${NUM_INITS:-1}"
NUM_REVISIONS="${NUM_REVISIONS:-3}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SEED="${SEED:-42}"
MIN_PER_DB="${MIN_PER_DB:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/envs/vllm310/bin/python}

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "MODEL_PATH must be a standalone SQL-SFT model: ${MODEL_PATH}" >&2
  exit 1
fi

${PYTHON_BIN} "${SCRIPT_DIR}/generate_phase1_traces.py" \
  --model-path "${MODEL_PATH}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --max-samples "${MAX_SAMPLES}" \
  --sampling-mode stratified \
  --min-per-db "${MIN_PER_DB}" \
  --seed "${SEED}" \
  --backend vllm \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --num-inits "${NUM_INITS}" \
  --num-revisions "${NUM_REVISIONS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --verifier-workers 16 \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}"
