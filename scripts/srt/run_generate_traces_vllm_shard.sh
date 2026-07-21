#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
INPUT_JSONL="${INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/data/ches_train_sft_train_4k.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_shard_vllm.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_shard_vllm_summary.json}"

MAX_SAMPLES="${MAX_SAMPLES:-100}"
NUM_INITS="${NUM_INITS:-4}"
SAMPLE_CHUNK_SIZE="${SAMPLE_CHUNK_SIZE:-16}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SEED="${SEED:-42}"
MIN_PER_DB="${MIN_PER_DB:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/envs/vllm310/bin/python}

${PYTHON_BIN} /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/srt/generate_phase1_traces.py \
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
  --num-inits "${NUM_INITS}" \
  --sample-chunk-size "${SAMPLE_CHUNK_SIZE}" \
  --verifier-workers 16 \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}"
