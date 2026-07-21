#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
INPUT_JSONL="${INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_benchmark_128_32init_feedbackfix.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_benchmark_128_32init_feedbackfix_summary.json}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-/home/pkuccadm/anaconda3/envs/vllm310/bin/python}

START_TS=$(date +%s)

echo "[benchmark] start_time=$(date '+%F %T')"
echo "[benchmark] max_samples=${MAX_SAMPLES}"
echo "[benchmark] output_jsonl=${OUTPUT_JSONL}"
echo "[benchmark] summary_json=${SUMMARY_JSON}"

${PYTHON_BIN} /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/srt/generate_phase1_traces.py \
  --model-path "${MODEL_PATH}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --max-samples "${MAX_SAMPLES}" \
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
  --sample-chunk-size 8 \
  --verifier-workers 16

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "[benchmark] end_time=$(date '+%F %T')"
echo "[benchmark] elapsed_seconds=${ELAPSED}"
