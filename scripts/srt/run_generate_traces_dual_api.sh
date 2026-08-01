#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# API model IDs are case-sensitive; EAS exposes the 4B model with this spelling.
INIT_MODEL="${INIT_MODEL:-Qwen3-4B-Instruct-2507}"
REVISION_MODEL="${REVISION_MODEL:-qwen3-coder-30b-a3b-instruct}"
PROMPT_TOKENIZER_PATH="${PROMPT_TOKENIZER_PATH:-/data/model/Qwen3-4B-Instruct-2507}"

INIT_API_BASE_URL="${PHASE1_INIT_API_BASE_URL:-${OPENAI_BASE_URL:-}}"
REVISION_API_BASE_URL="${PHASE1_REVISION_API_BASE_URL:-${INIT_API_BASE_URL}}"
if [[ -z "${INIT_API_BASE_URL}" ]]; then
  echo "Set PHASE1_INIT_API_BASE_URL or OPENAI_BASE_URL." >&2
  exit 1
fi

# A shared endpoint commonly uses one key. Keep phase-specific keys available for separate providers.
if [[ -z "${PHASE1_INIT_API_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export PHASE1_INIT_API_KEY="${OPENAI_API_KEY}"
fi
if [[ -z "${PHASE1_REVISION_API_KEY:-}" && -n "${PHASE1_INIT_API_KEY:-}" ]]; then
  export PHASE1_REVISION_API_KEY="${PHASE1_INIT_API_KEY}"
fi

INPUT_JSONL="${INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_api_4binit_coder30brev.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_api_4binit_coder30brev_summary.json}"
PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"

export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_phase1_traces.py" \
  --init-backend api \
  --revision-backend api \
  --init-model-path "${INIT_MODEL}" \
  --revision-model-path "${REVISION_MODEL}" \
  --init-tokenizer-path "${PROMPT_TOKENIZER_PATH}" \
  --revision-tokenizer-path "${PROMPT_TOKENIZER_PATH}" \
  --init-api-base-url "${INIT_API_BASE_URL}" \
  --revision-api-base-url "${REVISION_API_BASE_URL}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --sampling-mode stratified \
  --min-per-db 5 \
  --seed 42 \
  --batch-size "${BATCH_SIZE:-8}" \
  --init-api-max-concurrency "${INIT_API_MAX_CONCURRENCY:-8}" \
  --revision-api-max-concurrency "${REVISION_API_MAX_CONCURRENCY:-8}" \
  --api-timeout "${API_TIMEOUT:-120}" \
  --api-max-retries "${API_MAX_RETRIES:-5}" \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 1.0 \
  --num-inits 1 \
  --num-revisions 3 \
  --max-model-len 8192 \
  --verifier-workers "${VERIFIER_WORKERS:-16}" \
  "${EXTRA_ARGS[@]}"
