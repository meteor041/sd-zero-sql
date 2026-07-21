#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_MODEL="${BASE_MODEL:-/data/model/Qwen3-4B-Instruct-2507}"
SQL_SFT_ADAPTER="${SQL_SFT_ADAPTER:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_lora_8k}"
TRACE_FILE="${TRACE_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_1init_3revision.jsonl}"
DATA_DIR="${DATA_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt}"
DATA_PREFIX="${DATA_PREFIX:-ches_qwen3_4b_srt_joint}"
TRAIN_FILE="${DATA_DIR}/${DATA_PREFIX}_train.jsonl"
VALID_FILE="${DATA_DIR}/${DATA_PREFIX}_valid.jsonl"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_joint}"
ADAPTER_OUT="${ADAPTER_OUT:-${OUTPUT_ROOT}/adapter}"
MERGED_OUT="${MERGED_OUT:-${OUTPUT_ROOT}/merged}"

PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
GPU_SET="${GPU_SET:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-3}"

if [[ ! -f "${TRACE_FILE}" ]]; then
  echo "Missing Phase1 trace file: ${TRACE_FILE}" >&2
  echo "Generate traces with 1 init x 3 revisions before training." >&2
  exit 1
fi
if [[ ! -f "${SQL_SFT_ADAPTER}/adapter_config.json" ]]; then
  echo "Missing SQL-SFT adapter: ${SQL_SFT_ADAPTER}" >&2
  echo "Phase1 requires a competent SQL generator; set SQL_SFT_ADAPTER explicitly." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${ADAPTER_OUT}" "${MERGED_OUT}"
export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_phase1_multitask_data.py" \
  --input "${TRACE_FILE}" \
  --output-dir "${DATA_DIR}" \
  --prefix "${DATA_PREFIX}" \
  --validation-fraction 0.05 \
  --max-traces-per-question 3 \
  --max-correct-init-ratio 0.5 \
  --seed 42

${ACCELERATE_BIN} \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train_srt_stage.py" \
  --model-path "${BASE_MODEL}" \
  --adapter-path "${SQL_SFT_ADAPTER}" \
  --train-file "${TRAIN_FILE}" \
  --valid-file "${VALID_FILE}" \
  --output-dir "${ADAPTER_OUT}" \
  --max-length "${MAX_LENGTH}" \
  --overlength-policy error \
  --num-train-epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
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

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_lora_adapter.py" \
  --base-model "${BASE_MODEL}" \
  --adapter-path "${ADAPTER_OUT}" \
  --output-dir "${MERGED_OUT}" \
  --torch-dtype bfloat16 \
  --device-map auto

echo "Phase1 adapter: ${ADAPTER_OUT}"
echo "Phase1 standalone model: ${MERGED_OUT}"
