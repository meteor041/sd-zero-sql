#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/ches_qwen3_4b_sft_from_phase1_4init_correct_sql_dbsplit_train.jsonl}"
VALID_FILE="${VALID_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/ches_qwen3_4b_sft_from_phase1_4init_correct_sql_dbsplit_valid.jsonl}"
ADAPTER_OUT="${ADAPTER_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_from_phase1_4init_correctsql_dbsplit/adapter}"
MERGED_OUT="${MERGED_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_from_phase1_4init_correctsql_dbsplit/merged}"
PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
GPU_SET="${GPU_SET:-0,1,2,3}"
IFS=',' read -r -a GPU_INDICES <<< "${GPU_SET}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
POLL_SECONDS="${POLL_SECONDS:-180}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-sft_from_phase1_4init_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"

mkdir -p "${ADAPTER_OUT}" "${MERGED_OUT}" "${LOG_DIR}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] run_tag=${RUN_TAG}"
echo "[start] log_file=${LOG_FILE}"
echo "[config] gpu_set=${GPU_SET}"
echo "[config] train_file=${TRAIN_FILE}"
echo "[config] valid_file=${VALID_FILE}"

gpu_free() {
  mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  READY=1
  for idx in "${GPU_INDICES[@]}"; do
    line="$(printf '%s\n' "${GPU_LINES[@]}" | grep -E "^${idx},")"
    used="$(printf '%s' "${line}" | cut -d',' -f2 | tr -d ' ')"
    util="$(printf '%s' "${line}" | cut -d',' -f3 | tr -d ' ')"
    if [[ -z "${used}" || -z "${util}" ]]; then
      READY=0
      continue
    fi
    if (( used > 100 || util > 5 )); then
      READY=0
    fi
  done
  return $((1 - READY))
}

wait_for_gpus() {
  while true; do
    echo "[gpu-check] $(date '+%F %T')"
    for idx in "${GPU_INDICES[@]}"; do
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | grep -E "^${idx},"
    done
    if gpu_free; then
      echo "[gpu-check] target GPUs appear free"
      return 0
    fi
    echo "[gpu-check] GPUs busy, sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

wait_for_gpus

CUDA_VISIBLE_DEVICES="${GPU_SET}" \
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
echo "[done] run_tag=${RUN_TAG} end_time=$(date '+%F %T')"