#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/ches_qwen3_4b_sft_from_phase1_4init_correct_sql_dbsplit_train.jsonl}"
VALID_FILE="${VALID_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/ches_qwen3_4b_sft_from_phase1_4init_correct_sql_dbsplit_valid.jsonl}"
MODEL_OUT="${MODEL_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_from_phase1_4init_correctsql_dbsplit/full_model}"
PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
GPU_COUNT="${GPU_COUNT:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-${GPU_COUNT}}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-100}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-5}"
POLL_SECONDS="${POLL_SECONDS:-180}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-sft_from_phase1_4init_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"

mkdir -p "${MODEL_OUT}" "${LOG_DIR}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] run_tag=${RUN_TAG}"
echo "[start] log_file=${LOG_FILE}"
echo "[config] gpu_count=${GPU_COUNT}"
echo "[config] max_memory_used_mb=${MAX_MEMORY_USED_MB}"
echo "[config] max_gpu_utilization=${MAX_GPU_UTILIZATION}"
echo "[config] train_file=${TRAIN_FILE}"
echo "[config] valid_file=${VALID_FILE}"

select_free_gpus() {
  local line idx used util
  local -a selected=()
  mapfile -t GPU_LINES < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  for line in "${GPU_LINES[@]}"; do
    IFS=',' read -r idx used util <<< "${line}"
    idx="${idx// /}"
    used="${used// /}"
    util="${util// /}"
    if (( used <= MAX_MEMORY_USED_MB && util <= MAX_GPU_UTILIZATION )); then
      selected+=("${idx}")
      if (( ${#selected[@]} == GPU_COUNT )); then
        GPU_SET="$(IFS=,; printf '%s' "${selected[*]}")"
        return 0
      fi
    fi
  done
  return 1
}

wait_for_gpus() {
  while true; do
    echo "[gpu-check] $(date '+%F %T')"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    if select_free_gpus; then
      echo "[gpu-check] selected GPUs: ${GPU_SET}"
      return 0
    fi
    echo "[gpu-check] fewer than ${GPU_COUNT} GPUs free, sleeping ${POLL_SECONDS}s"
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
  --output-dir "${MODEL_OUT}" \
  --max-length 4096 \
  --overlength-policy drop \
  --tuning-mode full \
  --num-train-epochs 3 \
  --learning-rate 5e-6 \
  --weight-decay 1e-4 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --optim adamw_torch \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --sync-each-batch \
  --logging-steps 10 \
  --save-steps 200 \
  --eval-steps 200 \
  --save-total-limit 3 \
  --seed 42 \
  --gradient-checkpointing \
  --bf16 \
  --use-liger-kernel \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer \
  --report-to none

echo "SQL-SFT full model: ${MODEL_OUT}"
echo "[done] run_tag=${RUN_TAG} end_time=$(date '+%F %T')"
