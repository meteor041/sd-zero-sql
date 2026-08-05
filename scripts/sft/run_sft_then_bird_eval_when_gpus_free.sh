#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
TRAIN_FILE="${TRAIN_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/ches_qwen3_4b_sft_from_phase1_4init_correct_sql_dbsplit_train.jsonl}"
VALID_FILE="${VALID_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/ches_qwen3_4b_sft_from_phase1_4init_correct_sql_dbsplit_valid.jsonl}"
MODEL_OUT="${MODEL_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_from_phase1_4init_correctsql_dbsplit/full_model}"

GPU_SET="${GPU_SET:-4,5,6,7}"
IFS=',' read -r -a GPU_INDICES <<< "${GPU_SET}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
POLL_SECONDS="${POLL_SECONDS:-180}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-sft_then_bird_eval_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"

PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
VLLM310_BIN_DIR="${VLLM310_BIN_DIR:-/home/pkuccadm/anaconda3/envs/vllm310/bin}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/home/pkuccadm/huwenp/emb/lxy/csc_sql/bin/process/run_ches_qwen3_eval.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/bird_eval_sft_from_phase1_4init}"
DATA_ROOT="${DATA_ROOT:-/data/huwenp/emb/data/ches}"
DATAFILE_PATH="${DATAFILE_PATH:-/data/huwenp/emb/data/ches/dev.json}"
GOLD_PATH="${GOLD_PATH:-/data/huwenp/emb/data/ches/dev.sql}"
DATASET_PATH="${DATASET_PATH:-/data/huwenp/emb/data/ches/dev_databases}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

mkdir -p "${MODEL_OUT}" "${LOG_DIR}" "${OUTPUT_ROOT}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PATH="${VLLM310_BIN_DIR}:${PATH}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] run_tag=${RUN_TAG}"
echo "[start] log_file=${LOG_FILE}"
echo "[config] gpu_set=${GPU_SET}"
echo "[config] train_file=${TRAIN_FILE}"
echo "[config] valid_file=${VALID_FILE}"
echo "[config] model_out=${MODEL_OUT}"

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

run_sft() {
  echo "[sft] start $(date '+%F %T')"
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
  echo "[sft] done $(date '+%F %T')"
}

run_eval() {
  local mode="$1"
  local n_sql="$2"
  local temp="$3"
  local run_time="${RUN_TAG}_${mode}"
  local out_dir="${OUTPUT_ROOT}/${mode}"
  mkdir -p "${out_dir}"
  echo "[eval] mode=${mode} start=$(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" \
  MODEL_SQL_GENERATE="${MODEL_OUT}" \
  DATA_ROOT="${DATA_ROOT}" \
  DATAFILE_PATH="${DATAFILE_PATH}" \
  GOLD_PATH="${GOLD_PATH}" \
  DATASET_PATH="${DATASET_PATH}" \
  DATASET_NAME="bird" \
  DATASET_MODE="dev" \
  EVAL_MODE="${mode}" \
  N_SQL_GENERATE="${n_sql}" \
  TEMPERATURE_SQL_GENERATE="${temp}" \
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  OUTPUT_DIR="${out_dir}" \
  LOG_DIR="${LOG_DIR}" \
  RUN_TIME="${run_time}" \
  bash "${EVAL_SCRIPT}"
  echo "[eval] mode=${mode} done=$(date '+%F %T')"
}

wait_for_gpus
run_sft
wait_for_gpus
run_eval "greedy_search" 1 0.0
run_eval "major_voting" 8 0.8

echo "[done] run_tag=${RUN_TAG} end_time=$(date '+%F %T')"