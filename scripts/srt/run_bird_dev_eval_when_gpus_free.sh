#!/usr/bin/env bash
set -euo pipefail

EVAL_SCRIPT="${EVAL_SCRIPT:-/home/pkuccadm/huwenp/emb/lxy/csc_sql/bin/process/run_ches_qwen3_eval.sh}"
GPU_SET="${GPU_SET:-4,5,6,7}"
IFS=',' read -r -a GPU_INDICES <<< "${GPU_SET}"
POLL_SECONDS="${POLL_SECONDS:-180}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-bird_dev_eval_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"

STAGE1_MODEL_DIR="${STAGE1_MODEL_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_stage1_base_1init_stage1k4_gpu0to3}"
STAGE2_MODEL_DIR="${STAGE2_MODEL_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_stage2_from_stage1_1init_stage1k4_gpu0to3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/bird_eval}"
DATA_ROOT="${DATA_ROOT:-/data/huwenp/emb/data/ches}"
DATAFILE_PATH="${DATAFILE_PATH:-/data/huwenp/emb/data/ches/dev.json}"
GOLD_PATH="${GOLD_PATH:-/data/huwenp/emb/data/ches/dev.sql}"
DATASET_PATH="${DATASET_PATH:-/data/huwenp/emb/data/ches/dev_databases}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM310_BIN_DIR="${VLLM310_BIN_DIR:-/home/pkuccadm/anaconda3/envs/vllm310/bin}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
export PATH="${VLLM310_BIN_DIR}:${PATH}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] run_tag=${RUN_TAG}"
echo "[start] log_file=${LOG_FILE}"
echo "[config] gpu_set=${GPU_SET}"

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

run_eval() {
  local model_dir="$1"
  local label="$2"
  local mode="$3"
  local n_sql="$4"
  local temp="$5"
  local run_time="${RUN_TAG}_${label}_${mode}"
  local out_dir="${OUTPUT_ROOT}/${label}_${mode}"

  mkdir -p "${out_dir}"
  echo "[eval] label=${label} mode=${mode} start=$(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" \
  MODEL_SQL_GENERATE="${model_dir}" \
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
  echo "[eval] label=${label} mode=${mode} done=$(date '+%F %T')"
}

wait_for_gpus
run_eval "${STAGE1_MODEL_DIR}" "stage1" "greedy_search" 1 0.0
run_eval "${STAGE1_MODEL_DIR}" "stage1" "major_voting" 8 0.8
run_eval "${STAGE2_MODEL_DIR}" "stage2" "greedy_search" 1 0.0
run_eval "${STAGE2_MODEL_DIR}" "stage2" "major_voting" 8 0.8

echo "[done] run_tag=${RUN_TAG} end_time=$(date '+%F %T')"