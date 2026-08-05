#!/usr/bin/env bash
set -euo pipefail

EVAL_SCRIPT="${EVAL_SCRIPT:-/home/pkuccadm/huwenp/emb/lxy/csc_sql/bin/process/run_ches_qwen3_eval.sh}"
ALL_GPUS=(0 1 2 3 4 5 6 7)
REQUIRED_GPUS="${REQUIRED_GPUS:-4}"
POLL_SECONDS="${POLL_SECONDS:-180}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-bird_dev_eval_any4_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"

SRT_MODEL_DIR="${SRT_MODEL_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_from_phase1_4init_correctsql_dbsplit/merged}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/bird_eval_any4}"
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
echo "[config] required_gpus=${REQUIRED_GPUS}"

select_free_gpus() {
  mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  FREE=()
  for idx in "${ALL_GPUS[@]}"; do
    line="$(printf '%s\n' "${GPU_LINES[@]}" | grep -E "^${idx},")"
    used="$(printf '%s' "${line}" | cut -d',' -f2 | tr -d ' ')"
    util="$(printf '%s' "${line}" | cut -d',' -f3 | tr -d ' ')"
    if [[ -z "${used}" || -z "${util}" ]]; then
      continue
    fi
    if (( used <= 100 && util <= 5 )); then
      FREE+=("${idx}")
    fi
  done
  if (( ${#FREE[@]} >= REQUIRED_GPUS )); then
    printf '%s\n' "${FREE[@]:0:${REQUIRED_GPUS}}"
    return 0
  fi
  return 1
}

wait_for_any_four() {
  while true; do
    echo "[gpu-check] $(date '+%F %T')"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    CHOSEN=()
    mapfile -t CHOSEN < <(select_free_gpus || true)
    if (( ${#CHOSEN[@]} >= REQUIRED_GPUS )); then
      GPU_SET="$(IFS=,; echo "${CHOSEN[*]:0:${REQUIRED_GPUS}}");"
      GPU_SET="${GPU_SET%;}"
      export GPU_SET
      export TENSOR_PARALLEL_SIZE="${REQUIRED_GPUS}"
      echo "[gpu-check] selected GPUs=${GPU_SET}"
      return 0
    fi
    echo "[gpu-check] no free 4-GPU set, sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

run_eval() {
  local mode="$1"
  local n_sql="$2"
  local temp="$3"
  local run_time="${RUN_TAG}_${mode}"
  local out_dir="${OUTPUT_ROOT}/${mode}"
  mkdir -p "${out_dir}"
  echo "[eval] mode=${mode} gpu_set=${GPU_SET} start=$(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" \
  MODEL_SQL_GENERATE="${SRT_MODEL_DIR}" \
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

wait_for_any_four
run_eval "greedy_search" 1 0.0
run_eval "major_voting" 8 0.8

echo "[done] run_tag=${RUN_TAG} end_time=$(date '+%F %T')"