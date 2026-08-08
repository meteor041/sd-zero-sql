#!/usr/bin/env bash
set -euo pipefail

# BIRD dev evaluation for a Qwen3-1.7B full-finetune model (Phase1 SRT output).
# Runs greedy (n=1) and major@8 (n=8), then extracts the REVISED SQL (the block
# after the p_r phrase; Phase1 models emit "init + p_r + revised" in one
# response) and rescore with the canonical major-vote metric.
# Waits for a couple of free GPUs; TP=1 for the small model.

MODEL="${MODEL:?MODEL must point to the trained full model dir}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/home/pkuccadm/huwenp/emb/lxy/csc_sql/bin/process/run_ches_qwen3_eval.sh}"
DATA_ROOT="${DATA_ROOT:-/data/huwenp/emb/data/ches}"
DATAFILE_PATH="${DATAFILE_PATH:-${DATA_ROOT}/dev.json}"
GOLD_PATH="${GOLD_PATH:-${DATA_ROOT}/dev.sql}"
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/dev_databases}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/bird_dev_eval_1p7b_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-bird_dev_eval_1p7b_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/pkuccadm/anaconda3/envs/vllm310/bin/python}"

POLL_SECONDS="${POLL_SECONDS:-180}"
FREE_STABILITY_SECONDS="${FREE_STABILITY_SECONDS:-60}"
RETRY_SECONDS="${RETRY_SECONDS:-180}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-100}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
REQUIRED_GPUS="${REQUIRED_GPUS:-2}"

for path in "${EVAL_SCRIPT}" "${DATAFILE_PATH}" "${GOLD_PATH}" "${MODEL}/config.json" "${MODEL}/model.safetensors.index.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Missing BIRD database directory: ${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
export PATH="/home/pkuccadm/anaconda3/envs/vllm310/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] run_tag=${RUN_TAG} time=$(date '+%F %T')"
echo "[config] model=${MODEL}"
echo "[config] dataset=${DATAFILE_PATH} examples=1534"
echo "[config] modes=greedy(n=1,temp=0.0),major_at_8(n=8,temp=0.8) + revised-SQL extract"
echo "[config] output_root=${OUTPUT_ROOT}"

select_free_gpus() {
  local line idx used util target
  local -a gpu_lines free
  mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  free=()
  for line in "${gpu_lines[@]}"; do
    IFS=',' read -r idx used util <<< "${line}"
    idx="${idx// /}"
    used="${used// /}"
    util="${util// /}"
    if (( used <= MAX_MEMORY_USED_MB && util <= MAX_GPU_UTILIZATION )); then
      free+=("${idx}")
    fi
  done
  if (( ${#free[@]} < REQUIRED_GPUS )); then
    return 1
  fi
  CHOSEN_GPUS=("${free[@]:0:${REQUIRED_GPUS}}")
  GPU_SET="$(IFS=,; printf '%s' "${CHOSEN_GPUS[*]}")"
  return 0
}

wait_for_gpus() {
  local label="$1"
  while true; do
    echo "[gpu-check] label=${label} time=$(date '+%F %T')"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    if select_free_gpus; then
      local first_selection="${GPU_SET}"
      echo "[gpu-check] label=${label} candidate=${first_selection}; confirming ${FREE_STABILITY_SECONDS}s"
      sleep "${FREE_STABILITY_SECONDS}"
      if select_free_gpus && [[ "${GPU_SET}" == "${first_selection}" ]]; then
        echo "[gpu-check] label=${label} selected=${GPU_SET}"
        return 0
      fi
    fi
    echo "[gpu-check] label=${label} fewer than ${REQUIRED_GPUS} stable GPUs; sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

run_raw_eval() {
  local mode_label="$1"
  local eval_mode="$2"
  local n_sql="$3"
  local temperature="$4"
  local run_time="${RUN_TAG}_${mode_label}"
  echo "[eval-start] mode=${mode_label} gpu=${GPU_SET} $(date '+%F %T')"
  if CUDA_VISIBLE_DEVICES="${GPU_SET}" \
    MODEL_SQL_GENERATE="${MODEL}" \
    DATA_ROOT="${DATA_ROOT}" \
    DATAFILE_PATH="${DATAFILE_PATH}" \
    GOLD_PATH="${GOLD_PATH}" \
    DATASET_PATH="${DATASET_PATH}" \
    DIFF_JSON_PATH="${DATAFILE_PATH}" \
    DATASET_NAME=bird \
    DATASET_MODE=dev \
    EVAL_MODE="${eval_mode}" \
    N_SQL_GENERATE="${n_sql}" \
    TEMPERATURE_SQL_GENERATE="${temperature}" \
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    NUM_CPUS=16 \
    META_TIME_OUT=30.0 \
    OUTPUT_DIR="${OUTPUT_ROOT}" \
    LOG_DIR="${LOG_DIR}" \
    RUN_TIME="${run_time}" \
    bash "${EVAL_SCRIPT}"; then
    echo "[eval-complete] mode=${mode_label} $(date '+%F %T')"
    return 0
  fi
  echo "[retry] mode=${mode_label} failed; sleeping ${RETRY_SECONDS}s"
  sleep "${RETRY_SECONDS}"
  return 1
}

extract_revised_and_score() {
  # Phase1-trained models emit "init SQL + p_r phrase + revised SQL" in one
  # response. The final answer is the revised SQL (after the p_r phrase), NOT
  # the whole blob. Rescore from the raw vLLM output, extracting the revised SQL.
  local mode_label="$1"
  local prefix="$2"
  local run_time="${RUN_TAG}_${mode_label}"
  local raw_json="${OUTPUT_ROOT}/${run_time}/${prefix}_ches_chat_sql_generate.json"
  local revised_json="${OUTPUT_ROOT}/${run_time}/${prefix}_ches_chat_sql_generate_revised_only.json"
  local metric_json="${revised_json%.json}_metric.json"
  if [[ ! -f "${raw_json}" ]]; then
    echo "[postprocess-skip] mode=${mode_label} raw missing: ${raw_json}"
    return 1
  fi
  echo "[postprocess-start] mode=${mode_label} raw=${raw_json}"
  PYTHONPATH="${PROJECT_ROOT}/src:/home/pkuccadm/huwenp/emb/lxy/csc_sql/src:${PYTHONPATH:-}" \
    "${VLLM_PYTHON}" "${PROJECT_ROOT}/scripts/eval/extract_revised_and_score.py" \
    "${raw_json}" "${revised_json}" "${GOLD_PATH}" "${DATASET_PATH}"
  if [[ -f "${metric_json}" ]]; then
    echo "[postprocess-complete] mode=${mode_label} metric=${metric_json}"
    return 0
  fi
  echo "[postprocess-failed] mode=${mode_label} metric missing: ${metric_json}"
  return 1
}

# greedy
while true; do
  wait_for_gpus greedy
  if run_raw_eval greedy greedy_search 1 0.0; then break; fi
done
extract_revised_and_score greedy greedy_search || true

# major@8
while true; do
  wait_for_gpus major_at_8
  if run_raw_eval major_at_8 major_voting 8 0.8; then break; fi
done
extract_revised_and_score major_at_8 sampling || true

echo "[done] run_tag=${RUN_TAG} model=${MODEL} time=$(date '+%F %T')"
