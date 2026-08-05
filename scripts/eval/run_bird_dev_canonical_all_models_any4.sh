#!/usr/bin/env bash
set -euo pipefail

EVAL_SCRIPT="${EVAL_SCRIPT:-/home/pkuccadm/huwenp/emb/lxy/csc_sql/bin/process/run_ches_qwen3_eval.sh}"
PHASE1_EVAL_SCRIPT="${PHASE1_EVAL_SCRIPT:-/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/eval/evaluate_phase1_revisions.py}"
CONFIG_RECORD="${CONFIG_RECORD:-/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/configs/bird_dev_canonical_all_models_20260804.json}"
INPUT_JSON="${INPUT_JSON:-/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/data/eval/ches_dev_input_seq.json}"
GOLD_PATH="${GOLD_PATH:-/data/huwenp/emb/data/ches/dev.sql}"
DATASET_PATH="${DATASET_PATH:-/data/huwenp/emb/data/ches/dev_databases}"
BASE_MODEL="${BASE_MODEL:-/data/model/Qwen3-4B-Instruct-2507}"
SFT_MODEL="${SFT_MODEL:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_phase1_dual_api_2init3rev_correctsql/full_model_gpu0to3}"
PHASE1_MODEL="${PHASE1_MODEL:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_phase1_dual_api_2init3rev_base_lora_gpu0to3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/bird_dev_eval_canonical_20260804}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-bird_dev_eval_canonical_any4_20260804}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
VLLM310_BIN_DIR="${VLLM310_BIN_DIR:-/home/pkuccadm/anaconda3/envs/vllm310/bin}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM310_BIN_DIR}/python}"
POLL_SECONDS="${POLL_SECONDS:-180}"
FREE_STABILITY_SECONDS="${FREE_STABILITY_SECONDS:-60}"
RETRY_SECONDS="${RETRY_SECONDS:-180}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-100}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ALL_GPUS=(0 1 2 3 4 5 6 7)
REQUIRED_GPUS=4

for path in "${EVAL_SCRIPT}" "${PHASE1_EVAL_SCRIPT}" "${CONFIG_RECORD}" "${INPUT_JSON}" "${GOLD_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "Missing BIRD database directory: ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -f "${BASE_MODEL}/config.json" || ! -f "${SFT_MODEL}/config.json" ]]; then
  echo "Missing standalone base or SFT model." >&2
  exit 1
fi
if [[ ! -f "${PHASE1_MODEL}/adapter_config.json" ]]; then
  echo "Missing Phase1 PEFT adapter: ${PHASE1_MODEL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
cp "${CONFIG_RECORD}" "${OUTPUT_ROOT}/evaluation_config.json"
export PATH="${VLLM310_BIN_DIR}:${PATH}"
export TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] run_tag=${RUN_TAG} time=$(date '+%F %T')"
echo "[config] input=${INPUT_JSON} examples=1534"
echo "[config] runs=base,sft_correct_sql,phase1_base_lora modes=greedy,major_at_8"
echo "[config] seed=42 tp=4 greedy_n=1 greedy_temp=0.0 major_n=8 major_temp=0.8"
echo "[config] phase1_postprocess=final_revision_sql"
echo "[config] gpu_policy=any_four stability_seconds=${FREE_STABILITY_SECONDS} retry_seconds=${RETRY_SECONDS}"

select_free_gpus() {
  local line idx used util target
  local -a gpu_lines free
  mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  free=()
  for target in "${ALL_GPUS[@]}"; do
    for line in "${gpu_lines[@]}"; do
      IFS=',' read -r idx used util <<< "${line}"
      idx="${idx// /}"
      used="${used// /}"
      util="${util// /}"
      if [[ "${idx}" == "${target}" ]]; then
        if (( used <= MAX_MEMORY_USED_MB && util <= MAX_GPU_UTILIZATION )); then
          free+=("${idx}")
        fi
        break
      fi
    done
  done
  if (( ${#free[@]} < REQUIRED_GPUS )); then
    return 1
  fi
  CHOSEN_GPUS=("${free[@]:0:${REQUIRED_GPUS}}")
  GPU_SET="$(IFS=,; printf '%s' "${CHOSEN_GPUS[*]}")"
  return 0
}

wait_for_any_four() {
  local label="$1"
  local first_selection
  while true; do
    echo "[gpu-check] label=${label} time=$(date '+%F %T')"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    if select_free_gpus; then
      first_selection="${GPU_SET}"
      echo "[gpu-check] label=${label} candidate=${first_selection}; confirming for ${FREE_STABILITY_SECONDS}s"
      sleep "${FREE_STABILITY_SECONDS}"
      if select_free_gpus && [[ "${GPU_SET}" == "${first_selection}" ]]; then
        echo "[gpu-check] label=${label} selected=${GPU_SET} after stability check"
        return 0
      fi
      echo "[gpu-check] label=${label} candidate changed or became busy"
    fi
    echo "[gpu-check] label=${label} fewer than four stable GPUs free; sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

run_eval() {
  local model_path="$1"
  local model_label="$2"
  local mode_label="$3"
  local eval_mode="$4"
  local n_sql="$5"
  local temperature="$6"
  local phase1_postprocess="$7"
  local run_time="${RUN_TAG}_${model_label}_${mode_label}"
  local prefix="sampling"
  if [[ "${eval_mode}" == "greedy_search" ]]; then
    prefix="greedy_search"
  fi
  local output_json="${OUTPUT_ROOT}/${run_time}/${prefix}_ches_chat_sql_generate.json"
  local raw_metric="${OUTPUT_ROOT}/${run_time}/${prefix}_ches_chat_sql_generate_metric.json"
  local final_metric="${raw_metric}"
  if [[ "${phase1_postprocess}" == "true" ]]; then
    final_metric="${OUTPUT_ROOT}/${run_time}/${prefix}_ches_chat_sql_generate_revision_extracted_metric.json"
  fi

  if [[ -s "${final_metric}" ]]; then
    echo "[skip] model=${model_label} mode=${mode_label} completed_metric=${final_metric}"
    return 0
  fi

  while true; do
    wait_for_any_four "${model_label}_${mode_label}"
    echo "[eval] model=${model_label} mode=${mode_label} gpu_set=${GPU_SET} start=$(date '+%F %T')"
    if CUDA_VISIBLE_DEVICES="${GPU_SET}" \
      MODEL_SQL_GENERATE="${model_path}" \
      DATAFILE_PATH="${INPUT_JSON}" \
      GOLD_PATH="${GOLD_PATH}" \
      DATASET_PATH="${DATASET_PATH}" \
      DIFF_JSON_PATH="${INPUT_JSON}" \
      DATASET_NAME=bird \
      DATASET_MODE=dev \
      EVAL_MODE="${eval_mode}" \
      N_SQL_GENERATE="${n_sql}" \
      TEMPERATURE_SQL_GENERATE="${temperature}" \
      TENSOR_PARALLEL_SIZE=4 \
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
      NUM_CPUS=16 \
      META_TIME_OUT=30.0 \
      OUTPUT_DIR="${OUTPUT_ROOT}" \
      LOG_DIR="${LOG_DIR}" \
      RUN_TIME="${run_time}" \
      bash "${EVAL_SCRIPT}" && [[ -s "${raw_metric}" ]]; then
      if [[ "${phase1_postprocess}" == "true" ]]; then
        echo "[postprocess] model=${model_label} mode=${mode_label} extracting final revisions"
        if ! PYTHONPATH="/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/src:/home/pkuccadm/huwenp/emb/lxy/csc_sql/src" \
          "${VLLM_PYTHON}" "${PHASE1_EVAL_SCRIPT}" \
          --input-json "${output_json}" \
          --gold-file "${GOLD_PATH}" \
          --db-path "${DATASET_PATH}" \
          --num-cpus 128 \
          --timeout 30; then
          echo "[retry] model=${model_label} mode=${mode_label} revision evaluation failed; sleeping ${RETRY_SECONDS}s"
          sleep "${RETRY_SECONDS}"
          continue
        fi
      fi
      if [[ -s "${final_metric}" ]]; then
        echo "[eval] model=${model_label} mode=${mode_label} gpu_set=${GPU_SET} done=$(date '+%F %T') metric=${final_metric}"
        return 0
      fi
    fi
    echo "[retry] model=${model_label} mode=${mode_label} failed_or_incomplete; sleeping ${RETRY_SECONDS}s"
    sleep "${RETRY_SECONDS}"
  done
}

run_eval "${BASE_MODEL}" base greedy greedy_search 1 0.0 false
run_eval "${BASE_MODEL}" base major_at_8 major_voting 8 0.8 false
run_eval "${SFT_MODEL}" sft_correct_sql greedy greedy_search 1 0.0 false
run_eval "${SFT_MODEL}" sft_correct_sql major_at_8 major_voting 8 0.8 false
run_eval "${PHASE1_MODEL}" phase1_base_lora greedy greedy_search 1 0.0 true
run_eval "${PHASE1_MODEL}" phase1_base_lora major_at_8 major_voting 8 0.8 true

echo "[done] run_tag=${RUN_TAG} time=$(date '+%F %T')"
