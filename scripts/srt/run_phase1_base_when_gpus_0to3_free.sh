#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="${BASE_MODEL:-/data/model/Qwen3-4B-Instruct-2507}"
SFT_OUTPUT="${SFT_OUTPUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_phase1_dual_api_2init3rev_correctsql/full_model_gpu0to3}"
SFT_COMPLETION_FILE="${SFT_COMPLETION_FILE:-${SFT_OUTPUT}/train_metrics.json}"
TRAIN_FILE="${TRAIN_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/phase1_dual_api_full_2init3rev_20260801_090031/derived/phase1_dual_api_2init3rev_srt_train.jsonl}"
VALID_FILE="${VALID_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/phase1_dual_api_full_2init3rev_20260801_090031/derived/phase1_dual_api_2init3rev_srt_valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_phase1_dual_api_2init3rev_base_lora_gpu0to3}"
GPU_SET="0,1,2,3"
GPU_INDICES=(0 1 2 3)
NUM_PROCESSES=4
POLL_SECONDS="${POLL_SECONDS:-180}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-100}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-5}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-phase1_base_2init3rev_gpu0to3_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-sd-zero-sql}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ ! -f "${BASE_MODEL}/config.json" ]]; then
  echo "Missing base model: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_FILE}" || ! -f "${VALID_FILE}" ]]; then
  echo "Missing Phase1 train or validation data." >&2
  exit 1
fi
if [[ "${BASE_MODEL}" == "${SFT_OUTPUT}" ]]; then
  echo "Phase1 BASE_MODEL must not be the SFT output." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT WANDB_RUN_NAME WANDB_MODE
export WANDB_WATCH="${WANDB_WATCH:-false}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"

echo "[start] run_tag=${RUN_TAG}"
echo "[config] gpu_set=${GPU_SET}"
echo "[config] base_model=${BASE_MODEL}"
echo "[config] initialization=fresh_lora_on_base_model"
echo "[config] sft_completion_gate=${SFT_COMPLETION_FILE}"
echo "[config] train_file=${TRAIN_FILE}"
echo "[config] valid_file=${VALID_FILE}"
echo "[config] output_dir=${OUTPUT_DIR}"
echo "[config] max_length=${MAX_LENGTH} overlength_policy=drop"
echo "[config] log_file=${LOG_FILE}"

gpus_are_free() {
  local line idx used util target found
  local -a gpu_lines
  mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  for target in "${GPU_INDICES[@]}"; do
    found=0
    for line in "${gpu_lines[@]}"; do
      IFS=',' read -r idx used util <<< "${line}"
      idx="${idx// /}"
      used="${used// /}"
      util="${util// /}"
      if [[ "${idx}" == "${target}" ]]; then
        found=1
        if (( used > MAX_MEMORY_USED_MB || util > MAX_GPU_UTILIZATION )); then
          return 1
        fi
        break
      fi
    done
    if (( found == 0 )); then
      return 1
    fi
  done
  return 0
}

while true; do
  echo "[poll] $(date '+%F %T')"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | grep -E '^(0|1|2|3),'
  if [[ ! -f "${SFT_COMPLETION_FILE}" ]]; then
    echo "[poll] SFT completion marker not found; waiting ${POLL_SECONDS}s"
  elif ! gpus_are_free; then
    echo "[poll] SFT completed but GPUs 0-3 are not all free; waiting ${POLL_SECONDS}s"
  else
    echo "[poll] SFT completed and GPUs 0-3 are free"
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[phase1] starting $(date '+%F %T')"
CUDA_VISIBLE_DEVICES="${GPU_SET}" \
${ACCELERATE_BIN} \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train_srt_stage.py" \
  --model-path "${BASE_MODEL}" \
  --train-file "${TRAIN_FILE}" \
  --valid-file "${VALID_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --overlength-policy drop \
  --num-train-epochs 3 \
  --learning-rate 2e-5 \
  --weight-decay 1e-4 \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 10 \
  --save-steps 200 \
  --eval-steps 200 \
  --save-total-limit 1 \
  --seed 42 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --bf16 \
  --report-to "${REPORT_TO}"

echo "[done] Phase1 adapter=${OUTPUT_DIR} end_time=$(date '+%F %T')"
