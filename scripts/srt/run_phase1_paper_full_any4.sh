#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_MODEL="/data/model/Qwen3-4B-Instruct-2507"
TRAIN_FILE="/data/huwenp/emb/lxy/sd-zero-sql/data/srt/phase1_dual_api_full_2init3rev_20260801_090031/derived/phase1_dual_api_2init3rev_srt_train.jsonl"
VALID_FILE="/data/huwenp/emb/lxy/sd-zero-sql/data/srt/phase1_dual_api_full_2init3rev_20260801_090031/derived/phase1_dual_api_2init3rev_srt_valid.jsonl"
TRAIN_SHA256="eec0ed9128a1e91bb3c128fdabd21e4b4a81852fd39c748a37e3221f1c687828"
VALID_SHA256="2f54bed684a2941c116b382fffb601da710632faed75dc5355ac7923b493d14c"
TRAIN_ROWS=7078
VALID_ROWS=358
OUTPUT_DIR="${OUTPUT_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_phase1_dual_api_2init3rev_base_paper_full_32k}"
PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
NUM_PROCESSES=4
GPU_COUNT=4
POLL_SECONDS="${POLL_SECONDS:-180}"
FREE_STABILITY_SECONDS="${FREE_STABILITY_SECONDS:-60}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-100}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-5}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-120}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-phase1_base_paper_full_32k_2init3rev_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-sd-zero-sql}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-online}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

for path in "${BASE_MODEL}/config.json" "${TRAIN_FILE}" "${VALID_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done
if [[ "$(sha256sum "${TRAIN_FILE}" | cut -d' ' -f1)" != "${TRAIN_SHA256}" ]]; then
  echo "Training JSONL hash differs from the pinned Phase1 dataset." >&2
  exit 1
fi
if [[ "$(sha256sum "${VALID_FILE}" | cut -d' ' -f1)" != "${VALID_SHA256}" ]]; then
  echo "Validation JSONL hash differs from the pinned Phase1 dataset." >&2
  exit 1
fi
if [[ "$(wc -l < "${TRAIN_FILE}")" -ne "${TRAIN_ROWS}" || "$(wc -l < "${VALID_FILE}")" -ne "${VALID_ROWS}" ]]; then
  echo "Phase1 dataset row counts differ from the pinned inputs." >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c "import liger_kernel" >/dev/null 2>&1; then
  echo "liger-kernel is required for the paper-aligned run." >&2
  exit 1
fi
if [[ "${REPORT_TO}" == "wandb" ]] && ! "${PYTHON_BIN}" -c "import wandb" >/dev/null 2>&1; then
  echo "wandb is required when REPORT_TO=wandb." >&2
  exit 1
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_CHECKPOINT}" != "${OUTPUT_DIR}"/checkpoint-* ]]; then
  echo "Resume checkpoint must belong to this Phase1 full-run output directory." >&2
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

select_free_gpus() {
  local line idx used util
  local -a selected=()
  mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  for line in "${gpu_lines[@]}"; do
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

has_enough_disk() {
  local available_bytes required_bytes
  available_bytes="$(df -B1 --output=avail "${OUTPUT_DIR}" | tr -dc '0-9')"
  required_bytes=$((MIN_FREE_DISK_GB * 1024 * 1024 * 1024))
  (( available_bytes >= required_bytes ))
}

wait_for_resources() {
  local candidate
  while true; do
    echo "[resource-check] $(date '+%F %T')"
    df -h "${OUTPUT_DIR}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    if has_enough_disk && select_free_gpus; then
      candidate="${GPU_SET}"
      echo "[resource-check] candidate=${candidate}; confirming for ${FREE_STABILITY_SECONDS}s"
      sleep "${FREE_STABILITY_SECONDS}"
      if has_enough_disk && select_free_gpus && [[ "${GPU_SET}" == "${candidate}" ]]; then
        echo "[resource-check] selected=${GPU_SET} disk_threshold_gb=${MIN_FREE_DISK_GB}"
        return 0
      fi
    fi
    echo "[resource-check] waiting for four stable GPUs and ${MIN_FREE_DISK_GB}GB free disk; sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

echo "[start] run_tag=${RUN_TAG}"
echo "[config] initialization=full_parameter_on_qwen3_4b_instruct_2507"
echo "[config] prompts_and_jsonl=unchanged train_rows=${TRAIN_ROWS} valid_rows=${VALID_ROWS}"
echo "[config] max_length=32768 overlength_policy=error global_batch=4"
echo "[config] output_dir=${OUTPUT_DIR} log_file=${LOG_FILE}"
wait_for_resources

CUDA_VISIBLE_DEVICES="${GPU_SET}" \
${ACCELERATE_BIN} \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train_srt_stage.py" \
  --model-path "${BASE_MODEL}" \
  --train-file "${TRAIN_FILE}" \
  --valid-file "${VALID_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length 32768 \
  --overlength-policy error \
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
  --save-total-limit 1 \
  --seed 42 \
  --full-finetune \
  --gradient-checkpointing \
  --use-liger-kernel \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer \
  --bf16 \
  --report-to "${REPORT_TO}" \
  --run-name "${WANDB_RUN_NAME}" \
  "${RESUME_ARGS[@]}"

echo "[done] Phase1 paper-aligned full model=${OUTPUT_DIR} end_time=$(date '+%F %T')"
