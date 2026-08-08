#!/usr/bin/env bash
set -euo pipefail

# Phase1 SRT joint generation/revision training on Qwen3-1.7B.
# Initializes from a configurable model (BASE or a prior SFT full-finetune),
# waits for GPU_PREF cards to be free, then trains with the canonical Phase1 config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INIT_MODEL="${INIT_MODEL:-/data/model/Qwen3-1.7B}"
RUN_TAG="${RUN_TAG:-phase1_1p7b_$(date +%Y%m%d_%H%M%S)}"
TRAIN_FILE="${TRAIN_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/phase1_qwen3_1p7b_8init3rev_20260806/ches_qwen3_1p7b_srt_train.jsonl}"
VALID_FILE="${VALID_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/phase1_qwen3_1p7b_8init3rev_20260806/ches_qwen3_1p7b_srt_valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/${RUN_TAG}}"
# Resume from an existing checkpoint (e.g. OUTPUT_DIR/checkpoint-NNNN) after a crash.
RESUME_FROM="${RESUME_FROM:-}"

# Candidate card groups for THIS run: whitespace-separated list of comma-separated
# card sets. The first group with GPU_COUNT stable free cards wins, so the run waits
# smartly instead of blocking on one fixed set. GPU_PREF is kept for a single set.
GPU_PREF="${GPU_PREF:-}"
GPU_GROUPS="${GPU_GROUPS:-4,5,6,7 0,1,2,3}"
if [[ -n "${GPU_PREF}" ]]; then
  GPU_GROUPS="${GPU_PREF}"
fi
GPU_COUNT="${GPU_COUNT:-4}"
# Distributed main-process port. R3 runs on the accelerate default (29500), so
# concurrent Phase1 runs must pick a different port to avoid EADDRINUSE.
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29510}"
POLL_SECONDS="${POLL_SECONDS:-180}"
FREE_STABILITY_SECONDS="${FREE_STABILITY_SECONDS:-60}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-100}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-5}"

PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-sd-zero-sql}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-online}"

for path in "${INIT_MODEL}/config.json" "${TRAIN_FILE}" "${VALID_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
done
if ! "${PYTHON_BIN}" -c "import liger_kernel" >/dev/null 2>&1; then
  echo "liger-kernel is required for the Phase1 run." >&2
  exit 1
fi
if [[ "${REPORT_TO}" == "wandb" ]] && ! "${PYTHON_BIN}" -c "import wandb" >/dev/null 2>&1; then
  echo "wandb is required when REPORT_TO=wandb." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
# Keep temp/compile caches off the root partition (which has repeatedly run
# out of space and killed training via /tmp exhaustion). Use a SHORT path on
# /data1: multiprocessing AF_UNIX sockets break when TMPDIR exceeds ~100 chars.
CACHE_ROOT="${CACHE_ROOT:-/data1/huwenp/tmp}"
mkdir -p "${CACHE_ROOT}"
export TMPDIR="${CACHE_ROOT}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
exec > >(tee -a "${LOG_FILE}") 2>&1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT WANDB_RUN_NAME WANDB_MODE
export WANDB_WATCH="${WANDB_WATCH:-false}"
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"

IFS=' ' read -r -a GPU_GROUP_LIST <<< "${GPU_GROUPS}"

select_free_gpus() {
  local group line idx used util
  local -a selected=()
  local -a targets=()
  local found
  mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  for group in "${GPU_GROUP_LIST[@]}"; do
    selected=()
    IFS=',' read -r -a targets <<< "${group}"
    for target in "${targets[@]}"; do
      found=0
      for line in "${gpu_lines[@]}"; do
        IFS=',' read -r idx used util <<< "${line}"
        idx="${idx// /}"
        used="${used// /}"
        util="${util// /}"
        if [[ "${idx}" == "${target}" ]]; then
          found=1
          if (( used <= MAX_MEMORY_USED_MB && util <= MAX_GPU_UTILIZATION )); then
            selected+=("${idx}")
          fi
          break
        fi
      done
      if (( found == 0 )); then
        break
      fi
    done
    if (( ${#selected[@]} == GPU_COUNT )); then
      GPU_SET="$(IFS=,; printf '%s' "${selected[*]}")"
      return 0
    fi
  done
  return 1
}

wait_for_resources() {
  local candidate
  while true; do
    echo "[resource-check] $(date '+%F %T') groups=${GPU_GROUPS}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    if select_free_gpus; then
      candidate="${GPU_SET}"
      echo "[resource-check] candidate=${candidate}; confirming for ${FREE_STABILITY_SECONDS}s"
      sleep "${FREE_STABILITY_SECONDS}"
      if select_free_gpus && [[ "${GPU_SET}" == "${candidate}" ]]; then
        echo "[resource-check] selected=${GPU_SET}"
        return 0
      fi
    fi
    echo "[resource-check] waiting for ${GPU_COUNT} stable free cards from groups [${GPU_GROUPS}]; sleep=${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

echo "[start] run_tag=${RUN_TAG} time=$(date '+%F %T')"
echo "[config] init_model=${INIT_MODEL}"
echo "[config] train_file=${TRAIN_FILE} rows=$(wc -l < "${TRAIN_FILE}")"
echo "[config] valid_file=${VALID_FILE} rows=$(wc -l < "${VALID_FILE}")"
echo "[config] output_dir=${OUTPUT_DIR}"
echo "[config] max_length=16384 overlength=drop global_batch=${GPU_COUNT} main_process_port=${MAIN_PROCESS_PORT}"
wait_for_resources

CUDA_VISIBLE_DEVICES="${GPU_SET}" \
${ACCELERATE_BIN} \
  --num_processes "${GPU_COUNT}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision bf16 \
  "${SCRIPT_DIR}/train_srt_stage.py" \
  --model-path "${INIT_MODEL}" \
  --train-file "${TRAIN_FILE}" \
  --valid-file "${VALID_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length 16384 \
  --overlength-policy drop \
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
  --load-best-model-at-end \
  --metric-for-best-model eval_loss \
  ${RESUME_FROM:+--resume-from-checkpoint "${RESUME_FROM}"} \
  --seed 42 \
  --full-finetune \
  --gradient-checkpointing \
  --use-liger-kernel \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer \
  --bf16 \
  --report-to "${REPORT_TO}" \
  --run-name "${WANDB_RUN_NAME}"

echo "[done] Phase1 run tag=${RUN_TAG} model=${OUTPUT_DIR} end_time=$(date '+%F %T')"
