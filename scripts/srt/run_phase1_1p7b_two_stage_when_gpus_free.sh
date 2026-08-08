#!/usr/bin/env bash
set -euo pipefail

# Phase1 SRT SEQUENTIAL TWO-STAGE training on Qwen3-1.7B.
#
# Stage 1: generation task   (x -> y_init + p_r + y_revised),      best-at-end
# Stage 2: revision task     (x + y_init + p_r -> y_revised),      init = Stage1 best
#
# The two stages share ONE GPU allocation (wait once, then train both back-to-back),
# so a run does not release cards between stages. Data must come from the SAME
# joint multitask trace split used by R2/R3 (split by task into stage1/stage2).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INIT_MODEL="${INIT_MODEL:-/data/model/Qwen3-1.7B}"
RUN_TAG="${RUN_TAG:-phase1_1p7b_twostage_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/phase1_qwen3_1p7b_8init3rev_20260806/twostage}"
STAGE1_TRAIN="${STAGE1_TRAIN:-${DATA_DIR}/stage1_train.jsonl}"
STAGE1_VALID="${STAGE1_VALID:-${DATA_DIR}/stage1_valid.jsonl}"
STAGE2_TRAIN="${STAGE2_TRAIN:-${DATA_DIR}/stage2_train.jsonl}"
STAGE2_VALID="${STAGE2_VALID:-${DATA_DIR}/stage2_valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/huwenp/emb/lxy/sd-zero-sql/outputs/${RUN_TAG}}"
STAGE1_OUT="${STAGE1_OUT:-${OUTPUT_DIR}/stage1}"
RESUME_STAGE1="${RESUME_STAGE1:-}"
RESUME_STAGE2="${RESUME_STAGE2:-}"

# Candidate card groups: whitespace-separated list of comma-separated card sets.
# First group with GPU_COUNT stable free cards wins. R5/R6 are fixed to distinct
# preferred groups to avoid the concurrent-launcher race; launch them staggered.
GPU_GROUPS="${GPU_GROUPS:-4,5,6,7 0,1,2,3}"
GPU_COUNT="${GPU_COUNT:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29513}"
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

for path in "${INIT_MODEL}/config.json" "${STAGE1_TRAIN}" "${STAGE1_VALID}" "${STAGE2_TRAIN}" "${STAGE2_VALID}"; do
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

mkdir -p "${OUTPUT_DIR}" "${STAGE1_OUT}" "${LOG_DIR}"
# Keep temp/compile caches off the root partition (which has repeatedly run
# out of space and killed training via /tmp exhaustion). SHORT path on /data1:
# multiprocessing AF_UNIX sockets break when TMPDIR exceeds ~100 chars.
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

train_stage() {
  local label="$1"
  local model_init="$2"
  local train_file="$3"
  local valid_file="$4"
  local out_dir="$5"
  local resume="$6"
  echo "[stage-start] ${label} model_init=${model_init} train=$(wc -l < "${train_file}") rows"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" \
  ${ACCELERATE_BIN} \
    --num_processes "${GPU_COUNT}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision bf16 \
    "${SCRIPT_DIR}/train_srt_stage.py" \
    --model-path "${model_init}" \
    --train-file "${train_file}" \
    --valid-file "${valid_file}" \
    --output-dir "${out_dir}" \
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
    ${resume:+--resume-from-checkpoint "${resume}"} \
    --seed 42 \
    --full-finetune \
    --gradient-checkpointing \
    --use-liger-kernel \
    --fsdp "full_shard auto_wrap" \
    --fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer \
    --bf16 \
    --report-to "${REPORT_TO}" \
    --run-name "${WANDB_RUN_NAME}_${label}"
  echo "[stage-done] ${label} out=${out_dir}"
}

echo "[start] run_tag=${RUN_TAG} time=$(date '+%F %T')"
echo "[config] init_model=${INIT_MODEL}"
echo "[config] stage1_train=${STAGE1_TRAIN} rows=$(wc -l < "${STAGE1_TRAIN}")"
echo "[config] stage2_train=${STAGE2_TRAIN} rows=$(wc -l < "${STAGE2_TRAIN}")"
echo "[config] output_dir=${OUTPUT_DIR} (stage1 in ${STAGE1_OUT})"
echo "[config] max_length=16384 overlength=drop global_batch=${GPU_COUNT} main_process_port=${MAIN_PROCESS_PORT}"
wait_for_resources

train_stage "stage1" "${INIT_MODEL}" "${STAGE1_TRAIN}" "${STAGE1_VALID}" "${STAGE1_OUT}" "${RESUME_STAGE1}"
train_stage "stage2" "${STAGE1_OUT}" "${STAGE2_TRAIN}" "${STAGE2_VALID}" "${OUTPUT_DIR}" "${RESUME_STAGE2}"

echo "[done] two-stage run tag=${RUN_TAG} final_model=${OUTPUT_DIR} end_time=$(date '+%F %T')"
