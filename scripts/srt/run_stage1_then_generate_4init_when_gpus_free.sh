#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/data/model/Qwen3-4B-Instruct-2507}"
TRAIN_STAGE1_FILE="${TRAIN_STAGE1_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/ches_qwen3_4b_srt_full_1init_feedbackfix_stage1k4_stage1.jsonl}"
VALID_STAGE1_FILE="${VALID_STAGE1_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/ches_qwen3_4b_srt_full_1init_feedbackfix_stage1k4_stage1.jsonl}"
STAGE1_OUT="${STAGE1_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_stage1_base_1init_stage1k4}"

TRACE_INPUT_JSONL="${TRACE_INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl}"
TRACE_OUTPUT_JSONL="${TRACE_OUTPUT_JSONL:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_4init_from_stage1_base.jsonl}"
TRACE_SUMMARY_JSON="${TRACE_SUMMARY_JSON:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_4init_from_stage1_base_summary.json}"

PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/home/pkuccadm/anaconda3/envs/vllm310/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
GPU_SET="${GPU_SET:-4,5,6,7}"
GPU_INDICES=(4 5 6 7)
POLL_SECONDS="${POLL_SECONDS:-180}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-stage1_then_4init_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"

mkdir -p "${LOG_DIR}" "${STAGE1_OUT}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] run_tag=${RUN_TAG}"
echo "[start] log_file=${LOG_FILE}"
echo "[config] gpu_set=${GPU_SET}"
echo "[config] stage1_out=${STAGE1_OUT}"
echo "[config] trace_output_jsonl=${TRACE_OUTPUT_JSONL}"

wait_for_gpus() {
  while true; do
    mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    APPS="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
    READY=1
    for idx in "${GPU_INDICES[@]}"; do
      line="$(printf '%s
' "${GPU_LINES[@]}" | grep -E "^${idx},")"
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

    echo "[gpu-check] $(date '+%F %T')"
    printf '%s
' "${GPU_LINES[@]}" | grep -E '^(4|5|6|7),'
    if [[ -n "${APPS}" ]]; then
      echo "[gpu-apps]"
      printf '%s
' "${APPS}"
    fi

    if (( READY == 1 )); then
      echo "[gpu-check] target GPUs appear free"
      return 0
    fi

    echo "[gpu-check] GPUs busy, sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

run_stage1_training() {
  echo "[stage1] launch_time=$(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" ${ACCELERATE_BIN} \
    --num_processes 4 \
    --mixed_precision bf16 \
    /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/srt/train_srt_stage.py \
    --model-path "${BASE_MODEL}" \
    --adapter-path "" \
    --train-file "${TRAIN_STAGE1_FILE}" \
    --valid-file "${VALID_STAGE1_FILE}" \
    --output-dir "${STAGE1_OUT}" \
    --max-length 4096 \
    --num-train-epochs 2 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --logging-steps 10 \
    --save-steps 200 \
    --eval-steps 200 \
    --save-total-limit 3 \
    --seed 42 \
    --gradient-checkpointing \
    --bf16 \
    --report-to none
  echo "[stage1] completed_time=$(date '+%F %T')"
}

run_generate_4init() {
  echo "[trace4] launch_time=$(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" "${VLLM_PYTHON_BIN}" \
    /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/srt/generate_phase1_traces.py \
    --model-path "${BASE_MODEL}" \
    --input-jsonl "${TRACE_INPUT_JSONL}" \
    --output-jsonl "${TRACE_OUTPUT_JSONL}" \
    --summary-json "${TRACE_SUMMARY_JSON}" \
    --sampling-mode stratified \
    --min-per-db 5 \
    --seed 42 \
    --backend vllm \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --batch-size 8 \
    --max-new-tokens 256 \
    --temperature 0.8 \
    --num-inits 4 \
    --sample-chunk-size 8 \
    --verifier-workers 16
  echo "[trace4] completed_time=$(date '+%F %T')"
}

wait_for_gpus
run_stage1_training
wait_for_gpus
run_generate_4init

echo "[done] run_tag=${RUN_TAG} end_time=$(date '+%F %T')"