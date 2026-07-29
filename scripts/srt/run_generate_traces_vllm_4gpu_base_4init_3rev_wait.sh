#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-4B-Instruct-2507}"
INPUT_JSONL="${INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_base_4init_3rev_temp08.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_base_4init_3rev_temp08_summary.json}"

GPU_SET="${GPU_SET:-4,5,6,7}"
IFS=',' read -r -a GPU_INDICES <<< "${GPU_SET}"
POLL_SECONDS="${POLL_SECONDS:-180}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-trace_base_4init_3rev_temp08_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/home/pkuccadm/anaconda3/envs/vllm310/bin/python}"

mkdir -p "${LOG_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] run_tag=${RUN_TAG}"
echo "[start] log_file=${LOG_FILE}"
echo "[config] gpu_set=${GPU_SET}"
echo "[config] output_jsonl=${OUTPUT_JSONL}"

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

wait_for_gpus

CUDA_VISIBLE_DEVICES="${GPU_SET}" "${VLLM_PYTHON_BIN}" \
  /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/srt/generate_phase1_traces.py \
  --model-path "${MODEL_PATH}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --sampling-mode stratified \
  --min-per-db 5 \
  --seed 42 \
  --backend vllm \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --batch-size 8 \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 1.0 \
  --num-inits 4 \
  --num-revisions 3 \
  --verifier-workers 16

echo "[done] $(date '+%F %T')"