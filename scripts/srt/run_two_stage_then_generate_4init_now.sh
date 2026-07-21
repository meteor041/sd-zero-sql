#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/data/model/Qwen3-4B-Instruct-2507}"
PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/home/pkuccadm/anaconda3/envs/vllm310/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${PYTHON_BIN} -m accelerate.commands.launch}"
GPU_SET="${GPU_SET:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
RUN_TAG="${RUN_TAG:-two_stage_then_4init_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"

STAGE1_TRAIN_FILE="${STAGE1_TRAIN_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/ches_qwen3_4b_srt_full_1init_feedbackfix_stage1k4_stage1.jsonl}"
STAGE2_TRAIN_FILE="${STAGE2_TRAIN_FILE:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/ches_qwen3_4b_srt_full_1init_feedbackfix_stage1k4_stage2.jsonl}"
STAGE1_OUT="${STAGE1_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_stage1_base_1init_stage1k4_gpu0to3}"
STAGE2_OUT="${STAGE2_OUT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_stage2_from_stage1_1init_stage1k4_gpu0to3}"

TRACE_INPUT_JSONL="${TRACE_INPUT_JSONL:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl}"
TRACE_OUTPUT_JSONL="${TRACE_OUTPUT_JSONL:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_4init_from_stage2_gpu0to3.jsonl}"
TRACE_SUMMARY_JSON="${TRACE_SUMMARY_JSON:-/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_4init_from_stage2_gpu0to3_summary.json}"

mkdir -p "${LOG_DIR}" "${STAGE1_OUT}" "${STAGE2_OUT}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] run_tag=${RUN_TAG}"
echo "[start] log_file=${LOG_FILE}"
echo "[config] gpu_set=${GPU_SET}"

echo "[stage1] launch_time=$(date '+%F %T')"
CUDA_VISIBLE_DEVICES="${GPU_SET}" ${ACCELERATE_BIN} \
  --num_processes 4 \
  --mixed_precision bf16 \
  /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/srt/train_srt_stage.py \
  --model-path "${BASE_MODEL}" \
  --adapter-path "" \
  --train-file "${STAGE1_TRAIN_FILE}" \
  --valid-file "${STAGE1_TRAIN_FILE}" \
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

echo "[stage2] launch_time=$(date '+%F %T')"
CUDA_VISIBLE_DEVICES="${GPU_SET}" ${ACCELERATE_BIN} \
  --num_processes 4 \
  --mixed_precision bf16 \
  /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/srt/train_srt_stage.py \
  --model-path "${BASE_MODEL}" \
  --adapter-path "${STAGE1_OUT}" \
  --train-file "${STAGE2_TRAIN_FILE}" \
  --valid-file "${STAGE2_TRAIN_FILE}" \
  --output-dir "${STAGE2_OUT}" \
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

echo "[stage2] completed_time=$(date '+%F %T')"

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
echo "[done] run_tag=${RUN_TAG} end_time=$(date '+%F %T')"