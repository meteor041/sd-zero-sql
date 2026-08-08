#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/data/model/Qwen3-1.7B}"
TRAIN_FILE="${TRAIN_FILE:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_train.jsonl}"
VALID_FILE="${VALID_FILE:-/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_valid.jsonl}"
RUN_TAG="${RUN_TAG:-qwen3_1p7b_bird_train_full_sft_gpu2_3_$(date +%Y%m%d_%H%M%S)}"
MODEL_OUT="${MODEL_OUT:-/dev/shm/${RUN_TAG}}"

GPU_SET="${GPU_SET:-2,3}"
IFS=',' read -r -a GPU_INDICES <<< "${GPU_SET}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_INDICES[@]}}"
MASTER_PORT="${MASTER_PORT:-29617}"
POLL_SECONDS="${POLL_SECONDS:-180}"
FREE_STABILITY_SECONDS="${FREE_STABILITY_SECONDS:-60}"
MAX_MEMORY_USED_MB="${MAX_MEMORY_USED_MB:-100}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-5}"

PYTHON_BIN="${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/pkuccadm/anaconda3/bin/torchrun}"
VLLM310_BIN_DIR="${VLLM310_BIN_DIR:-/home/pkuccadm/anaconda3/envs/vllm310/bin}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM310_BIN_DIR}/python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/home/pkuccadm/huwenp/emb/lxy/csc_sql/bin/process/run_ches_qwen3_eval.sh}"

INPUT_JSON="${INPUT_JSON:-${PROJECT_ROOT}/data/eval/ches_dev_input_seq.json}"
GOLD_PATH="${GOLD_PATH:-/data/huwenp/emb/data/ches/dev.sql}"
DATASET_PATH="${DATASET_PATH:-/data/huwenp/emb/data/ches/dev_databases}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/huwenp/emb/lxy/sd-zero-sql/outputs/bird_dev_eval_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-/data/huwenp/emb/lxy/sd-zero-sql/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"

MAX_LENGTH="${MAX_LENGTH:-16384}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-1000000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
REPORT_TO="${REPORT_TO:-none}"
RUN_MAJOR_AT_8="${RUN_MAJOR_AT_8:-0}"
EVAL_GPU_SET="${EVAL_GPU_SET:-${GPU_SET%%,*}}"
EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

mkdir -p "${MODEL_OUT}" "${OUTPUT_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PATH="${VLLM310_BIN_DIR}:${PATH}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

check_inputs() {
  require_dir "${MODEL_PATH}"
  require_file "${MODEL_PATH}/config.json"
  require_file "${TRAIN_FILE}"
  require_file "${VALID_FILE}"
  require_file "${INPUT_JSON}"
  require_file "${GOLD_PATH}"
  require_dir "${DATASET_PATH}"
  require_file "${EVAL_SCRIPT}"
  require_file "${SCRIPT_DIR}/train_sft.py"
}

gpu_free_once() {
  local line idx used util target
  local -a gpu_lines
  mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  for target in "${GPU_INDICES[@]}"; do
    local found=0
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

wait_for_gpus() {
  local first_ok
  while true; do
    echo "[gpu-check] $(date '+%F %T') target=${GPU_SET}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    if gpu_free_once; then
      first_ok=1
      echo "[gpu-check] target GPUs appear free; confirming ${FREE_STABILITY_SECONDS}s"
      sleep "${FREE_STABILITY_SECONDS}"
      if gpu_free_once; then
        echo "[gpu-check] selected=${GPU_SET}"
        return 0
      fi
      echo "[gpu-check] target GPUs became busy during confirmation"
    else
      first_ok=0
    fi
    echo "[gpu-check] waiting; first_ok=${first_ok}; sleep=${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

run_sft() {
  echo "[sft-start] $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="${GPU_SET}" "${TORCHRUN_BIN}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NUM_PROCESSES}" \
    "${SCRIPT_DIR}/train_sft.py" \
    --model-path "${MODEL_PATH}" \
    --train-file "${TRAIN_FILE}" \
    --valid-file "${VALID_FILE}" \
    --output-dir "${MODEL_OUT}" \
    --max-length "${MAX_LENGTH}" \
    --overlength-policy drop \
    --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
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
    --save-steps "${SAVE_STEPS}" \
    --eval-steps "${EVAL_STEPS}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT}" \
    --save-only-model \
    --seed 42 \
    --full-finetune \
    --gradient-checkpointing \
    --use-liger-kernel \
    --fsdp "full_shard auto_wrap" \
    --fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer \
    --bf16 \
    --report-to "${REPORT_TO}" \
    --run-name "${RUN_TAG}"
  echo "[sft-complete] $(date '+%F %T')"
}

check_model_outputs() {
  require_file "${MODEL_OUT}/config.json"
  require_file "${MODEL_OUT}/tokenizer.json"
  require_file "${MODEL_OUT}/model.safetensors.index.json"
  require_file "${MODEL_OUT}/train_metrics.json"
  require_file "${MODEL_OUT}/data_stats.json"
}

run_raw_eval() {
  local mode_label="$1"
  local eval_mode="$2"
  local n_sql="$3"
  local temperature="$4"
  local run_time="${RUN_TAG}_${mode_label}"
  echo "[eval-start] mode=${mode_label} gpu=${EVAL_GPU_SET} $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU_SET}" \
    MODEL_SQL_GENERATE="${MODEL_OUT}" \
    DATAFILE_PATH="${INPUT_JSON}" \
    GOLD_PATH="${GOLD_PATH}" \
    DATASET_PATH="${DATASET_PATH}" \
    DIFF_JSON_PATH="${INPUT_JSON}" \
    DATASET_NAME=bird \
    DATASET_MODE=dev \
    EVAL_MODE="${eval_mode}" \
    N_SQL_GENERATE="${n_sql}" \
    TEMPERATURE_SQL_GENERATE="${temperature}" \
    TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    NUM_CPUS=16 \
    META_TIME_OUT=30.0 \
    OUTPUT_DIR="${OUTPUT_ROOT}" \
    LOG_DIR="${LOG_DIR}" \
    RUN_TIME="${run_time}" \
    bash "${EVAL_SCRIPT}"
  echo "[eval-complete] mode=${mode_label} $(date '+%F %T')"
}

strip_think_and_score() {
  local mode_label="$1"
  local prefix="$2"
  local run_time="${RUN_TAG}_${mode_label}"
  local raw_json="${OUTPUT_ROOT}/${run_time}/${prefix}_ches_chat_sql_generate.json"
  local stripped_json="${OUTPUT_ROOT}/${run_time}/${prefix}_ches_chat_sql_generate_think_stripped.json"
  local metric_json="${stripped_json%.json}_metric.json"
  require_file "${raw_json}"
  echo "[postprocess-start] mode=${mode_label} raw=${raw_json}"
  PYTHONPATH="${PROJECT_ROOT}/src:/home/pkuccadm/huwenp/emb/lxy/csc_sql/src:${PYTHONPATH:-}" "${VLLM_PYTHON}" - "${raw_json}" "${stripped_json}" "${GOLD_PATH}" "${DATASET_PATH}" <<'PY'
import json
import sys
from pathlib import Path

from cscsql.utils.infer_utils import run_eval_major_vote
from sql_core.sql_normalizer import normalize_sql_output

raw_path = Path(sys.argv[1])
stripped_path = Path(sys.argv[2])
gold_file = sys.argv[3]
db_path = sys.argv[4]


def strip_think(text: str) -> str:
    text = str(text or "").strip()
    lower = text.lower()
    if lower.startswith("<think>"):
        text = text[len("<think>"):].lstrip()
        lower = text.lower()
    end = lower.find("</think>")
    if end != -1:
        text = text[end + len("</think>"):].lstrip()
    return normalize_sql_output(text)

rows = json.loads(raw_path.read_text(encoding="utf-8"))
changed = 0
with_think = 0
for row in rows:
    values = row.get("responses") or row.get("pred_sqls") or []
    stripped = []
    for value in values:
        if str(value).lstrip().lower().startswith("<think>") or "</think>" in str(value).lower():
            with_think += 1
        new_value = strip_think(value)
        if new_value != value:
            changed += 1
        stripped.append(new_value)
    row["responses"] = stripped
    row["pred_sqls"] = stripped

stripped_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
config = {
    "source": str(raw_path),
    "postprocess": "strip_leading_think_tags_then_normalize_sql_output",
    "candidate_count": sum(len(row.get("pred_sqls", [])) for row in rows),
    "candidate_with_think_tag_count": with_think,
    "changed_candidate_count": changed,
    "num_cpus": 128,
    "timeout_seconds": 30,
}
run_eval_major_vote(
    gold_file,
    str(stripped_path),
    db_path,
    num_cpus=128,
    timeout=30,
    pred_sql_key="pred_sqls",
    config=config,
)
PY
  require_file "${metric_json}"
  echo "[postprocess-complete] mode=${mode_label} metric=${metric_json} $(date '+%F %T')"
}

echo "[start] run_tag=${RUN_TAG} time=$(date '+%F %T')"
echo "[config] model_path=${MODEL_PATH}"
echo "[config] train_file=${TRAIN_FILE}"
echo "[config] valid_file=${VALID_FILE}"
echo "[config] model_out=${MODEL_OUT}"
echo "[config] gpu_set=${GPU_SET} num_processes=${NUM_PROCESSES} master_port=${MASTER_PORT}"
echo "[config] output_root=${OUTPUT_ROOT}"
echo "[config] eval_gpu_set=${EVAL_GPU_SET} eval_tp=${EVAL_TENSOR_PARALLEL_SIZE} run_major_at_8=${RUN_MAJOR_AT_8}"

check_inputs
wait_for_gpus
run_sft
check_model_outputs
run_raw_eval greedy greedy_search 1 0.0
strip_think_and_score greedy greedy_search
if [[ "${RUN_MAJOR_AT_8}" == "1" ]]; then
  run_raw_eval major_at_8 major_voting 8 0.8
  strip_think_and_score major_at_8 sampling
fi

echo "[done] run_tag=${RUN_TAG} time=$(date '+%F %T')"
