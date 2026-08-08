#!/usr/bin/env bash
set -euo pipefail

# Poll the running Phase1 1.7B training runs; once a run finishes (top-level
# model saved AND its training process exited), launch the BIRD dev eval
# (greedy + major@8 + think-strip) for that model in the background.

EVAL_LAUNCHER="/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/eval/run_bird_dev_eval_1p7b_fullmodel.sh"
LOG_DIR="/data/huwenp/emb/lxy/sd-zero-sql/logs"

# run_dir|label  -- run_dir is the ABSOLUTE output dir (may live on /data1),
# label is used in the eval run tag
# 2026-08-08 retrain: 3-epoch + load_best_model_at_end, all outputs on /data1.
# R5/R6 = sequential two-stage (stage2 top-level is the final best model).
declare -a RUNS=(
  "/data1/huwenp/emb/lxy/sd-zero-sql/outputs/phase1_1p7b_base_1to1_r2retrain_20260808|base"
  "/data1/huwenp/emb/lxy/sd-zero-sql/outputs/phase1_1p7b_sftinit_1to1_r3retrain_20260808|sftinit"
  "/data1/huwenp/emb/lxy/sd-zero-sql/outputs/phase1_1p7b_r4_merged_retrain_20260808|r4"
  "/data1/huwenp/emb/lxy/sd-zero-sql/outputs/phase1_1p7b_sftinit_twostage_r5_20260808|r5"
  "/data/huwenp/emb/lxy/sd-zero-sql/outputs/phase1_1p7b_base_twostage_r6_20260808|r6"
)

now="$(date '+%F %T')"
echo "[auto-eval] ${now} poll start"

for entry in "${RUNS[@]}"; do
  run="${entry%%|*}"
  label="${entry##*|}"
  out="${run}"
  marker="${out}/.eval_started"

  if [[ -f "${marker}" ]]; then
    echo "[auto-eval] ${label}: eval already started (marker present)"
    continue
  fi
  if [[ ! -f "${out}/config.json" || ! -f "${out}/model.safetensors.index.json" || ! -f "${out}/train_metrics.json" ]]; then
    echo "[auto-eval] ${label}: training not finished yet (top-level model missing)"
    continue
  fi
  if pgrep -f "train_srt_stage.py.*${run}" >/dev/null 2>&1; then
    echo "[auto-eval] ${label}: training process still alive, waiting for clean exit"
    continue
  fi

  echo "[auto-eval] ${label}: TRAINING DONE -> launching eval for ${out}"
  touch "${marker}"
  nohup env \
    MODEL="${out}" \
    RUN_TAG="bird_dev_eval_1p7b_${label}_$(date +%Y%m%d_%H%M%S)" \
    bash "${EVAL_LAUNCHER}" >>"${LOG_DIR}/auto_eval_${label}.log" 2>&1 &
  echo "[auto-eval] ${label}: eval pid=$!"
  # Post-training cleanup: the eval reads the TOP-LEVEL model (already saved),
  # so deleting intermediate checkpoint-* dirs now is safe and keeps /data1
  # from filling up (R4/R5/R6 still have to train after this run finishes).
  bash "/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/eval/cleanup_phase1_checkpoints.sh" "${out}" \
    >>"${LOG_DIR}/auto_eval_${label}.log" 2>&1 || \
    echo "[auto-eval] ${label}: checkpoint cleanup failed"
done

echo "[auto-eval] ${now} poll end"
