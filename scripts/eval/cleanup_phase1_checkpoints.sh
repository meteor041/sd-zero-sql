#!/usr/bin/env bash
set -euo pipefail

# Post-training checkpoint cleanup for Phase1 1.7B runs.
#
# Only safe to run AFTER a run's training has fully completed AND its top-level
# model has been written (i.e. the auto_eval trigger conditions are met): the
# top level holds the best model (load_best_model_at_end + save_model), and
# checkpoint-* dirs only hold resume state (weights + 13G optimizer).
#
# Policy (per user): keep the BEST checkpoint + the top-level model, delete all
# other intermediate checkpoint-* dirs. For two-stage runs (R5/R6) this is
# applied to stage1/ (best + top level) and to the top level (stage2 best + top).
#
# Usage: cleanup_phase1_checkpoints.sh <output_dir>
#   --dry-run : print what would be deleted without deleting.

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
RUN_DIR="${1:?usage: cleanup_phase1_checkpoints.sh [--dry-run] <output_dir>}"

cleanup_one_dir() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 0
  local best=""
  local ts
  local b
  for ts in "${dir}"/checkpoint-*/trainer_state.json; do
    [[ -f "${ts}" ]] || continue
    b="$("${PYTHON_BIN:-/home/pkuccadm/anaconda3/bin/python}" -c \
      "import json,sys; print(json.load(open('${ts}')).get('best_model_checkpoint') or '')" 2>/dev/null)" || b=""
    if [[ -n "${b}" ]]; then
      best="${b}"
    fi
  done
  local n_ckpt=0
  local ckpt
  for ckpt in "${dir}"/checkpoint-*; do
    [[ -d "${ckpt}" ]] || continue
    n_ckpt=$((n_ckpt + 1))
    if [[ -n "${best}" && "${ckpt}" != "${best}" ]]; then
      if (( DRY_RUN )); then
        echo "[cleanup:dry-run] ${ckpt} would be deleted"
      else
        echo "[cleanup] deleting ${ckpt}"
        rm -rf "${ckpt}"
      fi
    fi
  done
  if [[ -z "${best}" && "${n_ckpt}" -gt 1 ]]; then
    echo "[cleanup:warning] ${dir}: could not locate best_model_checkpoint (${n_ckpt} checkpoints kept); refusing to delete anything"
  fi
}

echo "[cleanup] run_dir=${RUN_DIR} dry_run=${DRY_RUN} $(date '+%F %T')"
# Two-stage runs keep stage1 under <out>/stage1; clean it first (its top level
# is the Stage-2 init, so keep that too), then the top level (Stage-2 result).
if [[ -d "${RUN_DIR}/stage1" ]]; then
  cleanup_one_dir "${RUN_DIR}/stage1"
fi
cleanup_one_dir "${RUN_DIR}"
echo "[cleanup] done"
