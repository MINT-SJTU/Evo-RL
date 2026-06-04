#!/bin/bash
set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-evo-rl-zyx}"
CONDA_SH="${CONDA_SH:-}"

if [[ -z "$CONDA_SH" ]]; then
  for CANDIDATE in \
    "/mnt/data/miniconda3/etc/profile.d/conda.sh" \
    "/mnt/data1/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$CANDIDATE" ]]; then
      CONDA_SH="$CANDIDATE"
      break
    fi
  done
fi

if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  fi
fi

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda.sh not found. Set CONDA_SH or check Miniconda installation."
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
set -u
conda activate "$CONDA_ENV_NAME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export DIST_TIMEOUT_SECONDS="${DIST_TIMEOUT_SECONDS:-7200}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

DATASET_0="/mnt/data3/dataset/lerobot_socks3000_16K_sync_recap1_1100left_v4"
DATASET_1="/mnt/data3/dataset/lerobot_socks3000_16k_sync_recap1_900right_v4"
DATASET_2="/mnt/data3/dataset/lerobot_socks3000_checked"
DATASET_ROOT="${DATASET_ROOT:-${DATASET_0},${DATASET_1},${DATASET_2}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/socks3000_multidataset}"
POLICY_REPO_ID="${POLICY_REPO_ID:-local/pi05_socks3000_8gpu}"
POLICY_PRETRAINED_PATH="${POLICY_PRETRAINED_PATH:-lerobot/pi05_base}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
OPTIMIZER_LR="${OPTIMIZER_LR:-5e-5}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-5e-6}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-50000}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
WEIGHTED_BC_ENABLE="${WEIGHTED_BC_ENABLE:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"
RUN_NAME="${RUN_NAME:-pi05_socks3000_8gpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/data3/Evo-RL-outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$OUTPUT_BASE/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"

if [[ -n "${HF_CACHE_DIR:-}" ]]; then
  export HF_CACHE_DIR
  export HF_HOME="$HF_CACHE_DIR"
  export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
  mkdir -p "$HF_CACHE_DIR"
fi

mkdir -p "$LOG_DIR"

echo "[INFO] run_name: $RUN_NAME"
echo "[INFO] output_dir: $OUTPUT_DIR"
echo "[INFO] log_file: $LOG_FILE"
echo "[INFO] conda_env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "[INFO] python: $(command -v python)"
echo "[INFO] accelerate: $(command -v accelerate)"
echo "[INFO] dist_timeout_seconds: $DIST_TIMEOUT_SECONDS"
echo "[INFO] main_process_port: $MAIN_PROCESS_PORT"
echo "[INFO] cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
echo "[INFO] dataset.root: $DATASET_ROOT"
echo "[INFO] dataset.repo_id: $DATASET_REPO_ID"
echo "[INFO] num_processes: $NUM_PROCESSES"
echo "[INFO] global_batch_size: $((BATCH_SIZE * NUM_PROCESSES))"
echo "[INFO] optimizer_lr: $OPTIMIZER_LR"
echo "[INFO] scheduler_decay_lr: $SCHEDULER_DECAY_LR"
echo "[INFO] scheduler_decay_steps: $SCHEDULER_DECAY_STEPS"
echo "[INFO] weighted_bc_enable: $WEIGHTED_BC_ENABLE"

accelerate launch \
  --multi_gpu \
  --num_machines 1 \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  -m lerobot.scripts.lerobot_train \
  --policy.type=pi05 \
  --policy.dtype=bfloat16 \
  --policy.pretrained_path="$POLICY_PRETRAINED_PATH" \
  --policy.repo_id="$POLICY_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --steps="$TRAIN_STEPS" \
  --batch_size="$BATCH_SIZE" \
  --log_freq="$LOG_FREQ" \
  --save_freq="$SAVE_FREQ" \
  --policy.optimizer_lr="$OPTIMIZER_LR" \
  --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR" \
  --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS" \
  --policy.gradient_checkpointing="$GRADIENT_CHECKPOINTING" \
  --policy.push_to_hub="$POLICY_PUSH_TO_HUB" \
  --weighted_bc.enable="$WEIGHTED_BC_ENABLE" \
  --wandb.enable="$WANDB_ENABLE" \
  --wandb.disable_artifact="$WANDB_DISABLE_ARTIFACT" \
  --job_name="$RUN_NAME" \
  --output_dir="$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"
