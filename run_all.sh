#!/bin/bash
set -euo pipefail

# One-click launcher for dual-node training on h20-1 (rank0) + h20-0 (rank1).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$PROJECT_DIR/run_sft.sh}"
LOG_DIR="${LOG_DIR:-/mnt/data1/ljh/Evo-RL-logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-false}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-10.0.112.9}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29600}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
HF_PREWARM="${HF_PREWARM:-auto}"
HF_PREFER_OFFLINE="${HF_PREFER_OFFLINE:-auto}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/data1/ljh/.cache/huggingface}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-false}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-6}"
DATASET_ROOT="${DATASET_ROOT:-}"
DATASET_REPO_ID="${DATASET_REPO_ID:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
POLICY_COMPILE="${POLICY_COMPILE:-false}"
POLICY_COMPILE_MODE="${POLICY_COMPILE_MODE:-reduce-overhead}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new}"
CLEANUP="${CLEANUP:-0}"
LOCAL_RANK="${LOCAL_RANK:-}"
REMOTE_HOST="${REMOTE_HOST:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-evo-rl_ljh}"
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
  echo "[ERROR] 找不到 conda.sh，请设置 CONDA_SH 或检查 Miniconda 安装路径"
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
set -u
conda activate "$CONDA_ENV_NAME" >/dev/null 2>&1 || {
  echo "[ERROR] 无法激活 conda 环境: $CONDA_ENV_NAME"
  exit 1
}

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
  echo "[ERROR] conda 环境未生效，当前环境: ${CONDA_DEFAULT_ENV:-<none>}"
  exit 1
fi

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ -z "$LOCAL_RANK" ]]; then
  case "$HOSTNAME_SHORT" in
    h20-1|iv-yekfzb6mtc5i3z43xs8h*)
      LOCAL_RANK=0
      DEFAULT_REMOTE_HOST="h20-0"
      ;;
    h20-0|iv-yek0jxwet*)
      LOCAL_RANK=1
      DEFAULT_REMOTE_HOST="h20-1"
      ;;
    *)
      echo "[ERROR] Unknown host '$HOSTNAME_SHORT'. Set LOCAL_RANK and REMOTE_HOST explicitly."
      exit 1
      ;;
  esac
fi

if [[ -z "$REMOTE_HOST" ]]; then
  if [[ -n "${DEFAULT_REMOTE_HOST:-}" ]]; then
    REMOTE_HOST="$DEFAULT_REMOTE_HOST"
  else
    if [[ "$LOCAL_RANK" == "0" ]]; then
      REMOTE_HOST="h20-0"
    else
      REMOTE_HOST="h20-1"
    fi
  fi
fi

if [[ "$LOCAL_RANK" != "0" && "$LOCAL_RANK" != "1" ]]; then
  echo "[ERROR] LOCAL_RANK must be 0 or 1, got: $LOCAL_RANK"
  exit 1
fi

REMOTE_RANK=1
if [[ "$LOCAL_RANK" == "1" ]]; then
  REMOTE_RANK=0
fi

if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
  echo "[ERROR] launch script not found: $LAUNCH_SCRIPT"
  exit 1
fi

mkdir -p "$LOG_DIR"

ssh_remote() {
  local cmd="$1"
  ssh -n $SSH_OPTS "$REMOTE_HOST" "$cmd"
}

echo "[INFO] host: $HOSTNAME_SHORT"
echo "[INFO] conda_env: ${CONDA_DEFAULT_ENV}"
echo "[INFO] python: $(command -v python)"
echo "[INFO] project_dir: $PROJECT_DIR"
echo "[INFO] launch_script: $LAUNCH_SCRIPT"
echo "[INFO] run_id: $RUN_ID"
echo "[INFO] local_rank: $LOCAL_RANK"
echo "[INFO] remote_host: $REMOTE_HOST"
echo "[INFO] remote_rank: $REMOTE_RANK"
echo "[INFO] main_process: $MAIN_PROCESS_IP:$MAIN_PROCESS_PORT"
echo "[INFO] resume: $RESUME"
echo "[INFO] wandb_mode: $WANDB_MODE"
echo "[INFO] wandb_enable: $WANDB_ENABLE"
echo "[INFO] wandb_disable_artifact: $WANDB_DISABLE_ARTIFACT"
echo "[INFO] train_steps: $TRAIN_STEPS"
echo "[INFO] batch_size: $BATCH_SIZE"
echo "[INFO] log_freq: $LOG_FREQ"
echo "[INFO] save_freq: $SAVE_FREQ"
echo "[INFO] optimizer_lr: $OPTIMIZER_LR"
echo "[INFO] scheduler_decay_lr: $SCHEDULER_DECAY_LR"
if [[ -n "$DATASET_ROOT" ]]; then
  echo "[INFO] dataset_root: $DATASET_ROOT"
fi
if [[ -n "$DATASET_REPO_ID" ]]; then
  echo "[INFO] dataset_repo_id: $DATASET_REPO_ID"
fi
if [[ -n "$POLICY_REPO_ID" ]]; then
  echo "[INFO] policy_repo_id: $POLICY_REPO_ID"
fi
echo "[INFO] policy_compile: $POLICY_COMPILE"
echo "[INFO] policy_compile_mode: $POLICY_COMPILE_MODE"
echo "[INFO] gradient_checkpointing: $GRADIENT_CHECKPOINTING"
echo "[INFO] policy_push_to_hub: $POLICY_PUSH_TO_HUB"
echo "[INFO] hf_prewarm: $HF_PREWARM"
echo "[INFO] hf_prefer_offline: $HF_PREFER_OFFLINE"
echo "[INFO] hf_cache_dir: $HF_CACHE_DIR"
echo "[INFO] hf_local_files_only: $HF_LOCAL_FILES_ONLY"
echo "[INFO] nccl_debug: ${NCCL_DEBUG:-<unset>}"
echo "[INFO] nccl_debug_subsys: ${NCCL_DEBUG_SUBSYS:-<unset>}"
echo "[INFO] nccl_socket_ifname: ${NCCL_SOCKET_IFNAME:-<unset>}"
echo "[INFO] nccl_ib_hca: ${NCCL_IB_HCA:-<unset>}"
echo "[INFO] gloo_socket_ifname: ${GLOO_SOCKET_IFNAME:-<unset>}"

ssh_remote "echo [INFO] remote_host_ok: \$(hostname -s)" >/dev/null

if [[ "$CLEANUP" == "1" ]]; then
  echo "[STEP] cleanup existing training processes (local + remote)"
  pkill -f 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run|python -' || true
  ssh_remote "pkill -f 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run|python -' || true" || true
fi

echo "[STEP] sync launch script to remote"
rsync -e "ssh $SSH_OPTS" -az "$LAUNCH_SCRIPT" "$REMOTE_HOST:$PROJECT_DIR/"

LOCAL_HOST_LABEL="h20-1"
if [[ "$LOCAL_RANK" == "1" ]]; then
  LOCAL_HOST_LABEL="h20-0"
fi
REMOTE_HOST_LABEL="h20-0"
if [[ "$REMOTE_RANK" == "0" ]]; then
  REMOTE_HOST_LABEL="h20-1"
fi

LOCAL_LOG="$LOG_DIR/run_sft_${LOCAL_HOST_LABEL}_${RUN_ID}.log"
REMOTE_LOG="$LOG_DIR/run_sft_${REMOTE_HOST_LABEL}_${RUN_ID}.log"
LAUNCH_NAME="$(basename "$LAUNCH_SCRIPT")"

echo "[STEP] start local rank=$LOCAL_RANK"
OPTIONAL_DISTRIBUTED_ENV=()
if [[ -n "${NCCL_DEBUG:-}" ]]; then
  OPTIONAL_DISTRIBUTED_ENV+=(NCCL_DEBUG="$NCCL_DEBUG")
fi
if [[ -n "${NCCL_DEBUG_SUBSYS:-}" ]]; then
  OPTIONAL_DISTRIBUTED_ENV+=(NCCL_DEBUG_SUBSYS="$NCCL_DEBUG_SUBSYS")
fi
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  OPTIONAL_DISTRIBUTED_ENV+=(NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME")
fi
if [[ -n "${NCCL_IB_HCA:-}" ]]; then
  OPTIONAL_DISTRIBUTED_ENV+=(NCCL_IB_HCA="$NCCL_IB_HCA")
fi
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
  OPTIONAL_DISTRIBUTED_ENV+=(GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME")
fi
OPTIONAL_DISTRIBUTED_ENV_STR=""
if [[ ${#OPTIONAL_DISTRIBUTED_ENV[@]} -gt 0 ]]; then
  printf -v OPTIONAL_DISTRIBUTED_ENV_STR " %q" "${OPTIONAL_DISTRIBUTED_ENV[@]}"
fi

nohup bash -lc "cd '$PROJECT_DIR' && exec env RUN_ID='$RUN_ID' RESUME='$RESUME' NODE_RANK='$LOCAL_RANK' MAIN_PROCESS_IP='$MAIN_PROCESS_IP' MAIN_PROCESS_PORT='$MAIN_PROCESS_PORT' WANDB_MODE='$WANDB_MODE' WANDB_ENABLE='$WANDB_ENABLE' WANDB_DISABLE_ARTIFACT='$WANDB_DISABLE_ARTIFACT' HF_PREWARM='$HF_PREWARM' HF_PREFER_OFFLINE='$HF_PREFER_OFFLINE' HF_CACHE_DIR='$HF_CACHE_DIR' HF_LOCAL_FILES_ONLY='$HF_LOCAL_FILES_ONLY' TRAIN_STEPS='$TRAIN_STEPS' BATCH_SIZE='$BATCH_SIZE' LOG_FREQ='$LOG_FREQ' SAVE_FREQ='$SAVE_FREQ' OPTIMIZER_LR='$OPTIMIZER_LR' SCHEDULER_DECAY_LR='$SCHEDULER_DECAY_LR' DATASET_ROOT='$DATASET_ROOT' DATASET_REPO_ID='$DATASET_REPO_ID' POLICY_REPO_ID='$POLICY_REPO_ID' POLICY_COMPILE='$POLICY_COMPILE' POLICY_COMPILE_MODE='$POLICY_COMPILE_MODE' GRADIENT_CHECKPOINTING='$GRADIENT_CHECKPOINTING' POLICY_PUSH_TO_HUB='$POLICY_PUSH_TO_HUB'$OPTIONAL_DISTRIBUTED_ENV_STR bash '$LAUNCH_NAME'" > "$LOCAL_LOG" 2>&1 < /dev/null &
LOCAL_PID=$!

echo "[STEP] start remote rank=$REMOTE_RANK"
set +e
REMOTE_START_OUT="$(ssh_remote "cd '$PROJECT_DIR' && mkdir -p '$LOG_DIR' && ( nohup bash -lc \"cd '$PROJECT_DIR' && exec env RUN_ID='$RUN_ID' RESUME='$RESUME' NODE_RANK='$REMOTE_RANK' MAIN_PROCESS_IP='$MAIN_PROCESS_IP' MAIN_PROCESS_PORT='$MAIN_PROCESS_PORT' WANDB_MODE='$WANDB_MODE' WANDB_ENABLE='$WANDB_ENABLE' WANDB_DISABLE_ARTIFACT='$WANDB_DISABLE_ARTIFACT' HF_PREWARM='$HF_PREWARM' HF_PREFER_OFFLINE='$HF_PREFER_OFFLINE' HF_CACHE_DIR='$HF_CACHE_DIR' HF_LOCAL_FILES_ONLY='$HF_LOCAL_FILES_ONLY' TRAIN_STEPS='$TRAIN_STEPS' BATCH_SIZE='$BATCH_SIZE' LOG_FREQ='$LOG_FREQ' SAVE_FREQ='$SAVE_FREQ' OPTIMIZER_LR='$OPTIMIZER_LR' SCHEDULER_DECAY_LR='$SCHEDULER_DECAY_LR' DATASET_ROOT='$DATASET_ROOT' DATASET_REPO_ID='$DATASET_REPO_ID' POLICY_REPO_ID='$POLICY_REPO_ID' POLICY_COMPILE='$POLICY_COMPILE' POLICY_COMPILE_MODE='$POLICY_COMPILE_MODE' GRADIENT_CHECKPOINTING='$GRADIENT_CHECKPOINTING' POLICY_PUSH_TO_HUB='$POLICY_PUSH_TO_HUB'$OPTIONAL_DISTRIBUTED_ENV_STR bash '$LAUNCH_NAME'\" > '$REMOTE_LOG' 2>&1 < /dev/null & ) ; disown -a >/dev/null 2>&1 || true ; echo REMOTE_PID:\$! ; exit 0" 2>&1)"
REMOTE_START_RC=$?
set -e

if [[ -n "$REMOTE_START_OUT" ]]; then
  echo "$REMOTE_START_OUT"
fi

if [[ $REMOTE_START_RC -ne 0 ]]; then
  echo "[WARN] remote launch command returned non-zero: $REMOTE_START_RC"
fi

echo "LOCAL_PID:$LOCAL_PID"
echo "LOCAL_LOG:$LOCAL_LOG"
echo "REMOTE_LOG:$REMOTE_LOG"

LOCAL_FOUND=0
REMOTE_FOUND=0
for _ in $(seq 1 10); do
  if kill -0 "$LOCAL_PID" >/dev/null 2>&1; then
    LOCAL_FOUND=1
  elif pgrep -af 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run|python -' >/dev/null 2>&1; then
    LOCAL_FOUND=1
  fi

  if ssh_remote "pgrep -af 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run|python -' >/dev/null 2>&1"; then
    REMOTE_FOUND=1
  fi

  if [[ $LOCAL_FOUND -eq 1 && $REMOTE_FOUND -eq 1 ]]; then
    break
  fi
  sleep 3
done

if [[ $LOCAL_FOUND -ne 1 ]]; then
  echo "[ERROR] local training process not found for RUN_ID=$RUN_ID"
  exit 1
fi

if [[ $REMOTE_FOUND -ne 1 ]]; then
  echo "[ERROR] remote training process not found for RUN_ID=$RUN_ID"
  exit 1
fi

echo "[STEP] process snapshot"
echo "--- local ---"
pgrep -af 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run|python -' || true
echo "--- remote ---"
ssh_remote "pgrep -af 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run|python -' || true" || true

echo "[DONE] dual-node launch command submitted."
