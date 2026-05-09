#!/bin/bash
set -euo pipefail

# User-facing entrypoint for launching h20-1 + h20-0 multi-node training.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
RUN_ALL_SCRIPT="${RUN_ALL_SCRIPT:-$PROJECT_DIR/run_all.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-evo-rl_ljh}"
CONDA_SH="${CONDA_SH:-}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-10.0.112.9}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
WANDB_MODE="${WANDB_MODE:-disabled}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/data1/ljh/.cache/huggingface}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-false}"
LEROBOT_DATASET_LOAD_MODE="${LEROBOT_DATASET_LOAD_MODE:-auto}"
RESUME="${RESUME:-false}"
CLEANUP="${CLEANUP:-0}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
NUM_MACHINES="${NUM_MACHINES:-2}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
DATASET_ROOT="${DATASET_ROOT:-}"
DATASET_REPO_ID="${DATASET_REPO_ID:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
RUN_ID="${RUN_ID:-dual_$(date +%Y%m%d_%H%M%S)}"

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

if [[ ! -f "$RUN_ALL_SCRIPT" ]]; then
  echo "[ERROR] 启动脚本不存在: $RUN_ALL_SCRIPT"
  exit 1
fi

HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
case "$HOSTNAME_SHORT" in
  h20-1|iv-yekfzb6mtc5i3z43xs8h*)
    REMOTE_HOST="${REMOTE_HOST:-h20-0}"
    ;;
  h20-0|iv-yek0jxwet*)
    REMOTE_HOST="${REMOTE_HOST:-h20-1}"
    ;;
  *)
    echo "[ERROR] 当前主机不是 h20-0 或 h20-1: $HOSTNAME_SHORT"
    echo "[ERROR] 请在 h20-0 / h20-1 上运行，或显式设置 REMOTE_HOST"
    exit 1
    ;;
esac

pick_free_port() {
  local remote_host="$1"
  local port
  for port in $(seq 29600 29699); do
    if ! ss -ltnH "( sport = :$port )" | grep -q .; then
      if ssh $SSH_OPTS "$remote_host" "! ss -ltnH '( sport = :$port )' | grep -q ." >/dev/null 2>&1; then
        echo "$port"
        return 0
      fi
    fi
  done
  return 1
}

if [[ -z "$MAIN_PROCESS_PORT" ]]; then
  MAIN_PROCESS_PORT="$(pick_free_port "$REMOTE_HOST")" || {
    echo "[ERROR] 无法在本机和 $REMOTE_HOST 上找到共同空闲端口"
    exit 1
  }
fi

echo "[INFO] host: $HOSTNAME_SHORT"
echo "[INFO] conda_env: ${CONDA_DEFAULT_ENV}"
echo "[INFO] remote_host: $REMOTE_HOST"
echo "[INFO] main_process: $MAIN_PROCESS_IP:$MAIN_PROCESS_PORT"
echo "[INFO] run_id: $RUN_ID"
echo "[INFO] resume: $RESUME"
echo "[INFO] wandb_enable: $WANDB_ENABLE"
echo "[INFO] wandb_disable_artifact: $WANDB_DISABLE_ARTIFACT"
echo "[INFO] hf_cache_dir: $HF_CACHE_DIR"
echo "[INFO] hf_local_files_only: $HF_LOCAL_FILES_ONLY"
echo "[INFO] lerobot_dataset_load_mode: $LEROBOT_DATASET_LOAD_MODE"
echo "[INFO] cleanup: $CLEANUP"
echo "[INFO] save_freq: $SAVE_FREQ"
echo "[INFO] num_machines: $NUM_MACHINES"
echo "[INFO] num_processes: $NUM_PROCESSES"
if [[ -n "$DATASET_ROOT" ]]; then
  echo "[INFO] dataset_root: $DATASET_ROOT"
fi
if [[ -n "$DATASET_REPO_ID" ]]; then
  echo "[INFO] dataset_repo_id: $DATASET_REPO_ID"
fi
if [[ -n "$POLICY_REPO_ID" ]]; then
  echo "[INFO] policy_repo_id: $POLICY_REPO_ID"
fi

cd "$PROJECT_DIR"
exec env \
  RUN_ID="$RUN_ID" \
  RESUME="$RESUME" \
  CLEANUP="$CLEANUP" \
  SAVE_FREQ="$SAVE_FREQ" \
  NUM_MACHINES="$NUM_MACHINES" \
  NUM_PROCESSES="$NUM_PROCESSES" \
  DATASET_ROOT="$DATASET_ROOT" \
  DATASET_REPO_ID="$DATASET_REPO_ID" \
  POLICY_REPO_ID="$POLICY_REPO_ID" \
  WANDB_ENABLE="$WANDB_ENABLE" \
  WANDB_DISABLE_ARTIFACT="$WANDB_DISABLE_ARTIFACT" \
  WANDB_MODE="$WANDB_MODE" \
  HF_CACHE_DIR="$HF_CACHE_DIR" \
  HF_LOCAL_FILES_ONLY="$HF_LOCAL_FILES_ONLY" \
  LEROBOT_DATASET_LOAD_MODE="$LEROBOT_DATASET_LOAD_MODE" \
  MAIN_PROCESS_IP="$MAIN_PROCESS_IP" \
  MAIN_PROCESS_PORT="$MAIN_PROCESS_PORT" \
  REMOTE_HOST="$REMOTE_HOST" \
  bash "$RUN_ALL_SCRIPT"
