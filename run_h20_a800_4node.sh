#!/bin/bash
set -euo pipefail

# One-click launcher for h20-1 + h20-0 + cluster_0 + cluster_3.
# This script intentionally does not change the existing dual-node launch path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-$PROJECT_DIR/run_sft_4node_32gpu.sh}"
LOG_DIR="${LOG_DIR:-/mnt/data1/ljh/Evo-RL-logs}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-10.0.112.9}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-}"
RUN_ID="${RUN_ID:-4node32gpu_$(date +%Y%m%d_%H%M%S)}"
NODE_HOSTS_CSV="${NODE_HOSTS_CSV:-h20-1,h20-0,cluster_0,cluster_3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-evo-rl_ljh}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
RESUME="${RESUME:-false}"
CLEANUP="${CLEANUP:-0}"
SYNC_PROJECT="${SYNC_PROJECT:-1}"
SYNC_DATASET="${SYNC_DATASET:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-6}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data1/ljh/dataset/lerobot_towel_mid300+left_right_50}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/towel_mid300+left_right_50}"
POLICY_COMPILE="${POLICY_COMPILE:-false}"
POLICY_COMPILE_MODE="${POLICY_COMPILE_MODE:-reduce-overhead}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"

IFS=',' read -r -a NODE_HOSTS <<< "$NODE_HOSTS_CSV"
NUM_MACHINES="${#NODE_HOSTS[@]}"
NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_MACHINES * 8))}"

if [[ "$NUM_MACHINES" -ne 4 ]]; then
  echo "[ERROR] NODE_HOSTS_CSV 需要 4 台机器，当前是 $NUM_MACHINES: $NODE_HOSTS_CSV"
  exit 1
fi

if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
  echo "[ERROR] launch script not found: $LAUNCH_SCRIPT"
  exit 1
fi

mkdir -p "$LOG_DIR"

ssh_host() {
  local host="$1"
  local cmd="$2"
  if [[ "$host" == "h20-1" || "$host" == "$(hostname -s 2>/dev/null || hostname)" ]]; then
    bash -lc "$cmd"
  else
    ssh -n $SSH_OPTS "$host" "$cmd"
  fi
}

rsync_to_host() {
  local host="$1"
  local src="$2"
  local dst="$3"
  if [[ "$host" == "h20-1" || "$host" == "$(hostname -s 2>/dev/null || hostname)" ]]; then
    mkdir -p "$dst"
    rsync -az "$src" "$dst"
  else
    ssh -n $SSH_OPTS "$host" "mkdir -p '$dst'"
    rsync -e "ssh $SSH_OPTS" -az "$src" "$host:$dst"
  fi
}

pick_free_port() {
  local port host
  for port in $(seq 29600 29699); do
    local ok=1
    for host in "${NODE_HOSTS[@]}"; do
      if ! ssh_host "$host" "! ss -ltnH '( sport = :$port )' | grep -q ." >/dev/null 2>&1; then
        ok=0
        break
      fi
    done
    if [[ "$ok" == "1" ]]; then
      echo "$port"
      return 0
    fi
  done
  return 1
}

if [[ -z "$MAIN_PROCESS_PORT" ]]; then
  MAIN_PROCESS_PORT="$(pick_free_port)" || {
    echo "[ERROR] 无法在四台机器上找到共同空闲端口"
    exit 1
  }
fi

echo "[INFO] project_dir: $PROJECT_DIR"
echo "[INFO] launch_script: $LAUNCH_SCRIPT"
echo "[INFO] hosts: $NODE_HOSTS_CSV"
echo "[INFO] num_machines: $NUM_MACHINES"
echo "[INFO] num_processes: $NUM_PROCESSES"
echo "[INFO] global_batch_size: $((BATCH_SIZE * NUM_PROCESSES))"
echo "[INFO] main_process: $MAIN_PROCESS_IP:$MAIN_PROCESS_PORT"
echo "[INFO] run_id: $RUN_ID"
echo "[INFO] cleanup: $CLEANUP"
echo "[INFO] sync_project: $SYNC_PROJECT"
echo "[INFO] sync_dataset: $SYNC_DATASET"
echo "[INFO] wandb_enable: $WANDB_ENABLE"
echo "[INFO] wandb_disable_artifact: $WANDB_DISABLE_ARTIFACT"

echo "[STEP] preflight hosts"
for idx in "${!NODE_HOSTS[@]}"; do
  host="${NODE_HOSTS[$idx]}"
  echo "--- rank $idx / $host ---"
  ssh_host "$host" "hostname; nvidia-smi --query-gpu=name,index,memory.total --format=csv,noheader | head -8; test -f /mnt/data/miniconda3/etc/profile.d/conda.sh -o -f /mnt/data1/miniconda3/etc/profile.d/conda.sh && echo CONDA_OK || echo CONDA_MISSING"
done

if [[ "$SYNC_PROJECT" == "1" ]]; then
  echo "[STEP] sync project to remote hosts"
  for host in "${NODE_HOSTS[@]}"; do
    [[ "$host" == "h20-1" ]] && continue
    rsync_to_host "$host" "$PROJECT_DIR/" "$PROJECT_DIR/"
  done
else
  echo "[STEP] skip project sync"
fi

if [[ "$SYNC_DATASET" == "1" ]]; then
  DATASET_PARENT="$(dirname "$DATASET_ROOT")"
  echo "[STEP] sync dataset to remote hosts: $DATASET_ROOT"
  for host in "${NODE_HOSTS[@]}"; do
    [[ "$host" == "h20-1" ]] && continue
    rsync_to_host "$host" "$DATASET_ROOT/" "$DATASET_ROOT/"
  done
else
  echo "[STEP] skip dataset sync"
fi

echo "[STEP] validate project and dataset paths"
for host in "${NODE_HOSTS[@]}"; do
  ssh_host "$host" "test -f '$PROJECT_DIR/$(basename "$LAUNCH_SCRIPT")' && echo PROJECT_OK:$host || { echo PROJECT_MISSING:$host; exit 1; }"
  ssh_host "$host" "test -d '$DATASET_ROOT' && echo DATASET_OK:$host || { echo DATASET_MISSING:$host:$DATASET_ROOT; exit 1; }"
done

if [[ "$CLEANUP" == "1" ]]; then
  echo "[STEP] cleanup existing training processes on all hosts"
  for host in "${NODE_HOSTS[@]}"; do
    ssh_host "$host" "pkill -f 'run_sft_4node_32gpu.sh|run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run|python -' || true" || true
  done
fi

OPTIONAL_DISTRIBUTED_ENV=()
for name in NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_SOCKET_IFNAME NCCL_IB_HCA GLOO_SOCKET_IFNAME; do
  if [[ -n "${!name:-}" ]]; then
    OPTIONAL_DISTRIBUTED_ENV+=("$name=${!name}")
  fi
done
OPTIONAL_DISTRIBUTED_ENV_STR=""
if [[ ${#OPTIONAL_DISTRIBUTED_ENV[@]} -gt 0 ]]; then
  printf -v OPTIONAL_DISTRIBUTED_ENV_STR " %q" "${OPTIONAL_DISTRIBUTED_ENV[@]}"
fi

LAUNCH_NAME="$(basename "$LAUNCH_SCRIPT")"
echo "[STEP] start all ranks"
for idx in "${!NODE_HOSTS[@]}"; do
  host="${NODE_HOSTS[$idx]}"
  label="${host//[^a-zA-Z0-9._-]/_}"
  log="$LOG_DIR/run_sft_${label}_${RUN_ID}.log"
  echo "[INFO] rank $idx host $host log $log"
  ssh_host "$host" "mkdir -p '$LOG_DIR' && cd '$PROJECT_DIR' && ( nohup bash -lc \"cd '$PROJECT_DIR' && exec env RUN_ID='$RUN_ID' RESUME='$RESUME' NODE_RANK='$idx' MAIN_PROCESS_IP='$MAIN_PROCESS_IP' MAIN_PROCESS_PORT='$MAIN_PROCESS_PORT' NUM_MACHINES='$NUM_MACHINES' NUM_PROCESSES='$NUM_PROCESSES' WANDB_MODE='$WANDB_MODE' WANDB_ENABLE='$WANDB_ENABLE' WANDB_DISABLE_ARTIFACT='$WANDB_DISABLE_ARTIFACT' TRAIN_STEPS='$TRAIN_STEPS' BATCH_SIZE='$BATCH_SIZE' LOG_FREQ='$LOG_FREQ' SAVE_FREQ='$SAVE_FREQ' OPTIMIZER_LR='$OPTIMIZER_LR' SCHEDULER_DECAY_LR='$SCHEDULER_DECAY_LR' DATASET_ROOT='$DATASET_ROOT' DATASET_REPO_ID='$DATASET_REPO_ID' POLICY_COMPILE='$POLICY_COMPILE' POLICY_COMPILE_MODE='$POLICY_COMPILE_MODE' GRADIENT_CHECKPOINTING='$GRADIENT_CHECKPOINTING' POLICY_PUSH_TO_HUB='$POLICY_PUSH_TO_HUB'$OPTIONAL_DISTRIBUTED_ENV_STR bash '$LAUNCH_NAME'\" > '$log' 2>&1 < /dev/null & ) ; disown -a >/dev/null 2>&1 || true"
done

echo "[STEP] process snapshot"
sleep 5
for idx in "${!NODE_HOSTS[@]}"; do
  host="${NODE_HOSTS[$idx]}"
  echo "--- rank $idx / $host ---"
  ssh_host "$host" "pgrep -af 'run_sft_4node_32gpu.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run|python -' || true"
done

echo "[DONE] 4-node launch command submitted."
echo "[INFO] Logs are under: $LOG_DIR/run_sft_<host>_${RUN_ID}.log"
