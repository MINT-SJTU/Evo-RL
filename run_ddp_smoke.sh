#!/bin/bash
set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-evo-rl_ljh}"
CONDA_SH="${CONDA_SH:-/mnt/data/miniconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda.sh not found: $CONDA_SH"
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
set -u
conda activate "$CONDA_ENV_NAME" >/dev/null 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-10.0.112.9}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29600}"
NUM_MACHINES="${NUM_MACHINES:-2}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
NODE_RANK="${NODE_RANK:?NODE_RANK must be set}"
SMOKE_BACKEND="${SMOKE_BACKEND:-nccl}"
SMOKE_ITERS="${SMOKE_ITERS:-3}"
SMOKE_MB="${SMOKE_MB:-1}"
NPROC_PER_NODE=$((NUM_PROCESSES / NUM_MACHINES))

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

echo "[SMOKE-LAUNCH] host=$(hostname)"
echo "[SMOKE-LAUNCH] node_rank=$NODE_RANK nnodes=$NUM_MACHINES nproc_per_node=$NPROC_PER_NODE"
echo "[SMOKE-LAUNCH] master=$MAIN_PROCESS_IP:$MAIN_PROCESS_PORT"
echo "[SMOKE-LAUNCH] backend=$SMOKE_BACKEND iters=$SMOKE_ITERS mb=$SMOKE_MB"
echo "[SMOKE-LAUNCH] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"
echo "[SMOKE-LAUNCH] GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<unset>}"
echo "[SMOKE-LAUNCH] NCCL_IB_HCA=${NCCL_IB_HCA:-<unset>}"
echo "[SMOKE-LAUNCH] NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-<unset>}"

cd "$SCRIPT_DIR"
exec accelerate launch \
  --multi_gpu \
  --num_machines "$NUM_MACHINES" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MAIN_PROCESS_IP" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --num_processes "$NUM_PROCESSES" \
  scripts/ddp_allreduce_smoke.py \
  --backend "$SMOKE_BACKEND" \
  --iters "$SMOKE_ITERS" \
  --mb "$SMOKE_MB"
