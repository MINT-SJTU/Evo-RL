#!/bin/bash
set -euo pipefail

# 0) 每次运行先激活 conda 环境
CONDA_ENV_NAME="evo-rl_ljh"
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

echo "[INFO] conda_env: ${CONDA_DEFAULT_ENV}"
echo "[INFO] python: $(command -v python)"

# 1) 局部环境变量：屏蔽代理并启用内网直连
export no_proxy="localhost,127.0.0.1,10.0.112.8,10.0.112.9"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"
# 注意：你的机器有 eth1-eth8 高速网卡，不需要禁用 IB/RoCE

# 2) 训练与集群参数
DATASET_ROOT="${DATASET_ROOT:-/mnt/data1/ljh/dataset/lerobot_towel_mid300+left_right_50}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/towel_mid300+left_right_50}"
POLICY_REPO_ID="${POLICY_REPO_ID:-local/pi05_towel_mid300+left_right_50}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-10.0.112.9}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29600}"
NUM_MACHINES="${NUM_MACHINES:-2}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-5}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-6}"
POLICY_COMPILE="${POLICY_COMPILE:-false}"
POLICY_COMPILE_MODE="${POLICY_COMPILE_MODE:-reduce-overhead}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-false}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-false}"

# 2.1) 尽量自动选择正确的通信网卡，避免走到不可达的 IPv6/错误接口导致 rendezvous 超时。
# 允许用户显式设置 NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME 覆盖。
NCCL_SOCKET_IFNAME_ORIG="${NCCL_SOCKET_IFNAME:-}"
GLOO_SOCKET_IFNAME_ORIG="${GLOO_SOCKET_IFNAME:-}"
AUTO_IFNAME_METHOD=""
if [[ -z "${NCCL_SOCKET_IFNAME:-}" || -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
  if command -v ip >/dev/null 2>&1; then
    AUTO_IFNAME="$(ip -o route get "$MAIN_PROCESS_IP" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
    if [[ -n "${AUTO_IFNAME:-}" && "${AUTO_IFNAME:-}" != "lo" ]]; then
      AUTO_IFNAME_METHOD="route"
    fi

    # If MAIN_PROCESS_IP is local (rank0 machine), route lookup often returns dev=lo.
    # Never force NCCL/GLOO to use loopback; instead, pick the interface that owns MAIN_PROCESS_IP.
    if [[ "${AUTO_IFNAME:-}" == "lo" || -z "${AUTO_IFNAME:-}" ]]; then
      AUTO_IFNAME="$(ip -o -4 addr show to "$MAIN_PROCESS_IP" 2>/dev/null | awk '{print $2; exit}')"
      if [[ -n "${AUTO_IFNAME:-}" && "${AUTO_IFNAME:-}" != "lo" ]]; then
        AUTO_IFNAME_METHOD="addr"
      fi
    fi

    if [[ -n "${AUTO_IFNAME:-}" && "${AUTO_IFNAME:-}" != "lo" ]]; then
      export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$AUTO_IFNAME}"
      export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$AUTO_IFNAME}"
    fi
  fi
fi

if [[ -n "${AUTO_IFNAME_METHOD:-}" && ( -z "${NCCL_SOCKET_IFNAME_ORIG:-}" || -z "${GLOO_SOCKET_IFNAME_ORIG:-}" ) ]]; then
  echo "[INFO] comm_ifname(auto:${AUTO_IFNAME_METHOD} to ${MAIN_PROCESS_IP}): NCCL=${NCCL_SOCKET_IFNAME:-<unset>} GLOO=${GLOO_SOCKET_IFNAME:-<unset>}"
else
  echo "[INFO] comm_ifname(manual/default): NCCL=${NCCL_SOCKET_IFNAME:-<unset>} GLOO=${GLOO_SOCKET_IFNAME:-<unset>}"
fi

# 根分区已满，将运行时写入重定向到大盘 /mnt/data
RUN_BASE="/mnt/data/ljh"
CACHE_BASE="/mnt/data1/ljh/.cache"
if [[ -z "$HF_CACHE_DIR" ]]; then
  HF_CACHE_DIR="$CACHE_BASE/huggingface"
fi
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$RUN_BASE/Evo-RL-outputs/pi05_dist_towel_mid300_352ep_$RUN_ID}"
RUN_OUTPUT_PARENT="$(dirname "$RUN_OUTPUT_DIR")"
export TMPDIR="$RUN_BASE/tmp"
export WANDB_DIR="$RUN_BASE/wandb"
export XDG_CACHE_HOME="$CACHE_BASE"
export HF_CACHE_DIR
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HF_LOCAL_FILES_ONLY
if [[ "$HF_LOCAL_FILES_ONLY" == "true" || "$HF_LOCAL_FILES_ONLY" == "1" || "$HF_LOCAL_FILES_ONLY" == "yes" || "$HF_LOCAL_FILES_ONLY" == "on" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
export TORCH_HOME="$CACHE_BASE/torch"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$WANDB_DIR" "$RUN_OUTPUT_PARENT" "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

# 2.5) 优先使用当前仓库源码，避免双机节点加载到不同版本的 site-packages
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for SRC_ROOT in "$SCRIPT_DIR/src" /mnt/data1/ljh/Evo-RL-git/src /mnt/data1/ljh/Evo-RL/src /mnt/data/whs/Evo-RL/src /mnt/data1/whs/Evo-RL/src; do
  if [[ -d "$SRC_ROOT/lerobot" ]]; then
    export PYTHONPATH="$SRC_ROOT:${PYTHONPATH:-}"
    break
  fi
done

if ! python -c "import lerobot" >/dev/null 2>&1; then
  echo "[ERROR] 无法导入 lerobot，请检查环境或源码路径"
  exit 1
fi

# 3) 自动判定 Rank
# 优先允许外部覆盖：NODE_RANK=0/1 bash run_sft.sh
if [[ -z "${NODE_RANK:-}" ]]; then
  HOSTNAME_STR="$(hostname)"
  case "$HOSTNAME_STR" in
    *h20-1*|*iv-yekfzb6mtc5i3z43xs8h*) NODE_RANK=0 ;;
    *h20-0*|*iv-yek0jxwet*) NODE_RANK=1 ;;
    *cluster_0*|*cluster-0*|*iv-yegtvzd340xjd1ug9m6d*) NODE_RANK=0 ;;
    *cluster_3*|*cluster-3*|*vol-cluster-3*) NODE_RANK=1 ;;
    *)
      echo "[ERROR] 无法从 hostname 自动判定 NODE_RANK: $HOSTNAME_STR"
      echo "[ERROR] 请显式指定，例如：NODE_RANK=0 bash run_sft.sh"
      exit 1
      ;;
  esac
fi

echo "[INFO] hostname: $(hostname)"
echo "[INFO] machine_rank: $NODE_RANK"
echo "[INFO] num_machines: $NUM_MACHINES"
echo "[INFO] num_processes: $NUM_PROCESSES"
echo "[INFO] main_process_ip: $MAIN_PROCESS_IP:$MAIN_PROCESS_PORT"
echo "[INFO] dataset.root: $DATASET_ROOT"
echo "[INFO] dataset.repo_id: $DATASET_REPO_ID"
echo "[INFO] policy.repo_id: $POLICY_REPO_ID"
echo "[INFO] run_id: $RUN_ID"
echo "[INFO] output_dir: $RUN_OUTPUT_DIR"
echo "[INFO] resume: $RESUME"
echo "[INFO] wandb_enable: $WANDB_ENABLE"
echo "[INFO] wandb_disable_artifact: $WANDB_DISABLE_ARTIFACT"
echo "[INFO] train_steps: $TRAIN_STEPS"
echo "[INFO] batch_size: $BATCH_SIZE"
echo "[INFO] log_freq: $LOG_FREQ"
echo "[INFO] save_freq: $SAVE_FREQ"
echo "[INFO] optimizer_lr: $OPTIMIZER_LR"
echo "[INFO] scheduler_decay_lr: $SCHEDULER_DECAY_LR"
echo "[INFO] policy_compile: $POLICY_COMPILE"
echo "[INFO] policy_compile_mode: $POLICY_COMPILE_MODE"
echo "[INFO] hf_cache_dir: $HF_CACHE_DIR"
echo "[INFO] hf_local_files_only: $HF_LOCAL_FILES_ONLY"
echo "[INFO] nccl_debug: ${NCCL_DEBUG:-<unset>}"
echo "[INFO] nccl_debug_subsys: ${NCCL_DEBUG_SUBSYS:-<unset>}"
echo "[INFO] nccl_socket_ifname: ${NCCL_SOCKET_IFNAME:-<unset>}"
echo "[INFO] nccl_ib_hca: ${NCCL_IB_HCA:-<unset>}"
echo "[INFO] gloo_socket_ifname: ${GLOO_SOCKET_IFNAME:-<unset>}"
echo "[INFO] ddp_init_sync: ${DDP_INIT_SYNC:-<unset>}"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] 数据集目录不存在: $DATASET_ROOT"
  exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
  echo "[ERROR] 当前环境缺少 accelerate，请检查 $CONDA_ENV_NAME 依赖"
  exit 1
fi



# 4) 启动指令 (16卡双机)
accelerate launch \
  --multi_gpu \
  --num_machines $NUM_MACHINES \
  --machine_rank $NODE_RANK \
  --main_process_ip $MAIN_PROCESS_IP \
  --main_process_port $MAIN_PROCESS_PORT \
  --num_processes $NUM_PROCESSES \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id=$POLICY_REPO_ID \
  --dataset.root=$DATASET_ROOT \
  --dataset.repo_id=$DATASET_REPO_ID \
  --steps=$TRAIN_STEPS \
  --policy.optimizer_lr=$OPTIMIZER_LR \
  --policy.scheduler_decay_lr=$SCHEDULER_DECAY_LR \
  --batch_size=$BATCH_SIZE \
  --log_freq=$LOG_FREQ \
  --save_freq=$SAVE_FREQ \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=$GRADIENT_CHECKPOINTING \
  --policy.compile_model=$POLICY_COMPILE \
  --policy.compile_mode=$POLICY_COMPILE_MODE \
  --policy.push_to_hub=$POLICY_PUSH_TO_HUB \
  --policy.empty_cameras=0 \
  --resume=$RESUME \
  --job_name=pi05_${NUM_MACHINES}node${NUM_PROCESSES}gpu_towel_mid300_352ep_$RUN_ID \
  --output_dir=$RUN_OUTPUT_DIR \
  --wandb.enable=$WANDB_ENABLE \
  --wandb.disable_artifact=$WANDB_DISABLE_ARTIFACT
  
