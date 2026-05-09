# Evo-RL 多机训练说明

本文档是 Evo-RL 在 H20/A800 多机集群上启动 SFT 训练的统一说明。它合并了原来分散的启动步骤、DDP 解释、四机配置和训练命令解析文档。

## 推荐的 H20 双机启动方式

在任意一台 H20 节点的项目根目录执行 `run_h20_multinode.sh`。这个脚本会自动识别当前机器，通过 SSH 启动另一台机器，并把每台机器上的实际训练命令交给 `run_sft.sh` 执行。

```bash
cd /mnt/data1/ljh/Evo-RL

RUN_ID=socks_short1400_v3_lr5e5_bs512_log50_save2000_test0_compile_reduce_wandb_$(date +%Y%m%d_%H%M%S) \
MAIN_PROCESS_IP=10.0.112.9 \
MAIN_PROCESS_PORT=29622 \
NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 \
DDP_INIT_SYNC=false \
LEROBOT_DATASET_LOAD_MODE=main_first \
WANDB_ENABLE=true \
WANDB_MODE=online \
TRAIN_STEPS=20000 \
BATCH_SIZE=32 \
LOG_FREQ=50 \
SAVE_FREQ=2000 \
OPTIMIZER_LR=5e-5 \
SCHEDULER_DECAY_LR=1e-6 \
POLICY_COMPILE=false \
POLICY_COMPILE_MODE=reduce-overhead \
GRADIENT_CHECKPOINTING=true \
HF_LOCAL_FILES_ONLY=true \
DATASET_ROOT=/mnt/efs_1/lerobot_socks3000_clean_len300_750 \
DATASET_REPO_ID=local/socks3000_clean_len300_750 \
bash run_h20_multinode.sh

```

如果希望离线运行，只使用本地 cache，可以这样启动：

```bash
RUN_ID=socks_short1400_cacheonly \
WANDB_ENABLE=false \
WANDB_MODE=disabled \
HF_LOCAL_FILES_ONLY=true \
bash run_h20_multinode.sh
```

日志默认写入 `/mnt/data1/ljh/Evo-RL-logs`。本地和远端日志文件名会包含机器标识和 `RUN_ID`。

## 集群布局和路径约定

默认 H20 双机配置如下：

- Rank 0：`h20-1`
- Rank 1：`h20-0`
- 主进程 IP：`10.0.112.9`
- 两台机器上的项目路径：`/mnt/data1/ljh/Evo-RL`
- 日志路径：`/mnt/data1/ljh/Evo-RL-logs`
- Hugging Face cache：`/mnt/data1/ljh/.cache/huggingface`
- Conda 环境：`evo-rl_ljh`

四机脚本遵循相同约定，但每台机器都必须保持项目路径、conda 环境、数据路径、checkpoint 路径和 cache 布局一致。启动前要确保所有机器上的脚本版本一致。

启动前建议检查：

```bash
ssh h20-0 'hostname && test -d /mnt/data1/ljh/Evo-RL && test -d /mnt/data1/ljh/.cache/huggingface'
nvidia-smi
```

如果 SSH 主机别名不可用，可以显式设置 `REMOTE_HOST`、`LOCAL_RANK`、`MAIN_PROCESS_IP` 和 `MAIN_PROCESS_PORT`。

## 启动脚本链路

启动链路分三层：

1. `run_h20_multinode.sh`
   - 面向用户的 H20 双机训练入口。
   - 激活 conda 环境，选择远端机器，必要时选择可用端口，然后调用 `run_all.sh`。

2. `run_all.sh`
   - 使用 `nohup` 启动本机和远端 rank。
   - 通过 `rsync` 把启动脚本同步到远端机器。
   - 传递 W&B、Hugging Face cache、训练步数、batch size、学习率、compile、checkpoint 等环境变量。

3. `run_sft.sh`
   - 每台机器上的实际训练入口。
   - 使用 `accelerate launch` 启动训练，并传入机器 rank、主进程 IP/端口、进程数和 SFT 配置覆盖项。

当前分布式训练核心是 Accelerate + DDP。这个工作流没有启用 DeepSpeed。

## 重要环境变量

训练控制：

- `RUN_ID`：日志和输出标识后缀。
- `TRAIN_STEPS`：总训练步数。
- `BATCH_SIZE`：训练配置中的单进程 batch size。
- `LOG_FREQ`：日志打印间隔。
- `SAVE_FREQ`：checkpoint 保存间隔。
- `RESUME`：是否 resume。

数据集：

- `DATASET_ROOT`：数据集目录路径（两机/多机必须保持相同路径，并且每台机器上都存在）。
- `DATASET_REPO_ID`：数据集标识（本地数据集通常写 `local/<name>`，用于日志、输出目录和可能的 Hub 命名）。

优化相关：

- `OPTIMIZER_LR`：优化器学习率。
- `SCHEDULER_DECAY_LR`：scheduler 的最终或衰减学习率。
- `POLICY_COMPILE`：是否启用 `torch.compile`。
- `POLICY_COMPILE_MODE`：compile 模式，通常使用 `reduce-overhead`。
- `GRADIENT_CHECKPOINTING`：是否启用 gradient checkpointing，以减少显存占用。

W&B：

- `WANDB_ENABLE`：是否启用 W&B 日志。
- `WANDB_MODE`：可设为 `online`、`offline` 或 `disabled`。
- `WANDB_DISABLE_ARTIFACT`：为 true 时避免上传 artifact。

Hugging Face cache：

- `HF_CACHE_DIR`：模型和 tokenizer 加载时共用的 cache 目录。
- `HF_LOCAL_FILES_ONLY`：强制只使用本地 cache。
- `HF_PREWARM`：启动前是否预热模型/tokenizer cache。
- `HF_PREFER_OFFLINE`：优先尝试使用本地缓存文件。

分布式运行：

- `NODE_RANK`：当前机器 rank，供 `run_sft.sh` 使用。
- `MAIN_PROCESS_IP`：rank 0 机器 IP。
- `MAIN_PROCESS_PORT`：分布式 rendezvous 端口。
- `NCCL_SOCKET_IFNAME`、`GLOO_SOCKET_IFNAME`、`NCCL_IB_HCA`：可选的网络接口指定项。
- `DDP_INIT_SYNC=false`：跳过 DDP 初始化时的参数广播/形状同步检查。当前 H20 双机场景已验证这个设置可以避开 `accelerator.prepare` 阶段的小 allreduce 超时；前提是两台机器加载的是同一个模型和代码版本。
- `NCCL_DEBUG`、`NCCL_DEBUG_SUBSYS`：可选的 NCCL 调试日志。

## DDP 行为概要

当前工作流使用 Distributed Data Parallel，也就是 DDP：

- 每个 GPU 进程都有一份完整模型副本。
- 每个 rank 通过分布式 dataloader 读取不同的数据分片。
- backward 时不同 rank 之间同步梯度。
- 梯度同步完成后，每个 rank 各自执行 optimizer step。
- checkpoint 应只由主进程保存。
- W&B 应只由主进程初始化和写日志。

DeepSpeed 不同于 DDP，它可以通过 ZeRO 切分 optimizer state、梯度和参数，从而减少显存占用。但 DeepSpeed 会增加运行复杂度，当前启动链路没有使用它。

`lerobot_train.py` 中的训练统计使用单进程 batch size，避免在分布式训练中重复乘 `accelerator.num_processes`，从而导致 sample 数和 epoch 统计被重复放大。

## 启动前检查

启动前先检查脚本和 Python 文件：

```bash
cd /mnt/data1/ljh/Evo-RL
bash -n run_sft.sh
bash -n run_h20_multinode.sh
bash -n run_all.sh
python -m py_compile src/lerobot/scripts/lerobot_train.py
```

检查两台机器上的路径、环境和 GPU：

```bash
ssh h20-0 'cd /mnt/data1/ljh/Evo-RL && pwd && conda env list | grep evo-rl_ljh'
ssh h20-0 'nvidia-smi --query-gpu=index,name,memory.total --format=csv'
```

还要确认数据、cache 和输出路径在每台机器上都存在。如果使用 `HF_LOCAL_FILES_ONLY=true`，必须先保证所有需要的模型和 tokenizer 文件已经在本地 cache 中。

## 运行状态检查

查看训练进程：

```bash
pgrep -af 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run'
ssh h20-0 "pgrep -af 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run'"
```

查看 GPU 使用情况：

```bash
nvidia-smi
ssh h20-0 nvidia-smi
```

跟踪日志：

```bash
tail -f /mnt/data1/ljh/Evo-RL-logs/run_sft_h20-1_${RUN_ID}.log
tail -f /mnt/data1/ljh/Evo-RL-logs/run_sft_h20-0_${RUN_ID}.log
```

正常训练时，日志里应该能看到所有 rank 成功加入、周期性 loss/throughput 输出，以及 checkpoint 只由主进程保存。

## 常见问题

端口已被占用：

```bash
MAIN_PROCESS_PORT=29631 bash run_h20_multinode.sh
```

W&B 登录或 artifact 报错：

```bash
WANDB_ENABLE=false WANDB_MODE=disabled WANDB_DISABLE_ARTIFACT=true bash run_h20_multinode.sh
```

Hugging Face 下载、模型加载或 tokenizer 加载失败：

- 确认每台机器上都有 `HF_CACHE_DIR`。
- 只有在 cache 已经准备好之后，才使用 `HF_LOCAL_FILES_ONLY=true`。
- 保证每台机器上的模型和 tokenizer cache 路径一致。

NCCL timeout 或 rendezvous 失败：

- 确认每台机器都能访问 `MAIN_PROCESS_IP`。
- 确认 `MAIN_PROCESS_PORT` 没有被占用或被防火墙阻断。
- 可以设置 `NCCL_DEBUG=INFO` 查看更详细日志。
- 必要时指定 `NCCL_SOCKET_IFNAME` 或 `GLOO_SOCKET_IFNAME`。

GPU 利用率偏低：

- 查看日志中的 `data_s` 和 `updt_s`。
- 如果 `data_s` 很低而 `updt_s` 很高，瓶颈主要在计算，而不是数据读取。
- compile 和 gradient checkpointing 可以改善显存/吞吐之间的取舍，但可能增加启动和首次编译时间。

## 停止训练

停止本机和远端训练：

```bash
pkill -f 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run'
ssh h20-0 "pkill -f 'run_sft.sh|lerobot.scripts.lerobot_train|accelerate launch|torch.distributed.run'"
```

如果是四机训练，需要在每台机器上执行同类停止命令。

## 四机训练备注

建议先确认双机训练稳定，再使用四机脚本。四机训练需要满足：

- 每台机器都有唯一的 `NODE_RANK`。
- 所有机器使用相同的 `MAIN_PROCESS_IP` 和 `MAIN_PROCESS_PORT`。
- 项目路径、数据路径、cache 路径和输出路径保持一致。
- 启动机器可以通过 SSH 访问所有其他机器。
- Conda、CUDA、PyTorch 和 NCCL 环境版本一致。

四机训练和双机训练的 DDP 行为相同，只是参与的机器和进程数量更多。
