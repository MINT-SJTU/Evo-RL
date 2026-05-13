# Value Indicator 可视化

目标：用指定 checkpoint 对指定 episode 做 value / indicator 可视化，并且不修改原始数据集。

原始数据集不要直接跑 `lerobot-value-infer`，因为它会把 value / advantage / indicator 写回 `--dataset.root`。先复制一份工作副本，再对副本操作。

## 1. 准备副本

```bash
cd /mnt/data1/ljh/Evo-RL

ORIG=/mnt/data1/ljh/dataset/lerobot_socks_short1400_v3
WORK=/mnt/data2/ljh_acp_outputs/viz_work/lerobot_socks_short1400_v3_ep0
OUT=/mnt/data2/ljh_acp_outputs/value_viz_socks_ep0_ckpt012000

mkdir -p "$(dirname "$WORK")"
cp -a --reflink=auto "$ORIG" "$WORK"
```

## 2. 运行可视化

当前 parquet `data/chunk-000/file-000.parquet` 只包含 `episode_index=0`，所以这里指定 episode 0。

```bash
CUDA_VISIBLE_DEVICES=3 lerobot-value-infer \
  --dataset.repo_id=local/lerobot_socks_short1400_v3_ep0_viz \
  --dataset.root="$WORK" \
  --dataset.episodes='[0]' \
  --dataset.download_videos=false \
  --inference.checkpoint_path=/mnt/data2/ljh_acp_outputs/value_train_socks_sft_400_rl10K_400_retry/checkpoints/012000/pretrained_model \
  --inference.checkpoint_ref=last \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_ckpt012000 \
  --acp.advantage_field=complementary_info.advantage_ckpt012000 \
  --acp.indicator_field=complementary_info.acp_indicator_ckpt012000 \
  --viz.enable=true \
  --viz.episodes=0 \
  --viz.video_key=observation.images.base \
  --viz.overwrite=true \
  --output_dir="$OUT" \
  --job_name=value_viz_socks_ep0_ckpt012000
```

输出位置：

```bash
/mnt/data2/ljh_acp_outputs/value_viz_socks_ep0_ckpt012000/value/viz/
```

## 多相机版本

如果想看三路相机，把上面命令里的：

```bash
--viz.video_key=observation.images.base \
```

换成：

```bash
--viz.video_keys=observation.images.base,observation.images.left_wrist,observation.images.right_wrist \
```

## 只重新导出视频

如果 `$WORK` 已经跑过推理，后面只是改了可视化样式，可以跳过 checkpoint 推理：

```bash
CUDA_VISIBLE_DEVICES=3 lerobot-value-infer \
  --dataset.repo_id=local/lerobot_socks_short1400_v3_ep0_viz \
  --dataset.root="$WORK" \
  --dataset.episodes='[0]' \
  --dataset.download_videos=false \
  --inference.reuse_existing_value_field=true \
  --acp.enable=false \
  --acp.value_field=complementary_info.value_ckpt012000 \
  --acp.advantage_field=complementary_info.advantage_ckpt012000 \
  --acp.indicator_field=complementary_info.acp_indicator_ckpt012000 \
  --viz.enable=true \
  --viz.episodes=0 \
  --viz.video_key=observation.images.base \
  --viz.overwrite=true \
  --output_dir="$OUT" \
  --job_name=value_viz_socks_ep0_ckpt012000_viz_only
```

