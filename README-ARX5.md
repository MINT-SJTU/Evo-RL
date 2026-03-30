# ARX5 + pi05_base

This document records the ARX5 runtime that was added to Evo-RL for running LeRobot `pi05_base` directly on the robot.

## 1. Environment setup

Use the repository Conda environment first:

```bash
cd /home/user/workspace/whs/Evo-RL
conda activate lerobot
```

If `lerobot` is not available, use the fallback environment noted in `Evo-RL/AGENTS.md`.

Install the ARX5 runtime extras:

```bash
pip install -r requirements-arx5.txt
```

Point LeRobot to the local ARX SDK checkout:

```bash
export ARX_SDK_ROOT=~/workspace/ARX_X5/py/arx_x5_python
```

The ARX5 SDK itself is not installed by `pip`; it is loaded from `ARX_SDK_ROOT`.

## 2. Camera check

List RealSense devices:

```bash
lerobot-arx5-infer --list-cameras
```

The ARX5 runtime expects the same local camera names used in the dexbotic scripts:

- `side` -> LeRobot `base_0_rgb`
- `wrist` -> LeRobot `left_wrist_0_rgb`
- `front` -> LeRobot `right_wrist_0_rgb`

## 3. Optional dry run

Before touching the real arm, run a stub-only smoke test that still loads the policy stack and saves one inferred chunk:

```bash
lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path lerobot/pi05_base \
  --use-stub \
  --safe-mode \
  --record-dir ./tmp/arx5_stub_record \
  --cameras side:254522071216 wrist:150622073629 front:409122272986
```

If you do not want the script to access hardware cameras during the dry run, replace the three RealSense entries with USB camera indices and add `--use-usb-cams`.

## 4. Real robot inference

The command below mirrors the dexbotic deployment style but runs LeRobot `pi05_base` locally:

```bash
cd /home/user/workspace/whs/Evo-RL
conda activate lerobot
export ARX_SDK_ROOT=~/workspace/ARX_X5/py/arx_x5_python

lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path lerobot/pi05_base \
  --cameras side:254522071216 wrist:150622073629 front:409122272986 \
  --flip-cameras wrist front \
  --duration 0.1 \
  --safe-mode
```

Notes:

- `--duration 0.1` matches the 10 Hz chunk execution used in the dexbotic Pi0.5 client.
- `--safe-mode` gates each predicted chunk behind keyboard confirmation and clips each Cartesian translation step to `0.05 m` by default.
- The runtime uses the bundled quantile stats file `src/lerobot/robots/arx5_follower/pi05_arx5_default_stats.json` unless `--stats-path` is provided.

## 5. Keyboard controls

When keyboard control is enabled (the default):

- `Space`: hold current pose immediately
- `H`: go home, then hold
- `B`: enter teach mode / gravity compensation
- `N`: save the current teach pose
- `M`: move back to the saved teach pose
- `V`: switch to VR teleop mode (from STOPPED state)
- `R`: resume policy control
- `I`: in safe mode, request the next predicted chunk
- `Q`: quit

The script starts in a stopped state when keyboard control is enabled. Press `R` first, then `I` for each chunk when `--safe-mode` is active.

### 5.1 Raw training recording（录制用于训练的原始轨迹）

当你在命令行额外设置 `--raw-train-record-dir /path/to/raw_root` 时，会启用“raw training recording”。
此时键盘按键语义会在原有基础上叠加（主要用于原始轨迹采集）：

- `R`: 开始录制一个 episode，并恢复 policy（如果已经在录制中，则继续录制同一条 episode，且仍会把主循环切回 `RUNNING`）
- `D`: 结束当前录制 episode（如果此 episode 期间没有 VR 接管，则会进入等待你按 `0/1` 标注 success 的状态）
- `0`: 标注 `episode_success=0`（false）
- `1`: 标注 `episode_success=1`（true）
- `V`: 进入 VR teleop（如果你在录制过程中按下 `V`，会将当前 episode 标记为“人工接管(=VR takeover)”，并对该 episode 强制 success=true=1）

关于 VR takeover 标注：
- VR 段落会以 `segment_*_vr/` 的形式被记录
- 转换为 LeRobot Dataset 时，会把 `segment_*_vr` 映射为训练用的 `complementary_info.is_intervention=1`
- 若 episode 期间发生 VR takeover，则 `episode_success` 会被强制为 `1`，因此 `D` 结束后通常不会再等待你按 `0/1`

### VR switch behavior (`V`)

When switching from infer to VR with `V`, the runtime does:

1. Hold the current infer joint+gripper target.
2. Hand off the same ARX5 arm client to `Arx5Runtime` (no second CAN open, no arm `protect_mode` during handoff).
3. Start VR controller with a shared interface bound to that same low-level arm object.
4. Keep robot joint pose frozen until you hold the VR grip trigger to activate arm motion.
5. Keep gripper at pre-switch value until you start using the VR gripper trigger.

This avoids the jump/drop caused by resetting to home or reopening CAN.

### Exit VR and return to infer

While VR is running, press `Ctrl+C` in the VR terminal:

1. VR loop exits.
2. The arm client is taken back from `Arx5Runtime`.
3. Infer reconnects using `robot.connect(reuse_arm_client=...)` and reopens cameras.
4. Infer calls `hold_position()` and returns to keyboard STOPPED state.

After returning from VR, press `R` to continue infer policy execution.

### 5.2 Raw dataset format（raw 原始数据存储格式）

启用 `--raw-train-record-dir` 后，每次完整录制会生成一个 `episode_XXXX` 目录：

```text
raw_root/
  episode_0000/
    meta.json
    segments/
      segment_0000_policy/
        images/
          {camera_name}/
            frame_000000.png
            frame_000001.png
            ...
        actions.json   # [T, action_dim]，action_dim=7（6关节+夹爪）
        states.json    # [T, 7]，用于与 action 对齐的状态向量
      segment_0001_vr/
        images/
          {camera_name}/
            frame_000000.png
            frame_000001.png
            ...
        actions.json
        states.json
  episode_0001/
  ...
```

关键字段说明：
- `meta.json`
  - `episode_success`: 你的最终 success 标签（0/1），写入时间点是按 `D` 结束录制并完成标注后
  - `has_vr_takeover`: episode 是否发生过 VR 接管
  - `success_forced`: 若发生 VR 接管，则 success 会被强制为 1
- `segments/segment_*_policy` vs `segments/segment_*_vr`
  - `source` 由目录名体现：`policy`=自主运行，`vr`=人工接管
  - 转换脚本会据此生成训练用的 `complementary_info.is_intervention`

注：当前实现会在每个 segment 内把**每个 step 的所有相机帧都保存**到 `frame_XXXX.png`，因此能够保留完整逐步图像序列。

### 5.3 Convert raw -> LeRobotDataset（raw 转 LeRobot 训练格式）

转换脚本：

```bash
python3 src/lerobot/scripts/convert_arx5_raw_train_to_lerobot_dataset.py \
  --raw-record-dir /path/to/raw_root \
  --output-root /path/to/lerobot_dataset_root \
  --repo-id your_dataset_repo_id \
  --robot-type arx5
```

如果你希望训练集使用图片而不是视频（不生成 mp4）：

```bash
python3 src/lerobot/scripts/convert_arx5_raw_train_to_lerobot_dataset.py \
  --raw-record-dir /path/to/raw_root \
  --output-root /path/to/lerobot_dataset_root_images \
  --repo-id your_dataset_repo_id_images \
  --robot-type arx5 \
  --images-only
```

LeRobotDataset 存储位置与结构（简化理解）：
- `data/chunk-XXX/file-XXX.parquet`：逐帧数值数据（action/state 等）
- `meta/episodes/*.parquet` 与 `meta/info.json`：episode/索引/统计等元信息
- 若使用视频模式（默认）：`videos/` 下会存 mp4
- 若使用图片模式（`--images-only`）：`images/` 下存每帧 PNG（不走 mp4 编码）

## 6. Record-only mode

To inspect policy outputs without moving the robot, add `--record-dir`:



```bash
lerobot-arx5-infer \
  --task "Stack all the paper cups on the table together" \
  --policy-path checkpoints/multi_cups_test0/pretrained_model \
  --cameras front:335122271555 wrist:409122272986 \
  --safe-mode \
  --record-dir /home/user/workspace/whs/Evo-RL/inference_records
```

Each chunk is written under `./inference_records/round_XXXX/` with:

- `input/base.png`
- `input/right_wrist.png`
- `input/current_ee_state.json`
- `output/actions.json`

For `checkpoints/multi_cups_test0`, the ARX5 runtime resolves the local camera names as:

- `front` -> `observation.images.base`
- `wrist` -> `observation.images.right_wrist`
- `observation.images.empty_camera_0` is left empty and padded by the PI0.5 model

## 7. Useful overrides

Use a local checkpoint instead of the Hub:

```bash
lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path /path/to/local/pi05_base
```

Use a custom stats file:

```bash
lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path lerobot/pi05_base \
  --stats-path /path/to/arx5_stats.json
```

Shorten each executed chunk:

```bash
lerobot-arx5-infer \
  --task "place shoes on rack" \
  --policy-path lerobot/pi05_base \
  --execution-horizon 10
```

