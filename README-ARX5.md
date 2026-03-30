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

### 5.1 Raw training recording (record trajectories for training)

When you additionally set `--raw-train-record-dir /path/to/raw_root` in the command line, the script enables
"raw training recording".
In this mode, the keyboard semantics are overlaid on top of the existing controls (primarily for collecting
training trajectories):

- `R`: start recording a new `episode` and resume the policy (if you are already recording, it continues
  the same episode and the main loop stays in `RUNNING`)
- `D`: end the current recording episode (if there was no VR takeover during this episode, the script will
  wait for you to label success with `0/1`)
- `0`: set `episode_success=0` (false)
- `1`: set `episode_success=1` (true)
- `V`: enter VR teleop (if you press `V` while recording, the current episode is marked as "human takeover"
  (= VR takeover), and `episode_success` is forced to `1`)

VR takeover labeling:

- VR segments are stored under `segment_*_vr/`
- During conversion to LeRobot Dataset, `segment_*_vr` is mapped to
  `complementary_info.is_intervention=1`
- If VR takeover happens within the episode, `episode_success` is forced to `1`, so after you press `D`
  you usually will not need to press `0/1`.

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

### 5.2 Raw dataset format

After enabling `--raw-train-record-dir`, each full recording session generates an `episode_XXXX` directory:

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
        actions.json   # [T, action_dim], action_dim=7 (6 joints + gripper)
        states.json    # [T, 7] state vector aligned with action
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

Key fields:

- `meta.json`
  - `episode_success`: the final success label (0/1). It is written after you press `D` and complete labeling.
  - `has_vr_takeover`: whether the episode had a VR takeover
  - `success_forced`: if VR takeover happened, success is forced to 1
- `segments/segment_*_policy` vs `segments/segment_*_vr`
  - `source` is encoded in the directory name: `policy` = autonomous control, `vr` = human takeover
  - The conversion script uses it to generate training-time `complementary_info.is_intervention`

Note: the current implementation saves **all camera frames for every step** into `frame_XXXX.png` inside
each segment, so it preserves the full step-by-step visual sequence.

### 5.3 Convert raw -> LeRobotDataset

Conversion script:

```bash
python3 src/lerobot/scripts/convert_arx5_raw_train_to_lerobot_dataset.py \
  --raw-record-dir /path/to/raw_root \
  --output-root /path/to/lerobot_dataset_root \
  --repo-id your_dataset_repo_id \
  --robot-type arx5
```

If you want the dataset to store images instead of videos (no `mp4`):

```bash
python3 src/lerobot/scripts/convert_arx5_raw_train_to_lerobot_dataset.py \
  --raw-record-dir /path/to/raw_root \
  --output-root /path/to/lerobot_dataset_root_images \
  --repo-id your_dataset_repo_id_images \
  --robot-type arx5 \
  --images-only
```

LeRobotDataset storage location and structure (simplified):

- `data/chunk-XXX/file-XXX.parquet`: per-frame numeric data (e.g. action/state)
- `meta/episodes/*.parquet` and `meta/info.json`: episode index/statistics metadata
- If using video mode (default): `videos/` contains `mp4`
- If using image mode (`--images-only`): `images/` contains per-frame PNGs (no mp4 encoding)

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

