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

### 4.1 Dual-arm inference + intervention

The current repository also provides a dual-arm runtime with the same keyboard/VR takeover flow as the
single-arm script:

```bash
cd /home/user/workspace/zsj/Evo-RL
conda activate lerobot
export ARX_SDK_ROOT=~/workspace/ARX_X5/py/arx_x5_python

lerobot-arx5-dual-infer \
  --task "handover the object" \
  --policy-path lerobot/pi05_bimanual \
  --left-can-port can0 \
  --right-can-port can1 \
  --cameras base:254522071216 left_wrist:150622073629 right_wrist:409122272986 \
  --safe-mode

lerobot-arx5-dual-infer \
  --task "Put the slippers on the shelf" \
  --policy-path checkpoints/dual_towel/pretrained_model \
  --left-can-port can0 \
  --right-can-port can1 \
  --cameras base:150622073629 left_wrist:352122273179 right_wrist:409122272986 \
  --safe-mode

python -m lerobot.scripts.lerobot_arx5_dual_infer \
  --task "Fold the towel and put it on the edge of the table" \
  --policy-path checkpoints/dual_towel/pretrained_model/ \
  --left-can-port can0 \
  --right-can-port can1 \
  --cameras \
    base:150622073629 \
    left_wrist:352122273179 \
    right_wrist:409122272986 \
  --safe-mode \
  --record-dir /home/user/workspace/zsj/Evo-RL/inference_records
```

Notes:

- Dual-arm policy input/output state is fixed to 14 dims: left arm `state.0..6`, right arm `state.7..13`.
- `V` switches from infer to dual-arm VR teleop, reusing the same two ARX5 clients without reopening CAN.
- After pressing `Ctrl+C` in the VR process, the script reconnects infer cameras, holds both arms, and returns to keyboard `STOPPED` state.
- `--raw-train-record-dir` is also supported in dual-arm mode; policy segments and VR takeover segments are recorded in the same raw format as the single-arm script. **Dual-arm raw recording is gated by `S`/`D`** (see §5.1): press `S` to open a recording window, then `R` to run the policy; without `S`, inference and VR still work but trajectories are not written (the script logs a warning on `R` / `V`).

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
Keyboard semantics depend on which entrypoint you use:

**Dual-arm (`lerobot_arx5_dual_infer` / `lerobot-arx5-dual-infer`)**

- `S`: start a new raw `episode_`* directory (recording window). Policy does **not** run until you press `R`.
- `R`: resume policy control as usual; if no recording window is open, the run continues but **nothing is written**—the script logs a warning (same idea when entering VR with `V` without an active recording window).
- `D`: end the current recording episode and stop the policy; you must then finalize with `0` / `1` / `2` (see below).
- If `--raw-train-record-dir` is omitted and you press `S`, the script warns that recording did not start.
- `S` while the policy is running is ignored (stop with `Space` first, then `S` for a new window).

**Single-arm (`lerobot_arx5_infer`)**

- `R`: start recording a new `episode` (if allowed) **and** resume the policy in one step.
- `D` / `0` / `1` / `2`: same finalize behavior as below (shared `TrainRawEpisodeRecorder`).

**After `D`: labeling** (shared `TrainRawEpisodeRecorder`)

- `0`: failure (`episode_success=0`)
- `1`: success (`episode_success=1`)
- `2`: **abandon**—delete the entire `episode_XXXX` directory for this take (no `episode_success` written)

**VR takeover**

- `V` while a raw episode is active marks the episode with VR takeover (`has_vr_takeover`, `success_forced` in `meta.json`).
- After `D`, you are still prompted for `0` / `1` / `2`. If you press `0` after VR takeover, the script logs a **warning** and keeps waiting—you must choose `1` (keep as success) or `2` (abandon / delete).

Other details:

- VR segments are stored under `segment_*_vr/`
- During conversion to LeRobot Dataset, `segment_*_vr` is mapped to
`complementary_info.is_intervention=1`
- **`--duration`**: policy steps and **raw VR recording** both target this interval (VR uses a dedicated record thread with `precise_sleep`, decoupled from teleop control).
- **`--vr-control-hz`**: VR teleop IK / command rate only (default 50 Hz); does not set raw recording rate.
- **`--debug-timestamp`**: when raw recording is enabled, each segment also writes `timestamps.json` (per-frame `t_perf_s`, `t_wall_iso`, `t_rel_segment_s`) for debugging actual sample spacing.

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
        timestamps.json  # optional: --debug-timestamp (per-frame timing metadata)
      segment_0001_vr/
        images/
          {camera_name}/
            frame_000000.png
            frame_000001.png
            ...
        actions.json
        states.json
        timestamps.json  # optional: --debug-timestamp
  episode_0001/
  ...
```

Key fields:

- `meta.json`
  - `episode_success`: the final success label (`0` or `1`). Written after you press `D` and choose `0` or `1`. Abandoned episodes (`2`) remove the folder, so they do not appear here.
  - `has_vr_takeover`: whether the episode had a VR takeover
  - `success_forced`: set when VR takeover applies the stricter labeling rule (failure `0` not allowed after `D`; use `1` or `2`)
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

If you want   - `episode_success`: the final success label (`0` or `1`). Written after you press `D` and choose `0` or `1`. Abandoned episodes (`2`) remove the folder, so they do not appear here.

- `has_vr_takeover`: whether the episode had a VR takeover
- `success_forced`: set when VR takeover applies the stricter labeling rule (failure `0` not allowed after `D`; use `1` or `2`)
ied):
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

