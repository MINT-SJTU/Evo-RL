"""Record RLT human-in-loop data: RL rollout + human intervention.

In-process launcher that reads setup.json, builds config, and calls
the lerobot record() function with RLTDeployPolicy as the action source.

Keyboard controls during recording:
    r     - Start RL critical phase (switch to RL actor)
    SPACE - Toggle human intervention during RL phase
    s     - End critical phase (mark success), switch back to VLA
    f     - End critical phase (mark failure), switch back to VLA
    →     - End current episode
    ←     - Discard and re-record episode
    ESC   - Stop all recording

Data flow:
    Default: VLA prefix phase, VLA drives actions
      → [r] → Critical phase, RL actor drives
        → [SPACE] → Human teleop intervention ON
        → [SPACE] → Human intervention OFF, RL resumes
        → [s/f] → End CP, back to VLA

Usage (on zhaobo-4090-1, defaults match 278ep checkpoint):
    cd ~/code/hsy/Evo-RL
    conda activate evo-rl
    PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/record_rlt_hil.py --num-episodes 5

    # Or with explicit paths:
    PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/record_rlt_hil.py \
        --vla-model /home/zhaobo-4090-1/models/pi05_screw_271ep_sft_fp32 \
        --rl-token-ckpt checkpoints/rlt_271ep_sft/demo_adapt_checkpoint.pt \
        --ac-ckpt checkpoints/rlt_278ep_sft/rl_checkpoint.pt \
        --task "Insert the copper screw into the black sleeve." \
        --num-episodes 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dataset.setup_helpers import (
    get_sorted_followers,
    get_sorted_leaders,
    load_setup_json,
    resolve_dataset_root,
)

log = logging.getLogger(__name__)

# SO101 bilateral: 6 joints per arm × 2 = 12 DOF
_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record RLT human-in-loop data")
    # RLT model paths (defaults for 278ep SFT on zhaobo-4090-1)
    p.add_argument("--vla-model", type=str,
                    default="/home/zhaobo-4090-1/models/pi05_screw_271ep_sft_fp32")
    p.add_argument("--rl-token-ckpt", type=str,
                    default="checkpoints/rlt_271ep_sft/demo_adapt_checkpoint.pt")
    p.add_argument("--ac-ckpt", type=str,
                    default="checkpoints/rlt_278ep_sft/rl_checkpoint.pt")
    p.add_argument("--task", type=str, default="Insert the copper screw into the black sleeve.")
    # Recording
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--episode-time-s", type=int, default=3000)
    p.add_argument("--fps", type=int, default=30)
    # RLT inference
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--chunk-length", type=int, default=10)
    p.add_argument("--token-pool-size", type=int, default=64)
    p.add_argument("--deterministic", action="store_true", default=True)
    # Actor architecture (must match 278ep checkpoint)
    p.add_argument("--actor-hidden-dim", type=int, default=256)
    p.add_argument("--actor-num-layers", type=int, default=3)
    p.add_argument("--actor-residual", action="store_true", default=True)
    p.add_argument("--actor-activation", type=str, default="relu")
    p.add_argument("--actor-layer-norm", action="store_true", default=False)
    # Setup
    p.add_argument("--setup-json", default=None)
    p.add_argument("--dataset-tag", default="rlt_hil_test")
    p.add_argument("--vcodec", default="h264")
    p.add_argument("--no-teleop", action="store_true", default=False,
                    help="Skip leader arm teleop (disables human intervention)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_camera_configs(cameras: list[dict]) -> tuple[dict, dict]:
    """Split cameras into left/right dicts for BiSOFollower.

    Returns (left_cameras, right_cameras) as JSON-compatible dicts.
    """
    CAM_RENAME = {"left_wrist": "wrist", "right_wrist": "wrist", "top": "front"}
    LEFT_CAMS = {"left_wrist"}
    RIGHT_CAMS = {"right_wrist", "top"}

    left_cameras, right_cameras = {}, {}
    for cam in cameras:
        alias = cam["alias"]
        new_name = CAM_RENAME.get(alias, alias)
        cam_cfg = {
            "type": "opencv",
            "index_or_path": cam["port"],
            "width": cam.get("width", 640),
            "height": cam.get("height", 480),
            "fps": cam.get("fps", 30),
        }
        if cam.get("fourcc"):
            cam_cfg["fourcc"] = cam["fourcc"]
        if alias in LEFT_CAMS:
            left_cameras[new_name] = cam_cfg
        elif alias in RIGHT_CAMS:
            right_cameras[new_name] = cam_cfg
    return left_cameras, right_cameras


def main():
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load robot config
    setup = load_setup_json(args.setup_json)
    followers = get_sorted_followers(setup)
    leaders = get_sorted_leaders(setup)

    if len(followers) < 2:
        log.error("Need at least 2 follower arms, got %d", len(followers))
        sys.exit(1)

    # Generate dataset path
    now = datetime.now()
    date_folder = now.strftime("%m%d") + f"_{args.dataset_tag}"
    time_tag = now.strftime("%H%M%S")
    dataset_leaf = f"rlt_hil_{time_tag}"

    day_dir = resolve_dataset_root(setup) / date_folder
    day_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = day_dir / dataset_leaf
    dataset_name = f"local/{dataset_leaf}"

    log_file = day_dir / f"{dataset_leaf}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    log.info("=== record_rlt_hil started ===")
    log.info("Args: %s", vars(args))
    log.info("Dataset: %s -> %s", dataset_name, dataset_root)

    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)

    left_cameras, right_cameras = _build_camera_configs(setup.get("cameras", []))

    # Build teleop config for leader arms (needed for human intervention)
    teleop_argv = []
    if not args.no_teleop and len(leaders) >= 2:
        teleop_argv = [
            "--teleop.type=bi_so_leader",
            f"--teleop.left_arm_config.port={leaders[0]['port']}",
            "--teleop.left_arm_config.use_degrees=true",
            f"--teleop.right_arm_config.port={leaders[1]['port']}",
            "--teleop.right_arm_config.use_degrees=true",
            "--teleop.id=bimanual_leader",
        ]
        log.info("Teleop enabled: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])
    else:
        log.warning("Teleop disabled — human intervention not available")

    # Stage calibration files
    with TemporaryDirectory(prefix="rlt-hil-") as cal_dir:
        for side, arm in [("left", followers[0]), ("right", followers[1])]:
            serial = Path(arm["calibration_dir"]).name
            src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
            dst = Path(cal_dir) / f"bimanual_{side}.json"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                log.warning("Calibration file not found: %s", src)

        # Build sys.argv for the @parser.wrap() decorated record()
        sys.argv = [
            "record_rlt_hil",
            "--robot.type=bi_so_follower",
            "--robot.id=bimanual",
            f"--robot.calibration_dir={cal_dir}",
            f"--robot.left_arm_config.port={followers[0]['port']}",
            "--robot.left_arm_config.use_degrees=true",
            f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
            f"--robot.right_arm_config.port={followers[1]['port']}",
            "--robot.right_arm_config.use_degrees=true",
            f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
            *teleop_argv,
            f"--dataset.repo_id={dataset_name}",
            f"--dataset.root={dataset_root}",
            f"--dataset.single_task={args.task}",
            f"--dataset.num_episodes={args.num_episodes}",
            f"--dataset.episode_time_s={args.episode_time_s}",
            f"--dataset.fps={args.fps}",
            f"--dataset.vcodec={args.vcodec}",
            "--dataset.push_to_hub=false",
            # RLT config
            "--rlt.enable=true",
            f"--rlt.vla_model={args.vla_model}",
            f"--rlt.rl_token_ckpt={args.rl_token_ckpt}",
            f"--rlt.ac_ckpt={args.ac_ckpt}",
            f"--rlt.task_instruction={args.task}",
            "--rlt.phase_mode=manual",
            f"--rlt.device={args.device}",
            f"--rlt.chunk_length={args.chunk_length}",
            f"--rlt.token_pool_size={args.token_pool_size}",
            f"--rlt.deterministic={args.deterministic}",
            f"--rlt.actor_hidden_dim={args.actor_hidden_dim}",
            f"--rlt.actor_num_layers={args.actor_num_layers}",
            f"--rlt.actor_residual={args.actor_residual}",
            f"--rlt.actor_activation={args.actor_activation}",
            f"--rlt.actor_layer_norm={args.actor_layer_norm}",
            # Intervention via SPACE (configured in record() when rlt_hil_mode detected)
            "--intervention_state_machine_enabled=true",
            "--play_sounds=true",
        ]

        log.info("Calling record() with %d argv entries", len(sys.argv))
        print(f"\nDataset: {dataset_name} -> {dataset_root}")
        print(f"Log: {log_file}")
        print("RLT HIL mode: r=RL, SPACE=intervene, s=success, f=failure")
        print()

        from lerobot.scripts.lerobot_record import record
        record()

    log.info("=== record_rlt_hil finished ===")


if __name__ == "__main__":
    main()
