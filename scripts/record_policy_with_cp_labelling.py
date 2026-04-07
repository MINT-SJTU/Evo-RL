"""Record policy rollout with critical phase labelling.

Saves two datasets:
  - Full autonomy trajectory: 271ep_sft_autonomy_<timestamp>
  - Critical phase segments:  271ep_sft_cp_<timestamp>

During recording, press SPACE to mark start/end of critical phases.
After recording, marked segments are extracted into the CP dataset.

Usage:
    PYTHONPATH=src python scripts/record_policy_with_cp_labelling.py \
        --checkpoint Elvinky/pi05_screw_271ep_sft_fp32 \
        --task "Insert the copper screw into the black sleeve" \
        --num-episodes 1 --save-critical-phase
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory


def build_lerobot_record_argv(
    python: str,
    checkpoint: str,
    robot_setup: dict,
    dataset_repo_id: str,
    dataset_root: Path,
    task: str,
    num_episodes: int,
    episode_time_s: int,
    calibration_dir: str,
    enable_critical_phase: bool = False,
) -> list[str]:
    """Build the argv for lerobot-record subprocess."""
    arms = robot_setup["arms"]
    followers = [a for a in arms if "follower" in a["type"]]
    followers.sort(key=lambda a: (0 if "left" in a.get("alias", "") else 1))

    CAM_RENAME = {"left_wrist": "wrist", "right_wrist": "wrist", "top": "front"}
    LEFT_CAMS = {"left_wrist"}
    RIGHT_CAMS = {"right_wrist", "top"}

    left_cameras, right_cameras = {}, {}
    for cam in robot_setup.get("cameras", []):
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

    argv = [
        python, "-m", "lerobot.scripts.lerobot_record",
        "--robot.type=bi_so_follower",
        "--robot.id=bimanual",
        f"--robot.calibration_dir={calibration_dir}",
        f"--robot.left_arm_config.port={followers[0]['port']}",
        "--robot.left_arm_config.use_degrees=true",
        f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
        f"--robot.right_arm_config.port={followers[1]['port']}",
        "--robot.right_arm_config.use_degrees=true",
        f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
        f"--policy.path={checkpoint}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={task}",
        "--dataset.push_to_hub=false",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        "--play_sounds=false",
    ]
    if enable_critical_phase:
        argv.append("--enable_critical_phase_labeling=true")
        argv.append("--critical_phase_toggle_key= ")
    return argv


def main():
    parser = argparse.ArgumentParser(description="Record policy with critical phase labelling")
    parser.add_argument("--checkpoint", default="Elvinky/pi05_screw_271ep_sft_fp32")
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=int, default=500)
    parser.add_argument("--save-critical-phase", action="store_true")
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["HF_HUB_OFFLINE"] = "1"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_name = f"local/eval_271ep_sft_autonomy_{timestamp}"
    cp_name = f"local/eval_271ep_sft_cp_{timestamp}"

    setup_path = args.setup_json or str(Path.home() / ".roboclaw/workspace/embodied/setup.json")
    with open(setup_path) as fh:
        setup = json.load(fh)

    ds_root_base = setup.get("datasets", {}).get("root", "")
    if not ds_root_base:
        ds_root_base = str(Path.home() / ".roboclaw/workspace/embodied/datasets")
    full_ds_root = Path(ds_root_base).expanduser() / full_name.split("/", 1)[1]

    if full_ds_root.exists():
        shutil.rmtree(full_ds_root)

    arms = setup["arms"]
    followers = [a for a in arms if "follower" in a["type"]]
    followers.sort(key=lambda a: (0 if "left" in a.get("alias", "") else 1))

    with TemporaryDirectory(prefix="cp-labelling-") as cal_dir:
        for side, arm in [("left", followers[0]), ("right", followers[1])]:
            serial = Path(arm["calibration_dir"]).name
            src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
            dst = Path(cal_dir) / f"bimanual_{side}.json"
            shutil.copy2(src, dst)

        argv = build_lerobot_record_argv(
            python=args.python,
            checkpoint=args.checkpoint,
            robot_setup=setup,
            dataset_repo_id=full_name,
            dataset_root=full_ds_root,
            task=args.task,
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
            calibration_dir=cal_dir,
            enable_critical_phase=args.save_critical_phase,
        )

        print(f"\nFull dataset: {full_name} -> {full_ds_root}")
        if args.save_critical_phase:
            print(f"CP dataset:   {cp_name}")
        print("\n=== ARGV ===")
        for a in argv:
            print(f"  {a}")

        print("\n=== RECORDING ===", flush=True)
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        proc = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr, env=env)
        proc.wait()
        print(f"\nRecording exit code: {proc.returncode}")

    if not args.save_critical_phase or proc.returncode != 0:
        return

    cp_json = full_ds_root / "critical_phase_intervals.json"
    if not cp_json.exists():
        print("[CP] No critical_phase_intervals.json found — no phases were marked.")
        return

    with open(cp_json) as f:
        raw_intervals = json.load(f)
    intervals = [(iv["episode_index"], iv["start_frame"], iv["end_frame"]) for iv in raw_intervals]

    if not intervals:
        print("[CP] No critical phase intervals recorded.")
        return

    cp_ds_root = Path(ds_root_base).expanduser() / cp_name.split("/", 1)[1]

    from lerobot.utils.critical_phase_extraction import extract_critical_phase_dataset

    extract_critical_phase_dataset(
        source_repo_id=full_name,
        source_root=full_ds_root,
        output_repo_id=cp_name,
        output_root=cp_ds_root,
        intervals=intervals,
        task=args.task,
    )
    print(f"\n[CP] Done! CP dataset at: {cp_ds_root}")


if __name__ == "__main__":
    main()
