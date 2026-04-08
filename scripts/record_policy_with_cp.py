"""Record policy rollout with critical phase labelling.

During recording, press SPACE to mark start/end of critical phases,
S to mark success, F to mark failure.

After recording, the dataset directory contains critical_phase_intervals.json.
Use extract_cp_datasets.py separately to generate success/fail datasets.

Usage:
    PYTHONPATH=src python scripts/record_policy_with_cp.py \
        --checkpoint Elvinky/pi05_screw_271ep_sft_fp32 \
        --task "Insert the copper screw into the black sleeve" \
        --num-episodes 1
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

log = logging.getLogger(__name__)


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
    n_action_steps: int = 25,
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

    return [
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
        f"--policy.n_action_steps={n_action_steps}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={task}",
        "--dataset.push_to_hub=false",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        "--dataset.vcodec=h264",
        "--play_sounds=true",
        "--enable_critical_phase_labeling=true",
    ]


def diagnose_dataset(dataset_root: Path) -> None:
    """Log diagnostic info about the dataset state after recording."""
    log.info("--- Dataset Diagnostic ---")
    log.info("Path: %s", dataset_root)

    if not dataset_root.exists():
        log.warning("Dataset directory does not exist!")
        return

    # Check info.json
    info_json = dataset_root / "meta" / "info.json"
    if info_json.exists():
        with open(info_json) as f:
            info = json.load(f)
        log.info("total_episodes=%d, total_frames=%d", info.get("total_episodes", -1), info.get("total_frames", -1))
    else:
        log.warning("meta/info.json missing")

    # Check parquet
    data_dir = dataset_root / "data"
    if data_dir.exists():
        parquets = list(data_dir.rglob("*.parquet"))
        total_size = sum(p.stat().st_size for p in parquets)
        log.info("Parquet files: %d (%.1f MB)", len(parquets), total_size / 1e6)
    else:
        log.warning("data/ directory missing")

    # Check videos
    videos_dir = dataset_root / "videos"
    if videos_dir.exists():
        mp4s = list(videos_dir.rglob("*.mp4"))
        log.info("Video files: %d", len(mp4s))
    else:
        log.warning("videos/ directory missing (video encoding did not complete)")

    # Check raw images (present if save_episode didn't finish encoding)
    images_dir = dataset_root / "images"
    if images_dir.exists():
        png_count = sum(1 for _ in images_dir.rglob("*.png"))
        log.info("Raw PNG frames: %d (not encoded to video)", png_count)
    else:
        log.info("No leftover images/ (good: video encoding completed or no frames)")

    # Check CP intervals
    cp_json = dataset_root / "critical_phase_intervals.json"
    if cp_json.exists():
        with open(cp_json) as f:
            intervals = json.load(f)
        log.info("CP intervals: %d", len(intervals))
    else:
        log.info("No critical_phase_intervals.json")

    # Check episodes metadata
    episodes_jsonl = dataset_root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        with open(episodes_jsonl) as f:
            ep_count = sum(1 for _ in f)
        log.info("episodes.jsonl: %d entries", ep_count)
    else:
        log.warning("meta/episodes.jsonl missing (no episodes finalized)")

    log.info("--- End Diagnostic ---")


def main():
    parser = argparse.ArgumentParser(description="Record policy with critical phase labelling")
    parser.add_argument("--checkpoint", default="Elvinky/pi05_screw_271ep_sft_fp32")
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument("--n-action-steps", type=int, default=25, help="Action chunk size (default 25)")
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"

    now = datetime.now()
    date_folder = now.strftime("%m%d") + "_271ep_sft"
    time_tag = now.strftime("%H%M%S")
    dataset_leaf = f"eval_autonomy_{time_tag}"

    setup_path = args.setup_json or str(Path.home() / ".roboclaw/workspace/embodied/setup.json")
    with open(setup_path) as fh:
        setup = json.load(fh)

    ds_root_base = setup.get("datasets", {}).get("root", "")
    if not ds_root_base:
        ds_root_base = str(Path.home() / ".roboclaw/workspace/embodied/datasets")
    day_dir = Path(ds_root_base).expanduser() / date_folder
    day_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = day_dir / dataset_leaf
    dataset_name = f"local/{dataset_leaf}"

    # Setup logging to both console and file
    log_file = day_dir / f"{dataset_leaf}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file),
        ],
    )

    log.info("=== record_policy_with_cp started ===")
    log.info("Args: %s", vars(args))
    log.info("Date folder: %s, time tag: %s", date_folder, time_tag)
    log.info("Dataset: %s -> %s", dataset_name, dataset_root)
    log.info("Log file: %s", log_file)

    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)

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
            dataset_repo_id=dataset_name,
            dataset_root=dataset_root,
            task=args.task,
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
            calibration_dir=cal_dir,
            n_action_steps=args.n_action_steps,
        )

        log.info("ARGV: %s", " ".join(argv))

        print(f"\nDataset: {dataset_name} -> {dataset_root}")
        print(f"Log: {log_file}")
        print("CP labelling enabled (SPACE=toggle, S=success, F=failure)")

        log.info("=== Launching lerobot-record subprocess ===")
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"

        # Capture subprocess stdout/stderr to log file as well
        with open(log_file, "a") as lf:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
                lf.flush()
            proc.wait()

        log.info("Subprocess exit code: %d", proc.returncode)

    # Always run diagnostic, even on failure
    diagnose_dataset(dataset_root)

    if proc.returncode != 0:
        log.error("Recording failed with exit code %d", proc.returncode)
        log.info("Full log saved at: %s", log_file)
        sys.exit(proc.returncode)

    cp_json = dataset_root / "critical_phase_intervals.json"
    if cp_json.exists():
        with open(cp_json) as f:
            intervals = json.load(f)
        log.info("CP intervals: %d saved", len(intervals))
        print(f"\n[CP] {len(intervals)} critical phase intervals saved.")
        print(f"[CP] Run extract_cp_datasets.py --date {date_folder} to generate success/fail datasets.")
    else:
        log.info("No critical phase intervals marked")
        print("\n[CP] No critical phase intervals marked during recording.")

    log.info("=== record_policy_with_cp finished ===")


if __name__ == "__main__":
    main()
