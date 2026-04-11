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
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dataset import diagnose_dataset as _diagnose
from scripts.dataset.setup_helpers import (
    get_sorted_followers,
    load_setup_json,
    resolve_dataset_root,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record policy with critical phase labelling")
    parser.add_argument("--checkpoint", default="Elvinky/pi05_screw_271ep_sft_fp32")
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument("--n-action-steps", type=int, default=25, help="Action chunk size (default 25)")
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--folder-suffix", default=None, help="Date-folder suffix (default: derived from checkpoint)")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _derive_folder_suffix(checkpoint: str) -> str:
    """Extract a short suffix from checkpoint name for folder naming.

    Example: 'Elvinky/pi05_screw_271ep_sft_fp32' -> '271ep_sft'
    """
    name = checkpoint.rsplit("/", maxsplit=1)[-1]
    return name.removeprefix("pi05_screw_").removesuffix("_fp32").removesuffix("_fp16")


def resolve_paths(setup: dict, folder_suffix: str) -> tuple[Path, Path, str, str]:
    """Return (day_dir, dataset_root, dataset_name, date_folder)."""
    now = datetime.now()
    date_folder = f"{now:%m%d}_{folder_suffix}"
    dataset_leaf = f"eval_autonomy_{now:%H%M%S}"
    day_dir = resolve_dataset_root(setup) / date_folder
    day_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = day_dir / dataset_leaf
    dataset_name = f"local/{dataset_leaf}"
    return day_dir, dataset_root, dataset_name, date_folder


# ---------------------------------------------------------------------------
# Calibration staging
# ---------------------------------------------------------------------------


def stage_calibration_files(followers: list[dict], cal_dir: Path) -> None:
    """Copy per-arm calibration JSONs into *cal_dir* with bimanual naming."""
    if len(followers) != 2:
        raise ValueError(f"Expected 2 followers, got {len(followers)}")
    for side, arm in zip(("left", "right"), followers, strict=True):
        cal_path = Path(arm["calibration_dir"]).expanduser()
        src = cal_path / f"{cal_path.name}.json"
        dst = cal_dir / f"bimanual_{side}.json"
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Camera config
# ---------------------------------------------------------------------------

CAM_RENAME = {"left_wrist": "wrist", "right_wrist": "wrist", "top": "front"}
LEFT_ALIASES = {"left_wrist"}
RIGHT_ALIASES = {"right_wrist", "top"}


def build_camera_configs(cameras: list[dict]) -> tuple[dict, dict]:
    """Parse raw camera dicts into (left_cameras, right_cameras) config maps."""
    left: dict = {}
    right: dict = {}
    for cam in cameras:
        alias = cam["alias"]
        cam_cfg: dict = {
            "type": "opencv",
            "index_or_path": cam["port"],
            "width": cam.get("width", 640),
            "height": cam.get("height", 480),
            "fps": cam.get("fps", 30),
        }
        if cam.get("fourcc"):
            cam_cfg["fourcc"] = cam["fourcc"]
        if alias in LEFT_ALIASES:
            left[CAM_RENAME.get(alias, alias)] = cam_cfg
        elif alias in RIGHT_ALIASES:
            right[CAM_RENAME.get(alias, alias)] = cam_cfg
    return left, right


# ---------------------------------------------------------------------------
# Argv builder
# ---------------------------------------------------------------------------


def build_lerobot_record_argv(
    python: str,
    checkpoint: str,
    left_port: str,
    right_port: str,
    left_cameras: dict,
    right_cameras: dict,
    calibration_dir: str,
    dataset_repo_id: str,
    dataset_root: Path,
    task: str,
    num_episodes: int,
    episode_time_s: int,
    n_action_steps: int = 25,
) -> list[str]:
    """Build the argv for lerobot-record subprocess."""
    return [
        python, "-m", "lerobot.scripts.lerobot_record",
        "--robot.type=bi_so_follower",
        "--robot.id=bimanual",
        f"--robot.calibration_dir={calibration_dir}",
        f"--robot.left_arm_config.port={left_port}",
        "--robot.left_arm_config.use_degrees=true",
        f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
        f"--robot.right_arm_config.port={right_port}",
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


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def run_subprocess_with_log(argv: list[str], log_file: Path, env: dict) -> int:
    """Launch *argv*, tee stdout to console and *log_file*, handle SIGINT gracefully."""
    sigint_count = 0
    proc: subprocess.Popen[str] | None = None

    def _on_sigint(_signum: int, _frame: object) -> None:
        nonlocal sigint_count
        sigint_count += 1
        if sigint_count == 1:
            log.info("Ctrl+C — waiting for subprocess to save data...")
            print("\n[Ctrl+C] Waiting for save to complete (press again to force quit)...")
            return
        if proc is not None and proc.poll() is None:
            log.warning("Force killing subprocess")
            proc.kill()

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        with open(log_file, "a") as lf:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, text=True, bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
                lf.flush()
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        log.warning("Subprocess did not exit in 60s, killing")
        proc.kill()
        proc.wait()
    except BrokenPipeError:
        log.debug("Subprocess pipe closed")
    finally:
        signal.signal(signal.SIGINT, prev_handler)
    log.info("Subprocess exit code: %d", proc.returncode)
    return proc.returncode


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"

    setup = load_setup_json(args.setup_json)
    suffix = args.folder_suffix or _derive_folder_suffix(args.checkpoint)
    day_dir, dataset_root, dataset_name, date_folder = resolve_paths(setup, suffix)
    followers = get_sorted_followers(setup)
    left_cams, right_cams = build_camera_configs(setup.get("cameras", []))

    log_file = day_dir / f"{dataset_root.name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(log_file)],
        force=True,
    )
    log.info("=== record_policy_with_cp started ===")
    log.info("Args: %s", vars(args))
    log.info("Dataset: %s -> %s", dataset_name, dataset_root)

    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)

    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    print(f"\nDataset: {dataset_name} -> {dataset_root}")
    print(f"Log: {log_file}")
    print("CP labelling enabled (SPACE=toggle, S=success, F=failure)")

    with TemporaryDirectory(prefix="cp-labelling-") as cal_dir:
        stage_calibration_files(followers, Path(cal_dir))
        argv = build_lerobot_record_argv(
            python=args.python,
            checkpoint=args.checkpoint,
            left_port=followers[0]["port"],
            right_port=followers[1]["port"],
            left_cameras=left_cams,
            right_cameras=right_cams,
            calibration_dir=cal_dir,
            dataset_repo_id=dataset_name,
            dataset_root=dataset_root,
            task=args.task,
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
            n_action_steps=args.n_action_steps,
        )
        log.info("ARGV: %s", " ".join(argv))
        returncode = run_subprocess_with_log(argv, log_file, env)

    result = _diagnose(dataset_root)
    log.info("Diagnosis: %s (repairable=%s)", result.damage_type.value, result.repairable)

    if returncode != 0:
        log.error("Recording failed with exit code %d", returncode)
        sys.exit(returncode)

    if result.details["has_cp"]:
        with open(dataset_root / "critical_phase_intervals.json") as f:
            n_intervals = len(json.load(f))
        log.info("CP intervals: %d saved", n_intervals)
        print(f"\n[CP] {n_intervals} critical phase intervals saved.")
        print(f"[CP] Run extract_cp_datasets.py --date {date_folder} to generate success/fail datasets.")
    else:
        print("\n[CP] No critical phase intervals marked during recording.")

    log.info("=== record_policy_with_cp finished ===")


if __name__ == "__main__":
    main()
