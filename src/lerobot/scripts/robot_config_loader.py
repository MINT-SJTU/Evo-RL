"""Load robot + camera configuration from a JSON file.

Supports the roboclaw ``setup.json`` format and converts it into
lerobot ``BiSOFollowerConfig`` / ``SOFollowerRobotConfig`` dataclasses.

Camera aliases in the JSON use **final** feature names (e.g. ``left_wrist``).
The loader strips the ``left_`` / ``right_`` prefix automatically so that
``BiSOFollower``'s auto-prefixing produces the correct observation key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

logger = logging.getLogger(__name__)

# BiSOFollower prepends these prefixes to each arm's features.
_ARM_PREFIXES = {"left": "left_", "right": "right_"}


def _strip_arm_prefix(alias: str, prefix: str) -> str:
    """Strip the arm prefix so BiSOFollower's auto-prefix restores it."""
    if alias.startswith(prefix):
        return alias[len(prefix):]
    return alias


def _find_arm_port(arms: list[dict], side: str) -> str:
    """Find the follower port for the given side from the arms list."""
    side_lower = side.lower()
    for arm in arms:
        alias = arm.get("alias", "").lower()
        arm_type = arm.get("type", "").lower()
        if side_lower in alias and ("follower" in alias or "follower" in arm_type):
            return arm["port"]
    raise ValueError(
        f"Cannot find {side} follower arm in setup.json. "
        f"Available arms: {[a.get('alias') for a in arms]}"
    )


def _assign_camera_to_arm(
    alias: str,
    cam_cfg: OpenCVCameraConfig,
    left_cams: dict[str, OpenCVCameraConfig],
    right_cams: dict[str, OpenCVCameraConfig],
) -> None:
    """Assign a camera to left or right arm based on its alias prefix."""
    for side, prefix in _ARM_PREFIXES.items():
        if alias.startswith(prefix):
            short_name = _strip_arm_prefix(alias, prefix)
            if side == "left":
                left_cams[short_name] = cam_cfg
            else:
                right_cams[short_name] = cam_cfg
            return
    # No prefix match — default to right arm (convention: shared cameras go to right)
    right_cams[alias] = cam_cfg


def load_robot_config_from_json(path: str | Path) -> BiSOFollowerConfig:
    """Load a BiSOFollowerConfig from a roboclaw-compatible setup.json.

    Args:
        path: Path to the JSON config file.

    Returns:
        A fully populated ``BiSOFollowerConfig``.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    arms = data.get("arms", [])
    cameras = data.get("cameras", [])
    robot_id = data.get("robot_id", data.get("id"))

    left_port = _find_arm_port(arms, "left")
    right_port = _find_arm_port(arms, "right")

    left_cams: dict[str, OpenCVCameraConfig] = {}
    right_cams: dict[str, OpenCVCameraConfig] = {}

    for cam in cameras:
        alias = cam["alias"]
        cam_cfg = OpenCVCameraConfig(
            index_or_path=cam["port"],
            fps=cam.get("fps", 30),
            width=cam.get("width", 640),
            height=cam.get("height", 480),
        )
        _assign_camera_to_arm(alias, cam_cfg, left_cams, right_cams)

    logger.info(
        "Loaded robot config from %s: left_port=%s (%d cams), right_port=%s (%d cams)",
        path, left_port, len(left_cams), right_port, len(right_cams),
    )

    cfg = BiSOFollowerConfig(
        left_arm_config=SOFollowerConfig(port=left_port, cameras=left_cams),
        right_arm_config=SOFollowerConfig(port=right_port, cameras=right_cams),
    )
    if robot_id:
        cfg.id = robot_id
    return cfg
