import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from lerobot.scripts.lerobot_arx5_infer import _poll_vr_camera_frames, _warmup_vr_cameras_after_start


def test_poll_vr_camera_frames_maps_serial_to_slot():
    controller = SimpleNamespace(
        camera_interface=MagicMock(),
        camera_serial_to_name={"111": "base", "222": "left_wrist"},
    )
    controller.camera_interface.get_frames.return_value = {
        "111": {"color": np.zeros((4, 4, 3), dtype=np.uint8)},
        "222": {"color": np.ones((4, 4, 3), dtype=np.uint8)},
    }
    images = _poll_vr_camera_frames(controller, camera_names=["base", "left_wrist"])
    assert set(images) == {"base", "left_wrist"}


def test_poll_vr_camera_frames_applies_teleop_to_local():
    controller = SimpleNamespace(
        camera_interface=MagicMock(),
        camera_serial_to_name={"111": "base"},
    )
    controller.camera_interface.get_frames.return_value = {
        "111": {"color": np.zeros((4, 4, 3), dtype=np.uint8)},
    }
    images = _poll_vr_camera_frames(
        controller,
        camera_names=["side"],
        teleop_to_local={"base": "side"},
    )
    assert "side" in images


def test_warmup_skipped_when_zero():
    controller = SimpleNamespace(
        camera_interface=MagicMock(),
        camera_fps=30,
        _last_vr_images={"stale": np.zeros((2, 2, 3), dtype=np.uint8)},
    )
    _warmup_vr_cameras_after_start(controller, warmup_s=0.0, camera_names=["base"])
    controller.camera_interface.get_frames.assert_not_called()
    assert controller._vr_arm_control_enabled is True
    assert "stale" in controller._last_vr_images


def test_warmup_polls_and_clears_cache():
    controller = SimpleNamespace(
        camera_interface=MagicMock(),
        camera_serial_to_name={"111": "base"},
        camera_fps=100,
        _last_vr_images={},
    )
    controller.camera_interface.get_frames.return_value = {
        "111": {"color": np.zeros((4, 4, 3), dtype=np.uint8)},
    }
    t0 = time.perf_counter()
    _warmup_vr_cameras_after_start(controller, warmup_s=0.05, camera_names=["base"])
    elapsed = time.perf_counter() - t0
    assert controller.camera_interface.get_frames.call_count >= 1
    assert controller._last_vr_images == {}
    assert controller._vr_arm_control_enabled is True
    assert elapsed >= 0.04
