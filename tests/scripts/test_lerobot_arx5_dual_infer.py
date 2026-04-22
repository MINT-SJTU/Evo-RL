from types import SimpleNamespace

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType
from lerobot.scripts.lerobot_arx5_dual_infer import (
    LoopState,
    STATE_KEYS,
    _clip_safe_actions,
    _serialize_dual_arm_targets,
    _run_vr_teleop_session,
    _is_any_vr_arm_active,
    _make_camera_configs,
    _required_image_slot_names,
    _run_keyboard_command,
    _visual_image_slot_names,
)


def _visual_feature():
    return SimpleNamespace(type=FeatureType.VISUAL)


def test_dual_image_slots_ignore_empty_camera_padding():
    policy_cfg = SimpleNamespace(
        input_features={
            "observation.images.base": _visual_feature(),
            "observation.images.left_wrist": _visual_feature(),
            "observation.images.empty_camera_0": _visual_feature(),
            "observation.images.right_wrist": _visual_feature(),
        }
    )

    assert _visual_image_slot_names(policy_cfg) == ["base", "left_wrist", "empty_camera_0", "right_wrist"]
    assert _required_image_slot_names(policy_cfg) == ["base", "left_wrist", "right_wrist"]


def test_dual_camera_configs_build_usb_slots_and_flip_selected_camera():
    configs = _make_camera_configs(
        required_image_slot_names=["base", "left_wrist", "right_wrist"],
        camera_specs={"base": "0", "left_wrist": "1", "right_wrist": "2"},
        use_usb_cams=True,
        width=640,
        height=480,
        fps=30,
        flipped_cameras={"right_wrist"},
    )

    assert list(configs) == ["base", "left_wrist", "right_wrist"]
    assert all(isinstance(config, OpenCVCameraConfig) for config in configs.values())
    assert configs["right_wrist"].rotation == 180
    assert configs["base"].rotation == 0


def test_safe_mode_clips_joint_steps_for_both_arms_without_touching_grippers():
    current_state = [0.0] * 14
    current_state[6] = 1.5
    current_state[13] = 2.5
    action = {key: 0.0 for key in STATE_KEYS}
    action["state.0"] = 0.08
    action["state.7"] = -0.07
    action["state.6"] = 3.0
    action["state.13"] = 4.0

    clipped = _clip_safe_actions([action], current_state=current_state, max_joint_step=0.02)

    assert clipped[0]["state.0"] == 0.02
    assert clipped[0]["state.7"] == -0.02
    assert clipped[0]["state.6"] == 3.0
    assert clipped[0]["state.13"] == 4.0


def test_v_from_stopped_only_requests_vr_mode_without_marking_takeover():
    class RecorderStub:
        recording_active = True
        waiting_for_success_label = False

        def __init__(self):
            self.mark_calls = 0

        def handle_success_label_hotkey(self, key: str) -> bool:
            return False

        def mark_vr_takeover(self) -> None:
            self.mark_calls += 1

    recorder = RecorderStub()
    state, request_next_chunk, running, vr_requested = _run_keyboard_command(
        key="v",
        left_arm=object(),
        right_arm=object(),
        state=LoopState.STOPPED,
        safe_mode=False,
        request_next_chunk=False,
        recorder=recorder,
    )

    assert state == LoopState.STOPPED
    assert request_next_chunk is False
    assert running is True
    assert vr_requested is True
    assert recorder.mark_calls == 0


def test_serialize_dual_arm_targets_matches_dataset_order():
    q_cmd_by_arm = {
        "left_arm": [101, 102, 103, 104, 105, 106],
        "right_arm": [1, 2, 3, 4, 5, 6],
    }
    gripper_by_arm = {
        "left_arm": 107,
        "right_arm": 7,
    }

    assert _serialize_dual_arm_targets(q_cmd_by_arm, gripper_by_arm) == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
    ]
