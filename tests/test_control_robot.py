#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lerobot.datasets.dataset_tools import merge_datasets, remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_human_inloop_record import (
    _HumanInloopFailureResetController,
    _load_failure_reset_pose,
    _save_failure_reset_pose,
    _slow_reset_all_arms_to_pose,
    human_inloop_record,
)
from lerobot.scripts.lerobot_patch_hil_dataset_schema import PatchHilDatasetSchemaConfig, patch_hil_dataset_schema
from lerobot.scripts.lerobot_record import (
    ACPInferenceConfig,
    DatasetRecordConfig,
    PolicySyncDualArmExecutor,
    RecordConfig,
    _capture_policy_runtime_state,
    _predict_policy_action_with_acp_inference,
    record,
    record_loop,
)
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.utils.recording_annotations import EPISODE_SUCCESS
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_adds_episode_success_and_collector_policy_id(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    root = tmp_path / "record_with_annotations"
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=root,
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
        enable_episode_outcome_labeling=True,
        default_episode_success="failure",
        enable_collector_policy_id=True,
    )

    dataset = record(cfg)
    assert "complementary_info.collector_policy_id" in dataset.features

    reloaded = LeRobotDataset(DUMMY_REPO_ID, root=root)
    assert reloaded[0]["complementary_info.collector_policy_id"] == "human"
    assert "episode_success" in reloaded.meta.episodes.column_names
    assert reloaded.meta.episodes[0]["episode_success"] == "failure"


def test_human_inloop_record_works_without_policy_and_saves_annotations(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    root = tmp_path / "hil_no_policy"
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=root,
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = human_inloop_record(cfg)
    assert cfg.intervention_state_machine_enabled is False
    assert cfg.collector_policy_id_policy == "human"
    assert "complementary_info.collector_policy_id" in dataset.features
    assert "complementary_info.policy_action" in dataset.features
    assert "complementary_info.is_intervention" in dataset.features
    assert "complementary_info.state" in dataset.features

    reloaded = LeRobotDataset(DUMMY_REPO_ID, root=root)
    assert reloaded[0]["complementary_info.collector_policy_id"] == "human"
    torch.testing.assert_close(
        reloaded[0]["complementary_info.policy_action"],
        torch.zeros_like(reloaded[0]["action"]),
    )
    assert float(reloaded[0]["complementary_info.is_intervention"]) == 0.0
    assert float(reloaded[0]["complementary_info.state"]) == 0.0
    assert "episode_success" in reloaded.meta.episodes.column_names
    assert reloaded.meta.episodes[0]["episode_success"] == "failure"


def test_patch_hil_dataset_schema_restores_legacy_dataset_mergeability(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    current_root = tmp_path / "hil_current"
    legacy_root = tmp_path / "hil_legacy"
    patched_root = tmp_path / "hil_patched"
    merged_root = tmp_path / "hil_merged"
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=current_root,
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    current_dataset = human_inloop_record(cfg)
    legacy_dataset = remove_feature(
        current_dataset,
        feature_names=[
            "complementary_info.policy_action",
            "complementary_info.is_intervention",
            "complementary_info.state",
        ],
        output_dir=legacy_root,
        repo_id="dummy/repo_legacy",
    )
    assert "complementary_info.policy_action" not in legacy_dataset.features

    patched_dataset = patch_hil_dataset_schema(
        PatchHilDatasetSchemaConfig(
            repo_id="dummy/repo_legacy",
            root=str(legacy_root),
            output_repo_id="dummy/repo_patched",
            output_dir=str(patched_root),
        )
    )

    assert "complementary_info.policy_action" in patched_dataset.features
    assert "complementary_info.is_intervention" in patched_dataset.features
    assert "complementary_info.state" in patched_dataset.features
    patched_reloaded = LeRobotDataset("dummy/repo_patched", root=patched_root)
    torch.testing.assert_close(
        patched_reloaded[0]["complementary_info.policy_action"],
        torch.zeros_like(patched_reloaded[0]["action"]),
    )
    assert float(patched_reloaded[0]["complementary_info.is_intervention"]) == 0.0
    assert float(patched_reloaded[0]["complementary_info.state"]) == 0.0

    merged_dataset = merge_datasets(
        datasets=[current_dataset, patched_dataset],
        output_repo_id="dummy/repo_merged",
        output_dir=merged_root,
    )
    assert merged_dataset.meta.total_episodes == current_dataset.meta.total_episodes + patched_dataset.meta.total_episodes


def test_record_loop_sets_leader_manual_control_during_reset():
    class MockTeleopWithManualControl(MockTeleop):
        def __init__(self, config):
            super().__init__(config)
            self.manual_control_calls = []

        def set_manual_control(self, enabled: bool) -> None:
            self.manual_control_calls.append(enabled)

    robot = MockRobot(MockRobotConfig())
    teleop = MockTeleopWithManualControl(MockTeleopConfig())
    robot.connect()
    teleop.connect()
    try:
        record_loop(
            robot=robot,
            events={
                "exit_early": True,
                "rerecord_episode": False,
                "stop_recording": False,
                "toggle_intervention": False,
                "episode_outcome": None,
            },
            fps=30,
            teleop_action_processor=lambda x: x[0],
            robot_action_processor=lambda x: x[0],
            robot_observation_processor=lambda x: x,
            teleop=teleop,
            policy=None,
            control_time_s=0.1,
        )
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()

    assert teleop.manual_control_calls == [True]


def test_save_and_load_failure_reset_pose(tmp_path):
    robot = MockRobot(MockRobotConfig(n_motors=2, random_values=False, static_values=[12.5, -3.0]))
    robot.connect()
    pose_path = tmp_path / "failure_reset_pose.json"

    try:
        saved_pose = _save_failure_reset_pose(robot=robot, pose_path=pose_path)
    finally:
        if robot.is_connected:
            robot.disconnect()

    with open(pose_path) as f:
        payload = json.load(f)
    assert saved_pose == {"motor_1.pos": 12.5, "motor_2.pos": -3.0}
    assert payload["joint_pos"] == {"motor_1.pos": 12.5, "motor_2.pos": -3.0}


def test_load_failure_reset_pose_from_json(tmp_path):
    pose_path = tmp_path / "failure_reset_pose.json"
    payload = {
        "robot_type": "mock_robot",
        "joint_pos": {
            "motor_1.pos": 12.5,
            "motor_2.pos": -3.0,
            "non_joint_key": 999,
        },
    }
    with open(pose_path, "w") as f:
        json.dump(payload, f)

    loaded_pose = _load_failure_reset_pose(pose_path)
    assert loaded_pose == {"motor_1.pos": 12.5, "motor_2.pos": -3.0}


def test_human_inloop_failure_reset_controller_reuses_existing_pose(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "hil_with_policy",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    controller = _HumanInloopFailureResetController(cfg)
    controller.pose_path = tmp_path / "existing_failure_reset_pose.json"
    with open(controller.pose_path, "w") as f:
        json.dump({"joint_pos": {"motor_1.pos": 1.0, "motor_2.pos": -2.0}}, f)

    with (
        patch("builtins.input") as mock_input,
        patch("lerobot.scripts.lerobot_human_inloop_record._save_failure_reset_pose") as mock_save,
    ):
        controller.on_record_connected(robot=MagicMock(), teleop=MagicMock())

    assert controller.failure_reset_pose == {"motor_1.pos": 1.0, "motor_2.pos": -2.0}
    mock_input.assert_not_called()
    mock_save.assert_not_called()


def test_human_inloop_failure_reset_controller_resets_on_success(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "hil_with_policy_success_reset",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    controller = _HumanInloopFailureResetController(cfg)
    controller.failure_reset_pose = {"motor_1.pos": 1.0, "motor_2.pos": -2.0}

    with patch("lerobot.scripts.lerobot_human_inloop_record._slow_reset_all_arms_to_pose") as mock_reset:
        controller.on_episode_outcome(robot=MagicMock(), teleop=MagicMock(), episode_success=EPISODE_SUCCESS)

    mock_reset.assert_called_once()
    assert mock_reset.call_args.kwargs["target_pose"] == {"motor_1.pos": 1.0, "motor_2.pos": -2.0}


def test_slow_reset_all_arms_to_pose_uses_interpolation():
    robot = MockRobot(MockRobotConfig(n_motors=2, random_values=False, static_values=[0.0, 0.0]))
    teleop = MagicMock()
    robot.connect()
    robot.send_action = MagicMock(wraps=robot.send_action)
    target_pose = {"motor_1.pos": 11.0, "motor_2.pos": -22.0}

    try:
        _slow_reset_all_arms_to_pose(
            robot=robot,
            teleop=teleop,
            target_pose=target_pose,
            duration_s=0.2,
        )
    finally:
        if robot.is_connected:
            robot.disconnect()

    final_action = robot.send_action.call_args_list[-1].args[0]
    assert final_action == {"motor_1.pos": 11.0, "motor_2.pos": -22.0}
    assert robot.send_action.call_count > 1
    teleop.set_manual_control.assert_called_once_with(False)
    teleop.send_feedback.assert_called()


def test_record_and_replay(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)


def test_policy_sync_dual_arm_executor():
    robot = MagicMock()
    robot.send_action.return_value = {"motor_1.pos": 10.0}
    teleop = MagicMock()

    executor = PolicySyncDualArmExecutor(robot=robot, teleop=teleop, parallel_dispatch=True)
    action = {"motor_1.pos": 10.0}
    sent_action = executor.send_action(action)
    executor.shutdown()

    assert sent_action == action
    robot.send_action.assert_called_once_with(action)
    teleop.send_feedback.assert_called_once_with(action)


def test_record_config_rejects_cfg_without_acp_enable():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )

    with pytest.raises(ValueError, match="acp_inference.use_cfg=true"):
        RecordConfig(
            robot=robot_cfg,
            dataset=dataset_cfg,
            teleop=teleop_cfg,
            play_sounds=False,
            acp_inference=ACPInferenceConfig(enable=False, use_cfg=True, cfg_beta=0.6),
        )


def test_record_config_rejects_negative_cfg_beta():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )

    with pytest.raises(ValueError, match="cfg_beta"):
        RecordConfig(
            robot=robot_cfg,
            dataset=dataset_cfg,
            teleop=teleop_cfg,
            play_sounds=False,
            acp_inference=ACPInferenceConfig(enable=True, use_cfg=False, cfg_beta=-0.1),
        )


def test_acp_inference_without_cfg_appends_positive_prompt():
    class _StaticPolicy:
        def __init__(self, value: float):
            self.value = value
            self.tasks = []

        def select_action(self, batch):
            self.tasks.append(batch["task"])
            return torch.tensor([[self.value, self.value, self.value]], dtype=torch.float32)

    observation_frame = {"observation.state": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    policy = _StaticPolicy(value=2.0)

    action = _predict_policy_action_with_acp_inference(
        observation_frame=observation_frame,
        policy=policy,
        device=torch.device("cpu"),
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        use_amp=False,
        task="Pick and place",
        robot_type="mock_robot",
        acp_inference=ACPInferenceConfig(enable=True, use_cfg=False, cfg_beta=0.6),
    )

    assert torch.allclose(action, torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32))
    assert policy.tasks[-1] == "Pick and place\nAdvantage: positive"


def test_acp_inference_with_cfg_blends_cond_and_uncond_actions():
    class _StaticPolicy:
        def __init__(self):
            self.tasks = []

        def select_action(self, batch):
            self.tasks.append(batch["task"])
            value = 3.0 if "Advantage: positive" in batch["task"] else 1.0
            return torch.tensor([[value, value, value]], dtype=torch.float32)

    observation_frame = {"observation.state": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    policy = _StaticPolicy()
    cond_state = {}
    uncond_state = {}

    action = _predict_policy_action_with_acp_inference(
        observation_frame=observation_frame,
        policy=policy,
        device=torch.device("cpu"),
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        use_amp=False,
        task="Pick and place",
        robot_type="mock_robot",
        acp_inference=ACPInferenceConfig(enable=True, use_cfg=True, cfg_beta=0.5),
        cond_runtime_state=cond_state,
        uncond_runtime_state=uncond_state,
    )

    assert torch.allclose(action, torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32))
    assert policy.tasks == [
        "Pick and place\nAdvantage: positive",
        "Pick and place",
    ]


def test_acp_inference_with_cfg_uses_isolated_branch_queues():
    class _QueuePolicy:
        def __init__(self):
            self._action_queue = deque(maxlen=2)

        def select_action(self, batch):
            if len(self._action_queue) == 0:
                base = 10.0 if "Advantage: positive" in batch["task"] else 0.0
                self._action_queue.extend(
                    [
                        torch.tensor([[base + 1.0, base + 1.0, base + 1.0]], dtype=torch.float32),
                        torch.tensor([[base + 2.0, base + 2.0, base + 2.0]], dtype=torch.float32),
                    ]
                )
            return self._action_queue.popleft()

    observation_frame = {"observation.state": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    policy = _QueuePolicy()
    cond_state = _capture_policy_runtime_state(policy)
    uncond_state = _capture_policy_runtime_state(policy)

    action_1 = _predict_policy_action_with_acp_inference(
        observation_frame=observation_frame,
        policy=policy,
        device=torch.device("cpu"),
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        use_amp=False,
        task="Pick and place",
        robot_type="mock_robot",
        acp_inference=ACPInferenceConfig(enable=True, use_cfg=True, cfg_beta=0.5),
        cond_runtime_state=cond_state,
        uncond_runtime_state=uncond_state,
    )

    action_2 = _predict_policy_action_with_acp_inference(
        observation_frame=observation_frame,
        policy=policy,
        device=torch.device("cpu"),
        preprocessor=lambda x: x,
        postprocessor=lambda x: x,
        use_amp=False,
        task="Pick and place",
        robot_type="mock_robot",
        acp_inference=ACPInferenceConfig(enable=True, use_cfg=True, cfg_beta=0.5),
        cond_runtime_state=cond_state,
        uncond_runtime_state=uncond_state,
    )

    assert torch.allclose(action_1, torch.tensor([[6.0, 6.0, 6.0]], dtype=torch.float32))
    assert torch.allclose(action_2, torch.tensor([[7.0, 7.0, 7.0]], dtype=torch.float32))
