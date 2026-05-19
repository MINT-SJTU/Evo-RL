import json
import threading
import time
from pathlib import Path

import numpy as np

from lerobot.scripts.lerobot_arx5_infer import (
    TrainRawEpisodeRecorder,
    VrRecordBridge,
    _vr_record_loop,
)

_TEST_ACTION_KEYS = (
    "joint.1",
    "joint.2",
    "joint.3",
    "joint.4",
    "joint.5",
    "joint.6",
    "gripper.pos",
)


def test_vr_record_bridge_snapshot_requires_grip_and_images():
    bridge = VrRecordBridge()
    bridge.update_snapshot(
        grip_active=False,
        action_vector=[0.0] * 7,
        state_vector=[0.0] * 7,
        images={"cam": np.zeros((4, 4, 3), dtype=np.uint8)},
        images_complete=True,
    )
    assert bridge.copy_snapshot_for_record() is None

    bridge.update_snapshot(
        grip_active=True,
        action_vector=[1.0] * 7,
        state_vector=[2.0] * 7,
        images={"cam": np.ones((4, 4, 3), dtype=np.uint8)},
        images_complete=False,
    )
    assert bridge.copy_snapshot_for_record() is None

    bridge.update_snapshot(
        grip_active=True,
        action_vector=[1.0] * 7,
        state_vector=[2.0] * 7,
        images={"cam": np.ones((4, 4, 3), dtype=np.uint8)},
        images_complete=True,
    )
    snap = bridge.copy_snapshot_for_record()
    assert snap is not None
    assert snap["action_vector"][0] == 1.0
    assert snap["images"]["cam"].shape == (4, 4, 3)


def test_vr_record_loop_respects_duration(tmp_path: Path):
    bridge = VrRecordBridge()
    recorder = TrainRawEpisodeRecorder(
        raw_record_root=tmp_path,
        action_keys=_TEST_ACTION_KEYS,
        camera_names=["cam"],
        task="test",
        dt_s=0.05,
        debug_timestamp=True,
    )
    recorder.start_episode()
    recorder.start_vr_segment()

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    bridge.update_snapshot(
        grip_active=True,
        action_vector=[float(i) for i in range(7)],
        state_vector=[float(i) for i in range(7)],
        images={"cam": image},
        images_complete=True,
    )

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_vr_record_loop,
        kwargs={
            "bridge": bridge,
            "recorder": recorder,
            "stop_event": stop_event,
            "step_duration_s": 0.05,
        },
        daemon=True,
    )
    thread.start()
    time.sleep(0.32)
    stop_event.set()
    thread.join(timeout=2.0)

    recorder.finish_active_segment()
    segment_dir = next((tmp_path / "episode_0000" / "segments").glob("segment_*_vr"))
    actions = json.loads((segment_dir / "actions.json").read_text(encoding="utf-8"))
    timestamps = json.loads((segment_dir / "timestamps.json").read_text(encoding="utf-8"))
    assert len(actions) >= 5
    assert len(timestamps) == len(actions)
    deltas = [
        timestamps[index + 1]["t_perf_s"] - timestamps[index]["t_perf_s"]
        for index in range(len(timestamps) - 1)
    ]
    assert all(0.04 <= delta <= 0.08 for delta in deltas)


def test_debug_timestamp_optional(tmp_path: Path):
    recorder = TrainRawEpisodeRecorder(
        raw_record_root=tmp_path,
        action_keys=_TEST_ACTION_KEYS,
        camera_names=["cam"],
        task="test",
        dt_s=0.1,
        debug_timestamp=False,
    )
    recorder.start_episode()
    recorder.start_policy_segment()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    action_obs = {key: float(i) for i, key in enumerate(_TEST_ACTION_KEYS)}
    recorder.record_policy_step(
        action_dict=action_obs,
        observation={**action_obs, "cam": image},
        record_timestamp=time.perf_counter(),
    )
    recorder.finish_active_segment()
    segment_dir = next((tmp_path / "episode_0000" / "segments").glob("segment_*_policy"))
    assert not (segment_dir / "timestamps.json").exists()
