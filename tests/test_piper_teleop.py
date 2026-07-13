# ruff: noqa: N802
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import lerobot.robots.piper_follower.piper_follower as piper_follower_module
import lerobot.teleoperators.bi_piper_leader.bi_piper_leader as bi_piper_leader_module
import lerobot.teleoperators.piper_leader.piper_leader as piper_leader_module
import lerobot.utils.piper_sdk as piper_sdk_utils
from lerobot.motors import MotorCalibration
from lerobot.processor import make_default_processors
from lerobot.robots.bi_piper_follower import (
    BiPiperFollower,
    BiPiperFollowerConfig,
    BiPiperXFollower,
    BiPiperXFollowerConfig,
)
from lerobot.robots.piper_follower import (
    PiperFollower,
    PiperFollowerConfig,
    PiperFollowerConfigBase,
    PiperXFollower,
    PiperXFollowerConfig,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.teleoperators.bi_piper_leader import (
    BiPiperLeader,
    BiPiperLeaderConfig,
    BiPiperXLeader,
    BiPiperXLeaderConfig,
)
from lerobot.teleoperators.piper_leader import (
    PiperLeader,
    PiperLeaderConfig,
    PiperLeaderConfigBase,
    PiperXLeader,
    PiperXLeaderConfig,
    PiperXLeaderConfigBase,
)
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS


class FakeLogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SILENT = "SILENT"


class FakePiperInterface:
    def __init__(self, can_name, judge_flag=False, can_auto_init=True, logger_level=None):
        self.can_name = can_name
        self.judge_flag = judge_flag
        self.can_auto_init = can_auto_init
        self.logger_level = logger_level
        self.connected = False
        self.mode_commands = []
        self.role_commands = []
        self.last_joint = None
        self.last_gripper = None
        self.gripper_calls = []
        self.joint_mit_calls = []
        self.enable_calls = 0
        self.disable_calls = 0
        self.is_enabled = False
        self.connect_calls = []

        self._joint_ctrl = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            joint_ctrl=SimpleNamespace(
                joint_1=10000,
                joint_2=20000,
                joint_3=30000,
                joint_4=40000,
                joint_5=50000,
                joint_6=60000,
            ),
        )
        self._joint_state = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            joint_state=SimpleNamespace(
                joint_1=11000,
                joint_2=21000,
                joint_3=31000,
                joint_4=41000,
                joint_5=51000,
                joint_6=61000,
            ),
        )
        self._gripper_ctrl = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            gripper_ctrl=SimpleNamespace(grippers_angle=42000, grippers_effort=1500, status_code=0x01),
        )
        self._gripper_state = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            gripper_state=SimpleNamespace(grippers_angle=43000, grippers_effort=1400, status_code=0x01),
        )
        self._high_spd = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            motor_1=SimpleNamespace(motor_speed=0),
            motor_2=SimpleNamespace(motor_speed=0),
            motor_3=SimpleNamespace(motor_speed=0),
            motor_4=SimpleNamespace(motor_speed=0),
            motor_5=SimpleNamespace(motor_speed=0),
            motor_6=SimpleNamespace(motor_speed=0),
        )
        self._arm_status = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            arm_status=SimpleNamespace(ctrl_mode=0x01),
        )

    def ConnectPort(self, can_init=False, piper_init=True, start_thread=True):
        self.connect_calls.append(
            {"can_init": can_init, "piper_init": piper_init, "start_thread": start_thread}
        )
        self.connected = True

    def DisconnectPort(self, thread_timeout=0.1):
        del thread_timeout
        self.connected = False

    def MotionCtrl_2(self, *args):
        self.mode_commands.append(args)

    def MasterSlaveConfig(self, *args):
        self.role_commands.append(args)
        if args and args[0] == 0xFC:
            self._arm_status.arm_status.ctrl_mode = 0x01

    def EnablePiper(self):
        self.enable_calls += 1
        self.is_enabled = True
        return True

    def DisableArm(self, motor_num):
        del motor_num
        self.disable_calls += 1
        self.is_enabled = False

    def JointCtrl(self, *args):
        self.last_joint = args

    def JointMitCtrl(self, *args):
        self.joint_mit_calls.append(args)

    def GripperCtrl(self, *args):
        self.last_gripper = args
        self.gripper_calls.append(args)

    def GetArmJointCtrl(self):
        return self._joint_ctrl

    def GetArmJointMsgs(self):
        return self._joint_state

    def GetArmGripperCtrl(self):
        return self._gripper_ctrl

    def GetArmGripperMsgs(self):
        return self._gripper_state

    def GetArmHighSpdInfoMsgs(self):
        return self._high_spd

    def GetArmStatus(self):
        return self._arm_status


def patch_fake_sdk(monkeypatch):
    def fake_loader():
        return (FakePiperInterface, FakeLogLevel)

    monkeypatch.setattr(piper_sdk_utils, "get_piper_sdk", fake_loader)
    monkeypatch.setattr(piper_follower_module, "get_piper_sdk", fake_loader)
    monkeypatch.setattr(piper_leader_module, "get_piper_sdk", fake_loader)


def make_identity_calibration():
    return {
        key: MotorCalibration(
            id=idx,
            drive_mode=0,
            homing_offset=0,
            range_min=-200000,
            range_max=200000,
        )
        for idx, key in enumerate(PIPER_ACTION_KEYS)
    }


def wait_until(predicate, timeout_s: float = 0.2, poll_s: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


@pytest.mark.parametrize(
    ("teleop_cfg", "robot_cfg", "teleop_cls", "robot_cls"),
    [
        (
            PiperLeaderConfig(port="can1", manual_control=False, sync_gripper=True),
            PiperFollowerConfig(port="can0", sync_gripper=True),
            PiperLeader,
            PiperFollower,
        ),
        (
            PiperXLeaderConfig(port="can1", manual_control=False, sync_gripper=True),
            PiperXFollowerConfig(port="can0", sync_gripper=True),
            PiperXLeader,
            PiperXFollower,
        ),
    ],
)
def test_piper_leader_follower_teleop_roundtrip(monkeypatch, teleop_cfg, robot_cfg, teleop_cls, robot_cls):
    patch_fake_sdk(monkeypatch)

    teleop = make_teleoperator_from_config(teleop_cfg)
    robot = make_robot_from_config(robot_cfg)

    assert isinstance(teleop, teleop_cls)
    assert isinstance(robot, robot_cls)

    teleop.calibration = make_identity_calibration()
    robot.calibration = make_identity_calibration()

    teleop.connect(calibrate=False)
    robot.connect(calibrate=False)
    try:
        action = teleop.get_action()
        sent = robot.send_action(action)
        obs = robot.get_observation()

        assert robot.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert robot.arm.last_gripper == (
            42000,
            robot_cfg.gripper_effort_default,
            robot_cfg.gripper_status_code,
            0x00,
        )
        assert sent["joint_1.pos"] == 10.0
        assert sent["gripper.pos"] == 42.0
        assert obs["joint_1.pos"] == 11.0
        assert obs["gripper.pos"] == 43.0

        teleop.send_feedback(action)
        assert teleop.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert teleop.arm.last_gripper == (
            42000,
            teleop_cfg.gripper_effort_default,
            teleop_cfg.gripper_status_code,
            0x00,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()


def test_piper_requires_calibration(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = PiperLeader(PiperLeaderConfig(port="can1"))
    robot = PiperFollower(PiperFollowerConfig(port="can0"))

    assert not teleop.is_calibrated
    assert not robot.is_calibrated

    teleop.connect(calibrate=False)
    robot.connect(calibrate=False)
    try:
        with pytest.raises(RuntimeError, match="is not calibrated"):
            teleop.get_action()
        with pytest.raises(RuntimeError, match="is not calibrated"):
            robot.send_action({"joint_1.pos": 0.0})
    finally:
        teleop.disconnect()
        robot.disconnect()


def test_piper_leader_reconnect_reapplies_mode(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = PiperLeader(PiperLeaderConfig(port="can1", manual_control=False))
    teleop.calibration = make_identity_calibration()

    teleop.connect(calibrate=False)
    teleop.disconnect()
    teleop.connect(calibrate=False)
    try:
        assert teleop.arm.enable_calls == 2
    finally:
        teleop.disconnect()


def test_piper_follower_connect_rolls_back_connected_cameras(monkeypatch):
    patch_fake_sdk(monkeypatch)

    class FakeCamera:
        def __init__(self, should_fail: bool):
            self.should_fail = should_fail
            self.is_connected = False
            self.disconnect_calls = 0

        def connect(self):
            if self.should_fail:
                raise RuntimeError("camera connect failed")
            self.is_connected = True

        def disconnect(self):
            self.disconnect_calls += 1
            self.is_connected = False

        def async_read(self):
            return None

    cam_ok = FakeCamera(should_fail=False)
    cam_fail = FakeCamera(should_fail=True)
    monkeypatch.setattr(
        piper_follower_module,
        "make_cameras_from_configs",
        lambda _: {"cam_ok": cam_ok, "cam_fail": cam_fail},
    )

    robot = PiperFollower(PiperFollowerConfig(port="can0"))
    robot.calibration = make_identity_calibration()

    with pytest.raises(RuntimeError, match="camera connect failed"):
        robot.connect(calibrate=False)

    assert cam_ok.disconnect_calls == 1
    assert not cam_ok.is_connected
    assert not robot.is_connected


def test_piper_require_calibration_false_allows_uncalibrated_control(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = PiperLeader(PiperLeaderConfig(port="can1", require_calibration=False))
    robot = PiperFollower(PiperFollowerConfig(port="can0", require_calibration=False))

    teleop.connect(calibrate=False)
    robot.connect(calibrate=False)
    try:
        action = teleop.get_action()
        sent = robot.send_action(action)
        assert sent["joint_1.pos"] == 11.0
        assert sent["gripper.pos"] == 43.0
    finally:
        teleop.disconnect()
        robot.disconnect()


def test_piper_gravity_comp_defaults_are_model_specific():
    assert PiperLeaderConfig(port="can1").gravity_comp_tx_ratio == (0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
    assert PiperXLeaderConfig(port="can1").gravity_comp_tx_ratio == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_piper_leader_teaching_mode_allows_missing_status(monkeypatch):
    patch_fake_sdk(monkeypatch)
    seen = {}

    def fake_read_piper_ctrl_mode(arm, timeout_s=1.0, poll_s=0.02):
        del arm, poll_s
        seen["timeout_s"] = timeout_s
        return None

    monkeypatch.setattr(piper_leader_module, "read_piper_ctrl_mode", fake_read_piper_ctrl_mode)

    teleop = PiperLeader(
        PiperLeaderConfig(
            port="can1",
            read_only=True,
            allow_teaching_mode=True,
            teaching_mode_read_timeout_s=2.0,
            require_calibration=False,
        )
    )

    teleop.connect(calibrate=False)
    try:
        assert seen["timeout_s"] == 2.0
    finally:
        teleop.disconnect()


def test_piper_leader_read_only_prefers_leader_ctrl_frames(monkeypatch):
    patch_fake_sdk(monkeypatch)
    monkeypatch.setattr(piper_leader_module, "read_piper_ctrl_mode", lambda *args, **kwargs: None)

    teleop = PiperLeader(
        PiperLeaderConfig(
            port="can1",
            read_only=True,
            allow_teaching_mode=True,
            prefer_ctrl_messages=True,
            fallback_to_feedback=False,
            require_calibration=False,
        )
    )

    teleop.connect(calibrate=False)
    try:
        action = teleop.get_action()
        assert action["joint_1.pos"] == 10.0
        assert action["joint_2.pos"] == 20.0
        assert action["gripper.pos"] == 42.0
    finally:
        teleop.disconnect()


def test_piper_leader_read_only_requires_leader_ctrl_frames(monkeypatch):
    patch_fake_sdk(monkeypatch)
    monkeypatch.setattr(piper_leader_module, "read_piper_ctrl_mode", lambda *args, **kwargs: None)

    teleop = PiperLeader(
        PiperLeaderConfig(
            port="can1",
            read_only=True,
            allow_teaching_mode=True,
            prefer_ctrl_messages=True,
            fallback_to_feedback=False,
            read_only_action_timeout_s=0.01,
            require_calibration=False,
        )
    )

    teleop.arm._joint_ctrl.time_stamp = 0.0
    teleop.connect(calibrate=False)
    try:
        with pytest.raises(RuntimeError, match="no leader joint control frames"):
            teleop.get_action()
    finally:
        if teleop.is_connected:
            teleop.disconnect()


def test_piper_leader_manual_control_prefers_feedback_state(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = PiperLeader(
        PiperLeaderConfig(
            port="can1",
            manual_control=True,
            require_calibration=False,
        )
    )

    teleop.connect(calibrate=False)
    try:
        action = teleop.get_action()
        assert action["joint_1.pos"] == 11.0
        assert action["joint_2.pos"] == 21.0
        assert action["gripper.pos"] == 43.0
    finally:
        teleop.disconnect()


def test_piper_leader_manual_control_waits_for_fresh_feedback(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = PiperLeader(
        PiperLeaderConfig(
            port="can1",
            manual_control=True,
            require_calibration=False,
        )
    )

    teleop.connect(calibrate=False)
    try:
        initial = teleop.get_action()
        assert initial["joint_1.pos"] == 11.0

        def publish_fresh_feedback():
            time.sleep(0.01)
            teleop.arm._joint_state.time_stamp = 2.0
            teleop.arm._joint_state.joint_state.joint_1 = 12000
            teleop.arm._joint_state.joint_state.joint_2 = 22000
            teleop.arm._joint_state.joint_state.joint_3 = 32000
            teleop.arm._joint_state.joint_state.joint_4 = 42000
            teleop.arm._joint_state.joint_state.joint_5 = 52000
            teleop.arm._joint_state.joint_state.joint_6 = 62000
            teleop.arm._gripper_state.time_stamp = 2.0
            teleop.arm._gripper_state.gripper_state.grippers_angle = 44000

        publisher = threading.Thread(target=publish_fresh_feedback, daemon=True)
        publisher.start()

        t0 = time.monotonic()
        action = teleop.get_action()
        publisher.join(timeout=0.1)

        assert time.monotonic() - t0 >= 0.008
        assert action["joint_1.pos"] == 12.0
        assert action["joint_2.pos"] == 22.0
        assert action["gripper.pos"] == 44.0
    finally:
        teleop.disconnect()


def test_piper_leader_manual_control_does_not_unblock_on_gripper_feedback_only(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = PiperLeader(
        PiperLeaderConfig(
            port="can1",
            manual_control=True,
            require_calibration=False,
        )
    )

    teleop.connect(calibrate=False)
    try:
        initial = teleop.get_action()
        assert initial["joint_1.pos"] == 11.0

        def publish_feedback():
            time.sleep(0.005)
            teleop.arm._gripper_state.time_stamp = 2.0
            teleop.arm._gripper_state.gripper_state.grippers_angle = 45000
            time.sleep(0.01)
            teleop.arm._joint_state.time_stamp = 2.0
            teleop.arm._joint_state.joint_state.joint_1 = 13000
            teleop.arm._joint_state.joint_state.joint_2 = 23000
            teleop.arm._joint_state.joint_state.joint_3 = 33000
            teleop.arm._joint_state.joint_state.joint_4 = 43000
            teleop.arm._joint_state.joint_state.joint_5 = 53000
            teleop.arm._joint_state.joint_state.joint_6 = 63000

        publisher = threading.Thread(target=publish_feedback, daemon=True)
        publisher.start()

        t0 = time.monotonic()
        action = teleop.get_action()
        publisher.join(timeout=0.1)

        assert time.monotonic() - t0 >= 0.013
        assert action["joint_1.pos"] == 13.0
    finally:
        teleop.disconnect()


def test_piper_follower_connect_calibrates_then_reenables(monkeypatch, tmp_path):
    patch_fake_sdk(monkeypatch)

    robot = PiperFollower(
        PiperFollowerConfig(
            port="can0",
            id="connect_reenable_after_calibration",
            calibration_dir=tmp_path,
            require_calibration=True,
            enable_on_connect=True,
        )
    )

    def fake_calibrate():
        # Mirror real calibration's drag-mode behavior.
        robot.arm.DisableArm(7)
        robot.calibration = make_identity_calibration()

    monkeypatch.setattr(robot, "calibrate", fake_calibrate)

    robot.connect(calibrate=True)
    try:
        assert robot.arm.disable_calls == 1
        assert robot.arm.enable_calls == 1
        assert robot.arm.is_enabled
    finally:
        robot.disconnect()


def test_piper_follower_connect_without_calibration_still_enables(monkeypatch, tmp_path):
    patch_fake_sdk(monkeypatch)

    robot = PiperFollower(
        PiperFollowerConfig(
            port="can0",
            id="connect_enable_without_calibration",
            calibration_dir=tmp_path,
            require_calibration=True,
            enable_on_connect=True,
        )
    )

    robot.connect(calibrate=False)
    try:
        assert robot.arm.enable_calls == 1
        assert robot.arm.is_enabled
        assert robot.arm.disable_calls == 0
    finally:
        robot.disconnect()


def test_piper_leader_gravity_comp_manual_control_uses_mit(monkeypatch):
    patch_fake_sdk(monkeypatch)

    class FakeModel:
        def __init__(self):
            self.nq = 6
            self.nv = 6
            self.gravity = SimpleNamespace(linear=None)

        def createData(self):
            return object()

    class FakeRobotWrapper:
        @staticmethod
        def BuildFromURDF(urdf_path, package_dirs):
            del urdf_path, package_dirs
            return SimpleNamespace(model=FakeModel(), data=object())

    class FakePinocchio:
        RobotWrapper = FakeRobotWrapper

        @staticmethod
        def rnea(model, data, q, v, a):
            del model, data, q, v, a
            return [0.1] * 6

    monkeypatch.setitem(sys.modules, "pinocchio", FakePinocchio)

    teleop = PiperLeader(
        PiperLeaderConfig(
            port="can1",
            manual_control=True,
            gravity_comp_control_hz=200.0,
            mode_refresh_interval_s=0.01,
        )
    )
    teleop.calibration = make_identity_calibration()

    teleop.connect(calibrate=False)
    try:
        assert wait_until(lambda: len(teleop.arm.joint_mit_calls) > 0)
        assert teleop.arm.disable_calls == 0
        assert len(teleop.arm.joint_mit_calls) > 0
        assert len(teleop.arm.gripper_calls) > 0
        assert teleop.arm.gripper_calls[0][2] == 0x00

        teleop.set_manual_control(False)
        calls_after_stop = len(teleop.arm.joint_mit_calls)
        time.sleep(0.03)
        assert len(teleop.arm.joint_mit_calls) == calls_after_stop
        assert teleop.arm.gripper_calls[-1][2] == 0x01
    finally:
        teleop.disconnect()


@pytest.mark.parametrize(
    ("device_kind", "port", "robot_id"),
    [("leader", "can1", None), ("follower", "can0", "piper_role_guard")],
)
def test_piper_connect_fails_and_writes_follower_role_when_in_teach_mode(
    monkeypatch, device_kind, port, robot_id
):
    patch_fake_sdk(monkeypatch)

    if device_kind == "leader":
        device = PiperLeader(PiperLeaderConfig(port=port))
    else:
        device = PiperFollower(PiperFollowerConfig(port=port, id=robot_id))
    device.arm._arm_status.arm_status.ctrl_mode = 0x06

    with pytest.raises(RuntimeError, match="Follower role command .* sent.*Power-cycle"):
        device.connect(calibrate=False)

    assert device.arm.role_commands[-1] == (0xFC, 0x00, 0x00, 0x00)


def test_piper_lfs_pointer_urdf_raises_actionable_error(tmp_path):
    pointer_file = tmp_path / "lfs_pointer.urdf"
    pointer_file.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 123\n")

    with pytest.raises(RuntimeError, match="Git LFS pointer files"):
        piper_leader_module._ensure_not_lfs_pointer(
            pointer_file, "assets/piper_description/urdf/pointer.urdf"
        )


@pytest.mark.parametrize(
    (
        "teleop_cfg",
        "robot_cfg",
        "bi_teleop_cls",
        "bi_robot_cls",
        "left_teleop_cls",
        "right_teleop_cls",
        "left_robot_cls",
        "right_robot_cls",
    ),
    [
        (
            BiPiperLeaderConfig(
                left_arm_config=PiperLeaderConfigBase(port="can1", manual_control=False, sync_gripper=True),
                right_arm_config=PiperLeaderConfigBase(port="can3", manual_control=False, sync_gripper=True),
                process_isolation=False,
            ),
            BiPiperFollowerConfig(
                left_arm_config=PiperFollowerConfigBase(port="can0", sync_gripper=True),
                right_arm_config=PiperFollowerConfigBase(port="can2", sync_gripper=True),
            ),
            BiPiperLeader,
            BiPiperFollower,
            PiperLeader,
            PiperLeader,
            PiperFollower,
            PiperFollower,
        ),
        (
            BiPiperXLeaderConfig(
                left_arm_config=PiperXLeaderConfigBase(port="can1", manual_control=False, sync_gripper=True),
                right_arm_config=PiperXLeaderConfigBase(port="can3", manual_control=False, sync_gripper=True),
                process_isolation=False,
            ),
            BiPiperXFollowerConfig(
                left_arm_config=PiperFollowerConfigBase(port="can0", sync_gripper=True),
                right_arm_config=PiperFollowerConfigBase(port="can2", sync_gripper=True),
            ),
            BiPiperXLeader,
            BiPiperXFollower,
            PiperXLeader,
            PiperXLeader,
            PiperXFollower,
            PiperXFollower,
        ),
    ],
)
def test_bimanual_piper_leader_follower_roundtrip(
    monkeypatch,
    teleop_cfg,
    robot_cfg,
    bi_teleop_cls,
    bi_robot_cls,
    left_teleop_cls,
    right_teleop_cls,
    left_robot_cls,
    right_robot_cls,
):
    patch_fake_sdk(monkeypatch)

    teleop = make_teleoperator_from_config(teleop_cfg)
    robot = make_robot_from_config(robot_cfg)

    assert isinstance(teleop, bi_teleop_cls)
    assert isinstance(robot, bi_robot_cls)
    assert isinstance(teleop.left_arm, left_teleop_cls)
    assert isinstance(teleop.right_arm, right_teleop_cls)
    assert isinstance(robot.left_arm, left_robot_cls)
    assert isinstance(robot.right_arm, right_robot_cls)

    teleop.left_arm.calibration = make_identity_calibration()
    teleop.right_arm.calibration = make_identity_calibration()
    robot.left_arm.calibration = make_identity_calibration()
    robot.right_arm.calibration = make_identity_calibration()

    teleop.connect(calibrate=False)
    robot.connect(calibrate=False)
    try:
        action = teleop.get_action()
        assert "left_joint_1.pos" in action
        assert "right_joint_1.pos" in action

        sent = robot.send_action(action)
        obs = robot.get_observation()

        assert robot.left_arm.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert robot.right_arm.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert robot.left_arm.arm.last_gripper[0] == 42000
        assert robot.right_arm.arm.last_gripper[0] == 42000

        assert sent["left_joint_1.pos"] == 10.0
        assert sent["right_joint_1.pos"] == 10.0
        assert sent["left_gripper.pos"] == 42.0
        assert sent["right_gripper.pos"] == 42.0
        assert obs["left_joint_1.pos"] == 11.0
        assert obs["right_joint_1.pos"] == 11.0
        assert obs["left_gripper.pos"] == 43.0
        assert obs["right_gripper.pos"] == 43.0

        teleop.send_feedback(action)
        assert teleop.left_arm.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert teleop.right_arm.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert teleop.left_arm.arm.last_gripper[0] == 42000
        assert teleop.right_arm.arm.last_gripper[0] == 42000
    finally:
        teleop.disconnect()
        robot.disconnect()


def test_bimanual_piper_read_only_returns_side_that_has_first_frame(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = make_teleoperator_from_config(
        BiPiperLeaderConfig(
            left_arm_config=PiperLeaderConfigBase(
                port="can2",
                read_only=True,
                allow_teaching_mode=True,
                prefer_ctrl_messages=True,
                fallback_to_feedback=False,
                read_only_action_timeout_s=0,
                require_calibration=False,
            ),
            right_arm_config=PiperLeaderConfigBase(
                port="can3",
                read_only=True,
                allow_teaching_mode=True,
                prefer_ctrl_messages=True,
                fallback_to_feedback=False,
                read_only_action_timeout_s=0,
                require_calibration=False,
            ),
            process_isolation=False,
        )
    )

    teleop.connect(calibrate=False)
    try:
        teleop.left_arm.arm._joint_ctrl.time_stamp = 0.0
        teleop.right_arm.arm._joint_ctrl.time_stamp = 0.0
        action = teleop.get_action()
        assert action == {}

        teleop.left_arm.arm._joint_ctrl.time_stamp = 1.0
        action = teleop.get_action()
        assert "left_joint_1.pos" in action
        assert "right_joint_1.pos" not in action

        teleop.right_arm.arm._joint_ctrl.time_stamp = 1.0
        action = teleop.get_action()
        assert "left_joint_1.pos" in action
        assert "right_joint_1.pos" in action
    finally:
        teleop.disconnect()


def test_bimanual_piper_follower_action_features_are_available_without_connect(monkeypatch):
    patch_fake_sdk(monkeypatch)

    robot = make_robot_from_config(
        BiPiperFollowerConfig(
            left_arm_config=PiperFollowerConfigBase(port="can3", sync_gripper=True),
            right_arm_config=PiperFollowerConfigBase(port="can2", sync_gripper=True),
        )
    )

    action_features = robot.action_features

    assert action_features["left_joint_1.pos"] is float
    assert action_features["left_gripper.pos"] is float
    assert action_features["right_joint_1.pos"] is float
    assert action_features["right_gripper.pos"] is float


def test_bimanual_piper_teleop_loop_smoke(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = make_teleoperator_from_config(
        BiPiperLeaderConfig(
            left_arm_config=PiperLeaderConfigBase(port="can0", manual_control=False, sync_gripper=True),
            right_arm_config=PiperLeaderConfigBase(port="can1", manual_control=False, sync_gripper=True),
            process_isolation=False,
        )
    )
    robot = make_robot_from_config(
        BiPiperFollowerConfig(
            left_arm_config=PiperFollowerConfigBase(port="can3", sync_gripper=True),
            right_arm_config=PiperFollowerConfigBase(port="can2", sync_gripper=True),
        )
    )

    teleop.left_arm.calibration = make_identity_calibration()
    teleop.right_arm.calibration = make_identity_calibration()
    robot.left_arm.calibration = make_identity_calibration()
    robot.right_arm.calibration = make_identity_calibration()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    robot.get_observation = lambda: (_ for _ in ()).throw(
        AssertionError("teleop_loop should not fetch observation for identity processors without display")
    )

    teleop.connect(calibrate=False)
    robot.connect(calibrate=False)
    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=60,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_data=False,
            duration=0.0,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()

    assert robot.left_arm.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
    assert robot.right_arm.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
    assert robot.left_arm.arm.last_gripper[0] == 42000
    assert robot.right_arm.arm.last_gripper[0] == 42000


def test_bimanual_piper_leader_uses_process_proxy_by_default(monkeypatch):
    class DummyProxy:
        def __init__(self, arm_cls, arm_config):
            self.arm_cls = arm_cls
            self.arm_config = arm_config
            self.is_connected = False
            self.is_calibrated = True

        action_features = dict.fromkeys(PIPER_ACTION_KEYS, float)
        feedback_features = dict.fromkeys(PIPER_ACTION_KEYS, float)

        def connect(self, calibrate=True):
            del calibrate
            self.is_connected = True

        def calibrate(self):
            pass

        def configure(self):
            pass

        def setup_motors(self):
            pass

        def set_manual_control(self, enabled: bool):
            del enabled

        def get_action(self):
            return dict.fromkeys(PIPER_ACTION_KEYS, 0.0)

        def send_feedback(self, feedback):
            del feedback

        def disconnect(self):
            self.is_connected = False

    monkeypatch.setattr(bi_piper_leader_module, "_PiperLeaderProcessProxy", DummyProxy)

    teleop = make_teleoperator_from_config(
        BiPiperLeaderConfig(
            left_arm_config=PiperLeaderConfigBase(port="can0", manual_control=False, sync_gripper=True),
            right_arm_config=PiperLeaderConfigBase(port="can1", manual_control=False, sync_gripper=True),
        )
    )

    assert isinstance(teleop.left_arm, DummyProxy)
    assert isinstance(teleop.right_arm, DummyProxy)


def test_piper_follower_send_only_mode_skips_sdk_reader_threads(monkeypatch):
    patch_fake_sdk(monkeypatch)

    robot = make_robot_from_config(
        PiperFollowerConfig(port="can3", sync_gripper=True, require_calibration=False)
    )
    robot.calibration = make_identity_calibration()
    robot.set_teleop_send_only_mode(True)

    robot.connect(calibrate=False)
    try:
        assert robot.arm.connect_calls[-1]["start_thread"] is False
        with pytest.raises(RuntimeError, match="send-only mode"):
            robot.get_observation()
    finally:
        robot.disconnect()


def test_piper_follower_send_only_mode_skips_camera_connects_and_stays_connected(monkeypatch):
    patch_fake_sdk(monkeypatch)

    class FakeCamera:
        def __init__(self):
            self.is_connected = False
            self.connect_calls = 0
            self.disconnect_calls = 0

        def connect(self):
            self.connect_calls += 1
            self.is_connected = True

        def disconnect(self):
            self.disconnect_calls += 1
            self.is_connected = False

    camera = FakeCamera()
    monkeypatch.setattr(piper_follower_module, "make_cameras_from_configs", lambda _: {"front": camera})

    robot = make_robot_from_config(
        PiperFollowerConfig(port="can3", sync_gripper=True, require_calibration=False)
    )
    robot.calibration = make_identity_calibration()
    robot.set_teleop_send_only_mode(True)

    robot.connect(calibrate=False)
    try:
        assert robot.is_connected
        assert camera.connect_calls == 0
        assert not camera.is_connected
    finally:
        robot.disconnect()

    assert camera.disconnect_calls == 0


def test_bimanual_piper_send_only_mode_propagates_to_both_followers(monkeypatch):
    patch_fake_sdk(monkeypatch)

    robot = make_robot_from_config(
        BiPiperFollowerConfig(
            left_arm_config=PiperFollowerConfigBase(port="can3", sync_gripper=True, require_calibration=False),
            right_arm_config=PiperFollowerConfigBase(port="can2", sync_gripper=True, require_calibration=False),
        )
    )
    robot.left_arm.calibration = make_identity_calibration()
    robot.right_arm.calibration = make_identity_calibration()
    robot.set_teleop_send_only_mode(True)

    robot.connect(calibrate=False)
    try:
        assert robot.left_arm.arm.connect_calls[-1]["start_thread"] is False
        assert robot.right_arm.arm.connect_calls[-1]["start_thread"] is False
    finally:
        robot.disconnect()
