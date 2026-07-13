# ruff: noqa: N802
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
        self.command_log = []

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
        self.command_log.append(("motion_mode", args))

    def MasterSlaveConfig(self, *args):
        self.role_commands.append(args)
        self.command_log.append(("role", args))
        if args and args[0] == 0xFC:
            self._arm_status.arm_status.ctrl_mode = 0x01
        elif args and args[0] == 0xFA:
            self._arm_status.arm_status.ctrl_mode = 0x06
            self._joint_ctrl.time_stamp += 1.0
            self._gripper_ctrl.time_stamp += 1.0

    def EnablePiper(self):
        self.enable_calls += 1
        self.command_log.append(("enable", ()))
        self.is_enabled = True
        return True

    def DisableArm(self, motor_num):
        del motor_num
        self.disable_calls += 1
        self.is_enabled = False

    def JointCtrl(self, *args):
        self.last_joint = args
        self.command_log.append(("joint", args))

    def JointMitCtrl(self, *args):
        self.joint_mit_calls.append(args)

    def GripperCtrl(self, *args):
        self.last_gripper = args
        self.gripper_calls.append(args)
        self.command_log.append(("gripper", args))

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
    monkeypatch.setattr(piper_leader_module, "PIPER_ROLE_SWITCH_SETTLE_S", 0.0)
    monkeypatch.setattr(piper_leader_module, "PIPER_ACTION_READ_TIMEOUT_S", 0.03)


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


@pytest.mark.parametrize(
    ("teleop_cfg", "robot_cfg", "teleop_cls", "robot_cls"),
    [
        (
            PiperLeaderConfig(port="can1", manual_control=True, sync_gripper=True),
            PiperFollowerConfig(port="can0", sync_gripper=True),
            PiperLeader,
            PiperFollower,
        ),
        (
            PiperXLeaderConfig(port="can1", manual_control=True, sync_gripper=True),
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
        assert teleop.arm.role_commands[-1] == (0xFA, 0x00, 0x00, 0x00)
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
        assert teleop.arm.role_commands[-1] == (0xFC, 0x00, 0x00, 0x00)
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
    assert teleop.arm.role_commands[-1] == (0xFA, 0x00, 0x00, 0x00)


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
        assert sent["joint_1.pos"] == 10.0
        assert sent["gripper.pos"] == 42.0
    finally:
        teleop.disconnect()
        robot.disconnect()


def test_piper_leader_default_uses_hardware_leader_role(monkeypatch):
    patch_fake_sdk(monkeypatch)
    teleop = PiperLeader(PiperLeaderConfig(port="can1", require_calibration=False))

    teleop.connect(calibrate=False)
    try:
        assert teleop.arm.connect_calls[-1]["piper_init"] is False
        assert teleop.arm.role_commands == [(0xFA, 0x00, 0x00, 0x00)]
        assert teleop.arm.enable_calls == 0
        assert teleop.arm.mode_commands == []
        assert teleop.arm.gripper_calls == []
        assert teleop.arm.joint_mit_calls == []
        action = teleop.get_action()
        assert action["joint_1.pos"] == 10.0
        assert action["joint_2.pos"] == 20.0
        assert action["gripper.pos"] == 42.0
    finally:
        teleop.disconnect()


def test_piper_leader_switches_between_manual_and_policy_control(monkeypatch):
    patch_fake_sdk(monkeypatch)
    teleop = PiperLeader(PiperLeaderConfig(port="can1", require_calibration=False))
    teleop.connect(calibrate=False)
    try:
        teleop.arm.command_log.clear()
        teleop.set_manual_control(False)
        assert teleop.arm.role_commands[-1] == (0xFC, 0x00, 0x00, 0x00)
        assert teleop.arm.mode_commands[-1][:3] == (0x01, 0x01, teleop.config.command_speed_ratio)
        assert teleop.arm.enable_calls == 1
        assert [name for name, _ in teleop.arm.command_log[:3]] == ["role", "motion_mode", "enable"]

        role_count = len(teleop.arm.role_commands)
        action = {
            "joint_1.pos": 1.0,
            "joint_2.pos": 2.0,
            "joint_3.pos": 3.0,
            "joint_4.pos": 4.0,
            "joint_5.pos": 5.0,
            "joint_6.pos": 6.0,
            "gripper.pos": 7.0,
        }
        teleop.send_feedback(action)
        assert len(teleop.arm.role_commands) == role_count
        assert teleop.arm.last_joint == (1000, 2000, 3000, 4000, 5000, 6000)
        assert teleop.arm.last_gripper[0] == 7000

        teleop.set_manual_control(True)
        assert teleop.arm.role_commands[-1] == (0xFA, 0x00, 0x00, 0x00)
        assert teleop.arm.joint_mit_calls == []
    finally:
        teleop.disconnect()


def test_piper_leader_policy_mode_reads_feedback(monkeypatch):
    patch_fake_sdk(monkeypatch)
    teleop = PiperLeader(PiperLeaderConfig(port="can1", manual_control=False, require_calibration=False))

    teleop.connect(calibrate=False)
    try:
        action = teleop.get_action()
        assert action["joint_1.pos"] == 11.0
        assert action["joint_2.pos"] == 21.0
        assert action["gripper.pos"] == 43.0
    finally:
        teleop.disconnect()


def test_piper_leader_requires_fresh_complete_manual_frame(monkeypatch):
    patch_fake_sdk(monkeypatch)
    teleop = PiperLeader(PiperLeaderConfig(port="can1", require_calibration=False))
    teleop.connect(calibrate=False)
    try:
        teleop._leader_frames_ready = False
        teleop._leader_joint_timestamp_before_switch = teleop.arm._joint_ctrl.time_stamp
        teleop._leader_gripper_timestamp_before_switch = teleop.arm._gripper_ctrl.time_stamp
        teleop.arm._joint_ctrl.time_stamp += 1.0
        teleop.arm._joint_ctrl.Hz = 0.0
        teleop.arm._gripper_ctrl.time_stamp += 1.0
        with pytest.raises(RuntimeError, match="no complete Piper leader control frame"):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_piper_leader_failed_policy_switch_restores_manual_role(monkeypatch):
    patch_fake_sdk(monkeypatch)
    teleop = PiperLeader(PiperLeaderConfig(port="can1", enable_timeout_s=0.0, require_calibration=False))
    teleop.connect(calibrate=False)
    try:
        with pytest.raises(RuntimeError, match="did not enable"):
            teleop.set_manual_control(False)
        assert teleop._manual_control_enabled is True
        assert teleop.arm.role_commands[-1][0] == 0xFA
    finally:
        teleop.disconnect()


def test_piper_leader_role_restore_ignores_timestamp_read_errors(monkeypatch):
    patch_fake_sdk(monkeypatch)
    teleop = PiperLeader(PiperLeaderConfig(port="can1", require_calibration=False))

    def fail_timestamp_read():
        raise RuntimeError("timestamp unavailable")

    monkeypatch.setattr(teleop.arm, "GetArmJointCtrl", fail_timestamp_read)
    monkeypatch.setattr(teleop.arm, "GetArmGripperCtrl", fail_timestamp_read)

    teleop.connect(calibrate=False)
    try:
        assert teleop.arm.role_commands[-1][0] == 0xFA
        assert teleop._manual_control_enabled is True
    finally:
        teleop.disconnect()


def test_piper_leader_disconnect_disables_if_role_restore_fails(monkeypatch):
    patch_fake_sdk(monkeypatch)
    teleop = PiperLeader(PiperLeaderConfig(port="can1", manual_control=False, require_calibration=False))
    teleop.connect(calibrate=False)

    def fail_role_restore(*args):
        raise RuntimeError("leader role restore failed")

    monkeypatch.setattr(teleop.arm, "MasterSlaveConfig", fail_role_restore)

    with pytest.raises(RuntimeError, match="leader role restore failed"):
        teleop.disconnect()
    assert teleop.arm.disable_calls == 1
    assert not teleop.arm.connected


def test_piper_leader_auto_calibration_uses_manual_role_then_restores_config(monkeypatch, tmp_path):
    patch_fake_sdk(monkeypatch)
    teleop = PiperLeader(
        PiperLeaderConfig(
            port="can1",
            id="auto_calibrate_role",
            calibration_dir=tmp_path,
            manual_control=False,
            require_calibration=True,
        )
    )

    def fake_calibrate():
        assert teleop._manual_control_enabled is True
        assert teleop.arm.role_commands[-1][0] == 0xFA
        teleop.calibration = make_identity_calibration()

    monkeypatch.setattr(teleop, "calibrate", fake_calibrate)

    teleop.connect(calibrate=True)
    try:
        assert [command[0] for command in teleop.arm.role_commands] == [0xFA, 0xFC]
        assert teleop._manual_control_enabled is False
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


def test_piper_follower_connect_fails_and_writes_follower_role_when_in_teach_mode(monkeypatch):
    patch_fake_sdk(monkeypatch)
    device = PiperFollower(PiperFollowerConfig(port="can0", id="piper_role_guard"))
    device.arm._arm_status.arm_status.ctrl_mode = 0x06

    with pytest.raises(RuntimeError, match="Follower role command .* sent.*Power-cycle"):
        device.connect(calibrate=False)

    assert device.arm.role_commands[-1] == (0xFC, 0x00, 0x00, 0x00)


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
                left_arm_config=PiperLeaderConfigBase(port="can1", manual_control=True, sync_gripper=True),
                right_arm_config=PiperLeaderConfigBase(port="can3", manual_control=True, sync_gripper=True),
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
                left_arm_config=PiperXLeaderConfigBase(port="can1", manual_control=True, sync_gripper=True),
                right_arm_config=PiperXLeaderConfigBase(port="can3", manual_control=True, sync_gripper=True),
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
        assert teleop.left_arm.arm.role_commands[-1][0] == 0xFA
        assert teleop.right_arm.arm.role_commands[-1][0] == 0xFA
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
        assert teleop.left_arm.arm.role_commands[-1][0] == 0xFC
        assert teleop.right_arm.arm.role_commands[-1][0] == 0xFC
        assert teleop.left_arm.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert teleop.right_arm.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert teleop.left_arm.arm.last_gripper[0] == 42000
        assert teleop.right_arm.arm.last_gripper[0] == 42000
    finally:
        teleop.disconnect()
        robot.disconnect()
    assert teleop.left_arm.arm.role_commands[-1][0] == 0xFA
    assert teleop.right_arm.arm.role_commands[-1][0] == 0xFA


def test_bimanual_piper_get_action_requires_both_sides(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = make_teleoperator_from_config(
        BiPiperLeaderConfig(
            left_arm_config=PiperLeaderConfigBase(
                port="can2",
                require_calibration=False,
            ),
            right_arm_config=PiperLeaderConfigBase(
                port="can3",
                require_calibration=False,
            ),
            process_isolation=False,
        )
    )

    teleop.connect(calibrate=False)
    try:
        teleop.right_arm._leader_frames_ready = False
        teleop.right_arm._leader_joint_timestamp_before_switch = teleop.right_arm.arm._joint_ctrl.time_stamp
        teleop.right_arm._leader_gripper_timestamp_before_switch = (
            teleop.right_arm.arm._gripper_ctrl.time_stamp
        )
        with pytest.raises(RuntimeError, match=r"\[can3\].*leader control frame"):
            teleop.get_action()
    finally:
        teleop.disconnect()


def test_bimanual_piper_failed_policy_switch_restores_both_leaders(monkeypatch):
    patch_fake_sdk(monkeypatch)
    teleop = make_teleoperator_from_config(
        BiPiperLeaderConfig(
            left_arm_config=PiperLeaderConfigBase(port="can2", require_calibration=False),
            right_arm_config=PiperLeaderConfigBase(port="can3", require_calibration=False),
            process_isolation=False,
        )
    )
    original_right_switch = teleop.right_arm.set_manual_control

    def fail_policy_switch(enabled: bool):
        if not enabled:
            raise RuntimeError("right role switch failed")
        return original_right_switch(enabled)

    monkeypatch.setattr(teleop.right_arm, "set_manual_control", fail_policy_switch)

    teleop.connect(calibrate=False)
    try:
        with pytest.raises(RuntimeError, match="right role switch failed"):
            teleop.set_manual_control(False)
        assert teleop.left_arm.arm.role_commands[-1][0] == 0xFA
        assert teleop.right_arm.arm.role_commands[-1][0] == 0xFA
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
            left_arm_config=PiperLeaderConfigBase(port="can0", manual_control=True, sync_gripper=True),
            right_arm_config=PiperLeaderConfigBase(port="can1", manual_control=True, sync_gripper=True),
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


def test_bimanual_piper_process_proxy_reports_disconnect_error_after_cleanup(monkeypatch):
    class DummyConnection:
        def __init__(self):
            self.closed = False

        def send(self, request):
            del request

        def close(self):
            self.closed = True

    class DummyProcess:
        @staticmethod
        def is_alive():
            return False

    proxy = object.__new__(bi_piper_leader_module._PiperLeaderProcessProxy)
    proxy._parent_conn = DummyConnection()
    proxy._process = DummyProcess()
    proxy._is_connected = True
    monkeypatch.setattr(
        proxy,
        "_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("remote disconnect failed")),
    )

    connection = proxy._parent_conn
    with pytest.raises(RuntimeError, match="remote disconnect failed"):
        proxy.disconnect()
    assert connection.closed
    assert proxy._parent_conn is None
    assert proxy._process is None
    assert not proxy._is_connected


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
            left_arm_config=PiperFollowerConfigBase(
                port="can3", sync_gripper=True, require_calibration=False
            ),
            right_arm_config=PiperFollowerConfigBase(
                port="can2", sync_gripper=True, require_calibration=False
            ),
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
