#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate evo-rl
cd "$REPO_ROOT"
ip -br link show type can
duration_s="${1:-300}"
echo "Teleop duration: ${duration_s}s"
echo "CAN mapping: can0=left_follower can1=right_follower can2=left_leader can3=right_leader"
echo "Mode: followers use Evo-RL control (0xFC); leaders use AgileX built-in leader drag (0xFA only)."
echo "If teleop waits for leader frames, move each leader arm gently once."

python - <<'PY'
import re
import sys
import time

from piper_sdk import C_PiperInterface_V2, LogLevel

PORT_ROLES = (
    ("left_follower", "can0", "follower"),
    ("right_follower", "can1", "follower"),
    ("left_leader", "can2", "leader"),
    ("right_leader", "can3", "leader"),
)


def _read_firmware(arm):
    try:
        arm.SearchPiperFirmwareVersion()
        for _ in range(30):
            value = arm.GetPiperFirmwareVersion()
            if isinstance(value, str) and value.startswith("S-V"):
                return value
            time.sleep(0.1)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{exc}"
    return "UNKNOWN"


def _firmware_is_v189_or_newer(version):
    match = re.match(r"S-V(\d+)\.(\d+)-(\d+)", version)
    if match is None:
        return False
    major, minor, patch = (int(part) for part in match.groups())
    return (major, minor, patch) >= (1, 8, 9)


def _wait_live_status(arm, port, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    last_status = None
    while time.monotonic() < deadline:
        last_status = arm.GetArmStatus()
        if float(getattr(last_status, "time_stamp", 0.0) or 0.0) > 0.0:
            return last_status
        time.sleep(0.05)
    raise RuntimeError(f"{port} did not report live arm status within {timeout_s:.1f}s")


def _status_summary(status_msg):
    arm_status = getattr(status_msg, "arm_status", None)
    if arm_status is None:
        return "status=unavailable"
    return (
        f"ctrl_mode={getattr(arm_status, 'ctrl_mode', None)} "
        f"mode_feed={getattr(arm_status, 'mode_feed', None)} "
        f"teach={getattr(arm_status, 'teach_status', None)}"
    )


def _new_arm(port):
    arm = C_PiperInterface_V2(
        can_name=port,
        judge_flag=True,
        can_auto_init=True,
        logger_level=LogLevel.WARNING,
    )
    arm.ConnectPort(can_init=False, piper_init=False, start_thread=True)
    time.sleep(0.2)
    return arm


def _configure_follower(name, port, version):
    arm = _new_arm(port)
    try:
        arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
        time.sleep(0.3)
        status = _wait_live_status(arm, port)
        print(
            f"[OK] {name} {port}: firmware={version} role=follower_controlled(0xFC) "
            f"{_status_summary(status)}",
            flush=True,
        )
    finally:
        arm.DisconnectPort()


def _configure_leader(name, port, version):
    arm = _new_arm(port)
    try:
        arm.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00)
        time.sleep(0.2)
        print(
            f"[OK] {name} {port}: firmware={version} role=leader_drag(0xFA), "
            "restore_drag=skipped",
            flush=True,
        )
    finally:
        arm.DisconnectPort()


def _configure_port(name, port, role):
    arm = _new_arm(port)
    try:
        version = _read_firmware(arm)
    finally:
        arm.DisconnectPort()

    if not _firmware_is_v189_or_newer(version):
        print(f"[WARN] {name} {port}: expected >= S-V1.8-9, got {version}", flush=True)

    if role == "leader":
        _configure_leader(name, port, version)
    elif role == "follower":
        _configure_follower(name, port, version)
    else:
        raise ValueError(f"unknown role: {role}")


try:
    print("Preparing PiPER-X roles through piper_sdk...", flush=True)
    for item in PORT_ROLES:
        _configure_port(*item)
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] PiPER-X role setup failed: {exc}", file=sys.stderr, flush=True)
    sys.exit(1)
PY

exec lerobot-teleoperate \
  --robot.type=bi_piperx_follower \
  --robot.id=my_bi_piperx_follower \
  --robot.left_arm_config.port=can0 \
  --robot.right_arm_config.port=can1 \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --robot.left_arm_config.sync_gripper=true \
  --robot.right_arm_config.sync_gripper=true \
  --robot.left_arm_config.speed_ratio=10 \
  --robot.right_arm_config.speed_ratio=10 \
  --teleop.type=bi_piperx_leader \
  --teleop.id=my_bi_piperx_leader \
  --teleop.left_arm_config.port=can2 \
  --teleop.right_arm_config.port=can3 \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false \
  --teleop.left_arm_config.read_only=true \
  --teleop.right_arm_config.read_only=true \
  --teleop.left_arm_config.allow_teaching_mode=true \
  --teleop.right_arm_config.allow_teaching_mode=true \
  --teleop.left_arm_config.prefer_ctrl_messages=true \
  --teleop.right_arm_config.prefer_ctrl_messages=true \
  --teleop.left_arm_config.fallback_to_feedback=false \
  --teleop.right_arm_config.fallback_to_feedback=false \
  --teleop.left_arm_config.read_only_action_timeout_s=0 \
  --teleop.right_arm_config.read_only_action_timeout_s=0 \
  --teleop.left_arm_config.sync_gripper=true \
  --teleop.right_arm_config.sync_gripper=true \
  --fps=30 \
  --teleop_time_s="${duration_s}" \
  --fake_inference_key=i \
  --fake_inference_total_swing_deg=120 \
  --fake_inference_period_s=10 \
  --fake_inference_control_leaders=true \
  --fake_inference_leader_speed_ratio=10 \
  --display_data=false
