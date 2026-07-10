#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate evo-rl
cd "$REPO_ROOT"

port="${1:-can3}"
window_s="${2:-6}"

echo "PiPER-X leader recovery sequence diagnostic on ${port}"
echo "Each sequence sends role commands, then sniffs raw CAN IDs for ${window_s}s."
echo "During every sniff window, move ${port} gently. Do not force it if it is stiff."

python - "$port" "$window_s" <<'PY'
import socket
import struct
import sys
import time
from collections import Counter

from piper_sdk import C_PiperInterface_V2, LogLevel

port = sys.argv[1]
window_s = float(sys.argv[2])
leader_ids = {0x155, 0x156, 0x157, 0x159}


def _new_arm():
    arm = C_PiperInterface_V2(
        can_name=port,
        judge_flag=True,
        can_auto_init=True,
        logger_level=LogLevel.WARNING,
    )
    arm.ConnectPort(can_init=False, piper_init=False, start_thread=True)
    time.sleep(0.2)
    return arm


def _send(label, fn):
    arm = _new_arm()
    try:
        fn(arm)
        time.sleep(0.25)
    finally:
        arm.DisconnectPort()
    print(f"[OK] sent {label}", flush=True)


def _sniff(label):
    print(f"Window: {label}. Move {port} gently for {window_s:.1f}s.", flush=True)
    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.settimeout(0.05)
    sock.bind((port,))
    counts = Counter()
    examples = {}
    deadline = time.monotonic() + window_s
    try:
        while time.monotonic() < deadline:
            try:
                frame = sock.recv(16)
            except TimeoutError:
                continue
            can_id, can_dlc, data = struct.unpack("=IB3x8s", frame)
            can_id &= socket.CAN_EFF_MASK
            counts[can_id] += 1
            examples.setdefault(can_id, data[:can_dlc].hex())
    finally:
        sock.close()

    print(f"\n==== {label} ====")
    for can_id in sorted(leader_ids):
        example = f" example={examples[can_id]}" if can_id in examples else ""
        print(f"0x{can_id:03x}: {counts[can_id]}{example}")
    print("top raw ids:")
    for can_id, count in counts.most_common(12):
        print(f"  0x{can_id:03x}: {count} example={examples.get(can_id, '')}")
    ok = all(counts[x] > 0 for x in (0x155, 0x156, 0x157))
    print(f"leader_joint_frames_ok={ok}")
    return ok


def _snapshot(label):
    arm = _new_arm()
    try:
        msg = arm.GetArmJointCtrl()
        status = arm.GetArmStatus()
        jc = getattr(msg, "joint_ctrl", None)
        vals = None if jc is None else [getattr(jc, f"joint_{idx}", None) for idx in range(1, 7)]
        mode = getattr(getattr(status, "arm_status", None), "ctrl_mode", None)
        print(
            f"{label}: sdk_joint_ts={float(getattr(msg, 'time_stamp', 0.0) or 0.0):.3f} "
            f"sdk_joint={vals} status_ts={float(getattr(status, 'time_stamp', 0.0) or 0.0):.3f} "
            f"ctrl_mode={mode}",
            flush=True,
        )
    finally:
        arm.DisconnectPort()


results = {}

_snapshot("baseline before sequences")
results["baseline"] = _sniff("baseline/no command")

_send("leader_only: MasterSlaveConfig(0xFA)", lambda arm: arm.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00))
_snapshot("after leader_only")
results["leader_only"] = _sniff("after leader_only")

_send("restore_only: ReqMasterArmMoveToHome(0)", lambda arm: arm.ReqMasterArmMoveToHome(0))
_snapshot("after restore_only")
results["restore_only"] = _sniff("after restore_only")

_send("follower_only: MasterSlaveConfig(0xFC)", lambda arm: arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00))
_snapshot("after follower_only")
results["follower_only"] = _sniff("after follower_only")

_send("follower_then_leader: 0xFC -> 0xFA", lambda arm: (
    arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00),
    time.sleep(0.2),
    arm.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00),
))
_snapshot("after follower_then_leader")
results["follower_then_leader"] = _sniff("after follower_then_leader")

_send("follower_then_leader_restore: 0xFC -> 0xFA -> 0x191(0)", lambda arm: (
    arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00),
    time.sleep(0.2),
    arm.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00),
    time.sleep(0.2),
    arm.ReqMasterArmMoveToHome(0),
))
_snapshot("after follower_then_leader_restore")
results["follower_then_leader_restore"] = _sniff("after follower_then_leader_restore")

print("\n==== summary ====")
for name, ok in results.items():
    print(f"{name}: {ok}")
good = [name for name, ok in results.items() if ok]
if good:
    print("usable sequence(s):", ", ".join(good))
else:
    print("no tested sequence restored leader joint frames")
PY
