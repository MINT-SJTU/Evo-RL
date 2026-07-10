#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate evo-rl
cd "$REPO_ROOT"
duration_s="${1:-8}"
echo "Read-only leader frame diagnostic for ${duration_s}s."
echo "During this window, move LEFT leader (can2) and RIGHT leader (can3) gently."
echo "Looking for official leader joint frames 0x155/0x156/0x157."

python - "$duration_s" <<'PY'
import select
import socket
import struct
import sys
import time
from collections import Counter

from piper_sdk import C_PiperInterface_V2, LogLevel

duration_s = float(sys.argv[1])
ports = ("can2", "can3")
leader_ids = {0x155, 0x156, 0x157, 0x159}


def _open_can(port):
    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.setblocking(False)
    sock.bind((port,))
    return sock


def _read_sdk_snapshot(port):
    arm = C_PiperInterface_V2(
        can_name=port,
        judge_flag=True,
        can_auto_init=True,
        logger_level=LogLevel.WARNING,
    )
    try:
        arm.ConnectPort(can_init=False, piper_init=False, start_thread=True)
        time.sleep(0.25)
        joint = arm.GetArmJointCtrl()
        status = arm.GetArmStatus()
        joint_ts = float(getattr(joint, "time_stamp", 0.0) or 0.0)
        status_ts = float(getattr(status, "time_stamp", 0.0) or 0.0)
        joint_ctrl = getattr(joint, "joint_ctrl", None)
        vals = None
        if joint_ctrl is not None:
            vals = [getattr(joint_ctrl, f"joint_{idx}", None) for idx in range(1, 7)]
        mode = getattr(getattr(status, "arm_status", None), "ctrl_mode", None)
        return joint_ts, vals, status_ts, mode
    finally:
        arm.DisconnectPort()


socks = {port: _open_can(port) for port in ports}
counts = {port: Counter() for port in ports}
examples = {port: {} for port in ports}
deadline = time.monotonic() + duration_s

while time.monotonic() < deadline:
    readable, _, _ = select.select(list(socks.values()), [], [], 0.05)
    for sock in readable:
        port = next(name for name, item in socks.items() if item is sock)
        try:
            frame = sock.recv(16)
        except BlockingIOError:
            continue
        can_id, can_dlc, data = struct.unpack("=IB3x8s", frame)
        can_id &= socket.CAN_EFF_MASK
        counts[port][can_id] += 1
        if can_id in leader_ids and can_id not in examples[port]:
            examples[port][can_id] = data[:can_dlc].hex()

for sock in socks.values():
    sock.close()

for port in ports:
    print(f"\n==== {port} raw CAN ====")
    top = counts[port].most_common(20)
    if not top:
        print("no frames")
    for can_id, count in top:
        suffix = ""
        if can_id in examples[port]:
            suffix = f" example={examples[port][can_id]}"
        print(f"0x{can_id:03x} {count}{suffix}")
    missing = sorted(leader_ids - set(counts[port]))
    if missing:
        print("missing leader ids:", " ".join(f"0x{x:03x}" for x in missing))
    else:
        print("all expected leader ids observed")

for port in ports:
    joint_ts, vals, status_ts, mode = _read_sdk_snapshot(port)
    print(f"\n==== {port} piper_sdk snapshot ====")
    print(f"joint_ctrl_ts={joint_ts:.3f} joint_ctrl={vals}")
    print(f"status_ts={status_ts:.3f} ctrl_mode={mode}")
PY
