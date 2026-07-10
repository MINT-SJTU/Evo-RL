#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate evo-rl
cd "$REPO_ROOT"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 leader|follower can2 [can3 ...]"
  echo "Example: $0 leader can2 can3"
  echo "Example: $0 follower can2"
  exit 2
fi

role="$1"
shift

python - "$role" "$@" <<'PY'
import socket
import struct
import sys
import time
from collections import Counter

from piper_sdk import C_PiperInterface_V2, LogLevel

role = sys.argv[1].lower()
ports = sys.argv[2:]
if role not in {"leader", "follower"}:
    raise SystemExit("role must be leader or follower")

role_code = 0xFA if role == "leader" else 0xFC
leader_ids = {0x155, 0x156, 0x157, 0x159}


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


def _read_firmware(arm):
    try:
        arm.SearchPiperFirmwareVersion()
        for _ in range(30):
            version = arm.GetPiperFirmwareVersion()
            if isinstance(version, str) and version.startswith("S-V"):
                return version
            time.sleep(0.1)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR:{exc}"
    return "UNKNOWN"


def _sniff_ids(port, duration_s):
    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.settimeout(0.05)
    sock.bind((port,))
    counts = Counter()
    deadline = time.monotonic() + duration_s
    try:
        while time.monotonic() < deadline:
            try:
                frame = sock.recv(16)
            except TimeoutError:
                continue
            can_id, can_dlc, data = struct.unpack("=IB3x8s", frame)
            del can_dlc, data
            counts[can_id & socket.CAN_EFF_MASK] += 1
    finally:
        sock.close()
    return counts


for port in ports:
    arm = _new_arm(port)
    try:
        version = _read_firmware(arm)
        arm.MasterSlaveConfig(role_code, 0x00, 0x00, 0x00)
        time.sleep(0.1)
        if role == "leader":
            # Do not call ReqMasterArmMoveToHome(0) here. On some PiPER-X leader
            # arms it stops 0x155/0x156/0x157 leader frames after follower->leader switching.
            time.sleep(0.2)
        print(f"[OK] {port}: firmware={version} role={role}(0x{role_code:02X})", flush=True)
    finally:
        arm.DisconnectPort()

if role == "leader":
    print("\nMove each leader arm gently during the next 6s to confirm 0x155/0x156/0x157 output.")
    for port in ports:
        counts = _sniff_ids(port, 6.0)
        observed = sorted(leader_ids & set(counts))
        print(f"{port}: leader ids observed: {' '.join(f'0x{x:03x}' for x in observed) or 'none'}")
        for can_id in sorted(leader_ids):
            print(f"  0x{can_id:03x}: {counts[can_id]}")
else:
    print("\nFollower role command sent. Do not force the arm if it becomes stiff or controlled.")
PY
