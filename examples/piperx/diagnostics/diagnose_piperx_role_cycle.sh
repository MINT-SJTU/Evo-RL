#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate evo-rl
cd "$REPO_ROOT"

port="${1:-can3}"
leader_window_s="${2:-8}"

echo "PiPER-X role cycle diagnostic on ${port}"
echo "This sends only role-switch commands:"
echo "  1) ${port} -> leader 0xFA"
echo "  2) ${port} -> follower 0xFC"
echo "  3) ${port} -> leader 0xFA"
echo "Move the arm gently only during leader windows. Do not force it during follower window."

python - "$port" "$leader_window_s" <<'PY'
import socket
import struct
import sys
import time
from collections import Counter

from piper_sdk import C_PiperInterface_V2, LogLevel

port = sys.argv[1]
leader_window_s = float(sys.argv[2])
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


def _sniff(duration_s):
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


def _set_role(role):
    role_code = 0xFA if role == "leader" else 0xFC
    arm = _new_arm()
    try:
        arm.MasterSlaveConfig(role_code, 0x00, 0x00, 0x00)
        time.sleep(0.2)
    finally:
        arm.DisconnectPort()
    print(f"[OK] {port}: role={role}(0x{role_code:02X})", flush=True)


def _print_leader_counts(label, counts):
    print(f"\n==== {label} ====")
    for can_id in sorted(leader_ids):
        print(f"0x{can_id:03x}: {counts[can_id]}")
    missing = sorted(leader_ids - set(counts))
    if missing:
        print("missing:", " ".join(f"0x{x:03x}" for x in missing))
    else:
        print("all leader ids observed")


_set_role("leader")
print(f"Window A: move {port} gently for {leader_window_s:.1f}s.")
counts_a = _sniff(leader_window_s)
_print_leader_counts("after leader 0xFA", counts_a)

_set_role("follower")
print("Follower window: wait 3s. Do not force the arm if it is stiff.")
counts_b = _sniff(3.0)
_print_leader_counts("after follower 0xFC", counts_b)

_set_role("leader")
print(f"Window B: move {port} gently again for {leader_window_s:.1f}s.")
counts_c = _sniff(leader_window_s)
_print_leader_counts("after leader 0xFA again", counts_c)

if all(counts_a[x] > 0 for x in (0x155, 0x156, 0x157)) and all(
    counts_c[x] > 0 for x in (0x155, 0x156, 0x157)
):
    print("\nRESULT: leader -> follower -> leader switching produced leader frames again without power cycle.")
else:
    print("\nRESULT: switching did not reliably restore leader frames. Inspect firmware/state before integrating.")
PY
