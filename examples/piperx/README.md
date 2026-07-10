# PiPER-X Bimanual Teleoperation

Utilities for a four-CAN PiPER-X bimanual setup:

- `can0`: left follower arm
- `can1`: right follower arm
- `can2`: left leader arm
- `can3`: right leader arm

## Quick Start

```bash
cd /path/to/Evo-RL
./examples/piperx/setup_can_piperx.sh
./examples/piperx/run_bi_piperx_teleop.sh
```

The teleoperation script configures follower arms with SDK follower control
(`0xFC`) and leader arms with AgileX built-in leader drag mode (`0xFA`). It also
enables a debug fake-inference mode:

- Press `i` to request fake inference.
- If one side has not produced a leader frame yet, move that leader arm gently;
  fake inference starts automatically once both sides have a full pose.
- In fake inference, all joints hold the captured pose except `joint_1`, which
  sweeps slowly around the captured base angle.
- Press `i` again to restore the leader arms to drag mode and resume teleop.

## Diagnostics

Diagnostic scripts live in `examples/piperx/diagnostics/` and are intended for
checking leader frame publication and role-switch behavior on PiPER-X firmware.
