#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_samples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(f"No valid samples in trajectory file: {path}")
    return samples


def _to_series(samples: list[dict], arm_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray([float(item.get("t_rel_s", idx)) for idx, item in enumerate(samples)], dtype=np.float64)
    xyz = np.asarray([item.get(arm_key, [np.nan, np.nan, np.nan]) for item in samples], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"Invalid {arm_key} data shape: {xyz.shape}")
    return t, xyz[:, 0], xyz[:, 1], xyz[:, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot end-effector xyz trajectories from infer record JSON.")
    parser.add_argument("--traj-json", type=Path, required=True, help="Path to end_effector_traj_*.json")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window.")
    args = parser.parse_args()

    traj_json = args.traj_json.expanduser().resolve()
    samples = _load_samples(traj_json)

    t, lx, ly, lz = _to_series(samples, "left_xyz")
    _, rx, ry, rz = _to_series(samples, "right_xyz")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axis_names = ("x", "y", "z")
    left_series = (lx, ly, lz)
    right_series = (rx, ry, rz)

    for idx, ax in enumerate(axes):
        ax.plot(t, left_series[idx], label=f"left_{axis_names[idx]}", linewidth=1.6)
        ax.plot(t, right_series[idx], label=f"right_{axis_names[idx]}", linewidth=1.6)
        ax.set_ylabel(f"{axis_names[idx]} (m)")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"End-Effector Trajectory: {traj_json.name}")
    fig.tight_layout()

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else traj_json.with_name(traj_json.stem + "_xyz.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"[DONE] saved plot: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
