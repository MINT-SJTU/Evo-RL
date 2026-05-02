#!/usr/bin/env python
"""Visualize dual-arm joint velocity trajectories with action-chunk boundaries.

Reads a ``joint_vel_*.json`` file produced by ``lerobot_arx5_dual_infer.py``
when ``--joint-vel-dir`` is enabled and plots:

- Joint velocities over time (SDK 6-DoF and/or finite-difference 7-DoF).
- Vertical dashed lines + start/end labels at every action-chunk boundary.
- Alternating shaded background bands per chunk.
- Red ``x`` markers wherever the executed command was tagged as
  ``policy_safe_clipped`` (i.e. modified by the safe-mode joint-step clip),
  i.e. not a pure policy-inference action.

Usage examples::

    # Default: all joints, SDK velocity
    python Evo-RL/scripts/visualize_joint_vel.py --input traj/joint_vel/joint_vel_<ts>.json

    # Single joint, finite-difference velocity (gripper available only with diff source)
    python Evo-RL/scripts/visualize_joint_vel.py --input <file> --joint left_3 --source diff
    python Evo-RL/scripts/visualize_joint_vel.py --input <file> --joint right_grip --source diff

    # SDK + finite-difference together, save to PNG
    python Evo-RL/scripts/visualize_joint_vel.py --input <file> --source both --output out.png --no-show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


SDK_DIM = 6
DIFF_DIM = 7
ARMS = ("left", "right")


def _parse_joint_spec(spec: str, source: str) -> tuple[str, int]:
    """Parse ``left_3`` / ``right_grip`` style specs to ``(arm, channel_idx)``.

    Indices are 1-based in the spec ("joint 1" -> column 0). ``grip`` maps to
    column 6 and is only available when the chosen source includes the gripper
    channel (i.e. ``diff``; SDK velocities do not expose the gripper).
    """
    if "_" not in spec:
        raise ValueError(f"Invalid --joint spec '{spec}'. Expected e.g. left_1 or right_grip.")
    arm, name = spec.split("_", 1)
    arm = arm.lower()
    if arm not in ARMS:
        raise ValueError(f"Invalid arm '{arm}' in --joint spec. Use 'left' or 'right'.")
    if name == "grip":
        if source == "sdk":
            raise ValueError(
                "SDK velocity has no gripper channel. Use --source diff (or both) when "
                "selecting *_grip joints."
            )
        return arm, DIFF_DIM - 1
    try:
        idx_one_based = int(name)
    except ValueError as err:
        raise ValueError(f"Invalid joint index '{name}'. Expected integer 1..6 or 'grip'.") from err
    if idx_one_based < 1 or idx_one_based > 7:
        raise ValueError(f"Joint index out of range: {idx_one_based}. Expected 1..7.")
    if source == "sdk" and idx_one_based > SDK_DIM:
        raise ValueError(
            f"SDK velocity has only {SDK_DIM} channels; got joint index {idx_one_based}."
        )
    return arm, idx_one_based - 1


def _load(input_path: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in {input_path}.")
    t = np.asarray([s["t_rel_s"] for s in samples], dtype=np.float64)
    chunk_idx = np.asarray([int(s.get("chunk_index", -1)) for s in samples], dtype=np.int64)
    source = np.asarray([str(s.get("source", "policy")) for s in samples])
    velocities = {
        ("left", "sdk"): np.asarray([s["left_vel_sdk"] for s in samples], dtype=np.float64),
        ("right", "sdk"): np.asarray([s["right_vel_sdk"] for s in samples], dtype=np.float64),
        ("left", "diff"): np.asarray([s["left_vel_diff"] for s in samples], dtype=np.float64),
        ("right", "diff"): np.asarray([s["right_vel_diff"] for s in samples], dtype=np.float64),
    }
    return {
        "t": t,
        "chunk_idx": chunk_idx,
        "source": source,
        "velocities": velocities,
    }


def _chunk_segments(t: np.ndarray, chunk_idx: np.ndarray) -> list[tuple[int, int, int]]:
    """Return list of ``(start_sample, end_sample_inclusive, chunk_id)`` segments.

    Boundaries are detected wherever ``chunk_idx`` changes between consecutive
    samples; consecutive samples carrying the same chunk id are merged.
    """
    if len(chunk_idx) == 0:
        return []
    segments: list[tuple[int, int, int]] = []
    seg_start = 0
    for i in range(1, len(chunk_idx)):
        if chunk_idx[i] != chunk_idx[i - 1]:
            segments.append((seg_start, i - 1, int(chunk_idx[i - 1])))
            seg_start = i
    segments.append((seg_start, len(chunk_idx) - 1, int(chunk_idx[-1])))
    return segments


def _shade_chunks_and_mark_clipped(
    ax: Axes,
    t: np.ndarray,
    chunk_idx: np.ndarray,
    source: np.ndarray,
    y_values: np.ndarray,
) -> None:
    segments = _chunk_segments(t, chunk_idx)
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#CCCCCC", "#DDDDDD"])
    if not palette:
        palette = ["#CCCCCC", "#DDDDDD"]

    for seg_pos, (i0, i1, cid) in enumerate(segments):
        if cid < 0:
            # Sample produced outside any policy chunk; do not annotate as a chunk band.
            continue
        x0 = t[i0]
        x1 = t[i1] if i1 < len(t) - 1 else t[i1]
        # Light shaded band per chunk.
        ax.axvspan(x0, x1, color=palette[seg_pos % len(palette)], alpha=0.10, linewidth=0)
        # Start line + label.
        ax.axvline(x0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"chunk {cid} start",
            xy=(x0, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -10),
            textcoords="offset points",
            fontsize=7,
            color="black",
            rotation=90,
            va="top",
            ha="left",
        )
        # End line + label only if this is the last segment with this id.
        is_last_segment = (seg_pos == len(segments) - 1) or (segments[seg_pos + 1][2] != cid)
        if is_last_segment:
            ax.axvline(x1, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.annotate(
                f"chunk {cid} end",
                xy=(x1, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -10),
                textcoords="offset points",
                fontsize=7,
                color="gray",
                rotation=90,
                va="top",
                ha="left",
            )

    # Mark non-policy-pure samples with a red 'x'.
    non_pure = source != "policy"
    if np.any(non_pure):
        ax.scatter(
            t[non_pure],
            y_values[non_pure],
            marker="x",
            s=18,
            color="red",
            zorder=5,
            label="non-inference / clipped",
        )


def _plot_single_joint(
    data: dict[str, Any],
    joint_spec: str,
    sources: list[str],
    output: Path | None,
    show: bool,
) -> None:
    t = data["t"]
    chunk_idx = data["chunk_idx"]
    source = data["source"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for src in sources:
        arm, channel = _parse_joint_spec(joint_spec, src)
        y = data["velocities"][(arm, src)][:, channel]
        ax.plot(t, y, label=f"{arm} {src} ch{channel + 1}", linewidth=1.2)
    ax.set_xlabel("t_rel_s")
    ax.set_ylabel("joint velocity (rad/s)")
    ax.set_title(f"Joint velocity: {joint_spec} ({'+'.join(sources)})")

    # Reference y values for the red 'x' markers (use the first source).
    ref_arm, ref_channel = _parse_joint_spec(joint_spec, sources[0])
    _shade_chunks_and_mark_clipped(
        ax,
        t,
        chunk_idx,
        source,
        data["velocities"][(ref_arm, sources[0])][:, ref_channel],
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _plot_all_joints(
    data: dict[str, Any],
    sources: list[str],
    output: Path | None,
    show: bool,
) -> None:
    t = data["t"]
    chunk_idx = data["chunk_idx"]
    source = data["source"]
    n_cols = max((SDK_DIM if "sdk" in sources else 0), (DIFF_DIM if "diff" in sources else 0))
    n_rows = 2 * len(sources)  # rows: (left, right) per source
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for src_idx, src in enumerate(sources):
        dim = SDK_DIM if src == "sdk" else DIFF_DIM
        for arm_idx, arm in enumerate(ARMS):
            row = src_idx * 2 + arm_idx
            for col in range(n_cols):
                ax = axes[row][col]
                if col >= dim:
                    ax.axis("off")
                    continue
                y = data["velocities"][(arm, src)][:, col]
                ax.plot(t, y, linewidth=1.0)
                _shade_chunks_and_mark_clipped(ax, t, chunk_idx, source, y)
                ax.grid(True, alpha=0.3)
                channel_label = (
                    f"j{col + 1}" if not (src == "diff" and col == DIFF_DIM - 1) else "grip"
                )
                ax.set_title(f"{arm} {src} {channel_label}", fontsize=9)
                if row == n_rows - 1:
                    ax.set_xlabel("t_rel_s")
                if col == 0:
                    ax.set_ylabel("vel")

    fig.suptitle("Dual-arm joint velocities with action-chunk boundaries", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output is not None:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize dual-arm joint velocity trajectories with action-chunk boundaries. "
            "Use --joint to focus on a single channel; default plots all channels of the chosen source(s)."
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to joint_vel_*.json")
    parser.add_argument(
        "--joint",
        type=str,
        default=None,
        help="Single-joint spec, e.g. 'left_1', 'right_6', 'left_grip'. Defaults to plotting all joints.",
    )
    parser.add_argument(
        "--source",
        choices=("sdk", "diff", "both"),
        default="sdk",
        help="Which velocity source to display. 'both' overlays SDK + finite-difference.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG output path.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive matplotlib window.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")

    data = _load(args.input)
    sources: list[str] = ["sdk", "diff"] if args.source == "both" else [args.source]

    show = not args.no_show
    if args.joint is not None:
        _plot_single_joint(data, args.joint, sources, args.output, show)
    else:
        _plot_all_joints(data, sources, args.output, show)


if __name__ == "__main__":
    main()
