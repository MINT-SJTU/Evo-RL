#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_array(samples: list[dict], key: str) -> np.ndarray:
    if not samples:
        return np.zeros((0, 7), dtype=np.float64)
    return np.asarray([sample[key] for sample in samples], dtype=np.float64)


def _boundary_indices(num_samples: int, chunk_len: int) -> np.ndarray:
    # Boundary is between k and k+1 where (k+1) % chunk_len == 0.
    return np.arange(chunk_len - 1, num_samples - 1, chunk_len, dtype=np.int64)


def _print_boundary_stats(name: str, cmd: np.ndarray, chunk_len: int) -> None:
    deltas = np.abs(np.diff(cmd, axis=0))
    if deltas.shape[0] == 0:
        print(f"[{name}] not enough samples for delta statistics.")
        return

    boundaries = _boundary_indices(cmd.shape[0], chunk_len)
    mask = np.zeros(deltas.shape[0], dtype=bool)
    mask[boundaries] = True
    boundary_d = deltas[mask]
    intra_d = deltas[~mask]

    print(f"\n[{name}] chunk_len={chunk_len}, num_boundaries={len(boundaries)}")
    for joint_idx in range(cmd.shape[1]):
        b = boundary_d[:, joint_idx] if boundary_d.size else np.array([], dtype=np.float64)
        i = intra_d[:, joint_idx] if intra_d.size else np.array([], dtype=np.float64)
        b_mean = float(np.mean(b)) if b.size else float("nan")
        b_max = float(np.max(b)) if b.size else float("nan")
        i_p95 = float(np.percentile(i, 95)) if i.size else float("nan")
        ratio = b_mean / i_p95 if i.size and i_p95 > 0 else float("nan")
        print(
            f"  J{joint_idx + 1}: boundary_mean={b_mean:.6f}, "
            f"boundary_max={b_max:.6f}, intra_p95={i_p95:.6f}, mean/p95={ratio:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ARX5 dual-arm commanded vs actual joint trajectories."
    )
    parser.add_argument("--json", type=Path, required=True, help="Path to joint_traj_*.json")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path (e.g. traj.png). If omitted, show interactively.",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=None,
        help="If set, mark chunk boundaries and print command-jump statistics at boundaries.",
    )
    args = parser.parse_args()

    payload = _load_payload(args.json)
    samples = payload.get("samples", [])
    if not samples:
        raise RuntimeError(f"No samples found in {args.json}")

    t = np.asarray([float(sample.get("t_rel_s", idx)) for idx, sample in enumerate(samples)], dtype=np.float64)
    left_cmd = _to_array(samples, "left_cmd")
    left_actual = _to_array(samples, "left_actual")
    right_cmd = _to_array(samples, "right_cmd")
    right_actual = _to_array(samples, "right_actual")

    boundary_t = np.array([], dtype=np.float64)
    if args.chunk_len is not None:
        if args.chunk_len <= 0:
            raise ValueError("--chunk-len must be positive.")
        boundary_idx = _boundary_indices(len(samples), args.chunk_len)
        boundary_t = t[boundary_idx] if boundary_idx.size else np.array([], dtype=np.float64)
        _print_boundary_stats("Left CMD", left_cmd, args.chunk_len)
        _print_boundary_stats("Right CMD", right_cmd, args.chunk_len)

    fig, axes = plt.subplots(7, 2, figsize=(14, 20), sharex=True)
    fig.suptitle(f"Joint Trajectory: {args.json.name}", fontsize=14)

    for joint_idx in range(7):
        ax_l = axes[joint_idx, 0]
        ax_r = axes[joint_idx, 1]

        ax_l.plot(t, left_cmd[:, joint_idx], label="cmd", linewidth=1.5)
        ax_l.plot(t, left_actual[:, joint_idx], label="actual", linewidth=1.2)
        ax_l.set_ylabel(f"J{joint_idx + 1}")
        if joint_idx == 0:
            ax_l.set_title("Left Arm")
        if joint_idx == 6:
            ax_l.set_xlabel("t (s)")
        ax_l.grid(alpha=0.3)
        if joint_idx == 0:
            ax_l.legend(loc="best")
        if boundary_t.size:
            for bt in boundary_t:
                ax_l.axvline(bt, color="tab:red", linestyle="--", linewidth=0.7, alpha=0.35)

        ax_r.plot(t, right_cmd[:, joint_idx], label="cmd", linewidth=1.5)
        ax_r.plot(t, right_actual[:, joint_idx], label="actual", linewidth=1.2)
        if joint_idx == 0:
            ax_r.set_title("Right Arm")
        if joint_idx == 6:
            ax_r.set_xlabel("t (s)")
        ax_r.grid(alpha=0.3)
        if joint_idx == 0:
            ax_r.legend(loc="best")
        if boundary_t.size:
            for bt in boundary_t:
                ax_r.axvline(bt, color="tab:red", linestyle="--", linewidth=0.7, alpha=0.35)

    plt.tight_layout()
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200)
        print(f"Saved plot to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

