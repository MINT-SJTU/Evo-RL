from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.datasets.image_writer import image_array_to_pil_image


def extract_critical_phase_dataset(
    source_repo_id: str,
    source_root: Path,
    output_repo_id: str,
    output_root: Path,
    intervals: list[tuple[int, int, int] | tuple[int, int, int, str | None]],
    task: str,
    outcome_filter: str | None = None,
) -> Path | None:
    """Extract critical phase segments from a recorded dataset into a new dataset.

    Each interval becomes a separate episode in the output dataset.
    Intervals can be 3-tuples (ep_idx, start, end) or 4-tuples (..., outcome).

    Args:
        outcome_filter: If set, only extract intervals whose outcome matches
            this value (e.g. "success", "failure"). None means extract all.

    Returns output_root on success, None if no intervals.
    """
    if outcome_filter is not None:
        intervals = [iv for iv in intervals if len(iv) >= 4 and iv[3] == outcome_filter]

    if not intervals:
        logging.info("[CP Extraction] No intervals to extract (filter=%s).", outcome_filter)
        return None

    logging.info(f"[CP Extraction] Extracting {len(intervals)} segments...")

    source_ds = LeRobotDataset(source_repo_id, root=source_root, video_backend="pyav")

    # Determine which features to copy (exclude DEFAULT_FEATURES)
    copy_features = {k: v for k, v in source_ds.features.items() if k not in DEFAULT_FEATURES}

    output_ds = LeRobotDataset.create(
        output_repo_id,
        source_ds.fps,
        root=output_root,
        robot_type=source_ds.meta.robot_type,
        features=source_ds.features,
        use_videos=bool(source_ds.meta.video_keys),
    )

    for seg_idx, iv in enumerate(intervals):
        ep_idx, start_frame, end_frame = iv[0], iv[1], iv[2]
        n_frames = end_frame - start_frame
        logging.info(f"[CP Extraction] Segment {seg_idx}: ep={ep_idx} frames={start_frame}-{end_frame} ({n_frames})")

        ep_meta = source_ds.meta.episodes[ep_idx]
        ep_global_start = ep_meta["dataset_from_index"]

        for local_frame in range(start_frame, end_frame):
            global_idx = ep_global_start + local_frame
            item = source_ds[global_idx]

            frame = {"task": task}
            for key in copy_features:
                if key not in item:
                    continue
                val = item[key]
                if isinstance(val, torch.Tensor) and val.ndim == 3 and copy_features[key].get("dtype") in ("video", "image"):
                    frame[key] = image_array_to_pil_image(val.cpu().numpy())
                elif isinstance(val, torch.Tensor):
                    frame[key] = val.numpy()
                else:
                    frame[key] = val

            output_ds.add_frame(frame)

        output_ds.save_episode()

    output_ds.finalize()
    logging.info(f"[CP Extraction] Created {output_root} with {len(intervals)} episodes, {output_ds.num_frames} frames")
    return output_root
