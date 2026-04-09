"""GPU-accelerated critical phase extraction.

Three optimization levels over extract_critical_phase_dataset():

Level 1 (extract_critical_phase_dataset_fast):
  - Batch video decode per interval (1 call per camera vs N calls)
  - h264 CPU encode instead of libsvtav1

Level 2 (extract_critical_phase_dataset_direct):
  - Batch video decode (same as Level 1)
  - Direct tensor→video encode, skip PNG write/read entirely
  - ~14x speedup over original
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import av
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.datasets.image_writer import image_array_to_pil_image


def _encode_frames_direct(
    frames: torch.Tensor,
    video_path: Path,
    fps: int,
    vcodec: str = "h264",
    pix_fmt: str = "yuv420p",
    g: int = 2,
) -> None:
    """Encode (N, C, H, W) float32 [0,1] tensor directly to video file."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    n, c, h, w = frames.shape
    frames_u8 = (frames * 255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()

    opts = {"g": str(g)}
    if vcodec == "h264_nvenc":
        opts.update({"preset": "p1", "rc": "constqp", "qp": "23"})
    else:
        opts.update({"crf": "23", "preset": "ultrafast"})

    with av.open(str(video_path), "w") as out:
        stream = out.add_stream(vcodec, fps, options=opts)
        stream.pix_fmt = pix_fmt
        stream.width = w
        stream.height = h
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(frames_u8[i], format="rgb24")
            for pkt in stream.encode(frame):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)


def _batch_decode_interval(source_ds, ep_meta, video_keys, start_frame, end_frame):
    """Batch-decode all video frames for an interval across all cameras."""
    fps = source_ds.fps
    decoded = {}
    for vid_key in video_keys:
        from_ts = ep_meta[f"videos/{vid_key}/from_timestamp"]
        timestamps = [from_ts + f / fps for f in range(start_frame, end_frame)]
        vpath = source_ds.root / source_ds.meta.get_video_file_path(
            ep_meta["episode_index"], vid_key,
        )
        decoded[vid_key] = decode_video_frames(
            vpath, timestamps, source_ds.tolerance_s, source_ds.video_backend,
        )
    return decoded


def extract_critical_phase_dataset_fast(
    source_repo_id: str,
    source_root: Path,
    output_repo_id: str,
    output_root: Path,
    intervals: list[tuple[int, int, int] | tuple[int, int, int, str | None]],
    task: str,
    outcome_filter: str | None = None,
    vcodec: str = "h264",
) -> Path | None:
    """Level 1: batch decode + fast codec (still writes PNGs)."""
    if outcome_filter is not None:
        intervals = [iv for iv in intervals if len(iv) >= 4 and iv[3] == outcome_filter]
    if not intervals:
        return None

    total_frames = sum(iv[2] - iv[1] for iv in intervals)
    logging.info("[Fast CP] %d segments (%d frames), vcodec=%s", len(intervals), total_frames, vcodec)
    t_start = time.perf_counter()

    source_ds = LeRobotDataset(
        source_repo_id, root=source_root, video_backend="pyav", tolerance_s=0.04,
    )
    copy_features = {k: v for k, v in source_ds.features.items() if k not in DEFAULT_FEATURES}
    video_keys = list(source_ds.meta.video_keys)
    non_video_keys = [k for k in copy_features if k not in video_keys]

    output_ds = LeRobotDataset.create(
        output_repo_id, source_ds.fps, root=output_root,
        robot_type=source_ds.meta.robot_type, features=source_ds.features,
        use_videos=bool(video_keys), vcodec=vcodec,
    )
    source_ds._ensure_hf_dataset_loaded()

    t_decode = t_write = 0.0
    done = 0

    for seg_idx, iv in enumerate(intervals):
        ep_idx, sf, ef = iv[0], iv[1], iv[2]
        ep_meta = source_ds.meta.episodes[ep_idx]
        ep_start = ep_meta["dataset_from_index"]

        t0 = time.perf_counter()
        decoded = _batch_decode_interval(source_ds, ep_meta, video_keys, sf, ef)
        t_decode += time.perf_counter() - t0

        t0 = time.perf_counter()
        for i in range(ef - sf):
            hf_item = source_ds.hf_dataset[ep_start + sf + i]
            frame = {"task": task}
            for k in video_keys:
                frame[k] = image_array_to_pil_image(decoded[k][i].cpu().numpy())
            for k in non_video_keys:
                if k not in hf_item:
                    continue
                v = hf_item[k]
                frame[k] = v.numpy() if isinstance(v, torch.Tensor) else v
            output_ds.add_frame(frame)
        output_ds.save_episode()
        t_write += time.perf_counter() - t0
        done += ef - sf

    output_ds.finalize()
    total = time.perf_counter() - t_start
    logging.info(
        "[Fast CP] Done: %d frames in %.1fs (decode=%.1fs, write=%.1fs, fps=%.1f)",
        done, total, t_decode, t_write, done / total,
    )
    return output_root


def extract_critical_phase_dataset_direct(
    source_repo_id: str,
    source_root: Path,
    output_repo_id: str,
    output_root: Path,
    intervals: list[tuple[int, int, int] | tuple[int, int, int, str | None]],
    task: str,
    outcome_filter: str | None = None,
    vcodec: str = "h264",
) -> Path | None:
    """Level 2: batch decode + direct encode (no PNG intermediate).

    Decoded tensors are encoded directly to MP4 via PyAV, completely
    bypassing the PNG write/read/encode cycle.
    """
    if outcome_filter is not None:
        intervals = [iv for iv in intervals if len(iv) >= 4 and iv[3] == outcome_filter]
    if not intervals:
        return None

    total_frames = sum(iv[2] - iv[1] for iv in intervals)
    logging.info("[Direct CP] %d segments (%d frames), vcodec=%s", len(intervals), total_frames, vcodec)
    t_start = time.perf_counter()

    source_ds = LeRobotDataset(
        source_repo_id, root=source_root, video_backend="pyav", tolerance_s=0.04,
    )
    copy_features = {k: v for k, v in source_ds.features.items() if k not in DEFAULT_FEATURES}
    video_keys = list(source_ds.meta.video_keys)
    non_video_keys = [k for k in copy_features if k not in video_keys]

    output_ds = LeRobotDataset.create(
        output_repo_id, source_ds.fps, root=output_root,
        robot_type=source_ds.meta.robot_type, features=source_ds.features,
        use_videos=bool(video_keys), vcodec=vcodec,
    )
    source_ds._ensure_hf_dataset_loaded()

    # ── Monkey-patches for direct encode (skip PNG) ──
    # 1. Skip PNG writing in add_frame
    def noop_save_image(image, fpath, compress_level=1):
        fpath.parent.mkdir(parents=True, exist_ok=True)
    output_ds._save_image = noop_save_image

    # 2. Make sample_images use in-memory data instead of reading PNGs
    # Buffer: vid_key -> list of (C, H, W) uint8 numpy arrays
    frame_data_buffer: dict[str, list[np.ndarray]] = {}

    from lerobot.datasets import compute_stats
    original_sample_images = compute_stats.sample_images

    def patched_sample_images(image_paths: list[str]) -> np.ndarray:
        if not image_paths:
            return original_sample_images(image_paths)
        for vid_key, frames_list in frame_data_buffer.items():
            if vid_key in image_paths[0]:
                sampled_idx = compute_stats.sample_indices(len(frames_list))
                imgs = None
                for i, idx in enumerate(sampled_idx):
                    img = frames_list[idx]
                    img = compute_stats.auto_downsample_height_width(img)
                    if imgs is None:
                        imgs = np.empty((len(sampled_idx), *img.shape), dtype=np.uint8)
                    imgs[i] = img
                return imgs
        return original_sample_images(image_paths)

    compute_stats.sample_images = patched_sample_images

    # 3. Direct encode from tensor buffer instead of reading PNGs
    episode_frames: dict[str, torch.Tensor] = {}

    from lerobot.datasets import video_utils
    from lerobot.datasets import lerobot_dataset as ld_module
    original_encode_vu = video_utils.encode_video_frames
    original_encode_ld = ld_module.encode_video_frames

    def direct_encode_wrapper(imgs_dir, video_path, fps, vcodec="h264", **kwargs):
        imgs_dir_str = str(imgs_dir)
        for vid_key, frames in episode_frames.items():
            if vid_key in imgs_dir_str:
                _encode_frames_direct(frames, Path(video_path), fps, vcodec=vcodec)
                return
        raise RuntimeError(f"No frame buffer for {imgs_dir}")

    # Patch both import sites
    video_utils.encode_video_frames = direct_encode_wrapper
    ld_module.encode_video_frames = direct_encode_wrapper

    t_decode = t_rest = 0.0
    done = 0

    try:
        for seg_idx, iv in enumerate(intervals):
            ep_idx, sf, ef = iv[0], iv[1], iv[2]
            n_frames = ef - sf
            ep_meta = source_ds.meta.episodes[ep_idx]
            ep_start = ep_meta["dataset_from_index"]

            # Batch decode
            t0 = time.perf_counter()
            decoded = _batch_decode_interval(source_ds, ep_meta, video_keys, sf, ef)
            t_decode += time.perf_counter() - t0

            t0 = time.perf_counter()
            # Fill buffers for encode and stats
            episode_frames.clear()
            episode_frames.update(decoded)
            frame_data_buffer.clear()
            for vid_key in video_keys:
                frames_u8 = (decoded[vid_key] * 255).byte().cpu().numpy()  # (N,C,H,W)
                frame_data_buffer[vid_key] = [frames_u8[i] for i in range(n_frames)]

            # add_frame: PNG write is no-op, fills episode_buffer with paths
            for i in range(n_frames):
                hf_item = source_ds.hf_dataset[ep_start + sf + i]
                frame = {"task": task}
                for k in video_keys:
                    frame[k] = image_array_to_pil_image(decoded[k][i].cpu().numpy())
                for k in non_video_keys:
                    if k not in hf_item:
                        continue
                    v = hf_item[k]
                    frame[k] = v.numpy() if isinstance(v, torch.Tensor) else v
                output_ds.add_frame(frame)

            # save_episode: encode uses direct_encode_wrapper, stats use in-memory data
            output_ds.save_episode(parallel_encoding=False)
            t_rest += time.perf_counter() - t0
            done += n_frames

            if (seg_idx + 1) % 5 == 0 or seg_idx == len(intervals) - 1:
                elapsed = time.perf_counter() - t_start
                logging.info(
                    "[Direct CP] %d/%d segs, %d frames, %.1fs (decode=%.1fs, rest=%.1fs)",
                    seg_idx + 1, len(intervals), done, elapsed, t_decode, t_rest,
                )

        output_ds.finalize()
    finally:
        video_utils.encode_video_frames = original_encode_vu
        ld_module.encode_video_frames = original_encode_ld
        compute_stats.sample_images = original_sample_images

    total = time.perf_counter() - t_start
    logging.info(
        "[Direct CP] Done: %d frames in %.1fs (decode=%.1fs, rest=%.1fs, fps=%.1f)",
        done, total, t_decode, t_rest, done / total,
    )
    return output_root
