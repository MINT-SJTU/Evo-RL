#!/usr/bin/env python

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames


def _ensure_pil_rgb_frame(frame_obj) -> Image.Image:
    if isinstance(frame_obj, Image.Image):
        return frame_obj.convert("RGB")
    if isinstance(frame_obj, np.ndarray):
        arr = frame_obj
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                max_val = float(np.max(arr)) if arr.size > 0 else 1.0
                if max_val <= 1.0 + 1e-6:
                    arr = np.clip(arr, 0.0, 1.0) * 255.0
                else:
                    arr = np.clip(arr, 0.0, 255.0)
            else:
                arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported frame type for visualization: {type(frame_obj)}")


def _extract_image_frames(raw_dataset, image_key: str, positions: np.ndarray) -> list[Image.Image]:
    column_data = raw_dataset[image_key]
    frames: list[Image.Image] = []
    for pos in positions:
        frame_obj = column_data[int(pos)]
        frames.append(_ensure_pil_rgb_frame(frame_obj))
    return frames


def _smooth_1d(arr: np.ndarray, window: int = 1) -> np.ndarray:
    """Savitzky-Golay smoothing for 1-D array. Use window=1 to disable."""
    if window <= 1 or arr.shape[0] < 5:
        return arr.copy()

    w = min(int(window), int(arr.shape[0]))
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return arr.copy()

    return savgol_filter(arr, window_length=w, polyorder=3).astype(arr.dtype)


def _select_video_key(camera_keys: list[str], requested_video_key: str | None) -> str:
    if len(camera_keys) == 0:
        raise ValueError("No camera key found in dataset.")
    if requested_video_key is not None:
        if requested_video_key not in camera_keys:
            raise ValueError(
                f"Unknown video_key '{requested_video_key}'. Available camera keys: {camera_keys}"
            )
        return requested_video_key

    for key in camera_keys:
        lower = key.lower()
        if ".front" in lower or "_front" in lower or lower.endswith("front"):
            return key
    return camera_keys[0]


def _select_video_keys(
    camera_keys: list[str],
    requested_video_keys: str | None,
    requested_video_key: str | None,
) -> list[str]:
    if len(camera_keys) == 0:
        raise ValueError("No camera key found in dataset.")

    if requested_video_keys is None or not requested_video_keys.strip():
        return [_select_video_key(camera_keys, requested_video_key)]

    resolved: list[str] = []
    for raw_key in requested_video_keys.split(","):
        key = raw_key.strip()
        if not key:
            continue
        if key not in camera_keys:
            raise ValueError(f"Unknown video key '{key}'. Available camera keys: {camera_keys}")
        if key not in resolved:
            resolved.append(key)

    if not resolved:
        raise ValueError("'viz.video_keys' is empty after parsing. Provide comma-separated camera keys.")

    return resolved


def _parse_episodes_arg(episodes_arg: str, total_episodes: int) -> list[int]:
    value = episodes_arg.strip().lower()
    if value == "all":
        return list(range(total_episodes))

    parsed: set[int] = set()
    for token in episodes_arg.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range '{part}'.")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(part))

    episodes = sorted(parsed)
    for ep in episodes:
        if ep < 0 or ep >= total_episodes:
            raise ValueError(f"Episode index out of range: {ep}, total_episodes={total_episodes}.")
    return episodes


def _to_1d_float(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _to_1d_int(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _curve_points(
    values: np.ndarray,
    current_step: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    n = len(values)
    if n == 0:
        return []

    denom_x = max(1, n - 1)
    denom_y = max(1e-6, y_max - y_min)

    points = []
    last_step = min(current_step, n - 1)
    for i in range(last_step + 1):
        x = int(round(x0 + width * (i / denom_x)))
        y_norm = np.clip((float(values[i]) - y_min) / denom_y, 0.0, 1.0)
        y = int(round(y0 + (1.0 - y_norm) * height))
        points.append((x, y))
    return points


def _draw_overlay(
    frame: Image.Image,
    values: np.ndarray,
    current_step: int,
    advantage_t: float,
    acp_t: int,
    highlight_current_point: bool,
    y_min: float,
    y_max: float,
    indicators: np.ndarray | None = None,
) -> Image.Image:
    # Keep signature/call-sites stable; style does not use these signals.
    _ = (advantage_t, acp_t, highlight_current_point, indicators)

    rgba = frame.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = rgba.size
    n = len(values)
    if n == 0:
        return Image.alpha_composite(rgba, overlay).convert("RGB")

    margin_x = max(12, width // 96)
    margin_top = max(8, height // 72)
    margin_bottom = max(24, height // 36)
    chart_x0 = margin_x
    chart_x1 = width - margin_x
    chart_y0 = margin_top
    chart_y1 = height - margin_bottom
    chart_w = max(1, chart_x1 - chart_x0)
    chart_h = max(1, chart_y1 - chart_y0)

    denom_y = max(1e-6, y_max - y_min)
    padding = denom_y * 0.05
    val_min = y_min - padding
    val_max = y_max + padding

    last = min(current_step, n - 1)
    denom_x = max(1, n - 1)

    # Light grid + current-frame vertical cursor.
    grid_color = (255, 255, 255, 30)
    for frac in (0.25, 0.5, 0.75):
        gy = int(round(chart_y0 + frac * chart_h))
        draw.line([(chart_x0, gy), (chart_x1, gy)], fill=grid_color, width=1)
    cx = int(round(chart_x0 + chart_w * (last / denom_x)))
    draw.line([(cx, chart_y0), (cx, chart_y1)], fill=(255, 255, 255, 40), width=1)

    points = _curve_points(
        values=values,
        current_step=last,
        x0=chart_x0,
        y0=chart_y0,
        width=chart_w,
        height=chart_h,
        y_min=val_min,
        y_max=val_max,
    )

    curve_width = max(2, width // 600)
    if len(points) >= 2:
        draw.line(points, fill=(100, 200, 255, 220), width=curve_width)

    if len(points) >= 1:
        px, py = points[-1]
        radius = max(4, curve_width + 2)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(255, 255, 100, 255))
        draw.ellipse(
            (px - radius - 2, py - radius - 2, px + radius + 2, py + radius + 2),
            outline=(255, 255, 100, 180),
            width=1,
        )

        value_font = _load_font(max(16, height // 22))
        value_text = f"{float(values[last]):.4f}"
        value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
        text_w = value_bbox[2] - value_bbox[0]
        text_h = value_bbox[3] - value_bbox[1]
        tx = min(px + 10, width - text_w - 10)
        ty = max(py - text_h - 12, chart_y0 + 2)
        draw.rectangle((tx - 4, ty - 2, tx + text_w + 4, ty + text_h + 2), fill=(0, 0, 0, 160))
        draw.text((tx, ty), value_text, fill=(255, 255, 100, 255), font=value_font)

    small_font = _load_font(max(13, height // 30))
    frame_text = f"frame {last}/{n - 1}"
    frame_bbox = draw.textbbox((0, 0), frame_text, font=small_font)
    frame_w = frame_bbox[2] - frame_bbox[0]
    frame_h = frame_bbox[3] - frame_bbox[1]
    fx = width - frame_w - margin_x - 4
    fy = height - frame_h - 6
    draw.rectangle((fx - 3, fy - 1, fx + frame_w + 3, fy + frame_h + 1), fill=(0, 0, 0, 140))
    draw.text((fx, fy), frame_text, fill=(200, 200, 200, 220), font=small_font)

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _build_output_video_path(output_dir: Path, repo_id: str, video_key: str, episode_index: int) -> Path:
    repo_tag = repo_id.replace("/", "_")
    key_tag = video_key.replace(".", "_")
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{key_tag}.mp4"


def _build_output_video_path_multiview(
    output_dir: Path,
    repo_id: str,
    video_keys: list[str],
    episode_index: int,
) -> Path:
    repo_tag = repo_id.replace("/", "_")
    keys_tag = "__".join(key.replace(".", "_") for key in video_keys)
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{keys_tag}_multiview.mp4"


def _decode_frames_at_timestamps(
    video_file: Path,
    timestamps_s: np.ndarray,
    tolerance_s: float,
    backend: str | None,
) -> list[Image.Image]:
    if timestamps_s.size == 0:
        return []
    frames = decode_video_frames(
        video_path=video_file,
        timestamps=timestamps_s.tolist(),
        tolerance_s=tolerance_s,
        backend=backend,
    )
    np_frames = frames.detach().cpu().numpy()
    if np_frames.ndim != 4:
        raise ValueError(f"Unexpected decoded frame tensor shape: {np_frames.shape}")
    if np_frames.shape[1] in (1, 3):
        np_frames = np.transpose(np_frames, (0, 2, 3, 1))
    if np_frames.dtype != np.uint8:
        if np.issubdtype(np_frames.dtype, np.floating):
            max_val = float(np.max(np_frames)) if np_frames.size > 0 else 1.0
            if max_val <= 1.0 + 1e-6:
                np_frames = np.clip(np_frames, 0.0, 1.0) * 255.0
            else:
                np_frames = np.clip(np_frames, 0.0, 255.0)
        else:
            np_frames = np.clip(np_frames, 0, 255)
        np_frames = np_frames.astype(np.uint8)
    return [Image.fromarray(np_frames[i]) for i in range(np_frames.shape[0])]


def _get_episode_video_time_bounds(
    dataset: LeRobotDataset,
    episode_index: int,
    video_key: str,
) -> tuple[float, float | None]:
    episodes = getattr(dataset.meta, "episodes", None)
    if episodes is None:
        return 0.0, None
    episodes_ds = episodes.with_format(None)
    if "episode_index" not in episodes_ds.column_names:
        return 0.0, None

    episode_indices = np.asarray(episodes_ds["episode_index"], dtype=np.int64).reshape(-1)
    matched = np.flatnonzero(episode_indices == episode_index)
    if matched.size == 0:
        return 0.0, None
    row = int(matched[0])

    from_col = f"videos/{video_key}/from_timestamp"
    to_col = f"videos/{video_key}/to_timestamp"
    from_ts = 0.0
    to_ts: float | None = None
    if from_col in episodes_ds.column_names:
        from_ts = float(episodes_ds[from_col][row])
    if to_col in episodes_ds.column_names:
        to_ts = float(episodes_ds[to_col][row])
    return from_ts, to_ts


def _get_video_encode_options(vcodec: str) -> tuple[dict[str, str], str]:
    if vcodec == "libsvtav1":
        return {"g": "2", "crf": "30", "preset": "12"}, "yuv420p"
    if vcodec == "h264_nvenc":
        return {
            "preset": "p4",
            "rc": "vbr",
            "cq": "28",
            "b": "0",
            "g": "60",
        }, "yuv420p"
    return {"g": "2", "crf": "30"}, "yuv420p"


def _get_episode_value_bounds(ep_values: np.ndarray) -> tuple[float, float]:
    if ep_values.shape[0] == 0:
        raise ValueError("Cannot determine value bounds from an empty episode.")
    return float(np.min(ep_values)), float(np.max(ep_values))


def _encode_pil_to_video(
    frames: list[Image.Image],
    video_path: Path,
    fps: int,
    vcodec: str,
) -> None:
    """Encode PIL frames directly to video without writing intermediate PNG files."""
    import av as _av

    video_options, pix_fmt = _get_video_encode_options(vcodec)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with _av.open(str(video_path), "w") as output:
        out_stream = output.add_stream(vcodec, fps, options=video_options)
        out_stream.pix_fmt = pix_fmt
        out_stream.width = frames[0].width
        out_stream.height = frames[0].height
        for pil_img in frames:
            av_frame = _av.VideoFrame.from_image(pil_img.convert("RGB"))
            for packet in out_stream.encode(av_frame):
                output.mux(packet)
        for packet in out_stream.encode():
            output.mux(packet)


def _export_single_episode(
    src_video_path: Path,
    dst_video_path: Path,
    ep_values: np.ndarray,
    ep_advantages: np.ndarray,
    ep_indicators: np.ndarray,
    episode_timestamps_s: np.ndarray,
    fps: int,
    vcodec: str,
    tolerance_s: float,
    video_backend: str | None,
    frame_storage_mode: str = "memory",
    temp_dir_root: Path | None = None,
    smooth_window: int = 1,
) -> Path:
    ep_values = _smooth_1d(ep_values, smooth_window)
    ep_advantages = _smooth_1d(ep_advantages, smooth_window)
    y_min, y_max = _get_episode_value_bounds(ep_values)
    decoded_frames = _decode_frames_at_timestamps(
        video_file=src_video_path,
        timestamps_s=episode_timestamps_s,
        tolerance_s=tolerance_s,
        backend=video_backend,
    )
    n_frames = min(len(decoded_frames), len(ep_values))
    if n_frames == 0:
        raise ValueError(f"No decoded frames for video: {src_video_path}")

    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None,
            prefix=f"{dst_video_path.stem}-frames-",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            for i in range(n_frames):
                frame = decoded_frames[i]
                composed = _draw_overlay(
                    frame=frame,
                    values=ep_values,
                    current_step=i,
                    advantage_t=float(ep_advantages[i]),
                    acp_t=int(ep_indicators[i]),
                    highlight_current_point=(int(ep_indicators[i]) == 1),
                    y_min=y_min,
                    y_max=y_max,
                    indicators=ep_indicators,
                )
                composed.save(temp_path / f"frame-{i:06d}.png")

            encode_video_frames(
                imgs_dir=temp_path,
                video_path=dst_video_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=True,
            )
        return dst_video_path

    composed_frames: list[Image.Image] = []
    for i in range(n_frames):
        composed = _draw_overlay(
            frame=decoded_frames[i],
            values=ep_values,
            current_step=i,
            advantage_t=float(ep_advantages[i]),
            acp_t=int(ep_indicators[i]),
            highlight_current_point=(int(ep_indicators[i]) == 1),
            y_min=y_min,
            y_max=y_max,
            indicators=ep_indicators,
        )
        composed_frames.append(composed)

    _encode_pil_to_video(composed_frames, dst_video_path, fps, vcodec)
    return dst_video_path


def _export_single_episode_from_frames(
    decoded_frames: list[Image.Image],
    dst_video_path: Path,
    ep_values: np.ndarray,
    ep_advantages: np.ndarray,
    ep_indicators: np.ndarray,
    fps: int,
    vcodec: str,
    frame_storage_mode: str = "memory",
    temp_dir_root: Path | None = None,
    smooth_window: int = 1,
) -> Path:
    ep_values = _smooth_1d(ep_values, smooth_window)
    ep_advantages = _smooth_1d(ep_advantages, smooth_window)
    y_min, y_max = _get_episode_value_bounds(ep_values)
    n_frames = min(len(decoded_frames), len(ep_values))
    if n_frames == 0:
        raise ValueError("No decoded frames available for image-based visualization.")

    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None,
            prefix=f"{dst_video_path.stem}-frames-",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            for i in range(n_frames):
                composed = _draw_overlay(
                    frame=decoded_frames[i],
                    values=ep_values,
                    current_step=i,
                    advantage_t=float(ep_advantages[i]) if i < len(ep_advantages) else 0.0,
                    acp_t=int(ep_indicators[i]) if i < len(ep_indicators) else 0,
                    highlight_current_point=False,
                    y_min=y_min,
                    y_max=y_max,
                    indicators=ep_indicators,
                )
                composed.save(temp_path / f"frame-{i:06d}.png")

            encode_video_frames(
                imgs_dir=temp_path,
                video_path=dst_video_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=True,
            )
        return dst_video_path

    composed_frames: list[Image.Image] = []
    for i in range(n_frames):
        composed_frames.append(
            _draw_overlay(
                frame=decoded_frames[i],
                values=ep_values,
                current_step=i,
                advantage_t=float(ep_advantages[i]) if i < len(ep_advantages) else 0.0,
                acp_t=int(ep_indicators[i]) if i < len(ep_indicators) else 0,
                highlight_current_point=False,
                y_min=y_min,
                y_max=y_max,
                indicators=ep_indicators,
            )
        )

    _encode_pil_to_video(composed_frames, dst_video_path, fps, vcodec)
    return dst_video_path


def _export_single_episode_multiview(
    src_video_paths: list[Path],
    camera_labels: list[str],
    dst_video_path: Path,
    ep_values: np.ndarray,
    ep_advantages: np.ndarray,
    ep_indicators: np.ndarray,
    episode_timestamps_per_cam: list[np.ndarray],
    fps: int,
    vcodec: str,
    tolerance_s: float,
    video_backend: str | None,
    frame_storage_mode: str = "memory",
    temp_dir_root: Path | None = None,
    smooth_window: int = 1,
) -> Path:
    ep_values = _smooth_1d(ep_values, smooth_window)
    ep_advantages = _smooth_1d(ep_advantages, smooth_window)
    y_min, y_max = _get_episode_value_bounds(ep_values)

    all_cam_frames: list[list[Image.Image]] = []
    for src_path, ts in zip(src_video_paths, episode_timestamps_per_cam, strict=True):
        frames = _decode_frames_at_timestamps(
            video_file=src_path,
            timestamps_s=ts,
            tolerance_s=tolerance_s,
            backend=video_backend,
        )
        all_cam_frames.append(frames)

    n_frames = min(len(f) for f in all_cam_frames)
    n_frames = min(n_frames, len(ep_values))
    if n_frames == 0:
        raise ValueError(f"No decoded frames for multiview video: {dst_video_path}")

    single_w = all_cam_frames[0][0].width
    single_h = all_cam_frames[0][0].height
    n_cams = len(all_cam_frames)
    total_w = single_w * n_cams

    label_font_size = max(14, single_h // 30)
    label_font = _load_font(label_font_size)

    def _compose_frame(i: int) -> Image.Image:
        wide_frame = Image.new("RGB", (total_w, single_h))
        for cam_idx, cam_frames in enumerate(all_cam_frames):
            wide_frame.paste(cam_frames[i].resize((single_w, single_h)), (cam_idx * single_w, 0))

        label_draw = ImageDraw.Draw(wide_frame)
        for cam_idx, label in enumerate(camera_labels):
            short_label = label.split(".")[-1]
            lx = cam_idx * single_w + 8
            ly = 4
            bbox = label_draw.textbbox((lx, ly), short_label, font=label_font)
            label_draw.rectangle(
                (bbox[0] - 2, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
                fill=(0, 0, 0, 180),
            )
            label_draw.text((lx, ly), short_label, fill=(255, 255, 200), font=label_font)

        return _draw_overlay(
            frame=wide_frame,
            values=ep_values,
            current_step=i,
            advantage_t=float(ep_advantages[i]) if i < len(ep_advantages) else 0.0,
            acp_t=int(ep_indicators[i]) if i < len(ep_indicators) else 0,
            highlight_current_point=False,
            y_min=y_min,
            y_max=y_max,
            indicators=ep_indicators,
        )

    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None,
            prefix=f"{dst_video_path.stem}-frames-",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            for i in range(n_frames):
                _compose_frame(i).save(temp_path / f"frame-{i:06d}.png")

            encode_video_frames(
                imgs_dir=temp_path,
                video_path=dst_video_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=True,
            )
        return dst_video_path

    composed_frames: list[Image.Image] = []
    for i in range(n_frames):
        composed_frames.append(_compose_frame(i))

    _encode_pil_to_video(composed_frames, dst_video_path, fps, vcodec)
    return dst_video_path


def _export_single_episode_multiview_from_frames(
    all_cam_frames: list[list[Image.Image]],
    camera_labels: list[str],
    dst_video_path: Path,
    ep_values: np.ndarray,
    ep_advantages: np.ndarray,
    ep_indicators: np.ndarray,
    fps: int,
    vcodec: str,
    frame_storage_mode: str = "memory",
    temp_dir_root: Path | None = None,
    smooth_window: int = 1,
) -> Path:
    ep_values = _smooth_1d(ep_values, smooth_window)
    ep_advantages = _smooth_1d(ep_advantages, smooth_window)
    y_min, y_max = _get_episode_value_bounds(ep_values)

    n_frames = min(len(frames) for frames in all_cam_frames)
    n_frames = min(n_frames, len(ep_values))
    if n_frames == 0:
        raise ValueError(f"No frames available for multiview visualization: {dst_video_path}")

    single_w = all_cam_frames[0][0].width
    single_h = all_cam_frames[0][0].height
    n_cams = len(all_cam_frames)
    total_w = single_w * n_cams

    label_font_size = max(14, single_h // 30)
    label_font = _load_font(label_font_size)

    def _compose_frame(i: int) -> Image.Image:
        wide_frame = Image.new("RGB", (total_w, single_h))
        for cam_idx, cam_frames in enumerate(all_cam_frames):
            wide_frame.paste(cam_frames[i].resize((single_w, single_h)), (cam_idx * single_w, 0))

        label_draw = ImageDraw.Draw(wide_frame)
        for cam_idx, label in enumerate(camera_labels):
            short_label = label.split(".")[-1]
            lx = cam_idx * single_w + 8
            ly = 4
            bbox = label_draw.textbbox((lx, ly), short_label, font=label_font)
            label_draw.rectangle(
                (bbox[0] - 2, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
                fill=(0, 0, 0, 180),
            )
            label_draw.text((lx, ly), short_label, fill=(255, 255, 200), font=label_font)

        return _draw_overlay(
            frame=wide_frame,
            values=ep_values,
            current_step=i,
            advantage_t=float(ep_advantages[i]) if i < len(ep_advantages) else 0.0,
            acp_t=int(ep_indicators[i]) if i < len(ep_indicators) else 0,
            highlight_current_point=False,
            y_min=y_min,
            y_max=y_max,
            indicators=ep_indicators,
        )

    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None,
            prefix=f"{dst_video_path.stem}-frames-",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            for i in range(n_frames):
                _compose_frame(i).save(temp_path / f"frame-{i:06d}.png")

            encode_video_frames(
                imgs_dir=temp_path,
                video_path=dst_video_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=True,
            )
        return dst_video_path

    composed_frames: list[Image.Image] = []
    for i in range(n_frames):
        composed_frames.append(_compose_frame(i))

    _encode_pil_to_video(composed_frames, dst_video_path, fps, vcodec)
    return dst_video_path


def _export_overlay_videos(
    dataset: LeRobotDataset,
    value_field: str,
    advantage_field: str,
    indicator_field: str,
    viz_episodes: str,
    video_key: str | None,
    video_keys: str | None,
    output_dir: Path,
    overwrite: bool,
    vcodec: str,
    frame_storage_mode: str = "memory",
    smooth_window: int = 1,
) -> list[Path]:
    selected_video_keys = _select_video_keys(
        camera_keys=list(dataset.meta.camera_keys),
        requested_video_keys=video_keys,
        requested_video_key=video_key,
    )
    multiview_mode = len(selected_video_keys) > 1

    raw_dataset = dataset.hf_dataset.with_format(None)
    column_names = set(raw_dataset.column_names)

    if value_field not in column_names:
        raise KeyError(f"Missing value field '{value_field}' in dataset.")

    values_all = _to_1d_float(raw_dataset[value_field])
    if advantage_field in column_names:
        advantages_all = _to_1d_float(raw_dataset[advantage_field])
    else:
        advantages_all = np.zeros_like(values_all, dtype=np.float32)

    if indicator_field in column_names:
        indicators_all = _to_1d_int(raw_dataset[indicator_field])
    else:
        indicators_all = np.zeros(values_all.shape[0], dtype=np.int64)

    episode_indices_all = np.asarray(raw_dataset["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices_all = np.asarray(raw_dataset["frame_index"], dtype=np.int64).reshape(-1)
    if "timestamp" in column_names:
        timestamps_all = np.asarray(raw_dataset["timestamp"], dtype=np.float64).reshape(-1)
    else:
        timestamps_all = frame_indices_all.astype(np.float64) / float(dataset.fps)

    if dataset.episodes is not None:
        available_episodes = sorted(dataset.episodes)
    else:
        available_episodes = list(range(dataset.meta.total_episodes))

    if viz_episodes.strip().lower() == "all":
        episodes = available_episodes
    else:
        requested = _parse_episodes_arg(viz_episodes, dataset.meta.total_episodes)
        episodes = [ep for ep in requested if ep in set(available_episodes)]

    output_dir.mkdir(parents=True, exist_ok=True)

    fps = int(dataset.fps)
    tolerance_s = float(getattr(dataset, "tolerance_s", 1e-4))
    video_backend = getattr(dataset, "video_backend", None)
    written_paths: list[Path] = []

    if multiview_mode:
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] == 0:
                continue

            ep_frame_indices = frame_indices_all[ep_positions]
            if bool(np.any(np.diff(ep_frame_indices) < 0)):
                ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_values = values_all[ep_positions]
            if ep_values.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]
            cam_payloads: list[tuple[str, Path | list[Image.Image], np.ndarray | None]] = []
            for cam_key in selected_video_keys:
                cam_dtype = dataset.meta.features[cam_key]["dtype"]
                if cam_dtype == "image":
                    frames = _extract_image_frames(raw_dataset, cam_key, ep_positions)
                    cam_payloads.append((cam_dtype, frames, None))
                else:
                    src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, cam_key)
                    from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, cam_key)
                    cam_ts = from_ts + ep_timestamps
                    if to_ts is not None:
                        cam_ts = np.minimum(cam_ts, to_ts)
                    cam_payloads.append((cam_dtype, src_path, cam_ts))

            dst_path = _build_output_video_path_multiview(
                output_dir=output_dir,
                repo_id=dataset.repo_id,
                video_keys=selected_video_keys,
                episode_index=ep,
            )
            if dst_path.exists() and not overwrite:
                continue

            tasks.append(
                (
                    cam_payloads,
                    dst_path,
                    ep_values,
                    advantages_all[ep_positions],
                    indicators_all[ep_positions],
                )
            )

        desc_keys = ",".join(key.split(".")[-1] for key in selected_video_keys)
        for cam_payloads, dst, vals, advs, inds in tqdm(
            tasks,
            total=len(tasks),
            desc=f"Export overlay multiview [{desc_keys}]",
            leave=False,
        ):
            all_cam_frames: list[list[Image.Image]] = []
            for _, payload, ts in cam_payloads:
                if isinstance(payload, list):
                    all_cam_frames.append(payload)
                else:
                    if ts is None:
                        raise RuntimeError("Video timestamps are unexpectedly missing for multiview export.")
                    all_cam_frames.append(
                        _decode_frames_at_timestamps(
                            video_file=payload,
                            timestamps_s=ts,
                            tolerance_s=tolerance_s,
                            backend=video_backend,
                        )
                    )

            written_paths.append(
                _export_single_episode_multiview_from_frames(
                    all_cam_frames=all_cam_frames,
                    camera_labels=selected_video_keys,
                    dst_video_path=dst,
                    ep_values=vals,
                    ep_advantages=advs,
                    ep_indicators=inds,
                    fps=fps,
                    vcodec=vcodec,
                    frame_storage_mode=frame_storage_mode,
                    temp_dir_root=output_dir,
                    smooth_window=smooth_window,
                )
            )
    else:
        selected_video_key = selected_video_keys[0]
        tasks = []
        for ep in episodes:
            ep_positions = np.flatnonzero(episode_indices_all == ep)
            if ep_positions.shape[0] > 1:
                ep_frame_indices = frame_indices_all[ep_positions]
                if bool(np.any(np.diff(ep_frame_indices) < 0)):
                    ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

            ep_values = values_all[ep_positions]
            if ep_values.shape[0] == 0:
                continue

            ep_timestamps = timestamps_all[ep_positions]
            from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, selected_video_key)
            ep_video_timestamps = from_ts + ep_timestamps
            if to_ts is not None:
                ep_video_timestamps = np.minimum(ep_video_timestamps, to_ts)

            dst_path = _build_output_video_path(output_dir, dataset.repo_id, selected_video_key, ep)
            if dst_path.exists() and not overwrite:
                continue

            selected_dtype = dataset.meta.features[selected_video_key]["dtype"]
            if selected_dtype == "image":
                src_or_frames: Path | list[Image.Image] = _extract_image_frames(
                    raw_dataset, selected_video_key, ep_positions
                )
            else:
                src_or_frames = Path(dataset.root) / dataset.meta.get_video_file_path(ep, selected_video_key)
            tasks.append(
                (
                    src_or_frames,
                    dst_path,
                    ep_values,
                    advantages_all[ep_positions],
                    indicators_all[ep_positions],
                    ep_video_timestamps,
                )
            )

        for src_or_frames, dst, vals, advs, inds, ts in tqdm(
            tasks,
            total=len(tasks),
            desc=f"Export overlay [{selected_video_key}]",
            leave=False,
        ):
            if isinstance(src_or_frames, list):
                written_paths.append(
                    _export_single_episode_from_frames(
                        decoded_frames=src_or_frames,
                        dst_video_path=dst,
                        ep_values=vals,
                        ep_advantages=advs,
                        ep_indicators=inds,
                        fps=fps,
                        vcodec=vcodec,
                        frame_storage_mode=frame_storage_mode,
                        temp_dir_root=output_dir,
                        smooth_window=smooth_window,
                    )
                )
            else:
                written_paths.append(
                    _export_single_episode(
                        src_or_frames,
                        dst,
                        vals,
                        advs,
                        inds,
                        ts,
                        fps,
                        vcodec,
                        tolerance_s,
                        video_backend,
                        frame_storage_mode,
                        output_dir,
                        smooth_window=smooth_window,
                    )
                )

    return written_paths
