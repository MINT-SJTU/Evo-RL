#!/usr/bin/env python

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.configs.value import (
    ValueInferenceACPConfig,
    ValueInferenceCheckpointConfig,
    ValueInferenceDatasetConfig,
    ValueInferencePipelineConfig,
    ValueInferenceVizConfig,
)
from lerobot.scripts import lerobot_value_infer as value_infer, value_infer_viz


class _FakeHFDataset:
    def __init__(self, data: dict[str, list]):
        self._data = data
        self.column_names = list(data.keys())

    def with_format(self, _):
        return self

    def __len__(self):
        return len(next(iter(self._data.values())))

    def __getitem__(self, key: str):
        return self._data[key]


class _FakeAccelerator:
    def __init__(self):
        self.is_main_process = True
        self.num_processes = 1
        self.device = torch.device("cpu")

    def wait_for_everyone(self):
        return None

    def end_training(self):
        return None


class _FakeMeta:
    camera_keys = ["observation.images.front"]

    def get_video_file_path(self, ep_idx: int, vid_key: str) -> str:
        return f"videos/{vid_key}/episode-{ep_idx:06d}.mp4"


class _FakeDataset:
    repo_id = "dummy/repo"
    root = "/tmp"
    fps = 30
    episodes = [64, 101]
    meta = _FakeMeta()

    def __init__(self):
        self.hf_dataset = _FakeHFDataset(
            {
                "index": [1000, 1001, 1002, 5000, 5001],
                "episode_index": [64, 64, 64, 101, 101],
                "frame_index": [0, 1, 2, 0, 1],
                "complementary_info.value": [-0.8, -0.7, -0.6, -0.3, -0.2],
                "complementary_info.advantage": [0.1, 0.2, 0.3, 0.4, 0.5],
                "complementary_info.acp_indicator": [0, 1, 1, 0, 1],
            }
        )


def test_select_video_key_prefers_front_when_video_key_not_set():
    keys = [
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.right_front",
    ]
    assert value_infer_viz._select_video_key(keys, None) == "observation.images.right_front"


def test_export_overlay_videos_uses_episode_index_for_subset_alignment(monkeypatch, tmp_path: Path):
    dataset = _FakeDataset()

    def _fake_export_single_episode(
        src,
        dst,
        ep_values,
        ep_advantages,
        ep_indicators,
        episode_timestamps_s,
        fps,
        vcodec,
        tolerance_s,
        video_backend,
        frame_storage_mode,
        temp_dir_root,
        smooth_window=1,
    ):
        assert isinstance(src, Path)
        assert isinstance(dst, Path)
        assert isinstance(episode_timestamps_s, np.ndarray)
        assert fps == 30
        assert vcodec == "libsvtav1"
        assert tolerance_s > 0
        assert frame_storage_mode == "memory"
        assert temp_dir_root == tmp_path
        assert isinstance(ep_values, np.ndarray)
        assert isinstance(ep_advantages, np.ndarray)
        assert isinstance(ep_indicators, np.ndarray)
        assert smooth_window == 1
        return dst

    monkeypatch.setattr(value_infer_viz, "_export_single_episode", _fake_export_single_episode)

    written = value_infer_viz._export_overlay_videos(
        dataset=dataset,
        value_field="complementary_info.value",
        advantage_field="complementary_info.advantage",
        indicator_field="complementary_info.acp_indicator",
        viz_episodes="all",
        video_key=None,
        video_keys=None,
        output_dir=tmp_path,
        overwrite=True,
        vcodec="libsvtav1",
    )

    assert len(written) == 2
    names = [path.name for path in written]
    assert any("episode_0064" in name for name in names)
    assert any("episode_0101" in name for name in names)


def test_export_overlay_videos_passes_disk_storage_mode(monkeypatch, tmp_path: Path):
    dataset = _FakeDataset()

    def _fake_export_single_episode(
        src,
        dst,
        ep_values,
        ep_advantages,
        ep_indicators,
        episode_timestamps_s,
        fps,
        vcodec,
        tolerance_s,
        video_backend,
        frame_storage_mode,
        temp_dir_root,
        smooth_window=1,
    ):
        assert isinstance(episode_timestamps_s, np.ndarray)
        assert tolerance_s > 0
        assert frame_storage_mode == "disk"
        assert temp_dir_root == tmp_path
        assert smooth_window == 1
        return dst

    monkeypatch.setattr(value_infer_viz, "_export_single_episode", _fake_export_single_episode)

    written = value_infer_viz._export_overlay_videos(
        dataset=dataset,
        value_field="complementary_info.value",
        advantage_field="complementary_info.advantage",
        indicator_field="complementary_info.acp_indicator",
        viz_episodes="all",
        video_key=None,
        video_keys=None,
        output_dir=tmp_path,
        overwrite=True,
        vcodec="h264",
        frame_storage_mode="disk",
    )

    assert len(written) == 2


def test_get_video_encode_options_for_h264_nvenc():
    options, pix_fmt = value_infer_viz._get_video_encode_options("h264_nvenc")
    assert pix_fmt == "yuv420p"
    assert options["preset"] == "p4"
    assert options["rc"] == "vbr"
    assert options["cq"] == "28"
    assert options["b"] == "0"
    assert options["g"] == "60"
    assert "crf" not in options


def test_get_episode_value_bounds_returns_episode_min_max():
    y_min, y_max = value_infer_viz._get_episode_value_bounds(np.array([-0.7, -0.3, -0.5], dtype=np.float32))
    assert np.isclose(y_min, -0.7)
    assert np.isclose(y_max, -0.3)


def test_acp_disabled_skips_value_inference_and_uses_default_viz_dir(monkeypatch, tmp_path: Path):
    accelerator = _FakeAccelerator()
    dataset = _FakeDataset()
    captured: dict[str, Path] = {}

    def _fake_export_overlay_videos(**kwargs):
        out_dir = kwargs["output_dir"]
        captured["output_dir"] = out_dir
        return [out_dir / "dummy.mp4"]

    def _fail_resolve_pretrained(*args, **kwargs):
        raise AssertionError("checkpoint resolution should be skipped when acp.enable=false")

    monkeypatch.setattr(value_infer, "_create_accelerator", lambda cfg, acc: accelerator)
    monkeypatch.setattr(
        value_infer,
        "_init_runtime",
        lambda cfg, accelerator: (tmp_path / "runtime" / "value", torch.device("cpu")),
    )
    monkeypatch.setattr(value_infer, "_load_dataset_distributed", lambda cfg, accelerator: dataset)
    monkeypatch.setattr(value_infer, "_resolve_pretrained_model_dir", _fail_resolve_pretrained)
    monkeypatch.setattr(value_infer, "_export_overlay_videos", _fake_export_overlay_videos)

    cfg = ValueInferencePipelineConfig(
        dataset=ValueInferenceDatasetConfig(repo_id="dummy/repo"),
        inference=ValueInferenceCheckpointConfig(checkpoint_path="unused"),
        acp=ValueInferenceACPConfig(enable=False),
        viz=ValueInferenceVizConfig(enable=True, episodes="all"),
        output_dir=tmp_path / "pipeline_out",
        job_name="viz_skip_test",
    )

    result = value_infer.run_value_inference_pipeline(cfg)

    expected_viz_dir = tmp_path / "runtime" / "value" / "viz"
    assert captured["output_dir"] == expected_viz_dir
    assert result["main_process"] is True
    assert result["acp_enabled"] is False
    assert result["value_inference_skipped"] is True
    assert result["checkpoint"] is None
    assert result["viz_outputs"] == [str(expected_viz_dir / "dummy.mp4")]


def test_decode_frames_at_timestamps_scales_float_frames_to_uint8(monkeypatch, tmp_path: Path):
    fake = torch.full((1, 3, 2, 2), 0.5, dtype=torch.float32)

    def _fake_decode_video_frames(video_path, timestamps, tolerance_s, backend):
        return fake

    monkeypatch.setattr(value_infer_viz, "decode_video_frames", _fake_decode_video_frames)
    frames = value_infer_viz._decode_frames_at_timestamps(
        video_file=tmp_path / "dummy.mp4",
        timestamps_s=np.array([0.0], dtype=np.float64),
        tolerance_s=1e-4,
        backend="pyav",
    )

    assert len(frames) == 1
    assert isinstance(frames[0], Image.Image)
    arr = np.asarray(frames[0], dtype=np.uint8)
    assert int(arr.mean()) >= 120
