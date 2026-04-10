from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lerobot.rlt.interfaces import Observation

logger = logging.getLogger(__name__)


def normalize_quantiles(
    tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize from raw space to [-1, 1] using QUANTILES mode (matching lerobot convention).

    Formula: normalized = (raw - q01) / (q99 - q01) * 2.0 - 1.0
    """
    denom = q99 - q01
    denom = torch.where(denom.abs() < eps, torch.tensor(eps, dtype=denom.dtype), denom)
    return (tensor - q01) / denom * 2.0 - 1.0


class RLTDemoDataset(Dataset):
    """Wraps a LeRobotDataset to yield (images, proprio, expert_actions) for RLT demo adaptation.

    Loads action chunks via delta_timestamps so each sample contains a
    chunk_length-step action trajectory starting from the current frame.
    """

    def __init__(
        self,
        dataset_path: str,
        repo_id: str = "rlt_demo",
        chunk_length: int = 50,
        camera_keys: list[str] | None = None,
        image_size: tuple[int, int] = (224, 224),
        state_key: str = "observation.state",
        action_key: str = "action",
        normalize_actions: bool = False,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        fps = self._read_fps(dataset_path, repo_id)
        delta_timestamps = {
            action_key: [i / fps for i in range(chunk_length)],
        }
        self._dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_path,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )

        self._camera_keys = camera_keys or self._detect_camera_keys()
        self._image_size = image_size
        self._state_key = state_key
        self._action_key = action_key
        self._chunk_length = chunk_length

        # Action normalization: raw degrees -> [-1, 1] via QUANTILES
        self._normalize_actions = normalize_actions
        self._action_q01: torch.Tensor | None = None
        self._action_q99: torch.Tensor | None = None
        if normalize_actions:
            action_stats = self._dataset.meta.stats.get(action_key, {})
            q01_raw = action_stats.get("q01")
            q99_raw = action_stats.get("q99")
            if q01_raw is not None and q99_raw is not None:
                self._action_q01 = torch.as_tensor(q01_raw, dtype=torch.float32)
                self._action_q99 = torch.as_tensor(q99_raw, dtype=torch.float32)
                logger.info(
                    "Action normalization enabled: q01[:4]=%s, q99[:4]=%s",
                    self._action_q01[:4].tolist(), self._action_q99[:4].tolist(),
                )
            else:
                logger.warning("Dataset lacks q01/q99 stats for actions; skipping normalization")
                self._normalize_actions = False

        logger.info(
            "RLTDemoDataset: %d samples, cameras=%s, chunk=%d, normalize_actions=%s",
            len(self._dataset), self._camera_keys, chunk_length, self._normalize_actions,
        )

    def _read_fps(self, dataset_path: str, repo_id: str) -> float:
        """Read fps from the dataset metadata."""
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)
        return meta.fps

    def _detect_camera_keys(self) -> list[str]:
        """Auto-detect camera keys from dataset features."""
        return [k for k in self._dataset.features if k.startswith("observation.images.")]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self._dataset[idx]
        result = {}

        # Images: convert to float [0, 1] tensors of consistent size
        for cam_key in self._camera_keys:
            img = item[cam_key]
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img, dtype=torch.float32)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            # Ensure (C, H, W) format
            if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            # Resize if needed
            h, w = self._image_size
            if img.shape[1] != h or img.shape[2] != w:
                img = F.interpolate(img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
            # Use short camera name (e.g. "left_wrist" not "observation.images.left_wrist")
            short_name = cam_key.split(".")[-1]
            result[short_name] = img

        # Proprio state
        state = item[self._state_key]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        result["proprio"] = state.float()

        # Action chunk: (chunk_length, action_dim)
        action = item[self._action_key]
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        action = action.float()
        if self._normalize_actions and self._action_q01 is not None:
            action = normalize_quantiles(action, self._action_q01, self._action_q99)
        result["expert_actions"] = action

        return result


def rlt_demo_collate(batch: list[dict]) -> tuple[Observation, torch.Tensor]:
    """Collate a list of dataset items into (Observation, expert_actions).

    Returns:
        obs: Observation with batched images and proprio
        expert_actions: (B, chunk_length, action_dim)
    """
    # Gather all image keys from first item
    image_keys = [k for k in batch[0] if k not in ("proprio", "expert_actions")]

    images = {}
    for key in image_keys:
        images[key] = torch.stack([item[key] for item in batch])

    proprio = torch.stack([item["proprio"] for item in batch])
    expert_actions = torch.stack([item["expert_actions"] for item in batch])

    obs = Observation(images=images, proprio=proprio)
    return obs, expert_actions


def make_demo_loader(
    dataset_path: str,
    batch_size: int,
    chunk_length: int = 50,
    repo_id: str = "rlt_demo",
    camera_keys: list[str] | None = None,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 2,
    device: str = "cuda",
) -> Iterator[tuple[Observation, torch.Tensor]]:
    """Create an infinite-cycling DataLoader for demo adaptation.

    Yields (Observation, expert_actions) tuples moved to device.
    """
    dataset = RLTDemoDataset(
        dataset_path=dataset_path,
        repo_id=repo_id,
        chunk_length=chunk_length,
        camera_keys=camera_keys,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rlt_demo_collate,
        drop_last=True,
        pin_memory=(device != "cpu"),
    )

    def _cycle_with_cleanup() -> Iterator[tuple[Observation, torch.Tensor]]:
        """Restart the DataLoader each epoch to avoid memory accumulation."""
        import gc
        while True:
            for obs, expert_actions in loader:
                obs_device = Observation(
                    images={k: v.to(device) for k, v in obs.images.items()},
                    proprio=obs.proprio.to(device),
                )
                yield obs_device, expert_actions.to(device)
            gc.collect()

    return _cycle_with_cleanup()
