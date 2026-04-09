from __future__ import annotations

import numpy as np
import torch

from lerobot.rlt.interfaces import Observation


def _to_tensor(x: torch.Tensor | np.ndarray | float | int) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.tensor(x)


def robot_obs_to_rlt_obs(
    obs_dict: dict[str, torch.Tensor],
    camera_keys: list[str],
    proprio_keys: list[str],
    device: str = "cuda",
) -> Observation:
    """Convert a LeRobot observation dict to an RLT Observation.

    Handles both HWC uint8 (robot raw) and CHW float32 (preprocessed) image formats.
    All outputs are batched with B=1.

    Args:
        obs_dict: flat dict from robot.get_observation(), e.g.
            {"observation.images.left_wrist": Tensor, "observation.state.left_arm_pos": Tensor, ...}
        camera_keys: keys to extract as images, e.g. ["left_wrist", "right_wrist", "right_front"]
        proprio_keys: keys whose values are concatenated into the proprio vector
        device: target device

    Returns:
        Observation with images dict and (1, proprio_dim) proprio tensor
    """
    images = _extract_images(obs_dict, camera_keys, device)
    proprio = _extract_proprio(obs_dict, proprio_keys, device)
    return Observation(images=images, proprio=proprio)


def _extract_images(
    obs_dict: dict[str, torch.Tensor],
    camera_keys: list[str],
    device: str,
) -> dict[str, torch.Tensor]:
    """Extract and normalize camera images to (1, C, H, W) float32 in [0, 1]."""
    images: dict[str, torch.Tensor] = {}
    for key in camera_keys:
        raw = _to_tensor(_find_image_tensor(obs_dict, key))
        img = raw.to(device=device, dtype=torch.float32)
        # Ensure 4D: add batch dim if needed
        if img.ndim == 3:
            img = img.unsqueeze(0)
        # Convert HWC -> CHW if last dim is 3 and second dim is not 3
        if img.shape[-1] == 3 and img.shape[1] != 3:
            img = img.permute(0, 3, 1, 2)
        # Normalize uint8 range [0, 255] -> [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        images[key] = img
    return images


def _find_image_tensor(obs_dict: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    """Find image tensor in obs_dict by exact key or common LeRobot naming patterns."""
    # Try exact key first
    if key in obs_dict:
        return obs_dict[key]
    # Try observation.images.{key} pattern
    prefixed = f"observation.images.{key}"
    if prefixed in obs_dict:
        return obs_dict[prefixed]
    raise KeyError(f"Camera key '{key}' not found in obs_dict. Available: {list(obs_dict.keys())}")


def _extract_proprio(
    obs_dict: dict[str, torch.Tensor],
    proprio_keys: list[str],
    device: str,
) -> torch.Tensor:
    """Concatenate proprio values into (1, proprio_dim) tensor."""
    parts: list[torch.Tensor] = []
    for key in proprio_keys:
        val = _to_tensor(obs_dict[key]).to(device=device, dtype=torch.float32)
        if val.ndim == 0:
            val = val.unsqueeze(0)
        val = val.flatten()
        parts.append(val)
    proprio = torch.cat(parts, dim=-1).unsqueeze(0)  # (1, proprio_dim)
    return proprio


def rlt_action_to_robot_action(
    action_tensor: torch.Tensor,
    action_keys: list[str],
) -> dict[str, torch.Tensor]:
    """Map a flat action tensor to a named dict for robot.send_action().

    Args:
        action_tensor: (1, action_dim) or (action_dim,) action vector
        action_keys: list of key names, one per action dimension

    Returns:
        dict mapping each key to a scalar tensor
    """
    action = action_tensor.squeeze(0).detach().cpu()
    if len(action_keys) != action.shape[-1]:
        raise ValueError(
            f"action_keys length {len(action_keys)} != action_dim {action.shape[-1]}"
        )
    return {key: action[i] for i, key in enumerate(action_keys)}
