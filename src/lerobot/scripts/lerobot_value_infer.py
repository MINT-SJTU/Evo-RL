#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.value import ValueInferencePipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_info, write_info
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.value_infer_viz import (
    _export_overlay_videos,
)
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.recording_annotations import EPISODE_SUCCESS, resolve_episode_success_label
from lerobot.utils.utils import init_logging, inside_slurm
from lerobot.values.pistar06.configuration_pistar06 import Pistar06Config
from lerobot.values.pistar06.modeling_pistar06 import (
    EpisodeTargetInfo,
    Pistar06Policy,
    compute_normalized_value_targets,
)


class ContiguousDistributedEvalSampler(Sampler[int]):
    """Distributed eval sampler with contiguous per-rank shards and deterministic tail padding."""

    def __init__(self, dataset_size: int, num_replicas: int, rank: int):
        if dataset_size <= 0:
            raise ValueError(f"'dataset_size' must be > 0, got {dataset_size}.")
        if num_replicas <= 0:
            raise ValueError(f"'num_replicas' must be > 0, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"'rank' must be in [0, {num_replicas - 1}], got {rank}.")

        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        shard_start = self.rank * self.num_samples
        shard_end = min(shard_start + self.num_samples, self.dataset_size)
        if shard_start >= self.dataset_size:
            indices: list[int] = []
        else:
            indices = list(range(shard_start, shard_end))

        if len(indices) < self.num_samples:
            indices.extend([self.dataset_size - 1] * (self.num_samples - len(indices)))
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


def _set_infer_logger_levels() -> None:
    for logger_name in ["fsspec", "fsspec.local", "huggingface_hub", "datasets", "torchcodec"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _create_accelerator(cfg: ValueInferencePipelineConfig, accelerator: Accelerator | None) -> Accelerator:
    if accelerator is not None:
        return accelerator

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    force_cpu = cfg.runtime.device == "cpu"
    return Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs], cpu=force_cpu)


def _resolve_pretrained_model_dir(checkpoint_path: str, checkpoint_ref: str) -> Path:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    if (path / "model.safetensors").is_file() and (path / "config.json").is_file():
        return path

    if (path / PRETRAINED_MODEL_DIR / "model.safetensors").is_file() and (
        path / PRETRAINED_MODEL_DIR / "config.json"
    ).is_file():
        return path / PRETRAINED_MODEL_DIR

    checkpoints_root = path / CHECKPOINTS_DIR if (path / CHECKPOINTS_DIR).is_dir() else path
    step_ref = LAST_CHECKPOINT_LINK if checkpoint_ref == "last" else checkpoint_ref
    step_dir = checkpoints_root / step_ref

    if (step_dir / PRETRAINED_MODEL_DIR / "model.safetensors").is_file() and (
        step_dir / PRETRAINED_MODEL_DIR / "config.json"
    ).is_file():
        return step_dir / PRETRAINED_MODEL_DIR

    if (step_dir / "model.safetensors").is_file() and (step_dir / "config.json").is_file():
        return step_dir

    raise FileNotFoundError(
        f"Could not resolve pretrained model directory from checkpoint_path={path} checkpoint_ref={checkpoint_ref}."
    )


def _load_dataset_distributed(cfg: ValueInferencePipelineConfig, accelerator: Accelerator) -> LeRobotDataset:
    dataset_kwargs = {
        "repo_id": cfg.dataset.repo_id,
        "root": cfg.dataset.root,
        "episodes": cfg.dataset.episodes,
        "revision": cfg.dataset.revision,
        "download_videos": cfg.dataset.download_videos,
    }

    if accelerator.is_main_process:
        dataset = LeRobotDataset(**dataset_kwargs)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        dataset = LeRobotDataset(**dataset_kwargs)
    return dataset


def _init_runtime(
    cfg: ValueInferencePipelineConfig,
    accelerator: Accelerator,
) -> tuple[Path, torch.device]:
    output_dir = cfg.output_dir / "value"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    log_file = output_dir / "value_infer.log" if accelerator.is_main_process else None
    init_logging(log_file=log_file, file_level="INFO", accelerator=accelerator)
    _set_infer_logger_levels()

    if accelerator.is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    return output_dir, device


def _build_episode_info(
    dataset: LeRobotDataset,
    success_field: str,
    default_success: str,
    length_scale_quantile: float,
) -> tuple[dict[int, EpisodeTargetInfo], dict[int, float]]:
    episodes_ds = dataset.meta.episodes.with_format(None)
    episodes = episodes_ds[:]
    n_episodes = len(episodes_ds)
    has_success = success_field in episodes_ds.column_names

    episode_info: dict[int, EpisodeTargetInfo] = {}
    task_lengths: dict[int, list[int]] = {}
    for i in range(n_episodes):
        ep_idx = int(episodes["episode_index"][i])
        ep_length = int(episodes["length"][i])
        tasks = episodes["tasks"][i]
        task_name = tasks[0] if isinstance(tasks, list) else tasks
        if task_name not in dataset.meta.tasks.index:
            raise KeyError(f"Episode {ep_idx} references unknown task '{task_name}'.")
        task_index = int(dataset.meta.tasks.loc[task_name].task_index)

        explicit_success = episodes[success_field][i] if has_success else None
        resolved_success = resolve_episode_success_label(
            explicit_success,
            default_label=default_success,
            require_label=True,
        )
        ep_success = resolved_success == EPISODE_SUCCESS

        episode_info[ep_idx] = EpisodeTargetInfo(
            episode_index=ep_idx,
            task_index=task_index,
            length=ep_length,
            success=ep_success,
        )
        task_lengths.setdefault(task_index, []).append(ep_length)

    task_scales = Pistar06Policy._compute_task_length_scales(
        task_lengths=task_lengths,
        quantile=length_scale_quantile,
    )
    return episode_info, task_scales


def _compute_dense_rewards_from_targets(
    targets: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    rewards = np.zeros_like(targets, dtype=np.float32)
    n = targets.shape[0]

    for i in range(n):
        is_next_in_episode = (
            i + 1 < n
            and episode_indices[i + 1] == episode_indices[i]
            and frame_indices[i + 1] == frame_indices[i] + 1
        )
        if is_next_in_episode:
            rewards[i] = float(targets[i] - targets[i + 1])
        else:
            rewards[i] = float(targets[i])

    return rewards


def _compute_n_step_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    n_step: int,
) -> np.ndarray:
    if n_step <= 0:
        raise ValueError("'n_step' must be > 0.")

    n = rewards.shape[0]
    advantages = np.zeros(n, dtype=np.float32)

    for i in range(n):
        ep_i = episode_indices[i]
        fi = frame_indices[i]

        discounted_sum = 0.0
        j = i
        steps = 0
        while steps < n_step and j < n:
            same_episode = episode_indices[j] == ep_i
            contiguous = frame_indices[j] == fi + steps
            if not same_episode or not contiguous:
                break

            discounted_sum += float(rewards[j])
            steps += 1
            j += 1

        if steps == n_step and j < n and episode_indices[j] == ep_i and frame_indices[j] == fi + n_step:
            bootstrap = float(values[j])
        else:
            bootstrap = 0.0

        advantages[i] = float(discounted_sum + bootstrap - values[i])

    return advantages


def _compute_task_thresholds(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    positive_ratio: float,
) -> dict[int, float]:
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("'positive_ratio' must be within [0, 1].")

    thresholds: dict[int, float] = {}
    quantile = 1.0 - positive_ratio

    for task_idx in np.unique(task_indices):
        task_adv = advantages[task_indices == task_idx]
        if task_adv.size == 0:
            thresholds[int(task_idx)] = float("inf")
        else:
            thresholds[int(task_idx)] = float(np.quantile(task_adv, quantile))

    return thresholds


def _binarize_advantages(
    task_indices: np.ndarray,
    advantages: np.ndarray,
    thresholds: dict[int, float],
    interventions: np.ndarray,
    force_intervention_positive: bool,
) -> np.ndarray:
    indicators = np.zeros_like(advantages, dtype=np.int64)

    for i in range(advantages.shape[0]):
        task_idx = int(task_indices[i])
        threshold = thresholds[task_idx]
        indicators[i] = 1 if float(advantages[i]) >= threshold else 0

    if force_intervention_positive:
        intervention_mask = interventions.astype(np.float32) > 0.5
        indicators[intervention_mask] = 1

    return indicators


def _update_feature_metadata(dataset_root: Path, feature_infos: dict[str, dict[str, Any]]) -> None:
    info = load_info(dataset_root)
    for feature_name, feature_info in feature_infos.items():
        info["features"][feature_name] = {
            "dtype": feature_info["dtype"],
            "shape": tuple(feature_info["shape"]),
            "names": feature_info.get("names"),
        }
    write_info(info, dataset_root)


def _write_columns_in_place(
    dataset_root: Path,
    absolute_indices: np.ndarray,
    columns: dict[str, np.ndarray],
    feature_infos: dict[str, dict[str, Any]],
) -> None:
    if absolute_indices.ndim != 1:
        raise ValueError("'absolute_indices' must be rank-1.")

    max_index = int(np.max(absolute_indices))
    selected = np.zeros(max_index + 1, dtype=np.bool_)
    selected[absolute_indices] = True

    lookups: dict[str, np.ndarray] = {}
    for field, values in columns.items():
        lookup_dtype = np.float32 if feature_infos[field]["dtype"] == "float32" else np.int64
        lookup = np.zeros(max_index + 1, dtype=lookup_dtype)
        lookup[absolute_indices] = values.astype(lookup_dtype, copy=False)
        lookups[field] = lookup

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_root / 'data'}")

    for parquet_path in tqdm(data_files, desc="Writing annotations", leave=False):
        table = pq.read_table(parquet_path)
        idx_np = table["index"].to_numpy().astype(np.int64, copy=False)

        in_range = (idx_np >= 0) & (idx_np <= max_index)
        in_subset = np.zeros_like(in_range)
        in_subset[in_range] = selected[idx_np[in_range]]

        new_table = table
        for field, lookup in lookups.items():
            ftype = feature_infos[field]["dtype"]
            if ftype == "float32":
                default_value = np.nan
                target_dtype = np.float32
                pa_type = pa.float32()
            elif ftype == "int64":
                default_value = 0
                target_dtype = np.int64
                pa_type = pa.int64()
            else:
                raise ValueError(f"Unsupported annotation dtype '{ftype}' for field '{field}'.")

            if field in new_table.schema.names:
                current = new_table[field].to_numpy().astype(target_dtype, copy=True)
            else:
                current = np.full(idx_np.shape[0], default_value, dtype=target_dtype)

            if np.any(in_subset):
                subset_indices = idx_np[in_subset]
                current[in_subset] = lookup[subset_indices]

            array = pa.array(current, type=pa_type)
            if field in new_table.schema.names:
                col_idx = new_table.schema.names.index(field)
                new_table = new_table.set_column(col_idx, field, array)
            else:
                new_table = new_table.append_column(field, array)

        pq.write_table(new_table, parquet_path, compression="snappy")

    _update_feature_metadata(dataset_root=dataset_root, feature_infos=feature_infos)


def _load_value_policy_and_processors(
    cfg: ValueInferencePipelineConfig,
    dataset: LeRobotDataset,
    pretrained_dir: Path,
    device: torch.device,
):
    value_cfg = PreTrainedConfig.from_pretrained(pretrained_dir)
    if not isinstance(value_cfg, Pistar06Config):
        raise ValueError(
            f"Unsupported value config type '{type(value_cfg)}'. lerobot-value-infer currently supports only 'pistar06'."
        )

    value_cfg.pretrained_path = pretrained_dir
    value_cfg.device = device.type

    value_policy = make_policy(
        cfg=value_cfg,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=value_cfg,
        pretrained_path=pretrained_dir,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    return value_policy, value_cfg, preprocessor


def _export_visualization_outputs(
    dataset: LeRobotDataset,
    cfg: ValueInferencePipelineConfig,
    output_dir: Path,
) -> list[str]:
    viz_output_dir = output_dir / "viz"
    written_videos = _export_overlay_videos(
        dataset=dataset,
        value_field=cfg.acp.value_field,
        advantage_field=cfg.acp.advantage_field,
        indicator_field=cfg.acp.indicator_field,
        viz_episodes=cfg.viz.episodes,
        video_key=cfg.viz.video_key,
        video_keys=cfg.viz.video_keys,
        output_dir=viz_output_dir,
        overwrite=cfg.viz.overwrite,
        vcodec=cfg.viz.vcodec,
        frame_storage_mode=cfg.viz.frame_storage_mode,
        smooth_window=cfg.viz.smooth_window,
    )
    logging.info("Exported %d overlay videos to %s", len(written_videos), viz_output_dir)
    return [str(path) for path in written_videos]


def run_value_inference_pipeline(
    cfg: ValueInferencePipelineConfig,
    accelerator: Accelerator | None = None,
) -> dict[str, Any]:
    cfg.validate()

    accelerator = _create_accelerator(cfg, accelerator)
    output_dir, device = _init_runtime(cfg, accelerator)

    dataset = _load_dataset_distributed(cfg, accelerator)
    raw_frames = dataset.hf_dataset.with_format(None)
    frame_count = len(raw_frames)
    if frame_count == 0:
        raise ValueError("Dataset has no frames.")

    if not cfg.acp.enable and cfg.inference.reuse_existing_value_field:
        viz_outputs: list[str] = []
        if accelerator.is_main_process:
            logging.info(
                "ACP disabled; skipping value inference and reusing existing '%s' annotations.",
                cfg.acp.value_field,
            )
            if cfg.acp.value_field not in raw_frames.column_names:
                raise KeyError(
                    f"Missing value field '{cfg.acp.value_field}' in dataset while 'acp.enable=false'."
                )
            if cfg.viz.enable:
                viz_outputs = _export_visualization_outputs(dataset=dataset, cfg=cfg, output_dir=output_dir)

        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            result: dict[str, Any] = {
                "main_process": False,
                "world_size": int(accelerator.num_processes),
            }
            accelerator.end_training()
            return result

        result = {
            "main_process": True,
            "world_size": int(accelerator.num_processes),
            "num_frames": int(frame_count),
            "checkpoint": None,
            "value_field": cfg.acp.value_field,
            "acp_enabled": False,
            "value_inference_skipped": True,
            "indicator_positive_ratio": None,
            "thresholds": None,
            "viz_outputs": viz_outputs,
        }
        accelerator.end_training()
        return result

    pretrained_dir = _resolve_pretrained_model_dir(
        checkpoint_path=cfg.inference.checkpoint_path,
        checkpoint_ref=cfg.inference.checkpoint_ref,
    )
    value_policy, value_cfg, preprocessor = _load_value_policy_and_processors(
        cfg=cfg,
        dataset=dataset,
        pretrained_dir=pretrained_dir,
        device=device,
    )

    absolute_indices = np.asarray(raw_frames["index"], dtype=np.int64)

    if value_cfg.task_index_feature not in raw_frames.column_names:
        raise KeyError(f"Missing task feature '{value_cfg.task_index_feature}' in dataset columns.")

    task_indices = np.asarray(raw_frames[value_cfg.task_index_feature], dtype=np.int64)
    episode_indices = np.asarray(raw_frames["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(raw_frames["frame_index"], dtype=np.int64)

    if cfg.acp.intervention_field in raw_frames.column_names:
        interventions = np.asarray(raw_frames[cfg.acp.intervention_field], dtype=np.float32)
    else:
        interventions = np.zeros(frame_count, dtype=np.float32)

    eval_loader = DataLoader(
        dataset,
        batch_size=cfg.runtime.batch_size,
        shuffle=False,
        num_workers=cfg.runtime.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    value_policy = accelerator.prepare(value_policy)
    eval_loader = accelerator.prepare(eval_loader)

    if accelerator.is_main_process:
        max_abs_index = int(np.max(absolute_indices))
        prediction_lookup = np.zeros(max_abs_index + 1, dtype=np.float32)
        prediction_seen = np.zeros(max_abs_index + 1, dtype=np.bool_)
        logging.info(
            "Start value inference | world_size=%d batches=%d batch_size=%d checkpoint=%s",
            accelerator.num_processes,
            len(eval_loader),
            cfg.runtime.batch_size,
            pretrained_dir,
        )
    else:
        prediction_lookup = None
        prediction_seen = None

    value_policy.eval()
    eval_iter = tqdm(
        eval_loader,
        desc="Value inference",
        total=len(eval_loader),
        leave=False,
        disable=(not accelerator.is_main_process) or inside_slurm(),
    )

    with torch.no_grad():
        for raw_batch in eval_iter:
            batch_indices = raw_batch["index"]
            if not isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.as_tensor(batch_indices)
            batch_indices = batch_indices.to(device=device, dtype=torch.long, non_blocking=True)

            processed_batch = preprocessor(raw_batch)
            with accelerator.autocast():
                predicted_value = accelerator.unwrap_model(value_policy).predict_value(processed_batch)

            gathered_idx = accelerator.gather_for_metrics(batch_indices)
            gathered_val = accelerator.gather_for_metrics(predicted_value)

            if accelerator.is_main_process:
                idx_np = gathered_idx.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
                val_np = gathered_val.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                prediction_lookup[idx_np] = val_np
                prediction_seen[idx_np] = True

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if prediction_lookup is None or prediction_seen is None:
            raise RuntimeError("Prediction buffers unexpectedly missing on main process.")

        missing_mask = ~prediction_seen[absolute_indices]
        if bool(np.any(missing_mask)):
            missing_count = int(np.sum(missing_mask))
            raise RuntimeError(f"Inference is missing predictions for {missing_count} frames.")

        predicted_values = prediction_lookup[absolute_indices]
        logging.info(
            "Predicted value stats | min=%.6f max=%.6f mean=%.6f std=%.6f",
            float(np.min(predicted_values)),
            float(np.max(predicted_values)),
            float(np.mean(predicted_values)),
            float(np.std(predicted_values)),
        )

        columns: dict[str, np.ndarray] = {
            cfg.acp.value_field: predicted_values.astype(np.float32),
        }
        feature_infos: dict[str, dict[str, Any]] = {
            cfg.acp.value_field: {"dtype": "float32", "shape": (1,), "names": None},
        }

        indicator_positive_ratio: float | None = None
        thresholds: dict[int, float] | None = None

        if cfg.acp.enable:
            episode_info, task_scales = _build_episode_info(
                dataset=dataset,
                success_field=cfg.dataset.success_field,
                default_success=cfg.dataset.default_success,
                length_scale_quantile=cfg.acp.length_scale_quantile,
            )

            value_targets = compute_normalized_value_targets(
                episode_indices=episode_indices,
                frame_indices=frame_indices,
                episode_info=episode_info,
                task_scales=task_scales,
                c_fail_coef=cfg.acp.c_fail_coef,
                clip_min=value_cfg.bin_min,
                clip_max=value_cfg.bin_max,
            )
            rewards = _compute_dense_rewards_from_targets(value_targets, episode_indices, frame_indices)
            advantages = _compute_n_step_advantages(
                rewards=rewards,
                values=predicted_values,
                episode_indices=episode_indices,
                frame_indices=frame_indices,
                n_step=cfg.acp.n_step,
            )
            thresholds = _compute_task_thresholds(
                task_indices=task_indices,
                advantages=advantages,
                positive_ratio=cfg.acp.positive_ratio,
            )
            indicators = _binarize_advantages(
                task_indices=task_indices,
                advantages=advantages,
                thresholds=thresholds,
                interventions=interventions,
                force_intervention_positive=cfg.acp.force_intervention_positive,
            )

            indicator_positive_ratio = float(np.mean(indicators.astype(np.float32)))
            logging.info(
                "ACP stats | n_step=%d positive_ratio_target=%.4f positive_ratio_observed=%.4f",
                cfg.acp.n_step,
                cfg.acp.positive_ratio,
                indicator_positive_ratio,
            )

            columns[cfg.acp.advantage_field] = advantages.astype(np.float32)
            columns[cfg.acp.indicator_field] = indicators.astype(np.int64)
            feature_infos[cfg.acp.advantage_field] = {"dtype": "float32", "shape": (1,), "names": None}
            feature_infos[cfg.acp.indicator_field] = {"dtype": "int64", "shape": (1,), "names": None}

        _write_columns_in_place(
            dataset_root=Path(dataset.root),
            absolute_indices=absolute_indices,
            columns=columns,
            feature_infos=feature_infos,
        )

        logging.info("Wrote value annotations to dataset root: %s", dataset.root)

        # Sync computed columns into the in-memory hf_dataset so viz can read them
        for field, values in columns.items():
            if field in dataset.hf_dataset.column_names:
                dataset.hf_dataset = dataset.hf_dataset.remove_columns([field])
            dataset.hf_dataset = dataset.hf_dataset.add_column(field, values.tolist())

        viz_outputs: list[str] = []
        if cfg.viz.enable:
            viz_outputs = _export_visualization_outputs(dataset=dataset, cfg=cfg, output_dir=output_dir)

        result = {
            "main_process": True,
            "world_size": int(accelerator.num_processes),
            "num_frames": int(frame_count),
            "checkpoint": str(pretrained_dir),
            "value_field": cfg.acp.value_field,
            "acp_enabled": bool(cfg.acp.enable),
            "value_inference_skipped": False,
            "indicator_positive_ratio": indicator_positive_ratio,
            "thresholds": thresholds,
            "viz_outputs": viz_outputs,
        }
    else:
        result = {
            "main_process": False,
            "world_size": int(accelerator.num_processes),
        }

    # Keep non-main ranks alive until main rank finishes optional slow visualization export.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return result


@parser.wrap()
def value_infer(cfg: ValueInferencePipelineConfig):
    return run_value_inference_pipeline(cfg)


def main():
    register_third_party_plugins()
    value_infer()


if __name__ == "__main__":
    main()
