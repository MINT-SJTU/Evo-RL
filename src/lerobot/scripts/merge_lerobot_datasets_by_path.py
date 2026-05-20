#!/usr/bin/env python3

"""
Merge local LeRobotDataset directories by explicit filesystem paths.

This is a path-oriented wrapper around `lerobot.datasets.aggregate.aggregate_datasets`.
It does not require source datasets to be linked under the Hugging Face LeRobot cache.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_dataset_paths(paths: list[Path]) -> list[Path]:
    resolved = [path.expanduser().resolve() for path in paths]
    missing = [path for path in resolved if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(str(path) for path in missing))

    invalid = [path for path in resolved if not (path / "meta" / "info.json").is_file()]
    if invalid:
        raise FileNotFoundError(
            "These paths do not look like LeRobotDataset roots: "
            + ", ".join(str(path) for path in invalid)
        )

    return resolved


def _check_output_dir(output_dir: Path, dataset_paths: list[Path], overwrite: bool) -> None:
    for dataset_path in dataset_paths:
        if output_dir == dataset_path:
            raise ValueError(f"Output directory must differ from source dataset: {output_dir}")
        if dataset_path in output_dir.parents:
            raise ValueError(f"Output directory must not be inside source dataset {dataset_path}: {output_dir}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it.")
        logger.info("Deleting existing output directory: %s", output_dir)
        shutil.rmtree(output_dir)


def _repo_id_from_path(path: Path) -> str:
    return path.name


def merge_lerobot_datasets_by_path(
    *,
    dataset_paths: list[Path],
    output_dir: Path,
    output_repo_id: str | None = None,
    overwrite: bool = False,
) -> LeRobotDataset:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_paths = _resolve_dataset_paths(dataset_paths)
    output_dir = output_dir.expanduser().resolve()
    output_repo_id = output_repo_id or output_dir.name

    _check_output_dir(output_dir, dataset_paths, overwrite)

    repo_ids = [_repo_id_from_path(path) for path in dataset_paths]

    logger.info("Merging %d dataset(s)", len(dataset_paths))
    for repo_id, path in zip(repo_ids, dataset_paths, strict=True):
        logger.info("  %s <- %s", repo_id, path)
    logger.info("Output repo_id: %s", output_repo_id)
    logger.info("Output directory: %s", output_dir)

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=dataset_paths,
        aggr_root=output_dir,
    )

    return LeRobotDataset(output_repo_id, root=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge local LeRobotDataset directories by path.")
    parser.add_argument(
        "--dataset-paths",
        type=Path,
        nargs="+",
        required=True,
        help="Source LeRobotDataset root directories to merge, in the desired order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the merged LeRobotDataset.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Repo id written into dataset metadata. Default: basename of --output-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --output-dir if it already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    merged_dataset = merge_lerobot_datasets_by_path(
        dataset_paths=args.dataset_paths,
        output_dir=args.output_dir,
        output_repo_id=args.output_repo_id,
        overwrite=args.overwrite,
    )
    logger.info(
        "Done. episodes=%d frames=%d root=%s",
        merged_dataset.meta.total_episodes,
        merged_dataset.meta.total_frames,
        merged_dataset.root,
    )


if __name__ == "__main__":
    main()
