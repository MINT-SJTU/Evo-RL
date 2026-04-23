#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.acp_instance_table import write_acp_instance_table


def _default_repo_id(dataset_root: Path) -> str:
    return f"local/{dataset_root.name}"


def _default_output_path(
    dataset_root: Path,
    negative_bottom_ratio: float,
    positive_top_duplicate_ratio: float,
) -> Path:
    neg_pct = int(round(negative_bottom_ratio * 100))
    pos_pct = int(round(positive_top_duplicate_ratio * 100))
    filename = f"acp_instance_table_neg{neg_pct:02d}_posdup{pos_pct:02d}.parquet"
    return dataset_root / "meta" / filename


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ACP training instance table from value-inferred data.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to the source LeRobot dataset root.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repo id used for LeRobotDataset loading. Default: local/<dataset-root-name>.",
    )
    parser.add_argument(
        "--advantage-field",
        type=str,
        default="complementary_info.advantage",
        help="Advantage field written by value inference.",
    )
    parser.add_argument(
        "--negative-bottom-ratio",
        type=float,
        default=0.3,
        help="Bottom quantile promoted to explicit negative-tagged instances.",
    )
    parser.add_argument(
        "--positive-top-duplicate-ratio",
        type=float,
        default=0.3,
        help="Top quantile duplicated as explicit positive-tagged instances.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output parquet path. Default: <dataset-root>/meta/acp_instance_table_negXX_posdupYY.parquet",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output parquet.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(str(dataset_root))

    repo_id = args.repo_id if args.repo_id is not None else _default_repo_id(dataset_root)
    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else _default_output_path(
            dataset_root=dataset_root,
            negative_bottom_ratio=args.negative_bottom_ratio,
            positive_top_duplicate_ratio=args.positive_top_duplicate_ratio,
        )
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output path already exists: {output_path}")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        download_videos=False,
    )

    written_path = write_acp_instance_table(
        dataset=dataset,
        output_path=output_path,
        advantage_field=args.advantage_field,
        negative_bottom_ratio=float(args.negative_bottom_ratio),
        positive_top_duplicate_ratio=float(args.positive_top_duplicate_ratio),
    )
    print(written_path)


if __name__ == "__main__":
    main()
