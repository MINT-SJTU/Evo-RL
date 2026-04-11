"""Merge all cp_success datasets from multiple date folders into one dataset.

Scans 0407/0408/0409 _cp folders for eval_cp_success_* datasets, counts total
episodes, and merges them into a single dataset named
271ep_sft_success_critical_{N}ep.

Usage:
    PYTHONPATH=src python -m scripts.dataset.success_merge
    PYTHONPATH=src python -m scripts.dataset.success_merge --dry-run
    PYTHONPATH=src python -m scripts.dataset.success_merge --dates 0408_271ep_sft 0409_271ep_sft
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.aggregate import aggregate_datasets

from .setup_helpers import load_setup_json, resolve_dataset_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DATES = ["0407_271ep_sft", "0408_271ep_sft", "0409_271ep_sft"]
DEFAULT_TASK = "Insert the copper screw into the black sleeve."


def discover_success_datasets(ds_root: Path, dates: list[str]) -> list[tuple[Path, str]]:
    """Find all eval_cp_success_* datasets across date _cp folders.

    Returns list of (dataset_path, repo_id) tuples.
    """
    results = []
    for date in dates:
        cp_dir = ds_root / f"{date}_cp"
        if not cp_dir.is_dir():
            log.warning("CP dir not found: %s", cp_dir)
            continue
        for d in sorted(cp_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("eval_cp_success_"):
                continue
            info_path = d / "meta" / "info.json"
            if not info_path.exists():
                log.warning("Skipping %s: no info.json", d.name)
                continue
            results.append((d, d.name))
    return results


def count_episodes(datasets: list[tuple[Path, str]]) -> int:
    """Sum total_episodes across all datasets."""
    total = 0
    for ds_path, _ in datasets:
        info = json.loads((ds_path / "meta" / "info.json").read_text())
        total += info.get("total_episodes", 0)
    return total


def set_task_name(ds_path: Path, task: str) -> None:
    """Overwrite tasks.parquet with a single task entry."""
    tasks_path = ds_path / "meta" / "tasks.parquet"
    df = pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task"))
    pq.write_table(pa.Table.from_pandas(df), tasks_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all cp_success datasets into one")
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES, help="Date folder names")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--setup-json", default=None, help="Path to setup.json")
    parser.add_argument("--dry-run", action="store_true", help="Only count, don't merge")
    args = parser.parse_args()

    setup = load_setup_json(args.setup_json)
    ds_root = resolve_dataset_root(setup)

    datasets = discover_success_datasets(ds_root, args.dates)
    if not datasets:
        log.error("No cp_success datasets found")
        return

    total_eps = count_episodes(datasets)
    output_name = f"271ep_sft_success_critical_{total_eps}ep"
    output_path = ds_root / output_name

    log.info("Found %d cp_success datasets, %d total episodes", len(datasets), total_eps)
    for ds_path, repo_id in datasets:
        info = json.loads((ds_path / "meta" / "info.json").read_text())
        log.info("  %s: %d eps, %d frames", repo_id, info["total_episodes"], info["total_frames"])

    log.info("Output: %s", output_path)

    if args.dry_run:
        log.info("--dry-run: would merge into %s", output_name)
        return

    if output_path.exists():
        log.info("Removing existing output: %s", output_path)
        shutil.rmtree(output_path)

    repo_ids = [repo_id for _, repo_id in datasets]
    roots = [ds_path for ds_path, _ in datasets]

    log.info("Merging %d datasets...", len(datasets))
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_name,
        roots=roots,
        aggr_root=output_path,
    )

    set_task_name(output_path, args.task)

    merged_info = json.loads((output_path / "meta" / "info.json").read_text())
    log.info(
        "Done: %s — %d episodes, %d frames",
        output_name,
        merged_info["total_episodes"],
        merged_info["total_frames"],
    )


if __name__ == "__main__":
    main()
