"""Extract success/fail datasets from recorded datasets with CP annotations.

Datasets are organized in date folders like 0408_271ep_sft/.
Searches for eval_autonomy_* datasets, reads critical_phase_intervals.json,
and generates separate success/fail datasets in the same date folder.

Usage:
    # Extract all CP-annotated datasets in a date folder:
    PYTHONPATH=src python scripts/extract_cp_datasets.py --date 0408_271ep_sft

    # Extract from all date folders:
    PYTHONPATH=src python scripts/extract_cp_datasets.py --all

    # Extract from a specific dataset directory:
    PYTHONPATH=src python scripts/extract_cp_datasets.py --dataset-dir /path/to/0408_271ep_sft/eval_autonomy_143000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from lerobot.utils.critical_phase_extraction_fast import extract_critical_phase_dataset_direct as extract_critical_phase_dataset


def find_cp_datasets_in_dir(parent: Path) -> list[Path]:
    """Find eval_autonomy_* datasets containing critical_phase_intervals.json."""
    if not parent.is_dir():
        return []
    return sorted(
        d for d in parent.iterdir()
        if d.is_dir() and d.name.startswith("eval_autonomy_") and (d / "critical_phase_intervals.json").exists()
    )


def find_all_cp_datasets(ds_root_base: Path) -> list[Path]:
    """Search all date folders (MMDD_*) for CP-annotated datasets."""
    results = []
    for day_dir in sorted(ds_root_base.iterdir()):
        if day_dir.is_dir() and len(day_dir.name) >= 4 and day_dir.name[:4].isdigit():
            results.extend(find_cp_datasets_in_dir(day_dir))
    return results


def extract_from_dataset(source_dir: Path, task: str) -> None:
    """Extract success/fail CP datasets from a single source dataset."""
    cp_json = source_dir / "critical_phase_intervals.json"
    if not cp_json.exists():
        print(f"  [skip] No critical_phase_intervals.json in {source_dir.name}")
        return

    with open(cp_json) as f:
        raw_intervals = json.load(f)

    intervals = [
        (iv["episode_index"], iv["start_frame"], iv["end_frame"], iv.get("outcome"))
        for iv in raw_intervals
    ]

    if not intervals:
        print(f"  [skip] Empty intervals in {source_dir.name}")
        return

    day_dir = source_dir.parent
    # eval_autonomy_143000 -> time_tag = 143000
    time_tag = source_dir.name.replace("eval_autonomy_", "")
    source_repo_id = f"local/{source_dir.name}"

    print(f"\n  Source: {day_dir.name}/{source_dir.name} ({len(intervals)} intervals)")

    has_outcomes = any(iv[3] is not None for iv in intervals)

    if has_outcomes:
        for outcome in ("success", "failure"):
            matching = [iv for iv in intervals if iv[3] == outcome]
            if not matching:
                print(f"  [skip] No {outcome} intervals")
                continue

            out_name = f"eval_cp_{outcome}_{time_tag}"
            out_dir = day_dir / out_name
            if out_dir.exists():
                print(f"  [skip] {out_name} already exists")
                continue

            extract_critical_phase_dataset(
                source_repo_id=source_repo_id,
                source_root=source_dir,
                output_repo_id=f"local/{out_name}",
                output_root=out_dir,
                intervals=matching,
                task=task,
            )
            print(f"  [done] {outcome}: {len(matching)} segments -> {out_dir}")

        unlabeled = [iv for iv in intervals if iv[3] is None]
        if unlabeled:
            out_name = f"eval_cp_{time_tag}"
            out_dir = day_dir / out_name
            if not out_dir.exists():
                extract_critical_phase_dataset(
                    source_repo_id=source_repo_id,
                    source_root=source_dir,
                    output_repo_id=f"local/{out_name}",
                    output_root=out_dir,
                    intervals=unlabeled,
                    task=task,
                )
                print(f"  [done] unlabeled: {len(unlabeled)} segments -> {out_dir}")
    else:
        out_name = f"eval_cp_{time_tag}"
        out_dir = day_dir / out_name
        if out_dir.exists():
            print(f"  [skip] {out_name} already exists")
            return

        extract_critical_phase_dataset(
            source_repo_id=source_repo_id,
            source_root=source_dir,
            output_repo_id=f"local/{out_name}",
            output_root=out_dir,
            intervals=intervals,
            task=task,
        )
        print(f"  [done] all: {len(intervals)} segments -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract CP success/fail datasets from recorded data")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Date folder name (e.g. 0408_271ep_sft)")
    group.add_argument("--all", action="store_true", help="Process all date folders with CP annotations")
    group.add_argument("--dataset-dir", type=Path, help="Path to a specific dataset directory")
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve")
    parser.add_argument("--setup-json", default=None, help="Path to setup.json for dataset root")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset_dir:
        if not args.dataset_dir.exists():
            print(f"Error: {args.dataset_dir} does not exist")
            sys.exit(1)
        extract_from_dataset(args.dataset_dir, args.task)
        return

    setup_path = args.setup_json or str(Path.home() / ".roboclaw/workspace/embodied/setup.json")
    with open(setup_path) as fh:
        setup = json.load(fh)

    ds_root_base = setup.get("datasets", {}).get("root", "")
    if not ds_root_base:
        ds_root_base = str(Path.home() / ".roboclaw/workspace/embodied/datasets")
    ds_root_base = Path(ds_root_base).expanduser()

    if not ds_root_base.exists():
        print(f"Error: Dataset root {ds_root_base} does not exist")
        sys.exit(1)

    if args.all:
        datasets = find_all_cp_datasets(ds_root_base)
    else:
        day_dir = ds_root_base / args.date
        if not day_dir.exists():
            print(f"Error: Date folder {day_dir} does not exist")
            sys.exit(1)
        datasets = find_cp_datasets_in_dir(day_dir)

    if not datasets:
        print("No CP-annotated datasets found.")
        sys.exit(1)

    print(f"Found {len(datasets)} dataset(s) to process:")
    for ds in datasets:
        extract_from_dataset(ds, args.task)

    print("\nDone.")


if __name__ == "__main__":
    main()
