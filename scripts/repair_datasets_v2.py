"""Diagnose and auto-repair damaged LeRobot recording datasets (v2 modular CLI).

Usage:
    PYTHONPATH=src python scripts/repair_datasets_v2.py --dataset-dir /path/to/dataset_or_parent
    PYTHONPATH=src python scripts/repair_datasets_v2.py --dataset-dir /path/to/parent --force
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from lerobot.datasets.repair import (
    DamageType,
    DiagnosisResult,
    RepairResult,
    diagnose_dataset,
    find_datasets,
    repair_dataset,
)

log = logging.getLogger(__name__)


def log_diagnosis(diagnosis: DiagnosisResult) -> None:
    details = diagnosis.details
    tmp_tag = f" tmp_vid={details['n_tmp_videos']}" if details.get("n_tmp_videos", 0) > 0 else ""
    cp_tag = f" log_cp={details['n_log_cp']}" if details.get("n_log_cp", 0) > 0 else ""
    log.info(
        "%-40s %-18s repairable=%s recovery=%d min_images=%d parquet=%d videos=%d%s%s",
        diagnosis.dataset_dir.name,
        diagnosis.damage_type.value,
        diagnosis.repairable,
        details["n_recovery_lines"],
        details["min_images_per_camera"],
        details["n_parquet_rows"],
        details["n_video_files"],
        tmp_tag,
        cp_tag,
    )


def process_dataset(dataset_dir: Path, args: argparse.Namespace) -> RepairResult:
    """Diagnose and repair a single dataset. Returns RepairResult."""
    diagnosis = diagnose_dataset(dataset_dir)
    log_diagnosis(diagnosis)
    return repair_dataset(
        diagnosis, task=args.task, vcodec=args.vcodec, dry_run=args.dry_run, force=args.force,
    )


def print_summary(results: list[RepairResult]) -> None:
    healthy = sum(1 for r in results if r.outcome == "healthy")
    repaired = sum(1 for r in results if r.outcome == "repaired")
    skipped = sum(1 for r in results if r.outcome == "skipped")
    failed = sum(1 for r in results if r.outcome == "failed")
    log.info(
        "Summary: %d healthy, %d repaired, %d skipped, %d failed out of %d total",
        healthy, repaired, skipped, failed, len(results),
    )

    for r in results:
        if r.outcome != "failed":
            continue
        damage_type = r.damage_type.value if r.damage_type else "unknown"
        log.error("  FAILED: %s (%s)", r.dataset_dir.name, damage_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose and repair damaged LeRobot datasets")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset dir or parent directory")
    parser.add_argument(
        "--task",
        default="Insert the copper screw into the black sleeve",
        help="Task string used when rebuilding crash-no-save datasets",
    )
    parser.add_argument("--dry-run", action="store_true", help="Diagnose without applying repairs")
    parser.add_argument("--vcodec", default="h264", help="Video codec for rebuilt or encoded videos")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if _repaired already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.dataset_dir.exists():
        log.error("Path does not exist: %s", args.dataset_dir)
        sys.exit(1)

    datasets = find_datasets(args.dataset_dir)
    if not datasets:
        log.error("No dataset directories with meta/info.json found in %s", args.dataset_dir)
        sys.exit(1)

    log.info("Found %d dataset(s) to diagnose", len(datasets))
    results: list[RepairResult] = []

    for dataset_dir in datasets:
        try:
            result = process_dataset(dataset_dir, args)
        except Exception:
            log.exception("Failed to process %s", dataset_dir.name)
            result = RepairResult(
                dataset_dir,
                None,
                "failed",
                error=traceback.format_exc(),
            )
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
