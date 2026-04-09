#!/usr/bin/env python
"""
Run single_server_datasets_check_merge on multiple machines via SSH.

Copies the evo-rl repo's script to each remote machine and executes it.
Results are printed per-machine as they complete.

Examples:

```bash
# Merge zzz_20260402_* datasets on all configured machines
PYTHONPATH=src python -m lerobot.scripts.multi_server_datasets_check_merge \
    --prefix zzz_20260402 --output-name 0402 \
    --task "Insert the copper screw into the black sleeve."

# Only run on specific machines
PYTHONPATH=src python -m lerobot.scripts.multi_server_datasets_check_merge \
    --prefix zzz_20260402 --output-name 0402 --hosts 127 249

# Dry-run on all machines
PYTHONPATH=src python -m lerobot.scripts.multi_server_datasets_check_merge \
    --prefix zzz_20260402 --output-name 0402 --dry-run
```
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Host:
    alias: str
    ip: str
    user: str
    password: str
    conda_env: str = "roboclaw"
    repo_path: str = "~/code/hsy/evo-rl"
    datasets_root: str = "~/.roboclaw/workspace/embodied/datasets/local"


# ── Machine registry ───────────────────────────────────────────────────
# Edit this list to add/remove machines.

HOSTS: list[Host] = [
    Host(alias="127", ip="192.168.31.127", user="zhaobo", password="zhaobo"),
    Host(alias="230", ip="192.168.31.230", user="zhaobo", password="zhaobo"),
    Host(alias="100", ip="192.168.31.100", user="zhaobo-4090-1", password="zhaobo-4090-1"),
    Host(alias="10",  ip="192.168.31.10",  user="bozhao4060", password="bozhao4060"),
    Host(alias="249", ip="192.168.31.249", user="shuyuan", password="shuyuan"),
]


def run_on_host(
    host: Host,
    prefix: str,
    output_name: str,
    task: str | None,
    short_ratio: float,
    long_ratio: float,
    dry_run: bool,
) -> tuple[str, int, str]:
    """SSH into host and run single_server_datasets_check_merge. Returns (alias, returncode, output)."""

    # Build the remote Python command
    script_module = "lerobot.scripts.single_server_datasets_check_merge"
    remote_cmd_parts = [
        f"source ~/miniconda3/etc/profile.d/conda.sh",
        f"conda activate {host.conda_env}",
        f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH",
        f"cd {host.repo_path}",
        "HF_HUB_OFFLINE=1 PYTHONPATH=src python3 -m " + script_module
        + f" --prefix {prefix}"
        + f" --output-name {output_name}"
        + f" --datasets-root {host.datasets_root}"
        + f" --short-ratio {short_ratio}"
        + f" --long-ratio {long_ratio}"
        + (f' --task "{task}"' if task else "")
        + (" --dry-run" if dry_run else ""),
    ]
    remote_cmd = " && ".join(remote_cmd_parts)

    # Use sshpass for non-interactive password auth
    ssh_cmd = [
        "sshpass", "-p", host.password,
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        f"{host.user}@{host.ip}",
        remote_cmd,
    ]

    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=1800)
    output = result.stdout + result.stderr
    return host.alias, result.returncode, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dataset check+merge on multiple machines via SSH.",
    )
    parser.add_argument("--prefix", required=True, help="Dataset name prefix.")
    parser.add_argument("--output-name", required=True, help="Merged dataset name.")
    parser.add_argument("--task", default=None, help="Task description for merged dataset.")
    parser.add_argument("--short-ratio", type=float, default=0.3)
    parser.add_argument("--long-ratio", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--hosts", nargs="*", default=None,
        help="Specific host aliases to run on (e.g. '127 249'). Default: all.",
    )
    args = parser.parse_args()

    targets = HOSTS
    if args.hosts:
        alias_set = set(args.hosts)
        targets = [h for h in HOSTS if h.alias in alias_set]
        missing = alias_set - {h.alias for h in targets}
        if missing:
            print(f"WARNING: unknown host aliases: {missing}", file=sys.stderr)

    if not targets:
        print("No target hosts.", file=sys.stderr)
        sys.exit(1)

    print(f"Running on {len(targets)} host(s): {[h.alias for h in targets]}")
    print(f"Prefix: {args.prefix}, Output: {args.output_name}")
    print("=" * 60)

    results: list[tuple[str, int, str]] = []
    for host in targets:
        print(f"\n{'─' * 60}")
        print(f"[{host.alias}] {host.user}@{host.ip} ...")
        print(f"{'─' * 60}")

        alias, rc, output = run_on_host(
            host, args.prefix, args.output_name, args.task,
            args.short_ratio, args.long_ratio, args.dry_run,
        )
        results.append((alias, rc, output))

        # Stream output
        print(output)
        status = "OK" if rc == 0 else f"FAILED (exit={rc})"
        print(f"[{alias}] {status}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for alias, rc, _ in results:
        status = "OK" if rc == 0 else f"FAILED (exit={rc})"
        print(f"  [{alias}] {status}")

    failed = [alias for alias, rc, _ in results if rc != 0]
    if failed:
        print(f"\n{len(failed)} host(s) failed: {failed}")
        sys.exit(1)
    print(f"\nAll {len(results)} host(s) completed successfully.")


if __name__ == "__main__":
    main()
