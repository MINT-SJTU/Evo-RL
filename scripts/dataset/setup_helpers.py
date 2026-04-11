"""Shared setup.json helpers for dataset scripts."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SETUP_PATH = Path.home() / ".roboclaw/workspace/embodied/setup.json"
DEFAULT_DATASET_ROOT = Path.home() / ".roboclaw/workspace/embodied/datasets"


def load_setup_json(path: str | None = None) -> dict:
    """Load setup.json from the given path or the default location."""
    setup_path = path or str(DEFAULT_SETUP_PATH)
    with open(setup_path) as fh:
        return json.load(fh)


def resolve_dataset_root(setup: dict) -> Path:
    """Extract and expand the dataset root directory from setup config."""
    ds_root = setup.get("datasets", {}).get("root", "")
    if not ds_root:
        return DEFAULT_DATASET_ROOT
    return Path(ds_root).expanduser()


def get_sorted_followers(setup: dict) -> list[dict]:
    """Return follower arms sorted left-first from setup config."""
    followers = [a for a in setup["arms"] if "follower" in a["type"]]
    followers.sort(key=lambda a: (0 if "left" in a.get("alias", "") else 1))
    return followers


def get_sorted_leaders(setup: dict) -> list[dict]:
    """Return leader arms sorted left-first from setup config."""
    leaders = [a for a in setup["arms"] if "leader" in a["type"]]
    leaders.sort(key=lambda a: (0 if "left" in a.get("alias", "") else 1))
    return leaders
