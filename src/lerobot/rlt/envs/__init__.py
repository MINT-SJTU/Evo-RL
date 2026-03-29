from __future__ import annotations

from lerobot.rlt.collector import Environment
from lerobot.rlt.envs.reaching import ReachingEnvironment


def make_env(env_type: str, **kwargs) -> Environment:
    """Factory for creating environments by name."""
    if env_type == "reaching":
        return ReachingEnvironment(**kwargs)
    if env_type == "dummy":
        from lerobot.rlt.collector import DummyEnvironment
        return DummyEnvironment(**kwargs)
    raise ValueError(f"Unknown env_type: {env_type}")


__all__ = ["ReachingEnvironment", "make_env"]
