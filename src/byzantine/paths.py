"""Paths for local, single-user Byzantine research data."""

from __future__ import annotations

import os
from pathlib import Path


def app_data_dir() -> Path:
    """Return a user-owned data directory, optionally overridden for testing."""
    override = os.getenv("BYZANTINE_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    try:
        from platformdirs import user_data_dir
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("请安装 platformdirs：pip install -e \".[app]\"") from exc
    return Path(user_data_dir("Byzantine", "LifeArtist0102"))


def ensure_app_data_dir() -> Path:
    """Create the durable local data layout and return its root."""
    root = app_data_dir()
    for directory in (root, root / "documents", root / "qdrant"):
        directory.mkdir(parents=True, exist_ok=True)
    return root
