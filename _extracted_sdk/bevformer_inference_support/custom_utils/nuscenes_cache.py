"""Side-pickle cache for the parsed NuScenes devkit object.

The devkit's ``NuScenes(...)`` constructor parses ~2.5 GB of JSON every call
(``sample_data.json``, ``ego_pose.json``, ``sample_annotation.json``), which
dominates inference startup. ``create_data.py`` already builds a NuScenes
instance during annotation generation, so we pickle it there to a separate
file (distinct from ``nuscenes_infos_*.pkl``); subsequent inference loads it
in ~2 s instead of ~25 s.

The cache lives next to the info pkls (``out_path`` in the converter,
``ann_file``'s parent at inference time) — not the raw nuScenes ``data_root``,
which may be read-only or shared across many derivative annotation sets. The
cache is invalidated automatically if ``sample_data.json`` is newer than the
cache file (when the raw data root is known).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes


def cache_path(version: str, cache_dir: str | Path) -> Path:
    """``<cache_dir>/nuscenes_devkit_<version>.pkl`` — sibling to the info pkls for additional metadata."""
    return Path(cache_dir) / f"nuscenes_devkit_{version}.pkl"


def is_cache_fresh(
    version: str,
    cache_dir: str | Path,
    *,
    raw_data_root: str | Path | None = None,
) -> bool:
    """Cache is fresh if it exists and (when ``raw_data_root`` is given) is newer than ``sample_data.json``."""
    cp = cache_path(version, cache_dir)
    if not cp.is_file():
        return False
    if raw_data_root is not None:
        src = Path(raw_data_root) / version / "sample_data.json"
        if src.is_file() and cp.stat().st_mtime < src.stat().st_mtime:
            return False
    return True


def load_cached(
    version: str,
    cache_dir: str | Path,
    *,
    raw_data_root: str | Path | None = None,
) -> "NuScenes | None":
    """Return the pickled NuScenes if the cache is present and fresh; else ``None``."""
    if not is_cache_fresh(version, cache_dir, raw_data_root=raw_data_root):
        return None
    with open(cache_path(version, cache_dir), "rb") as f:
        return pickle.load(f)


def dump_cache(nusc, version: str, cache_dir: str | Path) -> Path:
    """Pickle ``nusc`` to ``<cache_dir>/nuscenes_devkit_<version>.pkl`` and return the path."""
    cp = cache_path(version, cache_dir)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "wb") as f:
        pickle.dump(nusc, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cp
