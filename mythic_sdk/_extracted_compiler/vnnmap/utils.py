import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from vnnmap.capnproto_interface import TensorDataType


def find_vid_repos_and_commits() -> dict[str, str | None]:
    """Find the vnnmap and vidort repos and return the currently checked out commit ids.

    Returns:
        dict[str, str | None]: A dictionary with keys "vidort" and "vnnmap" and values being the current commit id or None if not a git repo.
    """
    # This file is at src/vnnmap/utils.py within the vidORT project root
    current_directory = Path(__file__).parent
    vidort_directory = current_directory.parent.parent.parent
    vnnmap_directory = vidort_directory / "libs" / "vid_aimap"

    def get_current_commit_id(repo_path: Path) -> str | None:
        """Get the current commit ID of a Git repository."""
        try:
            commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path, text=True).strip()
            commit_id = commit_id[:8]
            return commit_id
        except subprocess.CalledProcessError:
            return None

    result = {
        "vidort": get_current_commit_id(vidort_directory),
        "vnnmap": get_current_commit_id(vnnmap_directory),
    }
    return result


def np_dtype_to_capnproto_dtype(dtype: np.dtype[Any]) -> TensorDataType:
    """Convert a numpy dtype to the corresponding capnproto TensorDataType."""
    match dtype:
        case np.int64:
            return TensorDataType.int64
        case np.float32:
            return TensorDataType.float32
        case _:
            msg = f"Data type {dtype} not supported-"
            raise RuntimeError(msg)
