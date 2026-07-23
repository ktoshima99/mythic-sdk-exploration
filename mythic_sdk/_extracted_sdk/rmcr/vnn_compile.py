# Copyright (C) 2026, Mythic Inc. All rights reserved.
#
"""v-NN SDK compilation step for the RMCR compiler pipeline."""

import logging
import shlex
import shutil

from pathlib import Path
from typing import Dict, Optional

from mythic.model_deployment.rmcr.compiler import CompilerDockerConfig, run_in_docker, _ensure_writable_output_dir

logger = logging.getLogger(__name__)

# VNN SDK layout inside the compiler docker image.
_VNN_SDK_SCRIPTS_DIR = Path("/mythic/vnnsdk/scripts")
_VNN_SDK_PYTHON_ENV = "/mythic/pyvnnsdk-env"


def resolve_off_chip_onnx(munc_artifact_dir: Path, munc_artifact_metadata: dict) -> Optional[Path]:
    """Return the off_chip_2 ONNX path, or None if absent."""
    onnx_files = munc_artifact_metadata["onnx_graphs"]
    off_chip_stages = [k for k in onnx_files if "off_chip_2" in k]
    if not off_chip_stages:
        return None
    assert len(off_chip_stages) == 1, "Multiple off_chip_2 stages found"
    return munc_artifact_dir / onnx_files[off_chip_stages[0]]


def vnn_compile(
    docker_config: CompilerDockerConfig,
    off_chip_onnx: Path,
    output_dir: Path,
    vnn_cfg: Dict,
    local_script: Optional[Path]
) -> None:
    """Run a v-NN SDK postprocessing script in the compiler docker image."""
    container_script = vnn_cfg.get("CONTAINER_SCRIPT")
    if not local_script and not container_script:
        raise RuntimeError("VNN must specify one of the script paths")
    if local_script:
        script_in_container = _stage_local_script(local_script, docker_config)
    else:
        script_in_container = str(_VNN_SDK_SCRIPTS_DIR / container_script)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_in_container = docker_config.local_path_to_container(off_chip_onnx)
    output_dir_in_container = docker_config.local_path_to_container(output_dir)

    logger.info("Running v-NN SDK script '%s' in compiler docker image", script_in_container)

    args = ["python3", script_in_container,
            "--source_onnx", onnx_in_container,
            "--result_directory", output_dir_in_container]
    run_command = (
        f"source {_VNN_SDK_PYTHON_ENV}/bin/activate && "
        + " ".join(shlex.quote(a) for a in args)
    )
    run_command = _ensure_writable_output_dir(run_command, output_dir_in_container)
    run_in_docker(docker_config, run_command, shm_size=vnn_cfg.get("SHM_SIZE"))


def _stage_local_script(local_script: Path, docker_config: CompilerDockerConfig) -> str:
    """Copy a local script into the docker work dir; return its in-container path."""
    if not local_script.is_absolute():
        raise ValueError(f"VNN.LOCAL_SCRIPT path must be absolute, got '{local_script}'.")
    if not local_script.exists():
        raise FileNotFoundError(f"VNN.LOCAL_SCRIPT '{local_script}' does not exist.")
    staging_dir = Path(docker_config.local_work_dir) / "compiler_config_files"
    staging_dir.mkdir(parents=True, exist_ok=True)
    dest = staging_dir / local_script.name
    shutil.copy2(local_script, dest)
    return docker_config.local_path_to_container(dest)
