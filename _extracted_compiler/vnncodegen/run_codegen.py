import logging
import os
import shutil
import site
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


from vnnort.models.vid_model import VidModel
from vnnort.utils.onnx_utils.meta_fields import _get_onnx_meta_field

logger = logging.getLogger(__name__)

_PACKAGE_NAME = "vnncodegen"  # This should match the actual package name of this module for finding executables / libs
CNN_CODEGEN_MODE = "hosted"
VNN_MAP_SUBDIRECTORY = "vnnmap"  # Subdirectory where vnnmap outputs are expected to be found
VNN_CODEGEN_SUBDIRECTORY = "vnncodegen"  # Subdirectory where vnncodegen outputs will be placed
VNNRTGEN_SUBDIRECTORY = "vnnruntime"  # Subdirectory where vnnrtgen outputs will be placed


def _find_in_site_packages(relative_path: str | Path) -> list[Path]:
    """Return candidate paths for `relative_path` across all site-packages directories."""
    candidates = [Path(d) / relative_path for d in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(Path(user_site) / relative_path)
    return candidates


def _find_vid_sdk() -> Path:
    candidates = _find_in_site_packages(f"{_PACKAGE_NAME}/vid_sdk")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"vid_sdk not found in any expected location: {[str(c) for c in candidates]}")


def _ensure_executable(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Not executable: {path}")


def _find_executable(name: str) -> Path:
    candidates = _find_in_site_packages(f"{_PACKAGE_NAME}/{name}")
    for candidate in candidates:
        if candidate.exists():
            _ensure_executable(candidate)
            return candidate
    raise FileNotFoundError(f"{name} executable not found in any expected location: {[str(c) for c in candidates]}")


def _find_vmpcode_dir() -> Path:
    candidates = _find_in_site_packages(f"{_PACKAGE_NAME}/vmpcode")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"vmpcode directory not found in any expected location: {[str(c) for c in candidates]}")


def _run_command(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    stdout_path: Optional[Path] = None,
) -> None:
    """Run a command with optional working directory, environment, and stdout redirection.

    Raises RuntimeError on failure.
    """
    stdout_handle = None
    try:
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = open(stdout_path, "w")

        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stdout_handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n" f"stderr:\n{result.stderr}")
    finally:
        if stdout_handle is not None:
            stdout_handle.close()


def _load_vid_sdk_environment_variables() -> dict[str, str]:
    """Emulate module load vid_sdk_internal/** by manually loading environment vars."""
    # Base environment copied from current process
    env = os.environ.copy()

    # vid_sdk_internal root
    root = str(_find_vid_sdk())

    # Prepend paths
    env["PATH"] = f"{root}:{root}/bin:" + env.get("PATH", "")

    # Set environment variables
    env.update(
        {
            "VID_SDK_ROOT": root,
            "VIDEANTIS_LICENSE_PATH": f"{root}/license",
            "VMPCC_ROOT": root,
            "VSPGCC_ROOT": root,
            "PKG_CONFIG_PATH": f"{root}/lib/pkgconfig",
            "VIDSDK_DIR": root,
            "LD_LIBRARY_PATH": f"{root}/lib:" + env.get("LD_LIBRARY_PATH", ""),
        }
    )
    return env


def _run_vnncodegen(
    input_vci: str | Path,
    output_directory: str | Path,
) -> None:
    input_vci = Path(input_vci)
    output_directory = Path(output_directory)
    executable = _find_executable("vnncodegen")
    vmpcode_dir = _find_vmpcode_dir()
    if output_directory.exists():
        logger.info("Removing existing vnncodegen result directory: %s", output_directory)
        shutil.rmtree(output_directory)
    tmp_build_directory = output_directory / "build"
    tmp_build_directory.mkdir(parents=True, exist_ok=True)
    model_name = input_vci.stem.split(".")[0]
    output_vcnn = output_directory / f"{model_name}.vcnn"

    cmd = [
        str(executable),
        "-o",
        str(output_vcnn),
        "-d",
        str(vmpcode_dir),
        "-b",
        str(tmp_build_directory),
        str(input_vci),
        f"--mode={CNN_CODEGEN_MODE}",
    ]
    env = _load_vid_sdk_environment_variables()
    _run_command(cmd, env=env)


def _run_vnnrtgen(
    input_vcnn_file: Path,
    input_file: Path,
    output_directory: str | Path,
    num_vmps: int,
) -> None:
    executable = _find_executable("vnnrtgen")

    cmd = [
        str(executable),
        "--num-vmps",
        str(num_vmps),
        f"--mode={CNN_CODEGEN_MODE}",
        str(input_vcnn_file),
        str(input_file),
    ]

    output_directory = Path(output_directory)
    if output_directory.exists():
        logger.info("Removing existing vnnrtgen result directory: %s", output_directory)
        shutil.rmtree(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    env = _load_vid_sdk_environment_variables()

    _run_command(cmd, cwd=output_directory, env=env)


def run_codegen(model: VidModel, n_mps: int = 1) -> None:
    """Run the compilation process by calling codegenerator and vnnrtgen.

    Pipeline:
        1) Expects v-NN Mapper output:
            - `model_directory/vnnmap_<n_mps>/<model>.part.vidir.vci`
            - `model_directory/vnnmap_<n_mps>/<model>.layer0.inp`
        2) `_run_vnncodegen(...)` generates `model_directory/codegen_<n_mps>/<model>.vcnn`
        3) `_run_vnnrtgen(...)` generates binaries in `model_directory/runtime_<n_mps>/<model>.vcnn`


    Args:
        model (VidModel): VidModel to be compiled. Should be in state QUANTIZED.
        n_mps (int): Number of MPs to map this model to.

    Returns:
        None: This function does not return anything.

    Raises:
        FileNotFoundError: If the .vci model file cannot be found.
        ValueError: If the number of MPs is not in [1, 4, 8].
        RuntimeError: If multiple .inp input files are found.
    """
    if n_mps not in [1, 4, 8]:
        raise ValueError("Number of MPs must be 1, 4 or 8.")

    model_directory = _get_onnx_meta_field(model._model_repr, "model_directory")
    model_directory = Path(model_directory)

    logger.info("Starting codegen.")
    vnnmap_result_directory = model_directory / (VNN_MAP_SUBDIRECTORY + f"_nMPs{n_mps}")
    vci_path = vnnmap_result_directory / (model.model_name + ".part.vidir.vci")
    if not vci_path.exists():
        raise FileNotFoundError(f"Could not find vci file at {vci_path}. Did you run `run_compilation`?")

    codegen_result_directory = model_directory / (VNN_CODEGEN_SUBDIRECTORY + f"_nMPs{n_mps}")
    _run_vnncodegen(input_vci=vci_path, output_directory=codegen_result_directory)

    runtime_result_directory = model_directory / (VNNRTGEN_SUBDIRECTORY + f"_nMPs{n_mps}")
    input_vcnn_file = codegen_result_directory / (model.model_name + ".vcnn")

    inp_files = list(vnnmap_result_directory.glob(model.model_name + ".layer???.inp"))
    if len(inp_files) > 1:
        raise RuntimeError(f"Multiple .inp input files found in {vnnmap_result_directory}: {inp_files}")
    if len(inp_files) == 0:
        raise FileNotFoundError(f"Could not find .inp input file in {vnnmap_result_directory}. Did you run `run_vnncodegen`?")
    data_input_file = inp_files[0]
    if not input_vcnn_file.exists():
        raise FileNotFoundError(f"Could not find vcnn file at {input_vcnn_file}. Did you run `run_codegen`?")

    _run_vnnrtgen(
        input_vcnn_file=input_vcnn_file,
        input_file=data_input_file,
        output_directory=runtime_result_directory,
        num_vmps=n_mps,
    )

    logger.info("Codegen successful.")
