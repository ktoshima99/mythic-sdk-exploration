import configparser
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from vnnmap.run_vnnmap import DEFAULT_OCRAM1
from vnnmap.run_vnnmap import run_vnnmap as _run_vnnmap
from vnnort.utils.onnx_utils.meta_fields import _get_onnx_meta_field

if TYPE_CHECKING:
    from vnnort.models.vid_model import VidModel

logger = logging.getLogger(__name__)

VNN_MAP_SUBDIRECTORY = "vnnmap"


def run_compilation(
    model: "VidModel",
    n_mps: int = 1,
    ocram1_bytes: int = DEFAULT_OCRAM1,
    system_config: Path | None = None,
) -> Dict[str, Any]:
    """Run the compilation process by calling the vnnmap executable.

    This takes into consideration the vidir file created in a previous step and calls the vnnmap executable.
    The resulting VCI file can be found under model_directory/vnnmap$nMPs.
    Afterwards you may may call `run_codegen` to generate the actual binaries to be run in the simulator or on HW.

    Args:
        model (VidModel): VidModel to be compiled. Should be in state QUANTIZED.
        n_mps (int): Number of MPs to map this model to. May be one of [1, 4, 8]. Defaults to 1.
            Ignored when system_config is provided (nMPs is read from the config file instead).
        ocram1_bytes (int): Size of the OCRAM1 in bytes to be used in the system config for vnnmap. Defaults to 0.
            Ignored when system_config is provided.
        system_config (Path | None): Path to a pre-existing system config file. When provided,
            this file is passed directly to vnnmap instead of generating one from n_mps/ocram1_bytes.

    Returns:
        Dict[str, Any]: Compilation metrics.

    Raises:
        ValueError: If the model is not in quantized state.
        FileNotFoundError: If the .vidir model file cannot be found.
        RuntimeError: If vnnmap does not generate a vci file.
    """
    # Inline imports to avoid circular dependencies
    from vnnort.models.vid_model import ModelState

    if system_config is not None:
        cfg = configparser.ConfigParser()
        cfg.read(system_config)
        n_mps = int(cfg["sys"]["nMPs"])

    # In the current configuration only 1, 4 and 8 MPs are supported.
    if n_mps not in [1, 4, 8]:
        raise ValueError("Number of MPs must be 1, 4 or 8.")

    # This can only be done for models already quantized and exported to .vidir
    if not model.state == ModelState.QUANTIZED:
        raise ValueError(f"Model is not in state {ModelState.QUANTIZED.name}")
    quantized_model = model._model_repr
    model_directory = Path(_get_onnx_meta_field(quantized_model, "model_directory"))
    vidir_path = model_directory / Path(model.model_name + ".vidir")
    if not vidir_path.exists():
        raise FileNotFoundError(f"v-NN Mapper serialized model not found at expected location: {str(vidir_path)}")

    # Run compilation
    logger.info("Starting compilation.")
    vnnmap_result_directory = model_directory / (VNN_MAP_SUBDIRECTORY + f"_nMPs{n_mps}")
    metrics: dict[str, Any]
    metrics, _ = _run_vnnmap(
        vidir_path,
        vnnmap_result_directory,
        n_mps=n_mps,
        ocram1_bytes=ocram1_bytes,
        generate_vci=True,
        system_config=system_config,
    )

    # Check that vci file was created
    vci_output_path = vnnmap_result_directory / (model.model_name + ".part.vidir.vci")
    if not vci_output_path.exists():
        raise RuntimeError(f"Unexpected Error. VCI file was not created by vnnmap call. Should be {vci_output_path}")

    logger.info("Compilation successful.")
    return metrics  # type: ignore
