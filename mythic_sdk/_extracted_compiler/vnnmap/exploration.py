import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from vnnmap.run_vnnmap import run_vnnmap as _run_vnnmap
from vnnort.utils.onnx_utils.meta_fields import _get_onnx_meta_field

if TYPE_CHECKING:
    from vnnort.models.vid_model import VidModel

logger = logging.getLogger(__name__)

VNN_MAPPER_SUBDIRECTORY = "vnnmap"


def explore_model(
    model: "VidModel",
    n_mps: int = 1,
    ocram1_bytes: int = 2_097_152,
    system_config: Path | None = None,
    advanced: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Profile the model and return key metrics from exploration mode.

    Args:
        model (VidModel): VidModel to be compiled. Should be in state QUANTIZED.
        n_mps (int): Number of MPs to map this model to. Ignored when system_config is provided.
        ocram1_bytes (int): Size of the OCRAM1 in bytes to be used in the system config for vnnmap. Default is 2MB.
            Ignored when system_config is provided.
        system_config (Path | None): Path to a pre-existing system config file. When provided,
            this file is passed directly to vnnmap instead of generating one from n_mps/ocram1_bytes.
        advanced (bool): Whether to enable advanced exploration mode in vnnmap. Default is False.
            Note this flag has no effect in the EVAL package of the v-NN SDK.
    Returns:
        tuple[dict[str, Any], pd.DataFrame]: Dictionary with profiling metrics

    Raises:
        ValueError: If the model is not in quantized state.
        FileNotFoundError: If the .vidir model file cannot be found.
    """
    from vnnort.models.vid_model import ModelState

    if not model.state == ModelState.QUANTIZED:
        raise ValueError(f"Model is not in state {ModelState.QUANTIZED.name}")
    quantized_model = model._model_repr
    model_directory = _get_onnx_meta_field(quantized_model, "model_directory")
    vidir_path = Path(model_directory) / Path(model.model_name + ".vidir")
    if not vidir_path.exists():
        raise FileNotFoundError(f"v-NN Mapper serialized model not found at expected location: {str(vidir_path)}")

    model_directory = Path(model_directory)

    logger.info("Starting exploration mode.")
    vnnmap_result_directory = model_directory / VNN_MAPPER_SUBDIRECTORY
    metrics, layerwise_metrics = _run_vnnmap(
        vidir_path,
        vnnmap_result_directory,
        n_mps=n_mps,
        ocram1_bytes=ocram1_bytes,
        generate_vci=False,
        system_config=system_config,
        advanced=advanced,
    )

    logger.info("Exploration successful.")

    return metrics, layerwise_metrics  # type: ignore
