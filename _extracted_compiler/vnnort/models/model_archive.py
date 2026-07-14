"""Defines the functionality to load and save instances of the VidModel class.

For this you can use the load_model() and save_model() functions.
"""

import importlib
import logging
import os
from pathlib import Path
from typing import Any, cast

import onnx

from vnnort.models.vid_model import ModelState, VidModel
from vnnort.utils.onnx_utils.meta_fields import _get_onnx_meta_field, _set_onnx_meta_field

logger = logging.Logger(__name__)

# Mapping between model states and their corresponding file suffixes
model_state_to_file_suffix = {
    ModelState.INITIALIZED: ".vidi.onnx",
    ModelState.OPTIMIZED: ".vido.onnx",
    ModelState.QUANTIZED: ".vidq.onnx",
}


def save_model(vid_model: "VidModel", model_path: str | Path) -> None:
    """Save the model and its current state to disk.

    Args:
        vid_model (VidModel): The model to save.
        model_path (str | Path ): The path to save the model to.
    Raises:
        ValueError: If Model State and file suffix differ.
        NotImplementedError: If Model State is not supported.
    Returns:
        None: This function does not return anything.
    """
    # Make sure directory exists
    model_path = Path(model_path)
    save_directory = model_path.parent
    save_directory.mkdir(parents=True, exist_ok=True)

    # Make sure naming is correct
    model_state = vid_model.state
    suffix = model_state_to_file_suffix[model_state]

    if not str(model_path).endswith(suffix):
        raise ValueError(f"Model is in state {model_state}, but filename does not end with {suffix}.")

    # Load all data into the model
    onnx_model = vid_model._model_repr
    onnx.load_external_data_for_model(onnx_model, str(save_directory))

    # Delete old data file (otherwise onnx will append all tensor data, even if it is already there)
    data_file_name = str(model_path.name)[:-5] + ".dat"
    data_file_path = save_directory / data_file_name
    if os.path.exists(data_file_path):
        os.remove(data_file_path)

    # Add model base directory to onnx meta data, so that external data can be found by by onnxruntime at later point
    _set_onnx_meta_field(onnx_model, "model_directory", str(save_directory.absolute()))

    # Save onnx model
    if (
        model_state == ModelState.INITIALIZED
        or model_state == ModelState.OPTIMIZED
        or model_state == ModelState.QUANTIZED
    ):
        onnx.save_model(
            onnx_model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file_name,
        )
    else:
        # TODO: Save compiled model
        raise NotImplementedError(f"Model state {model_state} not yet supported.")

    logger.info(f"Saved model to {model_path}")


def _load_model_class(class_path: str) -> Any:
    """Load the class at class_path.

    This function can be used to load a class using the identifier from class_path. This needs to be
    a fully importable module path together with the class name in that module.

    Example:
        class_path = "vnnort.models.model_zoo.image_classification.torchvision.squeezenet1_0.SqueezeNet1_0"
        Class = _load_model_class(class_path)

    Args:
        class_path (str): module_path to class

    Returns:
        Any: Resulting class
    """
    module_name, class_name = class_path.rsplit(".", 1)  # Split into module and class names
    module = importlib.import_module(module_name)  # Import the module
    VidModelClass = getattr(module, class_name)
    return VidModelClass


def load_model(model_path: str | Path) -> "VidModel":
    """Load the model from disk.

    Args:
        model_path (str | Path): The path to load the model from.

    Raises:
        FileNotFoundError: If the model_path does not point to a valid file.
        ValueError: If model_module_path is not set in the models meta field or if model has unknown file extension.

    Returns:
        VidModel: The loaded model.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Provided model path {model_path} does not exist.")

    if model_path.suffix == ".onnx":
        onnx_model = onnx.load(model_path, load_external_data=False)
        model_module_path = _get_onnx_meta_field(onnx_model, "model_module_path")
        if model_module_path is None:
            raise ValueError(f"Provided model path {model_path} has no model_module_path in meta field.")
        ModelClass = _load_model_class(model_module_path)
        model_state_string = _get_onnx_meta_field(onnx_model, "model_state") or "INITIALIZED"
        model_state = ModelState[model_state_string]

        # Initialize the object, but do not call the constructor
        # This way we do not have to convert the model to onnx every time we use the class
        model = object.__new__(ModelClass)
        model._model_repr = onnx_model
        model._state = model_state
        model.setup()
        model = cast(VidModel, model)

    else:
        raise ValueError(f"Provided model path {model_path} has an unknown file extension.")

    return model
