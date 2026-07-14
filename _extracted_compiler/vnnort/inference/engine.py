from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional, Type, cast

import numpy as np
from numpy.typing import NDArray

from vnnort.data.container import InputData, OutputData
from vnnort.data.utils import batch_model_input_data, unbatch_model_output_data
from vnnort.inference.runtime.base_runtime import BaseRuntime
from vnnort.inference.runtime.onnx_runtime import ONNXRuntime

if TYPE_CHECKING:
    from vnnort.models.vid_model import VidModel  # Only imported during type checking for type annotations

logger = logging.Logger(__name__)


class InferenceType(Enum):
    """Supported inference types."""

    ONNX_RUNTIME = 0
    SIMULATOR = 1  # Equivalent to v-NN Mapper representation?
    HARDWARE = 2
    # We may also add bittrue emulator from codegenerator output here?


class InferenceEngine:
    """An inference engine for VidORT.

    This class is intended to abstract away the runtime used for running model inference and provide a common interface
    this way.

    The engine as two modes of operation:
    1. Running ONNX models of state INITIALIZED, OPTIMIZED, QUANTIZED using the ONNX Runtime
    2. Running compiled models of state COMPILED on either the simulator or hardware.


    This can be controlled using the use_state and inference_type parameters.
    By default, the engine will use the current state of the model to determine which runtime to use.

    CURRENTLY ONLY ONNX RUNTIME IS SUPPORTED!

    Example::

        model = AnyVidModel()
        with InferenceEngine(model) as engine:
            engine.run(data)
    """

    def __init__(self, model: "VidModel", inference_type: Optional[InferenceType] = None):  # noqa  # type: ignore
        """Initialize the inference engine with a model.

        Args:
            model (VidModel): The model to use for inference.
            inference_type (InferenceType, optional): The inference type to use.
                Defaults to ONNX_RUNTIME.

        Raises:
            TypeError: If model is not of type VidModel.
            ValueError: If an unsupported inference_type is specified (currently only ONNX_RUNTIME is supported).
        """
        from vnnort.models.vid_model import VidModel

        if not isinstance(model, VidModel):
            raise TypeError("Model must be of type VidModel")
        if inference_type is not None and not isinstance(inference_type, InferenceType):
            raise TypeError("Inference type must be of type InferenceType")

        self._vid_model = model

        # For initialized, optimized, quantized state we use ONNX runtime
        # This is the only supported inference type for now
        if inference_type is None:
            inference_type = InferenceType.ONNX_RUNTIME
            # TODO: In the future we can initialize Simulator/Hardware here for compiled models
        else:
            if inference_type != InferenceType.ONNX_RUNTIME:
                raise ValueError("Currently only ONNX Runtime is supported")

        self._inference_type = inference_type
        self._runtime = self._load_runtime(self._inference_type)

    def run(
        self,
        input_data: InputData | list[InputData] | dict[str, NDArray[np.float32]] | list[dict[str, NDArray[np.float32]]],
    ) -> OutputData | list[OutputData] | dict[str, NDArray[np.float32]]:
        """Run a single inference step with the model and configured runtime.

        Args:
            input_data (InputData | list[InputData] | dict[str, NDArray[np.float32]] | list[dict[str, NDArray[np.float32]]]): The input data
                to the model
        Returns:
            OutputData | list[OutputData] | dict[str, NDArray[np.float32]]: The output data from the model

        Raises:
            ValueError: If the input data type is not supported.
        """
        # By default models processes multi batch data
        if isinstance(input_data, InputData):
            multi_batch_input = False
            if not isinstance(input_data, list):
                input_data = [input_data]
        else:
            multi_batch_input = True

        original_input_data = input_data

        # In case input_data is still of type InputData, the models preprocess function needs to be called
        input_of_type_input_data = False
        if isinstance(input_data, list) and all(isinstance(entry, InputData) for entry in input_data):
            input_data = [self._vid_model.preprocess(entry) for entry in input_data]
            if (
                isinstance(input_data, list)
                and all(isinstance(elem, list) for elem in input_data)
                and len(input_data) == 1
            ):
                input_data = input_data[0]
            input_of_type_input_data = True

        # At this point data should be either dict[str|NDarray] or list[dict[str|NDArray]]
        if isinstance(input_data, dict) and all(isinstance(entry, np.ndarray) for entry in input_data.values()):
            multi_batch_input = False
            input_data = [input_data]

        if not isinstance(input_data, list) and not all(isinstance(entry, dict) for entry in input_data):  # type: ignore
            raise ValueError("Wrong input data type. Please check your input data.")

        input_data = cast(List[dict[str, NDArray[np.float32]]], input_data)
        input_data = batch_model_input_data(input_data)
        model_output = self._runtime(input_data)  # type: ignore

        # In case we receive raw model input data we also return raw output data
        if not input_of_type_input_data:
            return model_output  # type: ignore
        unbatched_output = unbatch_model_output_data(model_output)  # type: ignore

        results = [
            self._vid_model.postprocess(output, input) for input, output in zip(original_input_data, unbatched_output)  # type: ignore
        ]

        # In case we provided non batched input, we also return non batched output
        if not multi_batch_input:
            return results[0]
        return results

    def _load_runtime(self, inference_type: InferenceType) -> BaseRuntime:
        if inference_type == InferenceType.ONNX_RUNTIME:
            return ONNXRuntime(self._vid_model._model_repr)
        else:
            raise ValueError("Currently only ONNX Runtime is supported")

    def __enter__(self) -> "InferenceEngine":
        """Enter the runtime context."""
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[Any]
    ) -> None:
        """Exit the runtime context."""
        if exc_type is not None:
            logger.error(f"An exception occurred: {exc_value}")
        self._runtime.cleanup()
