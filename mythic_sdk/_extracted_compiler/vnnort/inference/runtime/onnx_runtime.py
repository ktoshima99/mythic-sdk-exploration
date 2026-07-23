import logging
import os
import pathlib
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from onnx import ModelProto

from vnnort.data.container import InputData
from vnnort.inference.runtime.base_runtime import BaseRuntime
from vnnort.utils.onnx_utils.meta_fields import _get_onnx_meta_field

logger = logging.getLogger(__name__)

# This environment variable can be used to activate the usage of CUDA devices for inference
VNNORT_DEVICE = os.environ.get("VNNORT_DEVICE")  # Valid values are "CUDA", "CPU"
if VNNORT_DEVICE is not None and (VNNORT_DEVICE != "CPU" and VNNORT_DEVICE != "CUDA"):
    raise ValueError(f"Unsupported device: {VNNORT_DEVICE}. Use CPU or CUDA")


class ONNXRuntime(BaseRuntime):
    """Small wrapper around ONNX InferenceSession runtime for inference.

    It provides two additional functionalities:

    1. It allows the user to run models with batch dimension > 1.
    2. The user may use the CUDAExecutionProvider to accelerate inference.

    For this, either the ``VNNORT_DEVICE`` environment variable must be set to
    ``CUDA`` or the ONNXRuntime class needs to be initialized with
    ``use_cuda=True``. If both are set, the constructor parameter takes
    precedence.

    In order to use the GPU, ``LD_LIBRARY_PATH`` needs to point to the installed
    cuDNN library. See the ONNX Runtime documentation for more details. If you
    installed the NVIDIA libraries via pip, this could e.g. be:

    .. code-block:: bash

        export LD_LIBRARY_PATH=${PYTHONENV}/lib/python3.12/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH

    Example:

    .. code-block:: python

        # Running inference on any model (e.g. for image classification)
        model = load_onnx_model()
        runtime = ONNXRuntime(model)
        data = {"input": np.zeros([1, 3, 224, 224])}
        pred = runtime(data)
    """  # noqa: D412

    def __init__(self, model: ModelProto, use_cuda: bool | None = None):
        """Initialize the ONNX runtime with a model.

        Args:
            model (ModelProto): The model to use
            use_cuda (bool | None): Whether to use cuda or not. Takes precedence over VNNORT_DEVICE

        Raises:
            TypeError: If the model is not a ModelProto.
        """
        if not isinstance(model, ModelProto):
            raise TypeError("Model must be of type ModelProto")
        self.use_cuda = use_cuda

        # Make sure that batch sizes > 1 can be inserted into the model
        self._make_model_input_multi_batch(model)

        self.session = self._create_ort_inference_session(model)

    def __call__(self, model_input: InputData) -> Any:
        """Run a single inference step with a model.

        Args:
            model_input (InputData): The input data to the model

        Returns:
            Any: The output of the model
        """
        outputs = [x.name for x in self.session.get_outputs()]

        result = self.session.run(None, model_input)
        result = OrderedDict(zip(outputs, result))
        return result

    def _make_model_input_multi_batch(self, model: onnx.ModelProto) -> None:
        for model_input in model.graph.input:
            input_shape = model_input.type.tensor_type.shape
            if len(input_shape.dim) >= 1 and input_shape.dim[0].dim_value == 1:
                input_shape.dim[0].dim_value = -1

    def _create_ort_inference_session(self, model: onnx.ModelProto) -> ort.InferenceSession:
        # By default, we use the CPU
        providers: list[Any] = [
            "CPUExecutionProvider",
        ]

        # Add CUDA device, if requested and available
        if self._use_cuda():
            providers.insert(
                0,
                (
                    "CUDAExecutionProvider",  # Good arguments according to onnxruntime docs
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
            )
            logger.debug("Using CUDAExecutionProvider for inference")
        else:
            logger.debug("Using CPUExecutionProvider for inference")

        # Suppress runtime warnings (these are mostly related to shape errors)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # 3 = ERROR, hides warnings

        # Try to initialize session in memory
        # In our VidModel.save method we set the model_directory meta data field
        model_directory = _get_onnx_meta_field(model, "model_directory")

        with tempfile.TemporaryDirectory() as temp_dir:
            # For VidModels, we need to save the tmp model into the model_directory, where the corresponding weights are
            model_directory = _get_onnx_meta_field(model, "model_directory")
            if model_directory is None:
                model_directory = temp_dir

            model_path = pathlib.Path(model_directory) / "tmp_model.onnx"

            # Save the modified model in the temp_dir or model_directory
            if model_directory == temp_dir:
                onnx.save(model, model_path, save_as_external_data=True, location="weights.dat")
            else:
                # If we already have vid model, the weights are already there, no need to save them again
                onnx.save(model, model_path)

            session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)

            # Remove tmp model file
            os.remove(model_path)

        return session

    @staticmethod
    def _cudnn_library_path_set_correctly() -> bool:
        ld_library_paths = os.environ.get("LD_LIBRARY_PATH")
        if ld_library_paths is None:
            return False
        for ld_library_path_str in ld_library_paths.split(":"):
            ld_library_path = Path(ld_library_path_str)

            # Check that it is a directory and contains the cudnn libraries
            if ld_library_path.is_dir() and ld_library_path.joinpath("libcudnn.so.9").exists():
                return True
        return False

    @staticmethod
    def _onnx_gpu_installed_correctly() -> bool:
        return "CUDAExecutionProvider" in ort.get_available_providers()

    def _use_cuda(self) -> bool:
        """Return whether or not to use cuda.

        Returns:
            bool: Whether to use cuda
        """
        # Constructor parameter takes precedence
        use_cuda = self.use_cuda
        if not use_cuda and (VNNORT_DEVICE is None or VNNORT_DEVICE == "CPU"):
            return False

        use_cuda = use_cuda or VNNORT_DEVICE == "CUDA"

        if use_cuda:
            # Default to CUDA if available
            if not self._cudnn_library_path_set_correctly():
                logger.warning("CUDA library path is not set correctly. ")
                return False
            if not self._onnx_gpu_installed_correctly():
                logger.warning("Make sure to install the onnxruntime-gpu package.")
                return False
        return use_cuda

    def cleanup(self) -> None:
        """Not used."""
        # Not really necessary here, but relevant when allocating hw ressources
        pass
