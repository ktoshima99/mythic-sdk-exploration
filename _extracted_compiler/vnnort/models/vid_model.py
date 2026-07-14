from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional, Tuple

from boltons.typeutils import classproperty
from onnx import ModelProto

from vnnmap.network import CapnprotoNetwork
from vnnort.data.base_dataset import DatasetBase
from vnnort.data.container import InputData, OutputData
from vnnort.models.initialization_config import InitializationConfig
from vnnort.optimizer.optimization_config import OptimizationConfig
from vnnort.quantizer.quantization_config import QuantizationConfig
from vnnort.quantizer.report.report import QuantizationReport
from vnnort.utils.config.model_flow_config import ModelFlowConfig
from vnnort.utils.onnx_utils.meta_fields import _get_onnx_meta_field, _set_onnx_meta_field

logger = logging.getLogger(__name__)


# Enum for model states
class ModelState(IntEnum):
    """States of the VidModel."""

    INITIALIZED = 0
    OPTIMIZED = 1
    QUANTIZED = 3
    COMPILED = 4


class VidModel(ABC):
    """Base class for all models used in the VidORT interface.

    This class provides three main responsibilities.

    1. Providing a common interface for all models.

       The class defines multiple abstract methods that all models must implement:

       - preprocess(): Defines the expected input data format. All models receive
         task-specific data. For example, an image classifier receives
         ImageClassificationData. The data must be formatted so it can be fed into
         the ONNX model returned by ``_to_onnx()``.

       - postprocess(): Takes the raw model output and converts it back into a
         task-specific data container. For example, image classifiers must return
         ImageClassificationData outputs.

       - _to_onnx(): Returns a runnable ONNX ``ModelProto`` object. The source
         framework does not matter (PyTorch, TensorFlow, HuggingFace, etc.),
         as long as the model is executable and uses the correct opset version.

       - _setup(): Called during initialization. This may include loading
         tokenizers or other preprocessing artifacts.

       - _optimize_hook(): Called from ``optimize()``. Should modify
         ``self._model_repr`` so the model can continue through the
         quantize() and compile() stages.

       - default_model_flow_config(): Returns a ``ModelFlowConfig`` instance
         with recommended default parameters for the model flow.

    2. Loading and saving the model.

       Use the ``.load()`` and ``.save()`` methods.

    3. Tracking model states during the model flow.

       The class acts as a container tracking all model state transitions.
       During the flow, the model evolves through multiple stages. After each
       step, both the current and previous versions can be accessed via the
       ``*_model`` accessors.

       States:

       - INITIALIZED: After calling ``initialize()``.
       - OPTIMIZED: After calling ``optimize()``.
       - QUANTIZED: After calling ``quantize()``.
       - COMPILED: After calling ``compile()``.
    """

    def __init__(
        self, model_directory: str | Path, initialization_config: Optional[InitializationConfig] = None
    ) -> None:
        """Initialize the VidModel."""
        if initialization_config is None:
            initialization_config = InitializationConfig()
        # Create model directory
        model_directory = Path(model_directory)
        if model_directory.is_file():
            raise ValueError(f"model_directory {model_directory} is a file")
        if not model_directory.exists():
            logger.info(f"Model directory {model_directory} does not exist, creating it.")
            model_directory.mkdir(parents=True, exist_ok=True)

        self._model_repr: ModelProto = self.initialize_onnx(model_directory, initialization_config)
        self._state = ModelState.INITIALIZED

        # Set meta data in ONNX file so that the VidModel can be serialized
        self._update_onnx_meta()

        # Save the initialized model
        from vnnort.models.model_archive import model_state_to_file_suffix  # Inline import to avoid circular dep. issue

        initialized_path = model_directory / (self.model_name + model_state_to_file_suffix[ModelState.INITIALIZED])

        self.save(initialized_path)
        logger.info(f"Saved initialized model to {initialized_path}")

        # Setup the rest of the VidModel object so that it is fully initialized and can be used
        self.setup()

    def _update_onnx_meta(self) -> None:
        """Update the onnx meta data fields of our current ONNX model.

        We use ONNX metadata to serialize VidModels. It's thus possible to load a VidModel class
        purely from a .onnx without having to store any information about the class that produced it.
        This is accomplished by storing the module path (e.g. vnnort.models.model_zoo.image_classification...)
        and class name together with the model state.
        """
        model_module_path = self.__class__.__module__ + "." + self.__class__.__qualname__
        _set_onnx_meta_field(self._model_repr, "model_module_path", model_module_path)
        _set_onnx_meta_field(self._model_repr, "model_state", self.state.name)

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Define what format the input data is expected to be.

        All models can expect the input data to
        be of the "task type" they support. E.g. an image classifier, will receive instances of
        ImageClassificationData. The data needs to be formatted in way that it can be fed into the model
        exported by _to_onnx().
        """
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, model_output: Any, input_data: InputData) -> OutputData:
        """Transform the output of the model, as returned by the model exported from _to_onnx() to a common format.

        Depending on what kind of model it is, it is expected to return a certain data container type.
        E.g. image classifiers need to return ImageClassificationData outputs.

        Args:
            model_output(Any): The output as returned by the InferenceEngine.
            input_data(InputData): The InputData object also parsed to preprocess(). This might be needed in a
                postprocessing step.

        Raises:
            NotImplementedError: Subclasses need to implement this function.

        Returns:
            OutputData: The result of the postprocessing step specific to the model task type. E.g. ImageClassificationOutput for
            image classification models.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def initialize_onnx(
        cls, model_directory: str | Path, initialization_config: Optional[InitializationConfig] = None
    ) -> ModelProto:
        """Return a runable ONNX ModelProto object.

        It does not really matter, where
        this comes from (torch, huggingface, tensorflow, etc), as long as its runable and has the correct OPSET
        version.

        Args:
            model_directory (str | Path): The directory reserved for this model. It can be used to store any additional
                information requrired to run this model (e.g. ONNX preprocessing nodes).
            initialization_config (Optional[InitializationConfig], optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: Raised when this method is not implemented by subclass.

        Returns:
            ModelProto: The rununable ONNX ModelProto object
        """
        raise NotImplementedError()

    def setup(self) -> None:
        """Set up everything needed to run pre- and postprocessing.

        This function is called in the constructor and can be used to setup anything necessary to run the model.
        E.g. a NLP model needs to setup the tokenizer.
        """
        pass

    def optimize_hook(self, onnx_model: ModelProto) -> ModelProto:
        """Apply custom optimization/modification steps to the model after the optimization pipeline.

        This function is called at the end of the optimization pipeline, after all model graph optimizations and
        pattern matching operations are done. It can be used to apply custom modifications to the model. After
        this function is called, another shape inference step and name standardization functions are called.


        Args:
            onnx_model (ModelProto): The ONNX model at the latest optimization stage, which may be modified

        Returns:
            ModelProto: The modified model.
        """
        return onnx_model

    @classmethod
    @abstractmethod
    def load_default_dataset(cls) -> DatasetBase:
        """Return an initialized dataset that can be used to load data samples for this model."""
        raise NotImplementedError

    @classmethod
    def default_model_flow_config(cls) -> ModelFlowConfig:
        """Return a ModelFlowConfig instance with default parameters resulting in a "good" result for this model.

        This function can be overwritten to return a custom ModelFlowConfig instance.
        """
        config = ModelFlowConfig(model_name=cls.model_name)
        return config

    @classmethod
    def from_file(cls, model_path: str | Path) -> "VidModel":
        """Load a VidModel from file.

        Args:
            model_path(str|Path): path pointing to the model.
        Returns:
            VidModel: instance of the VidModel class stored.
        """
        from vnnort.models.model_archive import load_model  # Import here to avoid circular import

        return load_model(model_path)

    def save(self, path: str | Path) -> None:
        """Save a VidModel to file.

        Args:
            path(str|Path): path to save the model to.

        Returns:
            None: this function saves the model without returning it.

        """
        from vnnort.models.model_archive import save_model  # Import here to avoid circular import

        save_model(self, path)

    @property
    def state(self) -> ModelState:
        """Return the current state of the model."""
        return self._state

    def optimize(self, optimization_config: Optional[OptimizationConfig] = None) -> "VidModel":
        """
        Optimize the model and change its state from INITIALIZED to OPTIMIZED.

        Args:
            optimization_config (Optional[OptimizationConfig]): The configuration for optimization. Defaults to None.

        Returns:
            VidModel: The optimized model.
        """
        from vnnort.optimizer.optimizer import optimization_pipeline

        if optimization_config is None:
            optimization_config = OptimizationConfig()
        # Apply optimization

        self._model_repr = optimization_pipeline(self)

        # Update the model state and metadata
        self._state = ModelState.OPTIMIZED
        self._update_onnx_meta()

        return self

    def quantize(
        self,
        quantization_config: Optional[QuantizationConfig] = None,
        max_benchmark_samples: int = 0,
        n_workers: int = 8,
        dataset: DatasetBase | None = None,
    ) -> Tuple[QuantizationReport, CapnprotoNetwork]:
        """Quantize the model and change its state from OPTIMIZED->QUANTIZED.

        Args:
            quantization_config (Optional[QuantizationConfig]): The configuration for quantization. Defaults to None.
            max_benchmark_samples (int): Number of benchmark samples to be used. When 0, no benchmarks will be run.
                Defaults to 0.
            n_workers(int): Maximum number of worker threads used to perform processing tasks
            dataset (DatasetBase | None): Dataset used for calibration during quantization. Defaults to None.

        Raises:
            ValueError: if model is not in state OPTIMIZED
        Returns:
            Tuple[QuantizationReport, CapnprotoNetwork]: Returns the quantization report and CapnProtoNetwork.
        """
        if quantization_config is None:
            quantization_config = QuantizationConfig()

        if not self.state == ModelState.OPTIMIZED:
            raise ValueError("Model needs to be in state OPTIMIZED to be quantized")

        # Load model and dataset
        if dataset is None:
            dataset = self.load_default_dataset()

        from vnnort.quantizer.vid_quantizer import VidQuantizer

        quantizer = VidQuantizer(self, dataset, quantization_config, max_benchmark_samples, n_workers=n_workers)

        # Run actual quantization
        quantization_report, capnproto_network = quantizer.run()
        self._state = ModelState.QUANTIZED
        self._update_onnx_meta()

        # Save the .vidir file within the model directory. This way, we can directly compile the model after quant
        vidir_path = Path(_get_onnx_meta_field(self._model_repr, "model_directory")) / (self.model_name + ".vidir")
        capnproto_network.save(vidir_path)
        logger.info("Saved serialized model to: " + str(vidir_path))

        return quantization_report, capnproto_network

    def __eq__(self, other: object) -> bool:
        """Compare this model to another instance.

        They are the same, if the model state representations
        are the same.
        """
        if not isinstance(other, VidModel):
            raise NotImplementedError
        return self._model_repr == other._model_repr

    @classproperty  # type: ignore
    def model_name(cls) -> str:
        """Return the name of the model, which is defined as the class name."""
        model_name = str(cls.__name__)  # type: ignore
        return model_name

    def __call__(self, input_data: InputData) -> OutputData:
        """Process iput_data through the model in a single step.

        Args:
            input_data (InputData): input data to process

        Returns:
            OutputData: model specific ouptut data

        Raises:
            TypeError: if input_data is not of type InputData
        """
        if not isinstance(input_data, InputData):
            raise TypeError("input_data must be of type InputData")

        from vnnort.inference.engine import InferenceEngine

        engine = InferenceEngine(self)
        model_output = engine.run(input_data)

        if not isinstance(model_output, OutputData):
            raise TypeError("model_output must be of type OutputData")

        return model_output
