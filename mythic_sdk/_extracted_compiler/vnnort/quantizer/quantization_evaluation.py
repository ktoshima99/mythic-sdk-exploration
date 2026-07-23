import logging
from typing import Any

import numpy as np
from tqdm import tqdm

from vnnort.data.container import InputData
from vnnort.data.dataloader import Dataloader
from vnnort.inference.engine import InferenceEngine
from vnnort.models.vid_model import VidModel
from vnnort.quantizer.quant_utils import TensorQuantInfo
from vnnort.utils.onnx_utils.graph_helper import ONNXGraphHelper

TENSOR_RANGE_INITIALIZER_SUFFIX = "_QUANT_TENSOR_RANGE"
ZERO_POINT_INITIALIZER_SUFFIX = "_QUANT_ZERO_POINT"
TENSOR_SKIP_SUFFIX = "_SKIP"

logger = logging.getLogger(__name__)


class LayerwiseEvaluator:
    """This class can be used to evaluate a quantized model.

    Specifically, it goes over each layer and quantizes only that layer. All other layers are kept in fp32. Afterwards,
    a full evaluation is performed on the quantized model, and the quantization error is calculated. Because it is
    very costly to reinitialize the ONNX Runtime for each layer, it is instead initialized only once with all
    QDQLayers inplace. QDQLayers have an optional additional input which is used to skip quantization of a tensor.
    These are then controlled by model input parameters.
    """

    def __init__(self, vid_model: VidModel, tensor_quant_infos: dict[str, TensorQuantInfo]):
        """Initialize the LayerwiseEvaluator.

        Args:
            vid_model (VidModel): The quantized model
            tensor_quant_infos (dict[str, TensorQuantInfo]): Dict mapping tensor names to TensorQuantInfo objects
        """
        self.qdq_skip_inputs = {
            name + TENSOR_SKIP_SUFFIX: np.array(True, dtype=bool) for name in tensor_quant_infos.keys()
        }
        self.preprocessor = vid_model.preprocess

    def _preprocess_func(self, input_data: InputData) -> Any:
        """Add the skip inputs controlling the QDQLayers in the model.

        Args:
            input_data (InputData): The input data.

        Returns:
            Any: The input data with the skip inputs added
        """
        # Call the original preprocessing method
        input_data = self.preprocessor(input_data)

        # Add custom skip inputs to model input dict
        input_data.update(self.qdq_skip_inputs)

        return input_data

    def run_layerwise_evaluation(
        self,
        vid_model: VidModel,
        graph_helper: ONNXGraphHelper,
        tensor_quant_infos: dict[str, TensorQuantInfo],
        dataloader: Dataloader,
    ) -> dict[str, Any]:
        """Run layerwise quantization error evaluation.

        Args:
            vid_model (VidModel): The quantized model to use
            graph_helper (ONNXGraphHelper): The ONNXGraphHelper object of the ORIGINAL unquantized model
            tensor_quant_infos (dict[str, TensorQuantInfo]): Dict mapping tensor names to TensorQuantInfo objects
            dataloader (Dataloader): The dataloader to use for benchmarking

        Returns:
            dict[str, Any]: A dict with two entries. The first entry (fp32_accuracy) is the accuracy of the fp32
                model. The second entry (layerwise_quantization) is the quantization error per layer.
        """

        def get_first_metric(metric_dict: dict[str, float]) -> float:
            """Get the first metric from a metric dict."""
            return next(iter(metric_dict.values()))

        BenchmarkClass = dataloader.dataset.get_benchmark()
        engine = InferenceEngine(vid_model)
        benchmark = BenchmarkClass(engine, dataloader, verbose=False)
        # Calculate accuracy with fp32 model
        dataloader.preprocess_func = self._preprocess_func
        fp32_result = get_first_metric(benchmark.run())

        # Collect layers
        layers = graph_helper.nodes.values()
        layer_wise_results = {}  # Mapping layer names to metrics

        for layer in tqdm(layers, desc="Evaluating layerwise quantization errors."):
            tensors_to_be_quantized = []

            # Skip all tensors except for the current layer
            for input_tensor in layer.inputs:
                if input_tensor.name in tensor_quant_infos:
                    tensors_to_be_quantized.append(input_tensor.name)
            for output_tensor in layer.outputs:
                if output_tensor.name in tensor_quant_infos:
                    tensors_to_be_quantized.append(output_tensor.name)
            for tensor_name in tensors_to_be_quantized:
                self.qdq_skip_inputs[tensor_name + TENSOR_SKIP_SUFFIX] = np.array(False, dtype=bool)

            # Run evaluation
            result = get_first_metric(benchmark.run())
            layer_wise_results[layer.name] = result

            # Reset skip inputs
            for tensor_name in tensors_to_be_quantized:
                self.qdq_skip_inputs[tensor_name + TENSOR_SKIP_SUFFIX] = np.array(True, dtype=bool)
            logger.debug(layer.name, ": ", result)

        return {"fp32_result": fp32_result, "layer_wise_quantization": layer_wise_results}
