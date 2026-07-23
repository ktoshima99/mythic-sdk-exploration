import logging
from typing import Any, Tuple

import numpy as np

from vnnmap.network import CapnprotoNetwork
from vnnort.data.base_dataset import DatasetBase
from vnnort.data.dataloader import Dataloader
from vnnort.models.vid_model import ModelState, VidModel
from vnnort.quantizer.calibrator.calibrator import TensorStatisticCollector
from vnnort.quantizer.layer_handlers import QUANT_HANDLER_OP_REGISTRY, LayerHandlerBase
from vnnort.quantizer.qdq_helper import QDQHelper
from vnnort.quantizer.quant_utils import (
    TensorQuantInfo,
    calculate_quantization_range_from_data,
    calculate_quantization_range_from_histogram,
    fill_zero_entries,
    make_quantization_ranges_tensor_wide,
    power_of_two_values_to_exponents,
    round_up_to_power_of_two,
)
from vnnort.quantizer.quantization_config import QuantizationConfig
from vnnort.quantizer.quantization_evaluation import LayerwiseEvaluator
from vnnort.quantizer.report.report import QuantizationReport
from vnnort.utils.onnx_utils.graph_helper import ONNXGraphHelper, TensorType
from vnnort.utils.onnx_utils.unique_initializers import make_initializers_unique
from vnnort.utils.vnnmap_export import VNNMapExporter

logger = logging.getLogger(__name__)

DEFAULT_INPUT_TENSOR_N_BITS = 8
DEFAULT_INPUT_TENSOR_QUANTIZATION_AXIS = 1


class VidQuantizer:
    """Quantizer for a VidModel."""

    def __init__(
        self,
        model: VidModel,
        dataset: DatasetBase,
        config: QuantizationConfig,
        max_benchmark_samples: int = 0,
        n_workers: int = 8,
    ) -> None:
        """Initialize the VidQuantizer.

        Args:
            model (VidModel): model to be quantized
            dataset (DatasetBase): dataset to be used for calibration
            config (QuantizationConfig): quantization configuration
            max_benchmark_samples (int): Number of benchmark samples to be used. When 0, no benchmarks will be run.
                Defaults to 0.
            n_workers (int): Number of workers to be used for the dataloading. Defaults to 8.

        Raises:
            ValueError: if model is not in state OPTIMIZED
        """
        if not model.state == ModelState.OPTIMIZED:
            raise ValueError("Model needs to be in state OPTIMIZED or COMPRESSED to be quantized")
        self.config = config

        # Keep track of model
        self._model = model

        # Quantization requires all weights and biases to be used by only a single node
        make_initializers_unique(model._model_repr)

        # Setup dataloading
        self._dataset = dataset
        self._calibration_dataloader = Dataloader(
            dataset,
            model.preprocess,
            max_samples=config.calibration_dataset_size,
            n_workers=n_workers,
        )
        self._max_benchmark_samples = max_benchmark_samples

        # Use the ONNXGraphHelper class for easier handling of ONNX graph nodes and edges
        self._graph_helper = ONNXGraphHelper(self._model._model_repr)

        # For each layer op type we have one handler that specifies how to quantize incoming and outgoing tensors
        self._layer_handlers = self._register_layer_handlers()

        # This holds all of the information required to quantize the tensors and will be populated troughout the process
        self._tensor_quant_infos: dict[str, TensorQuantInfo] = {}

        self.n_workers = n_workers

    def run(self) -> Tuple[QuantizationReport, CapnprotoNetwork]:
        """Run the quantization process."""
        logger.info("Gathering tensor quantization info")
        self._gather_tensor_quant_infos()
        logger.info("Collecting tensor statistics")
        self._collect_tensor_statistics()
        logger.info("Calculating quantization ranges")
        self._calculate_quantization_ranges()
        logger.info("Adjusting quantization ranges")
        self._quantization_range_postprocessing()

        self._add_fake_quant_layers()
        skip_benchmark = self._max_benchmark_samples == 0
        if not skip_benchmark:
            logger.info("Run layerwise quantization evaluation")
            evaluation_results = self._evaluate_quantization()
        else:
            evaluation_results = None

        logger.info("Creating quantization report")
        report = self._create_quantization_report(evaluation_results)

        logger.info("Exporting to v-NN Mapper")
        exporter = VNNMapExporter(self._model.model_name, self._graph_helper, self._tensor_quant_infos)

        capnproto_network = exporter.export()
        return report, capnproto_network

    def _register_layer_handlers(self) -> dict[str, LayerHandlerBase]:
        """Register all layer handlers for the model.

        Returns:
            dict[str, LayerHandlerBase]: Dict mapping node names to layer handlers
        """
        handlers = {}
        for node in self._graph_helper.nodes.values():
            # Load the correct handler corresponding to this op_type
            op_type = node.op_type
            handler = (
                QUANT_HANDLER_OP_REGISTRY[op_type]
                if op_type in QUANT_HANDLER_OP_REGISTRY
                else QUANT_HANDLER_OP_REGISTRY["default"]
            )(node)
            handlers[node.name] = handler
        return handlers

    def _gather_tensor_quant_infos(self) -> None:
        """Gather all tensor quant infos from all layer handlers.

        Raises:
            RuntimeError: If a tensor already has a responsible node.
            ValueError: If an unsupported number of bits is provided.
        Returns:
            None: Tensors are gathered internally.
        """
        tensor_quant_infos: dict[str, TensorQuantInfo] = {}

        # Go over all layer handlers and gather their tensor quant infos
        for handler in self._layer_handlers.values():
            current_quant_infos = handler.tensor_quant_infos()

            for tensor_name, quant_info in current_quant_infos.items():
                if tensor_name in tensor_quant_infos:
                    if quant_info == tensor_quant_infos[tensor_name]:
                        continue
                    else:
                        raise RuntimeError(f"Tensor {tensor_name} already has a responsible node.")

                tensor_quant_infos[tensor_name] = quant_info

        # # Manually add model input tensors
        for graph_input in self._model._model_repr.graph.input:
            input_tensor = self._graph_helper.tensors[graph_input.name]
            # Skip if this is Integer input
            if np.issubdtype(input_tensor.dtype, np.integer):
                continue

            quant_info = TensorQuantInfo(
                input_tensor,
                n_bits=DEFAULT_INPUT_TENSOR_N_BITS,
                n_fraction_bits=7,
                axis=DEFAULT_INPUT_TENSOR_QUANTIZATION_AXIS,
            )
            tensor_quant_infos[input_tensor.name] = quant_info

        # Override config arguments
        for tensor_name, n_bits in self.config.tensor_n_bits:
            if tensor_name not in tensor_quant_infos:
                msg = f"Provided tensor {tensor_name} with {n_bits} bits does not exist."
                raise RuntimeError(msg)
            if n_bits not in [8, 16]:
                msg = f"Currently only 8 and 16 bits are supported. Set tensor {tensor_name} to {n_bits}"
                raise ValueError(msg)
            logger.debug(f"Overriding tensor {tensor_name} number of bits with {n_bits}")
            tensor_quant_infos[tensor_name].n_bits = n_bits
            # By default set this to n-1. We may need to also make this configurable
            tensor_quant_infos[tensor_name].n_fraction_bits = n_bits - 1

        self._tensor_quant_infos = tensor_quant_infos

    def _collect_tensor_statistics(self) -> None:
        """Run the calibration process and update the tensor quant infos with the results.

        Returns:
            None: collection of tensor statistics is done internally
        """
        # Collect statistics for all intermediate tensors
        tensor_quant_infos = self._tensor_quant_infos
        intermediate_quant_infos = {
            n: info for n, info in tensor_quant_infos.items() if info.tensor.tensor_type != TensorType.INITIALIZER
        }

        calibrator = TensorStatisticCollector(
            self._model,
            self._calibration_dataloader,
            intermediate_quant_infos,
            self.config.percentile_histogram_bins,
            n_workers=self.n_workers,
        )
        calibrator.run()

    def _calculate_quantization_ranges(self) -> None:
        """Run tensor range initialization and update the tensor quant infos with the results.

        Raises:
            RuntimeError: If both histogram and data are missing

        Returns:
            None: Initilization does not return a value
        """
        percentile = self.config.percentile
        for quant_info in self._tensor_quant_infos.values():
            # Depending on whether we quantize weights or intermediate tensors, we use the actual tensor data or
            # collected histogram
            if quant_info.tensor_histograms is not None:
                quant_info.quantization_ranges = calculate_quantization_range_from_histogram(
                    quant_info.tensor_histograms, percentile
                )
            elif quant_info.tensor.data is not None:
                quant_info.quantization_ranges = calculate_quantization_range_from_data(
                    quant_info.tensor.data, quant_info.axis, percentile
                )
            else:
                raise RuntimeError(f"Either histogram or data needs to be present for {quant_info.tensor.name}.")

    def _quantization_range_postprocessing(self) -> None:
        """Run tensor range postprocessing and update the tensor quant infos with the results."""
        # Tensor wise adjustments
        for quant_info in self._tensor_quant_infos.values():
            if quant_info.quantization_ranges is None:
                raise RuntimeError("Quantization ranges should be set at this point.")
            quant_info.quantization_ranges = fill_zero_entries(quant_info.quantization_ranges)
            if self.config.disable_last_layer_channelwise:
                # This is very important for classification models. In case not all classes are present in the
                # calibration dataset, channelwise quantization will give very bad results
                if quant_info.tensor.tensor_type == TensorType.GRAPH_OUTPUT:
                    quant_info.quantization_ranges = make_quantization_ranges_tensor_wide(
                        quant_info.quantization_ranges
                    )

            quant_info.power_of_two_scaling_only = True  # Always use power of two scaling for now
            quant_info.quantization_ranges = round_up_to_power_of_two(quant_info.quantization_ranges)

            # Calculate the v-NN Mapper MaxExponents at this point
            # The adjusted max exponents are the ones actually used for quantization
            # These are calculated in layer specific adjustements below
            # Keep track of the other one for debugging purposes

            quant_info.max_exponents = power_of_two_values_to_exponents(quant_info.quantization_ranges)
            quant_info.adjusted_max_exponents = np.copy(quant_info.max_exponents)

            # Remove quantization range at this point to avoid confusion
            # quant_info.quantization_ranges = None

        # Layerwise adjustments
        for layer_handler in self._layer_handlers.values():
            layer_handler.tensor_range_postprocess(self._tensor_quant_infos)

    def _evaluate_quantization(self) -> dict[str, Any]:
        """Evaluate the quantization of this model.

        This function evaluates quantization of this model, by performing layerwise quantization error estimatation.
        For, this it iterates over all layers, and for each layer, it quantizes only that layer. All other layers
        are kept in fp32. Then, a full evaluation is performed on the quantized model, and the quantization error
        is calculated.

        Returns:
            dict[str, Any]: A dictionary containing the quantization accuracy for each layer and the full fp32
                accuracy.
        """
        evaluator = LayerwiseEvaluator(self._model, self._tensor_quant_infos)
        return evaluator.run_layerwise_evaluation(
            self._model,
            self._graph_helper,
            self._tensor_quant_infos,
            self._calibration_dataloader,
        )

    def _add_fake_quant_layers(self) -> None:
        qdq_helper = QDQHelper(self._model._model_repr)
        for quant_info in self._tensor_quant_infos.values():
            qdq_helper.add_qdq_node(quant_info)

    def _create_quantization_report(self, evaluation_results: dict[str, Any] | None) -> QuantizationReport:
        """Create quantization report.

        Args:
            evaluation_results (dict[str, Any] | None): A dictionary containing the quantization accuracy for each layer and
                the full fp32 accuracy.

        Returns:
            QuantizationReport: The quantization report.

        Raises:
            RuntimeError:
                - If neither histograms nor data are present in TensorQuantInfo.
                - If quantization_ranges is not set in TensorQuantInfo.
                - If any computed array (percentile_50, percentile_99, max_values, quantization_ranges) is not one-dimensional.
        """
        tensor_statistics = {}
        for tensor_name, quant_info in self._tensor_quant_infos.items():
            tensor_name = quant_info.tensor.name
            histograms = quant_info.tensor_histograms

            if histograms is None and quant_info.tensor.data is not None:
                tensor_data = quant_info.tensor.data.reshape([quant_info.tensor.data.shape[0], -1])
                percentile_50 = np.percentile(tensor_data, 50.0, axis=1)  # type: ignore
                percentile_99 = np.percentile(tensor_data, 99.0, axis=1)  # type: ignore
                max_values = np.max(tensor_data, axis=1)
            elif histograms is not None:
                percentile_50 = histograms.ndim_percentile(50.0)
                percentile_99 = histograms.ndim_percentile(99.0)
                max_values = histograms.ndim_percentile(100.0)
            else:
                raise RuntimeError("Either histograms or data need to be present in TensorQuantInfo")

            # For weights quantization_ranges, may be 2D -> aggregate results
            if quant_info.quantization_ranges is None:
                raise RuntimeError("quantization_ranges should be present in TensorQuantInfo")
            quantization_ranges = quant_info.quantization_ranges.reshape([max_values.shape[0], -1]).max(-1)

            if len(percentile_50.shape) != 1:
                raise RuntimeError(f"Expected percentile_50 to be a 1D array, got shape: {percentile_50.shape}")
            if len(percentile_99.shape) != 1:
                raise RuntimeError(f"Expected percentile_99 to be a 1D array, got shape: {percentile_99.shape}")
            if len(max_values.shape) != 1:
                raise RuntimeError(f"Expected max_values to be a 1D array, got shape: {max_values.shape}")
            if len(quantization_ranges.shape) != 1:
                raise RuntimeError(
                    f"Expected quantization_ranges to be a 1D array, got shape: {quantization_ranges.shape}"
                )

            tensor_statistics[tensor_name] = {
                "percentile_50": percentile_50,
                "percentile_99": percentile_99,
                "max_values": max_values,
                "quantization_ranges": quantization_ranges,
            }

            # Purge histograms
            quant_info.tensor_histograms = None

        report = QuantizationReport(
            self._graph_helper,
            tensor_statistics,
            layer_metrics=evaluation_results,
        )
        return report
