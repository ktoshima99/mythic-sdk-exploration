from typing import Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.quant_utils import TENSOR_NAME_QUANT_SUFFIX

from vnnort.optimizer.pattern_detection import find_node_by_name
from vnnort.quantizer.qdq_layer import QDQLayer  # type: ignore
from vnnort.quantizer.quant_utils import TensorQuantInfo
from vnnort.utils.onnx_utils import VIDEANTIS_ONNX_DOMAIN
from vnnort.utils.onnx_utils.graph_helper import Tensor, TensorType

MAX_EXPONENT_INITIALIZER_SUFFIX = "_QUANT_TENSOR_RANGE"
ZERO_POINT_INITIALIZER_SUFFIX = "_QUANT_ZERO_POINT"
TENSOR_SKIP_SUFFIX = "_SKIP"


class QDQHelper:
    """Helper class for QDQ quantization."""

    def __init__(self, model: onnx.ModelProto):
        """Initialize the QDQHelper class.

        Args:
            model (onnx.ModelProto): model to be quantized
        """
        # model.opset_import[0].version = 21
        self._onnx_model = model

        # Add qdq function definition
        self._qdq_layer_proto = self._make_qdq_functionproto()
        self._onnx_model.functions.extend([self._qdq_layer_proto])

        # Keep track of tensors with qdq layers
        self.tensors_with_qdq_layers: list[str] = []

    def add_qdq_node(self, quant_info: TensorQuantInfo) -> None:
        """Add QDQ node to the model, according to the tensor quantization info provided.

        Args:
            quant_info (TensorQuantInfo): Tensor quantization info

        Returns:
            None: QDQ nodes are added to model inplace.

        Raises:
            ValueError: If quant axis is None or not a list
            RuntimeError: If tensor ranges are not set.
        """
        tensor = quant_info.tensor
        max_exponents = quant_info.adjusted_max_exponents
        n_bits = quant_info.n_bits
        n_fraction_bits = quant_info.n_fraction_bits
        quant_axis: int | tuple[int, ...] = quant_info.axis
        if quant_axis is None:
            raise ValueError(f"Quant axis is None for tensor {tensor.name}")
        elif isinstance(quant_axis, int):
            quant_axis = (quant_axis,)
        elif not isinstance(quant_axis, (list, tuple)):
            raise ValueError(f"Quant axis is of type {type(quant_axis)} for tensor {tensor.name}")
        if max_exponents is None:
            raise RuntimeError("Tensor ranges should be set at this point.")
        if n_fraction_bits is None:
            raise RuntimeError("Number of fraction bits should be set at this point.")

        max_exponent_initializer = self._create_max_exponent_initializer(tensor, max_exponents, quant_axis)
        qdq_layer = self._make_qdq_node(tensor, max_exponent_initializer, n_bits, n_fraction_bits)
        self._insert_qdq_layer_for_tensor(tensor, qdq_layer)

        self.tensors_with_qdq_layers.append(tensor.name)

    def _create_max_exponent_initializer(
        self, tensor: Tensor, max_exponents: NDArray[Any], quant_axis: tuple[int, ...]
    ) -> TensorProto:
        """Create onnx initializer for tensor ranges and add it to graph.

        Args:
            tensor (Tensor): Tensor to quantize
            max_exponents (NDArray[Any]): Tensor ranges
            quant_axis (tuple[int, ...]): Quantization axis

        Raises:
            ValueError: If tensor shape is None

        Returns:
            TensorProto: initialized TensorProto
        """
        tensor_name = tensor.name
        tensor_shape = tensor.shape
        if tensor_shape is None:
            raise ValueError(f"Tensor shape is None for tensor {tensor_name}. Please run shape inference")

        # The tensor ranges initializer needs have same rank as tensor [1, 1, ..., dim1, ..., dimn ..., 1]
        tensor_rank = len(tensor_shape)
        initializer_shape = np.ones(tensor_rank, dtype=np.int64)
        for index, axis in enumerate(list(quant_axis)):
            initializer_shape[axis] = max_exponents.shape[index]
        # Create initializer
        reshaped_max_exponents = np.reshape(max_exponents, initializer_shape)
        name = tensor_name + MAX_EXPONENT_INITIALIZER_SUFFIX
        type = onnx_proto.TensorProto.INT8
        initializer = onnx.helper.make_tensor(
            name, type, initializer_shape.tolist(), reshaped_max_exponents.tobytes(), raw=True
        )

        # Add to graph
        self._onnx_model.graph.initializer.append(initializer)

        return initializer

    def _make_qdq_functionproto(self) -> onnx.FunctionProto:
        """Create QDQ function definition from onnxscript."""
        layer: onnx.FunctionProto = QDQLayer.to_function_proto()
        return layer

    def _make_qdq_node(
        self, tensor: Tensor, initializer: TensorProto, n_bits: int, n_fraction_bits: int
    ) -> onnx.NodeProto:
        """Create QDQ node from onnxscript.

        Args:
            tensor (Tensor): Tensor to quantize
            initializer (TensorProto): Tensor ranges
            n_bits (int): Number of bits
            n_fraction_bits (int): Number of fraction bits

        Returns:
            onnx.NodeProto: QDQ node
        """
        tensor_name = tensor.name
        qdq_op_name = self._qdq_layer_proto.name
        quant_op_name = tensor_name + "_" + qdq_op_name
        qdq_output_name = tensor_name + TENSOR_NAME_QUANT_SUFFIX

        # Create default tensor for skip argument (this can be overwritten by input)
        skip_name = tensor_name + TENSOR_SKIP_SUFFIX
        skip_type = onnx_proto.TensorProto.BOOL
        skip_initializer = onnx.helper.make_tensor(skip_name, skip_type, [], np.array([False], dtype=bool))
        self._onnx_model.graph.initializer.append(skip_initializer)
        n_fraction_initializer = onnx.helper.make_tensor(
            tensor_name + "_FRACTION_BITS", onnx.TensorProto.INT64, [], np.array([n_fraction_bits], dtype=np.int64)
        )
        self._onnx_model.graph.initializer.append(n_fraction_initializer)
        n_bits_initializer = onnx.helper.make_tensor(
            tensor_name + "_N_BITS", onnx.TensorProto.INT64, [], np.array([n_bits], dtype=np.int64)
        )
        self._onnx_model.graph.initializer.append(n_bits_initializer)

        # Add optional input for skip tensor
        skip_input = onnx.helper.make_tensor_value_info(skip_name, onnx.TensorProto.BOOL, [])  # Optional input
        self._onnx_model.graph.input.append(skip_input)

        # Output names cannot contain colons (auto generated by torch export..)
        qdq_layer = onnx.helper.make_node(
            self._qdq_layer_proto.name,
            # [tensor_name, initializer.name],
            [tensor_name, initializer.name, skip_name, n_bits_initializer.name, n_fraction_initializer.name],
            [qdq_output_name],
            quant_op_name,
            domain=VIDEANTIS_ONNX_DOMAIN,
        )
        return qdq_layer

    def _insert_qdq_layer_for_tensor(self, tensor: Tensor, qdq_layer: onnx.NodeProto) -> None:
        """Insert QDQ layer for tensor.

        Args:
            tensor (Tensor): Tensor to quantize
            qdq_layer (onnx.NodeProto): QDQ layer

        Raises:
            RuntimeError: If tensor has no producer

        Returns:
            None: inserts QDQ layers inplace
        """
        tensor_name = tensor.name
        qdq_output_name = tensor_name + TENSOR_NAME_QUANT_SUFFIX

        # Insert its after its producer or first consumer
        if tensor.tensor_type == TensorType.INITIALIZER or tensor.tensor_type == TensorType.GRAPH_INPUT:
            index, _ = find_node_by_name(self._onnx_model.graph, tensor.consumers[0].name)
        else:
            if tensor.producer is None:
                raise RuntimeError(f"Tensor {tensor_name} should have a producer.")
            index, _ = find_node_by_name(self._onnx_model.graph, tensor.producer.name)
            index = index + 1
        self._onnx_model.graph.node.insert(index, qdq_layer)

        # Replace the input of consumer nodes with the newly fake quantized outputs
        for node in tensor.consumers:
            node_name = node.name
            _, onnx_node = find_node_by_name(self._onnx_model.graph, node_name)
            for i, input_name in enumerate(onnx_node.input):
                if input_name == tensor_name:
                    onnx_node.input[i] = qdq_output_name
                    break

        # If this is a graph output tensor, replace model output
        if tensor.tensor_type == TensorType.GRAPH_OUTPUT:
            for i, output in enumerate(self._onnx_model.graph.output):
                if output.name == tensor_name:
                    output.name = qdq_output_name
                    break
