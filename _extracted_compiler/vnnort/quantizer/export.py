from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from vnnort.models.vid_model import ModelState, VidModel
from vnnort.quantizer.qdq_helper import MAX_EXPONENT_INITIALIZER_SUFFIX
from vnnort.quantizer.quant_utils import power_of_two_values_to_exponents, quantize_values
from vnnort.utils.onnx_utils.graph_helper import Node, ONNXGraphHelper, Tensor

VIDCONV_OPTYPE = "vidConv"


def _is_weight_tensor(qdq_node: Node) -> bool:
    """Check if a QDQLayer node corresponds to a weight tensor.

    Args:
        qdq_node (Node): The QDQLayer node to check.

    Raises:
        ValueError: If the node is not of type QDQLayer.
    Returns:
        bool: True if the QDQLayer node corresponds to a weight tensor, False otherwise.
    """
    if not qdq_node.op_type == "QDQLayer":
        raise ValueError(f"Expected Optype QDQLayer, got {qdq_node.op_type}")

    # This is a qdq node for a weight tensor if it has one consumer of optype vidConv and is its second input
    if (
        len(qdq_node.outputs) == 1
        and len(qdq_node.outputs[0].consumers) == 1
        and qdq_node.outputs[0].consumers[0].op_type == VIDCONV_OPTYPE
        and len(qdq_node.outputs[0].consumers[0].inputs) >= 2
        and qdq_node.outputs[0].consumers[0].inputs[1] == qdq_node.outputs[0]
    ):
        return True
    return False


def _is_bias_tensor(qdq_node: Node) -> bool:
    """Check if a QDQLayer node corresponds to a bias tensor.

    Args:
        qdq_node (Node): The QDQLayer node to check.

    Raises:
        ValueError: If the node is not of type QDQLayer.

    Returns:
        bool: True if the QDQLayer node corresponds to a bias tensor, False otherwise.
    """
    if not qdq_node.op_type == "QDQLayer":
        raise ValueError(f"Expected Optype QDQLayer, got {qdq_node.op_type}")

    # This is a qdq node for a bias tensor if it has one consumer of optype vidConv and is its third input
    if (
        len(qdq_node.outputs) == 1
        and len(qdq_node.outputs[0].consumers) == 1
        and qdq_node.outputs[0].consumers[0].op_type == VIDCONV_OPTYPE
        and len(qdq_node.outputs[0].consumers[0].inputs) == 3
        and qdq_node.outputs[0].consumers[0].inputs[2] == qdq_node.outputs[0]
    ):
        return True
    return False


def _export_weights(
    weights: NDArray[np.float32], tensor_ranges: NDArray[np.float32], n_bits: int
) -> tuple[NDArray[np.int16 | np.int8], NDArray[np.int8]]:
    """Export weights to mantissa + max_exponents representation.

    Args:
        weights (NDArray[np.float32]): The weights to export. Shape: [Cout, Cin, H, W] or [Cout, Cin]
        tensor_ranges (NDArray[np.float32]): The tensor ranges to use. Shape: [Cout, Cin]
        n_bits (int): The number of bits to use.

    Returns:
        tuple[NDArray[np.int16 | np.int8], NDArray[np.int8]]: The mantissa and max_exponents.
    """
    # tensor_ranges has shape [Cout, Cin, 1, 1] or [Cout, Cin]
    # The goal is to extract a max_exponents vector of size [Cout]
    # tensor_ranges contains seperate ranges for input channels as well to simulate the mantissa shift trick
    # used by our actual implementation. We compensate this now by taking the maximum over all input channels
    # while we lose some precision here, this will happen anyway on the vnnmap side, when the mantissas are shifted
    Cout, Cin = tensor_ranges.shape[:2]
    tensor_ranges = tensor_ranges.reshape((Cout, Cin))
    tensor_ranges = tensor_ranges.max(axis=1)

    max_exponents = power_of_two_values_to_exponents(tensor_ranges)
    mantissas = quantize_values(weights, max_exponents.astype(np.int32), n_bits)

    return mantissas, max_exponents


def _export_biases(
    biases: NDArray[np.float32], tensor_ranges: NDArray[np.float32], n_bits: int
) -> tuple[NDArray[np.int8 | np.int16], NDArray[np.int8]]:
    """Export biases to mantissa + max_exponents representation.

    Args:
        biases (NDArray[np.float32]): The biases to export. Shape: [Cout]
        tensor_ranges (NDArray[np.float32]): The tensor ranges to use. Shape: [Cout]
        n_bits (int): The number of bits to use.

    Returns:
        tuple[NDArray[np.int8 | np.int16], NDArray[np.int8]]: The mantissa and max_exponents.
    """
    max_exponents = power_of_two_values_to_exponents(tensor_ranges)
    mantissas = quantize_values(biases, max_exponents.astype(np.int32), n_bits)

    return mantissas, max_exponents


def _collect_qdq_nodes(model: VidModel) -> list[Node]:
    """Collect all QDQLayers in a VidModel.

    Args:
        model (VidModel): The model to collect QDQLayers from.

    Returns:
        list[Node]: A list of all QDQLayers in the model.
    """
    graph_helper = ONNXGraphHelper(model._model_repr)
    qdq_nodes = []
    for node in graph_helper.nodes.values():
        if node.op_type == "QDQLayer":
            qdq_nodes.append(node)
    return qdq_nodes


def export_quantization_parameters(  # noqa: C901
    model: VidModel | Path | str,
) -> tuple[Dict[str, NDArray[np.int8 | np.int16]], Dict[str, NDArray[np.int8]]]:
    """Export quantization parameters from a VidModel.

    Args:
        model (VidModel | Path | str): The model to export quantization parameters from.

    Returns:
        tuple[Dict[str, NDArray[np.int8 | np.int16]], Dict[str, NDArray[np.int8]]]: The mantissa and max_exponents.

    Raises:
        ValueError: If the model is not in quantized state or if the model is not a VidModel.
        RuntimeError: If the tensor ranges are not set.
        TypeError: If the input tensor data is None.
    """
    if isinstance(model, (Path, str)):
        model = VidModel.from_file(model)
    elif not isinstance(model, VidModel):
        raise ValueError("Model needs to be VidModel object or path to a saved VidModel.")
    if not model.state == ModelState.QUANTIZED:
        raise ValueError("Model needs to be in state quantized.")

    qdq_nodes = _collect_qdq_nodes(model)
    all_mantissas = {}
    all_max_exponents = {}

    for node in qdq_nodes:
        # QDQLayers have two inputs: [input_tensor, tensor_range] and one attribute: n_bits
        input_tensor: Tensor = node.inputs[0]
        tensor_range: Tensor = node.inputs[1]
        n_bits = node.attributes["n_bits"]

        # Depending on what tensor this is, export max_exponents and/or mantissas
        mantissas = None
        if tensor_range.data is None:
            raise RuntimeError("Tensor ranges should be set at this point.")
        if _is_weight_tensor(node):
            if input_tensor.data is None:
                raise TypeError("Input tensor data cannot be None here.")
            mantissas, max_exponents = _export_weights(
                input_tensor.data.astype(np.float32), tensor_range.data.astype(np.float32), n_bits
            )
        elif _is_bias_tensor(node):
            if input_tensor.data is None:
                raise TypeError("Input tensor data cannot be None here.")
            mantissas, max_exponents = _export_biases(
                input_tensor.data.astype(np.float32), tensor_range.data.astype(np.float32), n_bits
            )
        else:
            max_exponents = power_of_two_values_to_exponents(tensor_range.data.astype(np.float32))

        # Map back to the originl tensor name
        tensor_name = tensor_range.name.replace(MAX_EXPONENT_INITIALIZER_SUFFIX, "")
        if mantissas is not None:
            all_mantissas[tensor_name] = mantissas
        all_max_exponents[tensor_name] = max_exponents

    return all_mantissas, all_max_exponents
