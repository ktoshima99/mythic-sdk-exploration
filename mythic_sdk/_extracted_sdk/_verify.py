import numpy as np

from munc import _node_utils, _session, _session_tools
from munc._constants import HardwareType, ONNXType, SUPPORTED_ON_CHIP_ACTIVATIONS


MM_ATTRIBUTES = ["__pFSR", "__iFSR", "__multiplier", "__shift", "__activation"]
ADDSUM_ATTRIBUTES = ["__multiplier", "__shift", "__activation"]
MMA_ONNX_TYPES = [ONNXType.CONV, ONNXType.GEMM, ONNXType.BCMCONV2D, ONNXType.BCMLINEAR]
ADD_SUM_ONNX_TYPES = [ONNXType.ADD, ONNXType.SUM, ONNXType.BCMADD, ONNXType.BCMSUM]
CONV_TYPES = [ONNXType.CONV, ONNXType.BCMCONV2D]

INT_TOLERANCE = 0.001
MAX_WEIGHT_BIAS = 128


def verify_compiler_model(model, hardware):
    """Verify that the compiler model is valid.

    Parameters
    ----------
    model : munc._onnx_model.ONNXModel
        MUNC's ONNX model wrapper.
    hardware : dict
        Hardware configuration dictionary.
    """
    # Check if not compiler model
    model_type = _session.get_model_type(model)
    if model_type != _session.MODELType.BCM:
        raise Exception("A non-compiler model is provided to the verification function.")

    # Check inputs
    if hardware.name == HardwareType.BOREAS:
        external_inputs = model.get_external_input_names()
        n_external_inputs = len(external_inputs)
        if not (1 <= n_external_inputs <= 3):
            raise ValueError(f"Number of model external inputs should be between 1 and 3 for {HardwareType.BOREAS}."
                             f" Current number of inputs is {n_external_inputs}.")

    # Check outputs
    n_outputs = len(model.get_output_names())
    if n_outputs < 1:
        # TODO: determine maximum number of outputs supported by compiler.
        raise Exception("Number of model outputs should be greater than 1"
                        f" Current number of outputs is {n_outputs}.")

    # Check nodes
    for node in model.get_nodes():
        if _node_utils.is_off_chip(node):
            continue
        if len(node.output) == 0:
            raise ValueError(
                f"Node with name {node.name} and inputs {node.input} has no output. All nodes must have an output.")
        if node.name == "":
            raise Exception(f"Node with output {node.output[0]} has no name. A name must be given.")
        if not _session_tools.is_op_type_supported_on_chip(node.op_type, model.hwconfig.name):
            raise Exception(f"Node {node.name} has Op_type {node.op_type} which is not supported on chip.")

        # CONV and GEMM checks
        if node.op_type in MMA_ONNX_TYPES:
            for attribute_name in MM_ATTRIBUTES:
                if not _node_utils.is_attribute(node, attribute_name):
                    raise Exception(f"Can't find attribute {attribute_name} in Conv/GEMM node {node.name}.")

            # If a conv has group!=1, it will be processed in the SALU and doesn't require
            # pFSR, iFSR attributes
            is_salu_node = False
            if node.op_type in CONV_TYPES:
                group = _node_utils.get_attribute_value(node, "group", 1)
                if group != 1:
                    is_salu_node = True
            if not is_salu_node:
                pFSR = _node_utils.get_attribute_value(node, "__pFSR")
                if pFSR not in hardware.pFSR_values:
                    raise Exception(f"Node {node.name} pFSR {pFSR} is invalid. Valid pFSRs: {hardware.pFSR_values}")

                iFSR = _node_utils.get_attribute_value(node, "__iFSR")
                if iFSR not in hardware.iFSR_values:
                    raise Exception(f"Node {node.name} iFSR {iFSR} is invalid. Valid iFSRs: {hardware.iFSR_values}")

            if len(node.input) < 2:
                raise Exception(f"Node {node.name} has {len(node.input)} inputs. It needs to be >= 2.")
            edge_weight = node.input[1]
            weights = model.get_initializer_np(edge_weight)
            weights_max = np.max(np.abs(weights))
            if weights_max > MAX_WEIGHT_BIAS:
                raise Exception(f"Node {node.name} has weight > {MAX_WEIGHT_BIAS}.")

            if len(node.input) >= 3:
                edge_bias = node.input[2]
                biases = model.get_initializer_np(edge_bias)
                biases_max = np.max(np.abs(biases))
                if biases_max > MAX_WEIGHT_BIAS:
                    raise Exception(f"Node {node.name} has bias > {MAX_WEIGHT_BIAS}.")

        # ADD and SUM checks
        if node.op_type in ADD_SUM_ONNX_TYPES:
            for attribute_name in ADDSUM_ATTRIBUTES:
                if not _node_utils.is_attribute(node, attribute_name):
                    raise Exception(f"Can't find attribute {attribute_name} in Add/Sum node {node.name}.")

        # CONV, GEMM, ADD and SUM common checks
        if node.op_type in MMA_ONNX_TYPES + ADD_SUM_ONNX_TYPES:
            mul = _node_utils.get_attribute_value(node, "__multiplier")
            if mul < 0 or mul > 255:
                raise Exception(f"Node {node.name} multiplier is out of range with value {mul}")
            if np.abs(mul % 1) > INT_TOLERANCE:
                raise Exception(f"Node {node.name} multiplier is expected to be an integer but has value {mul}")

            shift = _node_utils.get_attribute_value(node, "__shift")
            min_shift = 0
            max_shift = hardware.ds_max_shift
            if shift < min_shift or shift > max_shift:
                raise Exception(f"Node {node.name} shift is out of range with value {shift}")
            if np.abs(shift % 1) > INT_TOLERANCE:
                raise Exception(f"Node {node.name} shift is expected to be an integer but has value {shift}")

            activation = _node_utils.get_attribute_value(node, "__activation")
            if activation not in SUPPORTED_ON_CHIP_ACTIVATIONS:
                raise Exception(f"Node {node.name} activation {activation} is not supported. "
                                f" Only {SUPPORTED_ON_CHIP_ACTIVATIONS} are supported.")
