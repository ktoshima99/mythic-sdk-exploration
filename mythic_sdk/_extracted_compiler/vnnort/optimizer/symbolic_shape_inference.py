import math
from typing import Sequence

import numpy as np
import onnx
from onnx import ModelProto, TensorProto

from vnnort.utils.onnx_utils.graph_helper import Node, ONNXGraphHelper, TensorType
from vnnort.utils.onnx_utils.utils import set_output_shape


def _resolve_dynamic_shape(shape: Sequence[int], source_shape: Sequence[int]) -> tuple[int, ...]:
    """Resolve shapes with dynamic entries.

    E.g. [-1, 20]) with source_shape [2, 4, 5] to [2, 20]

    Args:
        shape (Sequence[int]): Shape to resolve
        source_shape (Sequence[int]): Source shape with the same number of elements as the target tensor

    Raises:
        RuntimeError: In case of illegal shape

    Returns:
        tuple[int, ...]: Resolved shape
    """
    # Check that there are at most one -1 and all others are positive
    if sum(entry < 0 for entry in shape) > 1 and any(entry < -1 for entry in shape):
        raise RuntimeError("Only one -1 is allowed in shape resolving.")

    # Check that source tensor is already resolved
    if any(entry < 0 for entry in source_shape):
        raise RuntimeError("Source tensor shape must be fully defined.")

    total_n_elements = abs(int(np.prod(source_shape)))
    current_shape_elements = abs(int(np.prod(shape)))
    shape = list(shape)
    for i in range(len(shape)):
        if shape[i] == -1:
            if total_n_elements % current_shape_elements != 0:
                raise RuntimeError(f"Cannot resolve shape {shape} to tensor with {total_n_elements}")
            shape[i] = total_n_elements // current_shape_elements

    return tuple(int(entry) for entry in shape)


def _resolve_axis(axis: int, rank: int) -> int:
    """
    Normalize an axis index with respect to a tensor rank.

    Converts a possibly negative axis into its corresponding non-negative
    index for a tensor of the given rank. Negative axes are interpreted
    using Python-style indexing (i.e., counting from the end).

    Args:
        axis (int): The axis index to resolve. May be negative.
        rank (int): The number of dimensions of the tensor. Must be positive.

    Returns:
        int: A non-negative axis index in the range [0, rank - 1].

    Raises:
        TypeError: If ``axis`` is not an integer.
        ValueError: If ``rank`` is not positive or if the resolved axis is out of bounds for the given rank.

    Examples:
        >>> _resolve_axis(-1, 4)
        3
        >>> _resolve_axis(-2, 4)
        2
        >>> _resolve_axis(1, 4)
        1
    """
    if not isinstance(axis, int):
        raise TypeError("axis must be an integer")
    if rank <= 0:
        raise ValueError("rank must be positive")

    if axis < 0:
        axis += rank

    if axis < 0 or axis >= rank:
        raise ValueError(f"axis {axis} is out of bounds for tensor of rank {rank}")

    return axis


class _LayerShapeHandler:
    @staticmethod
    def vidConv(node: Node) -> Sequence[int]:
        if not node.op_type == "vidConv":
            raise RuntimeError(f"Called wrong handler for op type {node.op_type}")
        dim = node.attributes["dim"]
        if dim == 4:
            n, _, h_in, w_in = node.inputs[0].shape
            c_out, _, k_h, k_w = node.inputs[1].shape
        elif dim == 2:
            n, _ = node.inputs[0].shape[0], node.inputs[0].shape[1]
            h_in, w_in = 1, 1
            c_out, _ = node.inputs[1].shape
            k_h, k_w = 1, 1
        else:
            raise RuntimeError(f"Cannot handle vidConv with dim attribute {dim}")

        d_h, d_w = node.attributes["dilations"]  # (h, w)
        pad_top, pad_left, pad_bottom, pad_right = node.attributes["pads"]  # (pad_top, pad_left, pad_bottom, pad_right)
        s_h, s_w = node.attributes["strides"]  # (h, w)
        auto_pad = node.attributes["auto_pad"]

        # effective kernel size
        k_h_eff = (k_h - 1) * d_h + 1
        k_w_eff = (k_w - 1) * d_w + 1

        if auto_pad == "NOTSET":
            pad_top, pad_left, pad_bottom, pad_right = node.attributes["pads"]
            h_out = math.floor((h_in + pad_top + pad_bottom - k_h_eff) / s_h + 1)
            w_out = math.floor((w_in + pad_left + pad_right - k_w_eff) / s_w + 1)

        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            # "same" convolution keeps output = ceil(input / stride)
            h_out = math.ceil(h_in / s_h)
            w_out = math.ceil(w_in / s_w)

            # total padding required
            pad_needed_h = max((h_out - 1) * s_h + k_h_eff - h_in, 0)
            pad_needed_w = max((w_out - 1) * s_w + k_w_eff - w_in, 0)

            if auto_pad == "SAME_UPPER":
                pad_top = pad_needed_h // 2
                pad_bottom = pad_needed_h - pad_top
                pad_left = pad_needed_w // 2
                pad_right = pad_needed_w - pad_left
            else:  # SAME_LOWER
                pad_bottom = pad_needed_h // 2
                pad_top = pad_needed_h - pad_bottom
                pad_right = pad_needed_w // 2
                pad_left = pad_needed_w - pad_right

        elif auto_pad == "VALID":
            # no padding
            h_out = math.floor((h_in - k_h_eff) / s_h + 1)
            w_out = math.floor((w_in - k_w_eff) / s_w + 1)

        else:
            raise RuntimeError(f"Unknown auto_pad value {auto_pad}")

        out = [n, c_out, h_out, w_out]
        reshape_mode = node.attributes["reshape_mode"]
        match reshape_mode:
            case "None":
                pass
            case "TRANSFORMER_V":
                out = [out[1], out[3], 1, 1]
            case "TRANSFORMER_QK":
                reshape_mode_groups = node.attributes["reshape_mode_groups"][0]
                out = [out[3] * reshape_mode_groups, out[1] // reshape_mode_groups, 1, 1]
            case "FLATTEN_W":
                out = [out[0], out[1], 1, out[2] * out[3]]

            case _:
                raise RuntimeError(f"Reshape mode {reshape_mode} is not supported")

        if dim == 2:
            out = [out[0], out[1]]
        if out[0] == 0:
            out[0] = -1
        return tuple(out)

    @staticmethod
    def vidMaxPool(node: Node) -> Sequence[int]:
        n, c, h_in, w_in = node.inputs[0].shape
        k_h, k_w = node.attributes["kernel_shape"]
        d_h, d_w = node.attributes["dilations"]
        pad_top, pad_left, pad_bottom, pad_right = node.attributes["pads"]
        s_h, s_w = node.attributes["strides"]

        # Effective kernel size
        k_h_eff = (k_h - 1) * d_h + 1
        k_w_eff = (k_w - 1) * d_w + 1

        # Output spatial dims
        h_out = math.floor((h_in + pad_top + pad_bottom - k_h_eff) / s_h + 1)
        w_out = math.floor((w_in + pad_left + pad_right - k_w_eff) / s_w + 1)

        return (n, c, h_out, w_out)

    @staticmethod
    def Shortcut(node: Node) -> Sequence[int]:
        shape0 = node.inputs[0].shape
        shape1 = node.inputs[1].shape

        rev0 = list(shape0[::-1])
        rev1 = list(shape1[::-1])
        out = []

        for i in range(max(len(rev0), len(rev1))):
            dim0 = rev0[i] if i < len(rev0) else 1
            dim1 = rev1[i] if i < len(rev1) else 1

            if dim0 == dim1:
                out.append(dim0)
            elif dim0 == 1:
                out.append(dim1)
            elif dim1 == 1:
                out.append(dim0)
            elif dim0 == -1 and dim1 > 0:
                out.append(dim1)
            elif dim1 == -1 and dim0 > 0:
                out.append(dim0)
            elif dim0 == -1 and dim1 == -1:
                out.append(-1)
            else:
                raise ValueError(f"Shapes {shape0} and {shape1} are not broadcastable")

        out = out[::-1]
        reshape_mode = node.attributes["reshape_mode"]
        match reshape_mode:
            case "None":
                pass
            case "TRANSFORMER_QK":
                group = node.attributes["group"][0]
                channels_per_group = out[1] // group
                out = [out[3] * group, channels_per_group, 1, 1]
            case _:
                raise RuntimeError(f"Reshape mode {reshape_mode} is not supported")
        return out

    @staticmethod
    def vidAveragePool(node: Node) -> Sequence[int]:
        n, c, h_in, w_in = node.inputs[0].shape
        k_h, k_w = node.attributes["kernel_shape"]
        d_h, d_w = node.attributes["dilations"]
        pad_top, pad_left, pad_bottom, pad_right = node.attributes["pads"]
        s_h, s_w = node.attributes["strides"]

        # Effective kernel size
        k_h_eff = (k_h - 1) * d_h + 1
        k_w_eff = (k_w - 1) * d_w + 1

        # Output spatial dims
        h_out = math.floor((h_in + pad_top + pad_bottom - k_h_eff) / s_h + 1)
        w_out = math.floor((w_in + pad_left + pad_right - k_w_eff) / s_w + 1)

        output_dim = node.attributes["output_dim"]
        if output_dim == 2:
            if h_out != 1 or w_out != 1:
                raise RuntimeError("output_dim = 2 is not valid for spatial dimensions != 1")
            return (n, c)
        return (n, c, h_out, w_out)

    @staticmethod
    def vidFlatten(node: Node) -> Sequence[int]:
        input_shape = node.inputs[0].shape
        axis = node.attributes["axis"]

        shape = input_shape[:axis] + [int(np.prod(input_shape[axis:]))]
        return shape

    @staticmethod
    def Concat(node: Node) -> Sequence[int]:
        output_shape = list(node.inputs[0].shape)
        axis = node.attributes["axis"]
        concat_axis_size = 0
        for input_tensor in node.inputs:
            concat_axis_size += input_tensor.shape[axis]
        output_shape[axis] = concat_axis_size
        return tuple(output_shape)

    @staticmethod
    def Resize(node: Node) -> Sequence[int]:
        # Either scales or sizes may be used to define resize
        scales = node.inputs[2]

        if len(node.inputs) == 3:
            scales = node.inputs[2]
            # Transform scales into sizes
            tensor_shape = node.inputs[0].shape
            if len(tensor_shape) != len(scales.data):
                msg = f"Scales must match tensor shape. Got {scales.data} and {tensor_shape}"
                raise Exception(msg)
            output_shape = [int(scale * dim) for scale, dim in zip(scales.data, tensor_shape)]
        elif len(node.inputs) == 4 and node.inputs[3] is not None:
            output_shape = [int(v) for v in node.inputs[3].data]
        return output_shape

    @staticmethod
    def vidLayerNorm(node: Node) -> Sequence[int]:
        return node.inputs[0].shape

    @staticmethod
    def vidSoftmax(node: Node) -> Sequence[int]:
        return node.inputs[0].shape

    @staticmethod
    def Gather(node: Node) -> Sequence[int]:
        number_indices = node.inputs[1].shape[0]
        output_shape = list(node.inputs[0].shape)
        axis = node.attributes["axis"]
        output_shape[axis] = number_indices

        # The dimensions seems to disappear if number_indices = 1
        # if number_indices == 1:
        #     output_shape = [s for index, s in enumerate(output_shape) if index != axis]
        return tuple(output_shape)

    @staticmethod
    def Squeeze(node: Node) -> Sequence[int]:
        axes = node.inputs[1].data
        input_shape = node.inputs[0].shape
        output_shape = [s for index, s in enumerate(input_shape) if index not in axes]
        return output_shape

    @staticmethod
    def Reshape(node: Node) -> Sequence[int]:
        # This only works for static reshapes
        if node.inputs[1].tensor_type is not TensorType.INITIALIZER:
            raise RuntimeError("Symbolic reshape requires static input for Reshape.")
        input_shape = node.inputs[0].shape
        target_shape = [int(entry) for entry in node.inputs[1].data]

        # Make sure to handle -1 cases
        total_n_elements = abs(int(np.prod(input_shape)))
        current_shape_elements = abs(int(np.prod(target_shape)))
        for i in range(len(target_shape)):
            if target_shape[i] == -1:
                target_shape[i] = total_n_elements // current_shape_elements
        return tuple(target_shape)

    @staticmethod
    def Transpose(node: Node) -> Sequence[int]:
        permutation = node.attributes["perm"]
        current_shape = node.inputs[0].shape
        target_shape = [current_shape[index] for index in permutation]
        return tuple(target_shape)

    @staticmethod
    def Slice(node: Node) -> Sequence[int]:
        input_shape = node.inputs[0].shape
        starts = node.inputs[1].data
        ends = node.inputs[2].data
        axes = node.inputs[3].data

        output_shape = list(input_shape)
        for axis_index, axis in enumerate(axes):
            start_index = int(starts[axis_index])
            end_index = int(ends[axis_index])

            if start_index < 0:
                start_index += input_shape[axis]
            if end_index < 0:
                end_index += input_shape[axis]
            start_index = max(0, min(start_index, input_shape[axis]))
            end_index = max(0, min(end_index, input_shape[axis]))
            output_shape[axis] = max(0, end_index - start_index)

        return tuple(output_shape)

    @staticmethod
    def Constant(node: Node) -> Sequence[int]:
        data = None
        for name, attr in node.attributes.items():
            if attr is not None:
                data = onnx.numpy_helper.to_array(attr)
                break
        return [int(s) for s in data.shape]

    @staticmethod
    def Shape(node: Node) -> Sequence[int]:
        return (len(node.inputs[0].shape),)

    @staticmethod
    def vidRope(node: Node) -> Sequence[int]:
        return node.inputs[0].shape

    @staticmethod
    def RMSNormalization(node: Node) -> Sequence[int]:
        return node.inputs[0].shape

    @staticmethod
    def Softmax(node: Node) -> Sequence[int]:
        return node.inputs[0].shape

    @staticmethod
    def Expand(node: Node) -> Sequence[int]:
        target_shape = node.inputs[1].data
        return tuple(int(v) for v in target_shape)

    @staticmethod
    def RTRTransformation(node: Node) -> Sequence[int]:
        # Number of elements does not change from input tensor
        input_shape = node.inputs[0].shape

        # Output shape comes from last reshape shape
        reshape2_shape = [int(s) for s in node.inputs[-2].data]
        result_shape = _resolve_dynamic_shape(reshape2_shape, input_shape)

        return result_shape

    @staticmethod
    def RETRTransformation(node: Node) -> Sequence[int]:
        # Total number of elements is defined by expand shape
        expand_shape = [int(s) for s in node.inputs[2].data]

        # Output shape comes from last reshape shape
        reshape2_shape = [int(s) for s in node.inputs[-2].data]
        result_shape = _resolve_dynamic_shape(reshape2_shape, expand_shape)

        return result_shape

    @staticmethod
    def RERTransformation(node: Node) -> Sequence[int]:
        # Total number of elements is defined by expand shape
        expand_shape = [int(s) for s in node.inputs[2].data]

        # Output shape comes from last reshape shape
        reshape2_shape = [int(s) for s in node.inputs[-1].data]
        result_shape = _resolve_dynamic_shape(reshape2_shape, expand_shape)

        return result_shape

    @staticmethod
    def Gelu(node: Node) -> Sequence[int]:
        # Ouput shape = Input shape
        input_shape = node.inputs[0].shape
        return input_shape

    @staticmethod
    def Relu(node: Node) -> Sequence[int]:
        # Ouput shape = Input shape
        input_shape = node.inputs[0].shape
        return input_shape

    @staticmethod
    def Swish(node: Node) -> Sequence[int]:
        # Ouput shape = Input shape
        input_shape = node.inputs[0].shape
        return input_shape

    @staticmethod
    def Sigmoid(node: Node) -> Sequence[int]:
        # Ouput shape = Input shape
        input_shape = node.inputs[0].shape
        return input_shape

    @staticmethod
    def vidScatter(node: Node) -> Sequence[int]:
        # Output shape is the same as input shape
        return node.inputs[0].shape

    @staticmethod
    def vidGridSample(node: Node) -> Sequence[int]:
        x_shape = node.inputs[0].shape  # [N, C, H_in, W_in]
        grid_shape = node.inputs[1].shape  # [N, 2, H_out, W_out]
        return (x_shape[0], x_shape[1], grid_shape[2], grid_shape[3])


def symbolic_shape_inference(model: ModelProto) -> ModelProto:
    """Perform symbolic shape inference for all tensors in model.

    Args:
        model (ModelProto): Model for which shape inference is to be implemented

    Raises:
        RuntimeError: In case shape inference cannot be performed for any reason

    Returns:
        ModelProto: Modified (inplace) model with tensor shape infos
    """
    graph = ONNXGraphHelper(model)

    # Check that all input tensors have shape information
    for tensor in graph.get_input_tensors():
        if tensor.shape is None:
            raise RuntimeError("Model input tensor shapes must be set.")
        if any(dim == 0 or dim is None for dim in tensor.shape):
            raise RuntimeError("Model input tensor shapes must be fully defined and non-zero.")

    # Clear existing shape info
    model.graph.ClearField("value_info")

    # Go over all nodes and infer their output shapes
    for node in graph.nodes.values():
        op_type = node.op_type
        if not hasattr(_LayerShapeHandler, op_type):
            raise RuntimeError(f"No layer handler defined for layer {op_type}")
        layer_handler = getattr(_LayerShapeHandler, op_type)
        output_shape = layer_handler(node)
        if 0 in output_shape or sum(s < 0 for s in output_shape) > 1:
            raise RuntimeError(f"Invalid shape returned: {output_shape}")

        output_tensor = node.outputs[0]
        output_tensor.shape = output_shape

        # Add shape information to graph
        # Different handling if this is a graph output
        if output_tensor.tensor_type is TensorType.GRAPH_OUTPUT:
            set_output_shape(model, output_tensor.name, output_shape)
        else:
            value_info = onnx.helper.make_tensor_value_info(
                name=output_tensor.name, elem_type=TensorProto.FLOAT, shape=output_shape
            )
            model.graph.value_info.append(value_info)

    return model


# if __name__ == "__main__":
# model_path = "model_flow_results/Qwen2_5_VL_3B_Instruct/qwen.vidi.onnx"
# onnx_model = onnx.load(model_path)
# run_shape_inference(onnx_model)

# model_directory = Path("/data/tmp/nommensen/model_comparisons/new")
# for index, model_name in enumerate(os.listdir(model_directory)):
#     try:
#         if index < 25:
#             print("Skipping ", index, ": ", model_name)
#             continue

#         print(index, ": ", model_name)
#         model_path = model_directory / (model_name + "/" + model_name + ".vido.onnx")
#         model = VidModel.from_file(model_path)

#         onnx_model = model._model_repr
#         onnx_model.graph.ClearField("value_info")
#         symbolic_shape_inference(onnx_model)
#         new_graph = ONNXGraphHelper(onnx_model)
#         for node in new_graph.nodes.values():
#             if node.op_type == "vidConv":
#                 if node.attributes["activation"] == "hardsigmoid":
#                     print(node.attributes)
#     except:
#         print("Error for ", model_name)

# ds = model.load_default_dataset()
# entry = ds[0]
# model_input = model.preprocess(entry)
# model_input = [v for v in model_input.values()]

# onnx_model.graph.ClearField("value_info")
# infer_shapes_runtime(onnx_model, model_input)
# gt_graph = ONNXGraphHelper(onnx_model)

# for tensor in gt_graph.tensors.values():
#     # Only check main outputs
#     if tensor.producer is not None and tensor.producer.outputs.index(tensor) != 0:
#         continue
#     if tensor.tensor_type is TensorType.INITIALIZER:
#         continue
#     gt_shape = tensor.shape
#     new_shape = new_graph.tensors[tensor.name].shape
#     if new_shape[0] == -1:
#         new_shape[0] = 1
#     if gt_shape[0] == -1:
#         gt_shape[0] = 1
#     assert gt_shape == new_shape
