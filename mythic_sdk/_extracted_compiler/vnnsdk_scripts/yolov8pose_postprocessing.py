import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx.numpy_helper import from_array, to_array
from onnx_ir import Value
from onnxscript.rewriter.pattern import RewriteRuleClassBase

from vnnort import configure_logging
from vnnort.data.container import ImageDetectionInput, ImageDetectionOutput
from vnnort.models.initialization_config import InitializationConfig
from vnnort.models.vid_model import VidModel
from vnnort.optimizer.pattern_detection import vid_match_patterns_onnxscript
from vnnort.optimizer.utils import (
    get_wgt_by_name,
    infer_shapes_runtime,
    move_constants_to_wgts,
    move_static_cons_to_wgts,
    remove_unused_nodes,
    replace_wgt,
)
from vnnort.utils.onnx_utils.graph_helper import ONNXGraphHelper, Tensor, TensorType

# Python sets __package__=None when run as __main__, making relative imports fail.
# This ensures mythic_utils is on sys.path so it can be imported by name.
sys.path.insert(0, str(Path(__file__).parent))
from mythic_utils import DummyDataset  # noqa: E402

# Path to the source ONNX file, set by the CLI before instantiating the VidModel.
# This is very hacky but we do not have a better way right now to pass the CLI argument to the VidModel.
ONNX_PATH: str | Path = ""


def const_to_np(v: Value) -> NDArray[Any]:
    """Extract the concrete numpy array from a nested Constant IR value."""
    arr = to_array(v._producer.attributes["value"]._value.raw)
    return arr


class PoseEstimationPart2RewritePattern(RewriteRuleClassBase):
    """Rewrite the second part of pose estimation processing part to a format we can handle.

    Here the main requirement is to have the Sigmoid layer right after a convolution so that it
    can be merged and does not have to be implemented as a standalone layer. The other challenge is to avoid
    moving parts of the C dimension into the spatial dimension in the righthandside part of the graph.

    This requires splitting up the second convolution into three seperate convolutions and then processing
    individual paths from there.
    """

    level = 2

    @classmethod
    def pattern(
        cls,
        op: Any,
        x1: Any,
        x2: Any,
        x3: Any,
        w1: Any,
        b1: Any,
        w2: Any,
        b2: Any,
        w3: Any,
        b3: Any,
        a1: Any,
        a2: Any,
        a3: Any,
    ) -> Any:
        """Pattern to be matched."""
        # Bottom three convolutions of prev. part
        x1 = op.vidConv(x1, w1, b1, _domain="com.videantis")
        x1 = op.Reshape(x1, _allow_other_inputs=True)
        x2 = op.vidConv(x2, w2, b2, _domain="com.videantis")
        x2 = op.Reshape(x2, _allow_other_inputs=True)
        x3 = op.vidConv(x3, w3, b3, _domain="com.videantis")
        x3 = op.Reshape(x3, _allow_other_inputs=True)
        x = op.Concat(x1, x2, x3, axis=-1)
        x = op.Reshape(x, _allow_other_inputs=True)

        # Righthandside rescaling graph
        x1 = op.Slice(x, _allow_other_inputs=True)
        x1 = op.Mul(x1, a1)
        x1 = op.Shortcut(x1, a2, _domain="com.videantis")
        x1 = op.Mul(x1, a3)

        # # Lefthandside sigmoid graph
        x2 = op.Slice(x, _allow_other_inputs=True)
        x2 = op.Sigmoid(x2)

        x = op.Concat(x1, x2, axis=2)
        x = op.Reshape(x, _allow_other_inputs=True)

        return x

    @classmethod
    def rewrite(
        cls,
        op: Any,
        x1: Any,
        x2: Any,
        x3: Any,
        w1: Any,
        b1: Any,
        w2: Any,
        b2: Any,
        w3: Any,
        b3: Any,
        a1: Any,
        a2: Any,
        a3: Any,
    ) -> Any:
        """Rewrite graph."""
        # Split each conv into three seperate convs
        # x1
        w11 = op.Constant(value=from_array(const_to_np(w1)[:24]))
        w12 = op.Constant(value=from_array(const_to_np(w1)[24:48]))
        w13 = op.Constant(value=from_array(const_to_np(w1)[48:]))
        b11 = op.Constant(value=from_array(const_to_np(b1)[:24]))
        b12 = op.Constant(value=from_array(const_to_np(b1)[24:48]))
        b13 = op.Constant(value=from_array(const_to_np(b1)[48:]))

        x11 = op.vidConv(x1, w11, b11, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)
        x12 = op.vidConv(x1, w12, b12, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)
        x13 = op.vidConv(x1, w13, b13, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)

        # x2
        w21 = op.Constant(value=from_array(const_to_np(w2)[:24]))
        w22 = op.Constant(value=from_array(const_to_np(w2)[24:48]))
        w23 = op.Constant(value=from_array(const_to_np(w2)[48:]))
        b21 = op.Constant(value=from_array(const_to_np(b2)[:24]))
        b22 = op.Constant(value=from_array(const_to_np(b2)[24:48]))
        b23 = op.Constant(value=from_array(const_to_np(b2)[48:]))

        x21 = op.vidConv(x2, w21, b21, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)
        x22 = op.vidConv(x2, w22, b22, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)
        x23 = op.vidConv(x2, w23, b23, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)

        # x3
        w31 = op.Constant(value=from_array(const_to_np(w3)[:24]))
        w32 = op.Constant(value=from_array(const_to_np(w3)[24:48]))
        w33 = op.Constant(value=from_array(const_to_np(w3)[48:]))
        b31 = op.Constant(value=from_array(const_to_np(b3)[:24]))
        b32 = op.Constant(value=from_array(const_to_np(b3)[24:48]))
        b33 = op.Constant(value=from_array(const_to_np(b3)[48:]))

        x31 = op.vidConv(x3, w31, b31, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)
        x32 = op.vidConv(x3, w32, b32, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)
        x33 = op.vidConv(x3, w33, b33, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)

        # Concatenate on spatial dimensions to generate inputs for path1 and path2 (flatten first)
        x11 = op.Reshape(x11, op.Constant(value_ints=[1, 24, 1, -1]))
        x21 = op.Reshape(x21, op.Constant(value_ints=[1, 24, 1, -1]))
        x31 = op.Reshape(x31, op.Constant(value_ints=[1, 24, 1, -1]))
        x12 = op.Reshape(x12, op.Constant(value_ints=[1, 24, 1, -1]))
        x22 = op.Reshape(x22, op.Constant(value_ints=[1, 24, 1, -1]))
        x32 = op.Reshape(x32, op.Constant(value_ints=[1, 24, 1, -1]))

        x1 = op.Concat(x11, x21, x31, axis=-1)
        x2 = op.Concat(x12, x22, x32, axis=-1)

        # a2 needs to be split between path1 and path2
        a21 = op.Constant(value=from_array(a2.const_value.numpy()[:, 0]))
        a22 = op.Constant(value=from_array(a2.const_value.numpy()[:, 1]))

        # Path1
        x1 = op.Shortcut(x1, a1, mode="multiplication", _domain="com.videantis", _version=1)
        x1 = op.Shortcut(x1, a21, _domain="com.videantis", _version=1)
        x1 = op.Shortcut(x1, a3, mode="multiplication", _domain="com.videantis", _version=1)

        # Path 2
        x2 = op.Shortcut(x2, a1, mode="multiplication", _domain="com.videantis", _version=1)
        x2 = op.Shortcut(x2, a22, _domain="com.videantis", _version=1)
        x2 = op.Shortcut(x2, a3, mode="multiplication", _domain="com.videantis", _version=1)

        # Sigmoid is required to be right after conv, so no spatial concatenation beforehand
        # Path3
        x13 = op.Sigmoid(x13)
        x23 = op.Sigmoid(x23)
        x33 = op.Sigmoid(x33)
        # Flatten first
        x13 = op.Reshape(x13, op.Constant(value_ints=[1, 24, 1, -1]))
        x23 = op.Reshape(x23, op.Constant(value_ints=[1, 24, 1, -1]))
        x33 = op.Reshape(x33, op.Constant(value_ints=[1, 24, 1, -1]))

        x3 = op.Concat(x13, x23, x33, axis=-1)
        x = op.Concat(x1, x2, x3, axis=1)

        # x = op.Reshape(x, op.Constant(value_ints=[1, -1, 42840]))
        return x


class PoseEstimationPaddingRewritePattern(RewriteRuleClassBase):
    """Rewrite the first part of pose estimation processing part to a format we can handle.

    Specifically we need to pad all channels of the initial shortcut->relu->vidconv->relu->vidconv paths so
    that the channels can be split into three even parts, each of which is a multiple of 8. Currently, there are
    51 channels (3x17). So we require 3x24=72 channels.
    """

    level = 2

    @classmethod
    def pattern(cls, op: Any, x: Any, a2: Any, w1: Any, b1: Any, w2: Any, b2: Any) -> Any:
        """Pattern to be matched."""
        x = op.Slice(x, _allow_other_inputs=True)
        x = op.vidConv(x, w1, b1, _domain="com.videantis")
        x = op.Relu(x)
        x = op.vidConv(x, w2, b2, _domain="com.videantis")
        x = op.Shortcut(x, a2, _domain="com.videantis")
        x = op.Reshape(x, _allow_other_inputs=True)

        return x

    @classmethod
    def rewrite(cls, op: Any, x: Any, a2: Any, w1: Any, b1: Any, w2: Any, b2: Any) -> Any:
        """Rewrite graph."""
        # Pad first conv: input/output channels 51 → 56 (extra channels are zero, effectively ignored)
        w1_np_arr = w1.const_value.numpy()  # [51, 51, 3, 3]
        padded_w1 = np.pad(w1_np_arr, ((0, 5), (0, 5), (0, 0), (0, 0)), mode="constant")  # [56, 56, 3, 3]
        w1 = op.Constant(value=from_array(padded_w1))

        b1_np_arr = b1.const_value.numpy()  # [51]
        padded_b1 = np.pad(b1_np_arr, (0, 5), mode="constant")  # [56]
        b1 = op.Constant(value=from_array(padded_b1))

        # Merge the trailing Shortcut addition into b2, then apply interleaved padding so channels
        # can be split into three equal groups of 24 (each a multiple of 8).
        b2_merged = b2.const_value.numpy() + a2.const_value.numpy().reshape(51)  # [51]

        w2_np_arr = w2.const_value.numpy()  # [51, 51, 3, 3]
        w2_np_arr = w2_np_arr.reshape([17, 3, 51, 3, 3])
        padded_w2 = np.pad(w2_np_arr, ((0, 7), (0, 0), (0, 5), (0, 0), (0, 0)), mode="constant")  # [24, 3, 56, 3, 3]
        padded_w2 = np.transpose(padded_w2, [1, 0, 2, 3, 4])
        padded_w2 = padded_w2.reshape([72, 56, 3, 3])
        w2 = op.Constant(value=from_array(padded_w2))

        b2_merged = b2_merged.reshape([17, 3])
        padded_b2 = np.pad(b2_merged, ((0, 7), (0, 0)), mode="constant")  # [24, 3]
        padded_b2 = np.transpose(padded_b2, [1, 0])
        padded_b2 = padded_b2.reshape([72])
        b2 = op.Constant(value=from_array(padded_b2))

        # Slice is dropped: use full 56-channel input x directly
        x = op.vidConv(x, w1, b1, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)
        x = op.Relu(x)
        x = op.vidConv(x, w2, b2, kernel_shape=(3, 3), pads=(1, 1, 1, 1), _domain="com.videantis", _version=1)
        x = op.Reshape(x, op.Constant(value_ints=[1, 72, 1, -1]))  # Flatten spatial dimensions

        return x

    @classmethod  # type: ignore[override]
    def check(cls, op: Any, x: Any, a2: Any, w1: Any, b1: Any, w2: Any, b2: Any) -> Any:
        """Make sure this pattern is only applied once by checking the shape of a2."""
        channels = a2.const_value.shape[1]
        return channels == 51


class BoundingBoxRewritePattern(RewriteRuleClassBase):
    """Rewrite the bounding box processing part to a format we can handle."""

    level = 2

    @classmethod
    def pattern(
        cls,
        op: Any,
        x11: Any,
        x12: Any,
        x21: Any,
        x22: Any,
        x31: Any,
        x32: Any,
        m11: Any,
        m12: Any,
        m21: Any,
        m22: Any,
        m31: Any,
        m32: Any,
        w: Any,
        a1: Any,
        a2: Any,
        a3: Any,
    ) -> Any:
        """Pattern to be matched."""
        x12 = op.Slice(x12, _allow_other_inputs=True)
        x22 = op.Slice(x22, _allow_other_inputs=True)
        x32 = op.Slice(x32, _allow_other_inputs=True)

        x11 = op.Mul(x11, m11)
        x12 = op.Mul(x12, m12)
        x21 = op.Mul(x21, m21)
        x22 = op.Mul(x22, m22)
        x31 = op.Mul(x31, m31)
        x32 = op.Mul(x32, m32)

        x1 = op.Concat(x11, x12, axis=1)
        x2 = op.Concat(x21, x22, axis=1)
        x3 = op.Concat(x31, x32, axis=1)

        x1 = op.Reshape(x1, _allow_other_inputs=True)
        x2 = op.Reshape(x2, _allow_other_inputs=True)
        x3 = op.Reshape(x3, _allow_other_inputs=True)

        res = op.Concat(x1, x2, x3, axis=2)
        sig = op.Slice(res, _allow_other_inputs=True)
        sig = op.Sigmoid(sig)

        x = op.Slice(res, _allow_other_inputs=True)
        x = op.Reshape(x, _allow_other_inputs=True)

        x = op.Transpose(x)
        x = op.Softmax(x)
        x = op.Transpose(x)
        x = op.vidConv(x, w, None, _domain="com.videantis")

        x = op.Reshape(x, _allow_other_inputs=True)
        x1 = op.Slice(x, _allow_other_inputs=True, _allow_other_attributes=True)
        x2 = op.Slice(x, _allow_other_inputs=True)
        y1 = op.Sub(a1, x1)
        y2 = op.Add(a2, x2)
        z1 = op.Add(y1, y2)
        o1 = op.Div(z1, _allow_other_inputs=True)
        o2 = op.Sub(y2, y1)
        x = op.Concat(o1, o2, axis=1)
        x = op.Mul(x, a3)

        return x, sig

    @classmethod
    def rewrite(
        cls,
        op: Any,
        x11: Any,
        x12: Any,
        x21: Any,
        x22: Any,
        x31: Any,
        x32: Any,
        m11: Any,
        m12: Any,
        m21: Any,
        m22: Any,
        m31: Any,
        m32: Any,
        w: Any,
        a1: Any,
        a2: Any,
        a3: Any,
    ) -> Any:
        """Rewrite graph."""
        # Initial elementwise muls
        x11 = op.Shortcut(x11, m11, mode="multiplication", _domain="com.videantis", _version=1)
        x11 = op.Reshape(x11, op.Constant(value_ints=[1, 64, 1, -1]))  # Flatten spatial dimensions
        x21 = op.Shortcut(x21, m21, mode="multiplication", _domain="com.videantis", _version=1)
        x21 = op.Reshape(x21, op.Constant(value_ints=[1, 64, 1, -1]))  # Flatten spatial dimensions
        x31 = op.Shortcut(x31, m31, mode="multiplication", _domain="com.videantis", _version=1)
        x31 = op.Reshape(x31, op.Constant(value_ints=[1, 64, 1, -1]))  # Flatten spatial dimensions

        # Concatenate along spatial dims
        x = op.Concat(x11, x21, x31, axis=-1)  # [1, 64, 1, 42840]

        # Use our grouped vidSoftmax instead of reshape->transpose->softmax->transpose
        x = op.vidSoftmax(x, group=[4], _domain="com.videantis")

        # Remove transpose by slicing across channels [1, 64, 1, 42840] -> 4x [1, 16, 1, 42840]
        x1 = op.Slice(x, op.Constant(value_ints=[0]), op.Constant(value_ints=[16]), op.Constant(value_ints=[1]))
        x2 = op.Slice(x, op.Constant(value_ints=[16]), op.Constant(value_ints=[32]), op.Constant(value_ints=[1]))
        x3 = op.Slice(x, op.Constant(value_ints=[32]), op.Constant(value_ints=[48]), op.Constant(value_ints=[1]))
        x4 = op.Slice(x, op.Constant(value_ints=[48]), op.Constant(value_ints=[64]), op.Constant(value_ints=[1]))

        # Calculate padded conv -> [1, 8, 1, 42840] where only the first channel is valid
        # Pad w from [1, 16, 1, 1] to [8, 16, 1, 1] to meet our conv requirements
        w_np_arr = w.const_value.numpy()
        padded_w = np.pad(w_np_arr, ((0, 7), (0, 0), (0, 0), (0, 0)), mode="constant")
        w = op.Constant(value=from_array(padded_w))
        x1 = op.vidConv(x1, w, None, _domain="com.videantis", _version=1)
        x2 = op.vidConv(x2, w, None, _domain="com.videantis", _version=1)
        x3 = op.vidConv(x3, w, None, _domain="com.videantis", _version=1)
        x4 = op.vidConv(x4, w, None, _domain="com.videantis", _version=1)

        # We need to split up a1 and a2 to avoid the last reshape in this path merging channels and spatial dimensions
        a11 = op.Constant(value=from_array(a1.const_value.numpy()[:, 0]))
        a12 = op.Constant(value=from_array(a1.const_value.numpy()[:, 1]))
        a21 = op.Constant(value=from_array(a2.const_value.numpy()[:, 0]))
        a22 = op.Constant(value=from_array(a2.const_value.numpy()[:, 1]))

        # We need to negate x11 and x12 to simulate sub
        x1 = op.Shortcut(x1, op.Constant(value_float=-1.0), mode="multiplication", _domain="com.videantis", _version=1)
        x2 = op.Shortcut(x2, op.Constant(value_float=-1.0), mode="multiplication", _domain="com.videantis", _version=1)
        x1 = op.Shortcut(a11, x1, _domain="com.videantis", _version=1)
        x2 = op.Shortcut(a12, x2, _domain="com.videantis", _version=1)

        x3 = op.Shortcut(a21, x3, _domain="com.videantis", _version=1)
        x4 = op.Shortcut(a22, x4, _domain="com.videantis", _version=1)

        x1 = op.Concat(x1, x2, axis=1)
        x2 = op.Concat(x3, x4, axis=1)

        y1 = op.Shortcut(x1, x2, _domain="com.videantis", _version=1)  # Add
        x1 = op.Shortcut(x1, op.Constant(value_float=-1.0), mode="multiplication", _domain="com.videantis", _version=1)
        y2 = op.Shortcut(x2, x1, _domain="com.videantis", _version=1)  # Sub

        # Simulate div by 2 by multiplying with 0.5
        y1 = op.Shortcut(y1, op.Constant(value_float=0.5), mode="multiplication", _domain="com.videantis", _version=1)

        bbox_path = op.Concat(y1, y2, axis=1)
        bbox_path = op.Shortcut(bbox_path, a3, mode="multiplication", _domain="com.videantis", _version=1)

        # Sigmoid paths
        x12 = op.Shortcut(x12, m12, mode="multiplication", _domain="com.videantis", _version=1)
        x12 = op.Sigmoid(x12)
        x22 = op.Shortcut(x22, m22, mode="multiplication", _domain="com.videantis", _version=1)
        x22 = op.Sigmoid(x22)
        x32 = op.Shortcut(x32, m32, mode="multiplication", _domain="com.videantis", _version=1)
        x32 = op.Sigmoid(x32)

        # Flatten spatial dimensions
        x12 = op.Reshape(x12, op.Constant(value_ints=[1, 8, 1, -1]))
        x22 = op.Reshape(x22, op.Constant(value_ints=[1, 8, 1, -1]))
        x32 = op.Reshape(x32, op.Constant(value_ints=[1, 8, 1, -1]))

        sig_out = op.Concat(x12, x22, x32, axis=-1)

        # TMP
        # bbox_path = op.Reshape(bbox_path, op.Constant(value_ints=[1, -1, 42840]))
        # sig_out = op.Reshape(sig_out, op.Constant(value_ints=[1, -1, 42840]))

        return bbox_path, sig_out


def _expand_to_4D_shape(arr: NDArray[Any]) -> NDArray[Any]:
    """
    Expand a NumPy array to 4 dimensions for broadcasting purposes.

    The function reshapes the input array by left-padding its shape with
    singleton dimensions (size 1) until it has exactly 4 dimensions.
    The relative order and sizes of the original dimensions are preserved.

    Args:
        arr (NDArray[Any]): Input array with 0 to 4 dimensions.

    Returns:
        NDArray[Any]: A view of the input array reshaped to 4 dimensions.

    Raises:
        ValueError: If the input array has more than 4 dimensions.

    Examples:
        >>> expand_to_4D_shape(np.empty((1, 400))).shape
        (1, 1, 1, 400)

        >>> expand_to_4D_shape(np.empty((1,))).shape
        (1, 1, 1, 1)

        >>> expand_to_4D_shape(np.empty((2, 1, 200))).shape
        (1, 2, 1, 200)
    """
    arr = np.array(arr, dtype=np.float32)  # also handles scalars
    current_shape = arr.shape

    if len(current_shape) > 4:
        raise ValueError("Input array has more than 4 dimensions.")

    new_shape = (1,) * (4 - len(current_shape)) + current_shape
    return arr.reshape(new_shape)


def _ensure_tensor_is_4D(model: onnx.ModelProto, input_tensor: Tensor) -> None:
    if input_tensor.tensor_type == TensorType.NODE_OUTPUT or input_tensor.tensor_type == TensorType.GRAPH_INPUT:
        return  # Dynamic tensor
    elif input_tensor.tensor_type == TensorType.INITIALIZER:
        initializer_data = get_wgt_by_name(model, input_tensor.name)
        new_data = _expand_to_4D_shape(initializer_data)
        model = replace_wgt(model, new_data, input_tensor.name)

    else:
        raise RuntimeError("Unknown tensor type.")


def generate_random_inputs(input_value_protos: Any, batch_size: int = 1) -> dict[str, Any]:
    """Generate a dict of random input tensors for an ONNX model.

    Dynamic or symbolic batch dimension is replaced with batch_size.
    """
    input_dict: dict[str, Any] = {}

    for inp in input_value_protos:
        name = inp.name
        tensor_type = inp.type.tensor_type

        # Resolve shape
        shape = []
        for i, dim in enumerate(tensor_type.shape.dim):
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                # treat unknown/symbolic dimension as batch dimension
                if i == 0:
                    shape.append(batch_size)
                else:
                    raise ValueError(f"Cannot resolve dimension {i} for input '{name}'.")

        # Generate random data
        data = np.random.randn(*shape).astype(np.float32) * 20.0
        input_dict[name] = data

    return input_dict


class MythicYoloV8PosePostprocessing(VidModel):
    """VidModel definition for the Mythic YoloV8Pose postprocessing part."""

    @classmethod
    def initialize_onnx(
        cls, model_directory: str | Path, config: Optional[InitializationConfig] = None
    ) -> onnx.ModelProto:
        """Return a runable ONNX ModelProto of the  model."""
        return onnx.load(ONNX_PATH)

    def setup(self) -> None:
        """Extract inputs to be used for random input generation."""
        self.input_value_protos = [inp for inp in self._model_repr.graph.input]

    def preprocess(self, input_data: ImageDetectionInput) -> Any:
        """Preprocess an image by resizing and normalizing."""
        example_data = generate_random_inputs(self.input_value_protos, batch_size=1)
        return example_data

    def postprocess(self, model_output: Any, _: ImageDetectionInput) -> ImageDetectionOutput:  # type: ignore
        """Remove channel padding and restore original channel order.

        The graph rewrites introduce two padding transformations that expand 56 → 112 channels
        and insert a singleton spatial dimension:

          BoundingBoxRewritePattern:
            - 4 bbox channels padded to 32 (valid at indices [0, 8, 16, 24])
            - 1 obj channel padded to 8 (valid at index [32])

          PoseEstimationPaddingRewritePattern + PoseEstimationPart2RewritePattern:
            - 51 pose channels (interleaved [x, y, vis] per keypoint) padded to 72 and
              regrouped so all x-coords are together [40:57], y-coords [64:81],
              visibility [88:105]

        This method undoes both: selects valid channels and re-interleaves pose back to
        [kp0_x, kp0_y, kp0_vis, kp1_x, ...].
        """
        assert len(model_output) == 1, f"Expected 1 output tensor, got {len(model_output)}."
        tensor = next(iter(model_output.values()))  # [1, 112, 1, 42840]

        bbox = tensor[:, [0, 8, 16, 24], :, :]  # [1, 4,  1, 42840]
        obj = tensor[:, [32], :, :]  # [1, 1,  1, 42840]
        pose_x = tensor[:, 40:57, :, :]  # [1, 17, 1, 42840]  kp0_x..kp16_x
        pose_y = tensor[:, 64:81, :, :]  # [1, 17, 1, 42840]  kp0_y..kp16_y
        pose_v = tensor[:, 88:105, :, :]  # [1, 17, 1, 42840]  kp0_vis..kp16_vis

        # Stack → [1, 17, 3, 1, 42840], reshape → [1, 51, 1, 42840]: gives [kp0_x, kp0_y, kp0_vis, kp1_x, ...]
        pose = np.stack([pose_x, pose_y, pose_v], axis=2).reshape(1, 51, 1, -1)

        result = np.concatenate([bbox, obj, pose], axis=1)  # [1, 56, 1, 42840]
        return {next(iter(model_output.keys())): result.squeeze(2)}  # type: ignore  # [1, 56, 42840]

    @classmethod
    def load_default_dataset(cls) -> DummyDataset:
        """Return an initialized dataset that can be used to load data samples for this model."""
        return DummyDataset()

    def optimize_hook(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Rewrite most of the graph so it can be processed by us."""
        # Apply some manual changes to the graph after the optimizer has run.
        rule1 = BoundingBoxRewritePattern.rule()  # type: ignore[no-untyped-call]
        rule2 = PoseEstimationPaddingRewritePattern.rule()  # type: ignore[no-untyped-call]
        rule3 = PoseEstimationPart2RewritePattern.rule()  # type: ignore[no-untyped-call]
        model, count1 = vid_match_patterns_onnxscript(model, rule1, verbose=0, commute=True)
        model, count2 = vid_match_patterns_onnxscript(model, rule2, verbose=0, commute=True)
        model, count3 = vid_match_patterns_onnxscript(model, rule3, verbose=0, commute=True)

        # Check that the patterns were matched the expected number of times to ensure they are only applied where intended
        expected_count1 = 1
        expected_count2 = 3
        expected_count3 = 1
        if count1 != expected_count1:
            raise ValueError(f"Expected to match pattern once, but matched {count1} times.")
        if count2 != expected_count2:
            raise ValueError(f"Expected to match pattern thrice, but matched {count2} times.")
        if count3 != expected_count3:
            raise ValueError(f"Expected to match pattern once, but matched {count3} times.")

        # Most of our toolbox can't handle constants properly so move them to initializers
        model = move_constants_to_wgts(model)

        # Ensure static inputs of Shortcut nodes have correct input rank so that no automatic broadcasting
        # to higher rank is required
        graph_helper = ONNXGraphHelper(model)
        for node in graph_helper.nodes.values():
            if node.op_type == "Shortcut":
                input_tensor1 = node.inputs[0]
                input_tensor2 = node.inputs[1]
                _ensure_tensor_is_4D(model, input_tensor1)
                _ensure_tensor_is_4D(model, input_tensor2)

        # Clean up graph one last time and infer shapes
        ds = self.load_default_dataset()
        data1, data2 = self.preprocess(ds[0]), self.preprocess(ds[50])
        model = move_static_cons_to_wgts(model, [data1, data2])
        model = remove_unused_nodes(model)  # type: ignore[arg-type]
        infer_shapes_runtime(model, inputs=data1)

        # One last sanity check
        onnx.checker.check_model(model, full_check=True, check_custom_domain=True)

        return model


if __name__ == "__main__":
    from mythic_utils import parse_arguments, run_vnn_flow

    args = parse_arguments()

    # Prepare/Check input/output paths
    result_directory: Path = args.result_directory
    result_directory.mkdir(parents=True, exist_ok=True)

    source_onnx: Path = args.source_onnx
    if not source_onnx.is_file():
        print(f"Error: '{source_onnx}' is not a valid file.")
        sys.exit(1)

    # This is very ugly but we do not have a better way right now to access the CLI path within the VidModel
    ONNX_PATH = source_onnx

    configure_logging()

    # Run v-NN ORT pipeline with this model
    model = MythicYoloV8PosePostprocessing(result_directory)
    run_vnn_flow(model, result_directory, system_config=Path(__file__).parent / "system_configs" / "yolov8pose.cfg")
