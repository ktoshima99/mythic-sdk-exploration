# type: ignore

import logging
from typing import Optional, Sequence

import onnx
from onnxscript import FLOAT, INT64, OnnxFunction
from onnxscript import opset20 as op
from onnxscript import script, values

from vnnort.utils.onnx_utils.utils import register_op_schema

logger = logging.getLogger("onnxscript.values")


class SuppressSpecificLog(logging.Filter):
    """
    A logging filter to suppress specific log messages containing a given substring.

    This filter is designed to ignore log messages that contain the phrase "Already defined".
    Useful for suppressing repetitive or non-critical log entries in the output.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore
        """
        Filter out log records that contain the phrase "Already defined".

        Args:
            record (logging.LogRecord): The log record to be filtered.

        Returns:
            bool: True if the record should be processed, False if it should be suppressed.
        """
        return "Already defined" not in record.getMessage()


logger.addFilter(SuppressSpecificLog())


# def script(opset):
#    def decorator(func):
#        @wraps(func)
#        def wrapper(*args, **kwargs):
#            # Additional behavior can be added here
#            logger.info(f"Wrapping function: {func.__name__} with opset: {opset}")
#            return onnx_script(opset)(func)(*args, **kwargs)
#        return wrapper
#    return decorator


@script(values.Opset("com.videantis", 1))
def reshapeToWgtsMulExpand(x: FLOAT) -> FLOAT:
    """Reshape and expand the input tensor for specific operations.

    This function performs a series of transformations on the input tensor `x`.
    It calculates the shape of `x`, extracts a specific dimension, and constructs
    an expanded shape. The tensor `x` is then expanded to match this shape, flattened,
    and an identity-like tensor is generated and multiplied with `x`.

    Args:
        x (FLOAT): Input tensor to be reshaped and expanded. Expected to be in shape ["N", "C", "H", "W"].

    Returns:
        FLOAT: The transformed tensor with the same shape as the input.

    Operations:
        - `op.Shape`: Extracts the shape of the tensor `x`.
        - `op.Constant`: Generates constants used in gathering dimensions and reshaping.
        - `op.Gather`: Selects a specific dimension of `x`’s shape.
        - `op.Concat`: Concatenates dimensions to form the desired shape.
        - `op.Expand`: Expands `x` to the constructed shape.
        - `op.Flatten`: Flattens `x` for further operations.
        - `op.EyeLike`: Creates an identity matrix-like tensor matching `flattened_x`.
        - `op.Unsqueeze`: Adds new dimensions to `eyelike`.
        - `op.Mul`: Multiplies `x` by `eyelike`.

    Note:
        This function is intended for use within an ONNX-like operator framework.

    """
    x_shape = op.Shape(x)
    one = op.Constant(value_ints=[1])
    dim = op.Gather(x_shape, one)
    expand_shape = op.Concat(dim, dim, one, one, axis=0)
    x = op.Expand(x, expand_shape)
    flattened_x = op.Flatten(x)
    eyelike = op.EyeLike(flattened_x)
    minus_one = op.Constant(value_ints=[-1])
    eyelike = op.Unsqueeze(eyelike, minus_one)
    eyelike = op.Unsqueeze(eyelike, minus_one)
    x = op.Mul(x, eyelike)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def Shortcut(
    x: FLOAT,
    y: FLOAT,
    mode: str = "addition",
    reshape_mode: str = "None",
    group: Sequence[int] = [1],
) -> FLOAT:
    """
    Perform element-wise addition of two inputs with optional activation and reshaping.

    Args:
        x (FLOAT): The first input tensor.
        y (FLOAT): The second input tensor.
        mode (str): possible options are 'addition' and 'multiplication'.
        reshape_mode (str): Reshape mode, such as "TRANSFORMER_QK" for transformer operations.
        group (Sequence[int]): Group size for reshaping in "TRANSFORMER_QK" mode. Defaults to [1].

    Returns:
        FLOAT: The output tensor after optional activation and reshape.
    """
    z = x
    if mode == "addition":
        z = op.Add(x, y)
    if mode == "multiplication":
        z = op.Mul(x, y)
    if mode == "division":
        z = op.Div(x, y)

    if reshape_mode == "TRANSFORMER_QK":
        three = op.Constant(value_ints=[3])
        k_seq_length = op.Shape(x)
        k_seq_length = op.Gather(k_seq_length, three)

        one = op.Constant(value_ints=[1])
        k_seq_length = op.Mul(k_seq_length, one)
        inter_dim = op.Div(op.Shape(x)[1], group)
        grp_reshape1 = op.Concat(one, group, inter_dim, k_seq_length, one, axis=0)

        full_val = op.Mul(group, k_seq_length)
        grp_reshape2 = op.Concat(full_val, inter_dim, one, one, axis=0)
        z = op.Reshape(z, grp_reshape1)
        z = op.Transpose(z, perm=(1, 3, 2, 0, 4))
        z = op.Reshape(z, grp_reshape2)
    return z


@script(values.Opset("com.videantis", 1))  # type: ignore
def Swish(x: FLOAT) -> FLOAT:
    """Apply the Swish activation function.

    The Swish activation function is defined as `x * sigmoid(x)`, which gives smooth, non-linear behavior.

    Args:
        x (FLOAT): Input tensor for the Swish activation.

    Returns:
        FLOAT: The result of applying Swish activation to the input tensor.
    """
    x1 = op.Sigmoid(x)
    x = op.Mul(x, x1)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def Mish(x: FLOAT) -> FLOAT:
    """Apply the Mish activation function.

    Args:
        x (FLOAT): Input tensor for the Mish activation.

    Returns:
        FLOAT: The result of applying Mish activation to the input tensor.
    """
    one = op.Constant(value_floats=[1.0])

    x1 = op.Exp(x)
    x1 = op.Add(one, x1)
    x1 = op.Log(x1)
    x1 = op.Tanh(x1)
    x = op.Mul(x, x1)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def Relu6(x: FLOAT) -> FLOAT:
    """Apply the ReLU6 activation function.

    The ReLU6 activation function clips the values of the input tensor between 0 and 6.

    Args:
        x (FLOAT): Input tensor for the ReLU6 activation.

    Returns:
        FLOAT: The result of applying ReLU6 activation, with values clipped between 0 and 6.
    """
    clip_min = op.Constant(value_float=0.0)
    clip_max = op.Constant(value_float=6.0)
    x = op.Clip(x, clip_min, clip_max)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidPartitionWindowsShiftedReverse(
    x: FLOAT,
    reshape1: INT64,
    reshape2: INT64,
    reshape3: INT64,
    starts1: INT64,
    ends1: INT64,
    axes1: INT64,
    steps1: INT64,
    starts2: INT64,
    ends2: INT64,
    axes2: INT64,
    steps2: INT64,
    starts1_1: INT64,
    ends1_1: INT64,
    axes1_1: INT64,
    starts1_2: INT64,
    ends1_2: INT64,
    axes1_2: INT64,
    starts2_1: INT64,
    ends2_1: INT64,
    axes2_1: INT64,
    starts2_2: INT64,
    ends2_2: INT64,
    axes2_2: INT64,
    dim: int = 3,
) -> FLOAT:
    """
    Reverse the partitioning of shifted windows by applying a series of transpositions, reshapes, slicing, and concatenation operations.

    Args:
        x (FLOAT): The input tensor.
        reshape1 (INT64): Target shape for the initial reshape.
        reshape2 (INT64): Target shape for the intermediate reshape.
        reshape3 (INT64): Target shape for the final reshape.
        starts1 (INT64): Start indices for the first slicing operation.
        ends1 (INT64): End indices for the first slicing operation.
        axes1 (INT64): Axes for the first slicing operation.
        steps1 (INT64): Steps for the first slicing operation.
        starts2 (INT64): Start indices for the second slicing operation.
        ends2 (INT64): End indices for the second slicing operation.
        axes2 (INT64): Axes for the second slicing operation.
        steps2 (INT64): Steps for the second slicing operation.
        starts1_1 (INT64): Start indices for the first additional slicing (part 1).
        ends1_1 (INT64): End indices for the first additional slicing (part 1).
        axes1_1 (INT64): Axes for the first additional slicing (part 1).
        starts1_2 (INT64): Start indices for the first additional slicing (part 2).
        ends1_2 (INT64): End indices for the first additional slicing (part 2).
        axes1_2 (INT64): Axes for the first additional slicing (part 2).
        starts2_1 (INT64): Start indices for the second additional slicing (part 1).
        ends2_1 (INT64): End indices for the second additional slicing (part 1).
        axes2_1 (INT64): Axes for the second additional slicing (part 1).
        starts2_2 (INT64): Start indices for the second additional slicing (part 2).
        ends2_2 (INT64): End indices for the second additional slicing (part 2).
        axes2_2 (INT64): Axes for the second additional slicing (part 2).
        dim (int): Dimension attribute to define IO dimensions (default=3).

    Returns:
        FLOAT: The output tensor after reversing the partitioning of shifted windows.
    """
    # zero = op.Constant(value_ints=[0])
    two = op.Constant(value_ints=[2])
    if dim == 4:
        x = op.Squeeze(x)
    x = op.Transpose(x, perm=[0, 2, 1])
    x = op.Reshape(x, reshape1)
    x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = op.Reshape(x, reshape2)

    x1_1 = op.Slice(x, starts1_1, ends1_1, axes1_1)
    x1_2 = op.Slice(x, starts1_2, ends1_2, axes1_2)
    x = op.Concat(x1_1, x1_2, axis=1)

    x2_1 = op.Slice(x, starts2_1, ends2_1, axes2_1)
    x2_2 = op.Slice(x, starts2_2, ends2_2, axes2_2)
    x = op.Concat(x2_1, x2_2, axis=2)

    x = op.Slice(x, starts1, ends1, axes1, steps1)
    x = op.Slice(x, starts2, ends2, axes2, steps2)
    x = op.Reshape(x, reshape3)
    if dim == 4:
        x = op.Transpose(x, perm=[0, 2, 1])
        x = op.Unsqueeze(x, two)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidPartitionWindowsShifted(
    x: FLOAT,
    reshape1: INT64,
    reshape2: INT64,
    reshape3: INT64,
    pad1: FLOAT,
    pad2: FLOAT,
    starts1: INT64,
    ends1: INT64,
    axes1: INT64,
    steps1: INT64,
    starts2: INT64,
    ends2: INT64,
    axes2: INT64,
    steps2: INT64,
    starts1_1: INT64,
    ends1_1: INT64,
    axes1_1: INT64,
    starts1_2: INT64,
    ends1_2: INT64,
    axes1_2: INT64,
    starts2_1: INT64,
    ends2_1: INT64,
    axes2_1: INT64,
    starts2_2: INT64,
    ends2_2: INT64,
    axes2_2: INT64,
    dim: int = 3,
) -> FLOAT:
    """
    Partition the input tensor into shifted windows using a series of reshaping, transposition, concatenation, and slicing operations.

    Args:
        x (FLOAT): The input tensor.
        reshape1 (INT64): Target shape for the initial reshape.
        reshape2 (INT64): Target shape for the intermediate reshape.
        reshape3 (INT64): Target shape for the final reshape.
        pad1 (FLOAT): Padding tensor for concatenation along the last axis.
        pad2 (FLOAT): Padding tensor for concatenation along the second-to-last axis.
        starts1 (INT64): Start indices for the first slicing operation.
        ends1 (INT64): End indices for the first slicing operation.
        axes1 (INT64): Axes for the first slicing operation.
        steps1 (INT64): Steps for the first slicing operation.
        starts2 (INT64): Start indices for the second slicing operation.
        ends2 (INT64): End indices for the second slicing operation.
        axes2 (INT64): Axes for the second slicing operation.
        steps2 (INT64): Steps for the second slicing operation.
        starts1_1 (INT64): Start indices for the first additional slicing (part 1).
        ends1_1 (INT64): End indices for the first additional slicing (part 1).
        axes1_1 (INT64): Axes for the first additional slicing (part 1).
        starts1_2 (INT64): Start indices for the first additional slicing (part 2).
        ends1_2 (INT64): End indices for the first additional slicing (part 2).
        axes1_2 (INT64): Axes for the first additional slicing (part 2).
        starts2_1 (INT64): Start indices for the second additional slicing (part 1).
        ends2_1 (INT64): End indices for the second additional slicing (part 1).
        axes2_1 (INT64): Axes for the second additional slicing (part 1).
        starts2_2 (INT64): Start indices for the second additional slicing (part 2).
        ends2_2 (INT64): End indices for the second additional slicing (part 2).
        axes2_2 (INT64): Axes for the second additional slicing (part 2).
        dim (int): Dimension attribute to specify IO dimensions.

    Returns:
        FLOAT: The output tensor after partitioning into shifted windows.
    """
    zero = op.Constant(value_ints=[0])
    two = op.Constant(value_ints=[2])
    if dim == 4:
        x = op.Squeeze(x)
        x = op.Transpose(x, perm=[1, 0])
        x = op.Unsqueeze(x, zero)
    x = op.Reshape(x, reshape1)
    x = op.Transpose(x, perm=[0, 3, 1, 2])

    x = op.Concat(x, pad1, axis=-1)
    x = op.Concat(x, pad2, axis=-2)

    x = op.Slice(x, starts1, ends1, axes1, steps1)
    x = op.Slice(x, starts2, ends2, axes2, steps2)

    x = op.Transpose(x, perm=[0, 2, 3, 1])

    x1_1 = op.Slice(x, starts1_1, ends1_1, axes1_1)
    x1_2 = op.Slice(x, starts1_2, ends1_2, axes1_2)
    x = op.Concat(x1_1, x1_2, axis=1)

    x2_1 = op.Slice(x, starts2_1, ends2_1, axes2_1)
    x2_2 = op.Slice(x, starts2_2, ends2_2, axes2_2)
    x = op.Concat(x2_1, x2_2, axis=2)

    x = op.Reshape(x, reshape2)
    x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = op.Reshape(x, reshape3)
    x = op.Transpose(x, perm=[0, 2, 1])
    if dim == 4:
        x = op.Unsqueeze(x, two)

    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidPartitionWindows(
    x: FLOAT,
    reshape1: INT64,
    reshape2: INT64,
    reshape3: INT64,
    pad1: FLOAT,
    pad2: FLOAT,
    starts1: INT64,
    ends1: INT64,
    axes1: INT64,
    steps1: INT64,
    starts2: INT64,
    ends2: INT64,
    axes2: INT64,
    steps2: INT64,
    dim: int = 3,
) -> FLOAT:
    """
    Partition the input tensor into windows using reshaping, transposition, concatenation, and slicing operations.

    This function takes an input tensor and performs the following steps:
      1. Reshape the tensor to a new shape.
      2. Transpose the dimensions to prepare for padding.
      3. Concatenate with provided padding tensors along the last and
         second-to-last axes.
      4. Apply two slicing operations to extract the desired window regions.
      5. Transpose and reshape again to organize the windows.
      6. Apply a final transpose to produce the output tensor with the desired
         window partitioning layout.

    Args:
        x (FLOAT): The input tensor.
        reshape1 (INT64): Target shape for the initial reshape.
        reshape2 (INT64): Target shape for the intermediate reshape.
        reshape3 (INT64): Target shape for the final reshape.
        pad1 (FLOAT): Padding tensor for concatenation along the last axis.
        pad2 (FLOAT): Padding tensor for concatenation along the second-to-last axis.
        starts1 (INT64): Start indices for the first slicing operation.
        ends1 (INT64): End indices for the first slicing operation.
        axes1 (INT64): Axes to slice for the first slicing operation.
        steps1 (INT64): Steps for the first slicing operation.
        starts2 (INT64): Start indices for the second slicing operation.
        ends2 (INT64): End indices for the second slicing operation.
        axes2 (INT64): Axes to slice for the second slicing operation.
        steps2 (INT64): Steps for the second slicing operation.
        dim (int): Dimension attribute to specify IO dimensions.


    Returns:
        FLOAT: The output tensor after partitioning into windows.
    """
    zero = op.Constant(value_ints=[0])
    two = op.Constant(value_ints=[2])
    if dim == 4:
        x = op.Squeeze(x)
        x = op.Transpose(x, perm=[1, 0])
        x = op.Unsqueeze(x, zero)
    x = op.Reshape(x, reshape1)

    x = op.Transpose(x, perm=[0, 3, 1, 2])

    x = op.Concat(x, pad1, axis=-1)
    x = op.Concat(x, pad2, axis=-2)

    x = op.Slice(x, starts1, ends1, axes1, steps1)
    x = op.Slice(x, starts2, ends2, axes2, steps2)

    x = op.Transpose(x, perm=[0, 2, 3, 1])

    x = op.Reshape(x, reshape2)

    x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])

    x = op.Reshape(x, reshape3)
    x = op.Transpose(x, perm=[0, 2, 1])
    if dim == 4:
        x = op.Unsqueeze(x, two)

    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidPartitionWindowsReverse(
    x: FLOAT,
    reshape1: INT64,
    reshape2: INT64,
    reshape3: INT64,
    starts1: INT64,
    ends1: INT64,
    axes1: INT64,
    steps1: INT64,
    starts2: INT64,
    ends2: INT64,
    axes2: INT64,
    steps2: INT64,
    dim: int = 3,
) -> FLOAT:
    """
    Reverse partition the input tensor into windows using reshaping, transposition, and slicing operations.

    Args:
        x (FLOAT): The input tensor.
        reshape1 (INT64): Target shape for the initial reshape.
        reshape2 (INT64): Target shape for the intermediate reshape.
        reshape3 (INT64): Target shape for the final reshape.
        starts1 (INT64): Start indices for the first slicing operation.
        ends1 (INT64): End indices for the first slicing operation.
        axes1 (INT64): Axes to slice for the first slicing operation.
        steps1 (INT64): Steps for the first slicing operation.
        starts2 (INT64): Start indices for the second slicing operation.
        ends2 (INT64): End indices for the second slicing operation.
        axes2 (INT64): Axes to slice for the second slicing operation.
        steps2 (INT64): Steps for the second slicing operation.
        dim (int): Dimension attribute to specify IO dimensions.

    Returns:
        FLOAT: The output tensor after reverse partitioning.
    """
    # zero = op.Constant(value_ints=[0])
    two = op.Constant(value_ints=[2])
    if dim == 4:
        x = op.Squeeze(x)
    x = op.Transpose(x, perm=[0, 2, 1])
    x = op.Reshape(x, reshape1)
    x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = op.Reshape(x, reshape2)
    x = op.Slice(x, starts1, ends1, axes1, steps1)
    x = op.Slice(x, starts2, ends2, axes2, steps2)
    x = op.Reshape(x, reshape3)
    if dim == 4:
        x = op.Transpose(x, perm=[0, 2, 1])
        x = op.Unsqueeze(x, two)

    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidConv(  # noqa: C901 ---ignores "too complex"-error
    x: FLOAT,
    w: FLOAT,
    b: Optional[FLOAT] = None,
    dilations: Sequence[int] = (1, 1),
    group: int = 1,
    kernel_shape: Sequence[int] = (1, 1),
    pads: Sequence[int] = (0, 0, 0, 0),
    strides: Sequence[int] = [1, 1],
    dim: int = 4,
    reshape_mode: str = "None",
    reshape_mode_groups: Sequence[int] = [1],
    reshape_swin: Sequence[int] = [0, 0, 0, 0],
    auto_pad: str = "NOTSET",
) -> FLOAT:
    """
    Apply convolution to an input tensor with optional reshaping.

    Args:
        x (FLOAT): The input tensor.
        w (FLOAT): The convolutional weight tensor.
        b (Optional[FLOAT]): The bias tensor for the convolution.
        dilations (Sequence[int]): Dilation factors for each spatial dimension. Defaults to (1, 1).
        group (int): Number of groups for the convolution. Defaults to 1.
        kernel_shape (Sequence[int]): Shape of the convolution kernel. Defaults to [1, 1].
        pads (Sequence[int]): Padding in the form (pad_top, pad_left, pad_bottom, pad_right). Defaults to (0, 0, 0, 0).
        strides (Sequence[int]): Strides for the convolution. Defaults to [1, 1].
        dim (int): Dimensionality of the convolution (e.g., 2D or 4D). Defaults to 4.
        reshape_mode (str): Reshape mode. Options:
            - "None": No reshaping is applied.
            - "MUL_EXPAND": Expands tensor dimensions to enable element-wise multiplication; needed to apply broadcast scaling.
            - "TRANSFORMER_V": Transposes tensor dimensions to align with transformer value projection requirements.
            - "TRANSFORMER_QK": Reshapes and groups channels to align with transformer query projection requirements.
            - "FLATTEN_W": Flattens [N,C,H,W] output to [N,C,1,(HxW)].
        reshape_mode_groups (Sequence[int]): Group sizes for reshaping in "TRANSFORMER_QK" mode. Defaults to [1].
        reshape_swin (Sequence[int]): specific reshape attribute for Swin type networks.
        auto_pad (str): Defines auto_pad mode (see https://onnx.ai/onnx/operators/onnx__Conv.html). Defaults to "NOTSET".

    Returns:
        FLOAT: The output tensor after convolution, activation, and reshape.
    """
    minus_one = op.Constant(value_ints=[-1])
    x = op.Identity(x)
    w = op.Identity(w)

    # Dimensional adjustment based on `dim` argument
    if dim == 2:
        x = op.Unsqueeze(x, minus_one)
        x = op.Unsqueeze(x, minus_one)
        w = op.Unsqueeze(w, minus_one)
        w = op.Unsqueeze(w, minus_one)
    else:
        x = op.Identity(x)
        x = op.Cast(x, to=onnx.TensorProto.FLOAT)
        w = op.Identity(w)
        w = op.Cast(w, to=onnx.TensorProto.FLOAT)
    if auto_pad == "NOTSET":
        x = op.Conv(
            x,
            w,
            b,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )
    else:
        x = op.Conv(
            x,
            w,
            b,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            strides=strides,
            auto_pad=auto_pad,
        )

    if dim == 2:
        x = op.Squeeze(x, minus_one)
        x = op.Squeeze(x, minus_one)
    else:
        x = op.Identity(x)
        x = op.Cast(x, to=onnx.TensorProto.FLOAT)

    # Applying reshape modes
    if reshape_mode == "MUL_EXPAND":
        x_shape = op.Shape(x)
        one = op.Constant(value_ints=[1])
        dim = op.Gather(x_shape, one)
        expand_shape = op.Concat(dim, dim, one, one, axis=0)
        x = op.Expand(x, expand_shape)
        flattened_x = op.Flatten(x)
        eyelike = op.EyeLike(flattened_x)
        minus_one = op.Constant(value_ints=[-1])
        eyelike = op.Unsqueeze(eyelike, minus_one)
        eyelike = op.Unsqueeze(eyelike, minus_one)
        x = op.Mul(x, eyelike)
    elif reshape_mode == "TRANSFORMER_V":
        x = op.Transpose(x, perm=(1, 3, 0, 2))
    elif reshape_mode == "TRANSFORMER_QK":
        three = op.Constant(value_ints=[3])
        k_seq_length = op.Shape(x)
        k_seq_length = op.Gather(k_seq_length, three)

        one = op.Constant(value_ints=[1])
        k_seq_length = op.Mul(k_seq_length, one)
        inter_dim = op.Div(op.Shape(x)[1], reshape_mode_groups)
        grp_reshape1 = op.Concat(one, reshape_mode_groups, inter_dim, k_seq_length, one, axis=0)

        full_val = op.Mul(reshape_mode_groups, k_seq_length)
        grp_reshape2 = op.Concat(full_val, inter_dim, one, one, axis=0)
        x = op.Reshape(x, grp_reshape1)
        x = op.Transpose(x, perm=(1, 3, 2, 0, 4))
        x = op.Reshape(x, grp_reshape2)
    elif reshape_mode == "FLATTEN_W":
        one = op.Constant(value_ints=[1])
        channels = op.Shape(w)[0]
        channels = op.CastLike(channels, one)
        channels = op.Reshape(channels, op.Shape(one))
        twofivesix = channels  # op.Constant(value_ints=[256])
        minus_one = op.Constant(value_ints=[-1])
        reshape1 = op.Concat(one, twofivesix, one, minus_one, axis=0)
        x = op.Reshape(x, reshape1)
    elif reshape_mode == "SWIN_QK":
        # windows = op.Div(group,reshape_mode_groups_swin)
        # three = op.Constant(value_ints=[3])
        # seq_len = op.Gather(op.Shape(x),three)
        # reshape3 = op.Concat(windows, reshape_mode_groups_swin, seq_len, seq_len, axis=0)
        reshape_swin_const = op.Constant(value_ints=reshape_swin)
        x = op.Reshape(x, reshape_swin_const)
        x = op.Transpose(x, perm=[0, 1, 3, 2])

    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidFlatten(x: FLOAT, axis: int = 1) -> FLOAT:
    """Flatten a tensor along the specified axis.

    Wraps the `Flatten` operation to reshape the input tensor while preserving
    all other dimensions.

    Args:
        x (FLOAT): The input tensor to be flattened.
        axis (int): The axis along which to flatten the tensor. Defaults to 1.

    Returns:
        FLOAT: The flattened tensor.
    """
    x = op.Flatten(x, axis=axis)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidConcat(x: FLOAT, y: FLOAT, axis: int = 1) -> FLOAT:
    """Concatenate two tensors along the specified axis.

    Wraps the `Concat` operation to combine two input tensors into a single tensor.

    Args:
        x (FLOAT): The first input tensor.
        y (FLOAT): The second input tensor.
        axis (int): The axis along which to concatenate the tensors. Defaults to 1.

    Returns:
        FLOAT: The concatenated tensor.
    """
    x = op.Concat(x, y, axis=axis)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidMaxPool(
    x: FLOAT,
    kernel_shape: Sequence[int],
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    dilations: Sequence[int] = [1, 1],
    pads: Sequence[int] = [0, 0, 0, 0],
    strides: Sequence[int] = [1, 1],
) -> FLOAT:
    """
    Apply max pooling to the input tensor.

    Args:
        x (FLOAT): The input tensor.
        kernel_shape (Sequence[int]): The size of the kernel along each axis.
        auto_pad (str): Padding mode. Options are 'NOTSET', 'SAME_UPPER', 'SAME_LOWER', or 'VALID'.
        ceil_mode (int): Whether to use ceil (1) or floor (0) to compute the output shape.
        dilations (Sequence[int]): Dilation value along each spatial axis of the filter.
        pads (Sequence[int]): Padding for the beginning and ending along each spatial axis.
        strides (Sequence[int]): Stride along each spatial axis.

    Returns:
        FLOAT: The tensor after applying max pooling.
    """
    # Apply max pooling operation
    x, _ = op.MaxPool(
        x,
        kernel_shape=kernel_shape,
        auto_pad=auto_pad,
        ceil_mode=ceil_mode,
        dilations=dilations,
        pads=pads,
        strides=strides,
    )
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidAveragePool(
    x: FLOAT,
    kernel_shape: Sequence[int],
    auto_pad: str = "NOTSET",
    ceil_mode: int = 0,
    count_include_pad: int = 0,
    dilations: Sequence[int] = [1, 1],
    pads: Sequence[int] = [0, 0, 0, 0],
    strides: Sequence[int] = [1, 1],
    output_dim: int = 4,
) -> FLOAT:
    """
    Apply average pooling to the input tensor.

    Args:
        x (FLOAT): The input tensor.
        kernel_shape (Sequence[int]): The size of the kernel along each axis. Defaults to [1, 1].
        auto_pad (str): Padding mode. Options are 'NOTSET', 'SAME_UPPER', 'SAME_LOWER', or 'VALID'.
            Defaults to 'NOTSET'.
        ceil_mode (int): Whether to use ceil (1) or floor (0) to compute the output shape. Defaults to 0.
        count_include_pad (int): Whether to include pad pixels when calculating values for the edges.
            Defaults to 0 (does not count pad).
        dilations (Sequence[int]): Dilation value along each spatial axis of the filter.
            Defaults to None, which means dilation of 1 along each axis.
        pads (Sequence[int]): Padding for the beginning and ending along each spatial axis.
            Defaults to None, which means padding of 0 along each spatial axis.
        strides (Sequence[int]): Stride along each spatial axis. Defaults to None, which means
            a stride of 1 along each spatial axis.
        output_dim (int): adds additional unsqueezes internally if dim !=4.

    Returns:
        FLOAT: The tensor after applying average pooling.
    """
    # Apply average pooling operation
    x = op.AveragePool(
        x,
        kernel_shape=kernel_shape,
        auto_pad=auto_pad,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        dilations=dilations,
        pads=pads,
        strides=strides,
    )
    if output_dim == 4:
        x = op.Identity(x)
    else:
        axes = op.Constant(value_int=0)
        x = op.Squeeze(x)
        x = op.Unsqueeze(x, axes)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidSoftmax(x: FLOAT, group: Sequence[int]) -> FLOAT:
    """Apply a (possible grouped) softmax to an input tensor and reshape it for transformer operations.

    This function works for all tensors with rank >= 2 and applies softmax along the second dimension (axis=1)
    after reshaping the input tensor to group channels together.
    The grouping allows for independent softmax computations across groups of channels, which is useful in
    transformer architectures where attention is computed separately for different heads.


    Args:
        x (FLOAT): The input tensor of shape [N, group*C, D_1, D_2, ..., D_M] to which softmax will be applied.
            Dimensions D_1, D_2, ..., D_M can be any number of additional dimensions (e.g., height and width for images).
        group(Sequence[int]): Number of groups for reshaping
    Returns:
        FLOAT: The transformed tensor after applying softmax and reshaping.
    """
    original_shape = op.Shape(x)
    # Extract constants
    batch_size = op.Gather(original_shape, op.Constant(value_ints=[0]))
    channels_times_group = op.Gather(original_shape, op.Constant(value_ints=[1]))

    # Per-group channel count: C = (C*group) / group
    channels_per_group = op.Div(channels_times_group, group)
    channels_per_group = op.Cast(channels_per_group, to=onnx.TensorProto.INT64)

    # Fold the group factor into the batch dimension so that each group
    # occupies its own independent slot along axis 0. In order to guarantee
    # uniqueness of the mapping, row-major layout guarantees that the specific
    # element [n, g*C+c, d] maps to [n*group+g, c, d] under a plain reshape.
    batch_size_times_group = op.Mul(batch_size, group)

    minus_one = op.Constant(value_ints=[-1])
    reshape_shape = op.Concat(batch_size_times_group, channels_per_group, minus_one, axis=0)

    # Reshape: [N, C*group, D_1..D_M] -> [N*group, C, D_1*...*D_M]
    x = op.Reshape(x, reshape_shape)

    # Softmax along axis=1 (per-group channels).  Because each group is
    # in a separate batch slot, groups are computed independently.
    x = op.Softmax(x, axis=1)

    # Restore the original shape: [N*group, C, D_1*...*D_M] -> [N, C*group, D_1..D_M]
    x = op.Reshape(x, original_shape)

    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidSoftmax_Mask(qk: FLOAT, heads_embed_dim: Sequence[int], mask: FLOAT) -> FLOAT:
    """
    Apply softmax to an input tensor with masking and reshape it for transformer operations.

    Args:
        qk (FLOAT): The input tensor for softmax and transformation operations.
        heads_embed_dim (Sequence[int]): Dimensions to reshape `qk` before applying softmax.
        mask (FLOAT): A masking tensor added to `qk` prior to softmax.

    Returns:
        FLOAT: The transformed tensor after applying masking, softmax, and reshaping.
    """
    unsqueeze = op.Constant(value_ints=[0])
    qk = op.Reshape(qk, heads_embed_dim)
    qk = op.Transpose(qk, perm=[0, 2, 1])
    qk = op.Unsqueeze(qk, unsqueeze)
    qk = op.Add(qk, mask)
    qk_softmax = op.Softmax(qk, axis=-1)
    qk_softmax = op.Squeeze(qk_softmax, unsqueeze)

    qk = qk_softmax
    qk_new = op.Transpose(qk, perm=[1, 0, 2])
    qk_new = op.Flatten(qk_new)
    qk_new = op.Transpose(qk_new, perm=[1, 0])
    unsqueeze = op.Constant(value_ints=[0, 2])
    qk_new = op.Unsqueeze(qk_new, unsqueeze)

    return qk_new


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidLayerNorm(
    x: FLOAT,
    scale: FLOAT,
    b: FLOAT,
    reshape_mode: str = "None",
    reshape: Sequence[int] = [0, 0, 0, 0],
    epsilon: float = 1e-6,
) -> FLOAT:
    """
    Apply layer normalization to an input tensor with scale and bias.

    Args:
        x (FLOAT): The input tensor to normalize.
        scale (FLOAT): The scale tensor used for normalization.
        b (FLOAT): The bias tensor used for normalization.
        reshape_mode (str): determines output reshaping mode.
        reshape (Sequence[int]): determines output reshape shape.
        epsilon (float): The epsilon to use to avoid zero division

    Returns:
        FLOAT: The normalized tensor after applying layer normalization.
    """
    x = op.Transpose(x, perm=(0, 2, 3, 1))
    x, _, _ = op.LayerNormalization(x, scale, b, epsilon=epsilon)
    x = op.Transpose(x, perm=(0, 3, 1, 2))
    if reshape_mode == "SWIN_RESHAPE":
        x = op.Reshape(x, reshape)
    return x


# @script(values.Opset("com.videantis", 1))  # type: ignore
# def vidNoNorm(x: FLOAT, w: FLOAT, b: FLOAT) -> FLOAT:
#     """
#     Apply nonorm normalization to an input tensor with scale and bias.

#     Args:
#         x (FLOAT): The input tensor to normalize.
#         w (FLOAT): The scale tensor used for normalization.
#         b (FLOAT): The bias tensor used for normalization.

#     Returns:
#         FLOAT: The normalized tensor after applying nonorm normalization.
#     """
#     if op.Shape(op.Shape(x)) == 4:
#         unsqueeze = op.Constant(value_ints=(0, -2, -1))
#         w = op.Unsqueeze(w, unsqueeze)
#         b = op.Unsqueeze(b, unsqueeze)
#     x = op.Mul(x, w)
#     x = op.Add(x, b)
#     return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def RMSNormalization(x: FLOAT, w: FLOAT, epsilon: float = 1e-6) -> FLOAT:
    """Normalize input tensor `x` using Root Mean Square (RMS) normalization.

    Args:
        x (FLOAT): Input tensor to be normalized.
        w (FLOAT): Weight tensor for scaling the normalized input.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        FLOAT: Normalized and scaled tensor.
    """
    x1 = op.Pow(x, 2)
    x1 = op.ReduceMean(x1, [1])
    x1 = op.Add(x1, epsilon)
    x1 = op.Sqrt(x1)
    x1 = op.Div(1, x1)
    x = op.Mul(x, x1)
    x = op.Mul(x, w)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidRMSNorm(x: FLOAT, scale: FLOAT, b: float) -> FLOAT:
    """
    Apply layer normalization to the input tensor `x` with a scale and bias.

    This function transposes `x`, Apply layer normalization using the provided `scale` and `b`,
    and then transposes it back to its original format.

    Parameters:
        x (FLOAT): The input tensor to be normalized.
        scale (FLOAT): The scale tensor used in normalization.
        b (float): The bias tensor used in normalization.

    Returns:
        FLOAT: The normalized tensor after applying layer normalization.
    """
    x = op.Transpose(x, perm=(0, 2, 3, 1))
    x = RMSNormalization(x, scale, b)
    x = op.Transpose(x, perm=(0, 3, 1, 2))
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidMultiQueryExpand(
    x: FLOAT,
    first_reshape: Sequence[INT64],
    unsqueeze_axes: Sequence[INT64],
    expand_shape: Sequence[INT64],
    reshape_shape: Sequence[INT64],
    two: INT64,
    reshape_mode: str = "None",
) -> FLOAT:
    """Expand Tensor according to "expand_shape". Used for multi-query attention.

    Args:
        x (FLOAT): The input tensor to be transformed.
        first_reshape (Sequence[INT64]): first reshape size.
        unsqueeze_axes (Sequence[INT64]): Axes along which to add new dimensions.
        expand_shape (Sequence[INT64]): Shape to which the tensor should be expanded.
        reshape_shape (Sequence[INT64]): Target shape for reshaping the tensor.
        two (INT64): just '2', needed for pattern matching.
        reshape_mode (str): controls output reshaping behaviour

    Returns:
        FLOAT: The transformed tensor after "multi-query" expansion.

    References:
        For further insights on grouped multi-query attention in large language models,
        see [LLMs Explained](https://medium.com/@pranjalkhadka/llama-explained-a70e71e706e9).

    """
    x = op.Transpose(x, perm=[0, 3, 2, 1])
    x = op.Reshape(x, first_reshape)
    x = op.Transpose(x, perm=[0, 2, 1, 3])
    x = op.Unsqueeze(x, unsqueeze_axes)
    x = op.Expand(x, expand_shape)
    x = op.Reshape(x, reshape_shape, allowzero=0)
    x = op.Transpose(x, perm=[0, 2, 1, 3])
    x = op.Flatten(x, axis=-2)
    x = op.Transpose(x, perm=[1, 0])
    x = op.Unsqueeze(x, two)
    if reshape_mode == "TRANSFORMER_QK":
        x = op.Transpose(x, perm=[0, 3, 2, 1])
        gather = op.Constant(value_ints=[0, 2, 1, 3])
        q_reshape = op.Gather(reshape_shape, gather)
        x = op.Reshape(x, q_reshape)
        x = op.Transpose(x, perm=[0, 3, 2, 1])
        x = op.Flatten(x, axis=-2)
        x = op.Unsqueeze(x, two)
        x = op.Transpose(x, perm=[3, 1, 2, 0])
    if reshape_mode == "TRANSFORMER_V":
        x = op.Transpose(x, perm=[1, 3, 0, 2])
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidFlip(
    x: FLOAT,
    start1: Sequence[INT64],
    end1: Sequence[INT64],
    axes1: Sequence[INT64],
    steps1: Sequence[INT64],
    start2: Sequence[INT64],
    end2: Sequence[INT64],
    axes2: Sequence[INT64],
    steps2: Sequence[INT64],
) -> FLOAT:
    """Flip and concatenate slices of the input tensor along specified axes.

    Args:
        x (FLOAT): Input tensor to be sliced and concatenated.
        start1 (Sequence[INT64]): Start indices for the first slice.
        end1 (Sequence[INT64]): End indices for the first slice.
        axes1 (Sequence[INT64]): Axes along which to take the first slice.
        steps1 (Sequence[INT64]): Step sizes for the first slice.
        start2 (Sequence[INT64]): Start indices for the second slice.
        end2 (Sequence[INT64]): End indices for the second slice.
        axes2 (Sequence[INT64]): Axes along which to take the second slice.
        steps2 (Sequence[INT64]): Step sizes for the second slice.

    Returns:
        FLOAT: Concatenated tensor after slicing along specified axes.
    """
    x_half1 = op.Slice(x, start1, end1, axes1, steps1)
    x_half2 = op.Slice(x, start2, end2, axes2, steps2)
    x = op.Concat(x_half1, x_half2, axis=-1)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidConv_ATTN_V(x: FLOAT, w: FLOAT, b: FLOAT) -> FLOAT:
    """
    Apply convolution to an input tensor with weights and bias, followed by transposition.

    Args:
        x (FLOAT): The input tensor.
        w (FLOAT): The convolutional weight tensor.
        b (FLOAT): The bias tensor for the convolution.

    Returns:
        FLOAT: The transformed tensor after applying convolution and transposition.
    """
    x = op.Conv(x, w, b)
    x = op.Transpose(x, perm=(1, 3, 0, 2))
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidDeconv(
    x: FLOAT,
    w: FLOAT,
    b: FLOAT,
    output_pads: Sequence[int],
    dilations: Sequence[int],
    group: int,
    kernel_shape: Sequence[int],
    pads: Sequence[int],
    strides: Sequence[int],
) -> FLOAT:
    """
    Apply a transposed convolution to an input tensor with additional output padding.

    Args:
        x (FLOAT): The input tensor.
        w (FLOAT): The weight tensor for the transposed convolution.
        b (FLOAT): The bias tensor for the transposed convolution.
        output_pads (Sequence[int]): Padding values applied to the output.
        dilations (Sequence[int]): Dilation values for the convolution.
        group (int): Number of groups for the convolution.
        kernel_shape (Sequence[int]): Shape of the convolution kernel.
        pads (Sequence[int]): Padding values for the convolution.
        strides (Sequence[int]): Stride values for the convolution.

    Returns:
        FLOAT: The tensor after applying transposed convolution and output padding.
    """
    x = op.ConvTranspose(
        x, w, b, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides
    )
    output_pads = op.Reshape(output_pads, [8])
    x = op.Pad(x, output_pads)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidConv_ATTN_QK(x: FLOAT, w: FLOAT, b: FLOAT, grp_reshape1: Sequence[int], grp_reshape2: Sequence[int]) -> FLOAT:
    """
    Apply convolution to an input tensor with weights and bias, followed by reshaping and transposition.

    Args:
        x (FLOAT): The input tensor.
        w (FLOAT): The convolutional weight tensor.
        b (FLOAT): The bias tensor for the convolution.
        grp_reshape1 (Sequence[int]): Dimensions for the first reshape operation.
        grp_reshape2 (Sequence[int]): Dimensions for the second reshape operation.

    Returns:
        FLOAT: The transformed tensor after convolution, reshaping, and transposition.
    """
    x = op.Conv(x, w)
    x = op.Add(x, b)
    x = op.Reshape(x, grp_reshape1)
    x = op.Transpose(x, perm=(1, 3, 2, 0, 4))
    x = op.Reshape(x, grp_reshape2)
    return x


@script(values.Opset("com.videantis", 1))  # type: ignore
def vidConv_ATTN_QK_s(x: FLOAT, w: FLOAT, b: FLOAT, grp_reshape1: Sequence[int], grp_reshape2: Sequence[int]) -> FLOAT:
    """
    Apply convolution to an input tensor with weights and bias, then reshape and transpose the result.

    Args:
        x (FLOAT): The input tensor.
        w (FLOAT): The convolutional weight tensor.
        b (FLOAT): The bias tensor for the convolution.
        grp_reshape1 (Sequence[int]): Dimensions for the first reshape operation.
        grp_reshape2 (Sequence[int]): Dimensions for the second reshape operation.

    Returns:
        FLOAT: The transformed tensor after convolution, reshaping, and transposition.
    """
    x = op.Conv(x, w, b)
    x = op.Reshape(x, grp_reshape1)
    x = op.Transpose(x, perm=(1, 3, 2, 0, 4))
    x = op.Reshape(x, grp_reshape2)
    return x


@script(values.Opset("com.videantis", 1))
def vidRope(x: FLOAT, rope_cos: FLOAT, rope_sin: FLOAT) -> FLOAT:
    """Apply Rotary Positional Embeddings (RoPE) to the input tensor.

    This function applies rotary positional embeddings to the input tensor `x` using the provided
    cosine (`rope_cos`) and sine (`rope_sin`) embedding tensors.

    Args:
        x (FLOAT): The input tensor to which RoPE will be applied. Shape: [1, n_roups * head_dim, 1, seq_length]
        rope_cos (FLOAT): The cosine embedding tensor. Shape: [1, head_dim, 1, seq_length]
        rope_sin (FLOAT): The sine embedding tensor. Shape: [1, head_dim, 1, seq_length]

    Returns:
        FLOAT: The tensor after applying rotary positional embeddings. Shape: [1, n_roups * head_dim, 1, seq_length]
    """
    original_shape = op.Shape(x)
    cos_shape = op.Shape(rope_cos)
    head_dim = op.Slice(cos_shape, starts=[1], ends=[2])
    tail_shape = op.Slice(original_shape, starts=[2], ends=[4])
    reshaped_x_shape = op.Concat(
        op.Constant(value_ints=[-1]),
        head_dim,
        tail_shape,
        axis=0,
    )

    reshaped_x = op.Reshape(x, reshaped_x_shape)
    # Split x into two halves along the channel dimension
    # and rotate the second half
    middle_index = op.Unsqueeze(
        op.Cast(reshaped_x_shape[1] / 2, to=onnx.TensorProto.INT64), op.Constant(value_ints=[0])
    )
    end_index = op.Unsqueeze(op.Cast(reshaped_x_shape[1], to=onnx.TensorProto.INT64), op.Constant(value_ints=[0]))
    x1 = op.Slice(reshaped_x, starts=[0], ends=middle_index, axes=[1])
    x2 = op.Slice(reshaped_x, starts=middle_index, ends=end_index, axes=[1])
    rotated_x = op.Concat(-x2, x1, axis=1)

    x_embed = (reshaped_x * rope_cos) + (rotated_x * rope_sin)

    # Reshape back to the original shape
    x_embed = op.Reshape(x_embed, original_shape)
    return x_embed


@script(values.Opset("com.videantis", 1))
def vidScatter(data: FLOAT, update: FLOAT, index: INT64) -> FLOAT:
    """Update data with update.

    This is a simplified version of the scatter operation used in the attention cache update. It updates
    the entry at data[..., index] with the provided update value. It is required to do it like this
    because TensorScatter is not yet supported in onnxruntime.

    Args:
        data (FLOAT): Shape [1, dim, 1, seq_len]
        update (FLOAT): Shape [1, dim, 1, 1]
        index (INT64): Axis index to update (single value)

    Returns:
        FLOAT: Updated data tensor.
    """
    original_shape = op.Shape(data)
    update = op.Reshape(update, shape=op.Constant(value=[-1]))
    data = op.Squeeze(data, axes=[0, 2])
    data = op.Transpose(data, perm=[1, 0])

    updated = op.ScatterND(data, indices=index, updates=update)
    updated = op.Transpose(updated, perm=[1, 0])
    updated = op.Reshape(updated, shape=original_shape)
    return updated


@script(values.Opset("com.videantis", 1))
def vidGridSample(
    x: FLOAT, grid: FLOAT, mode: str = "linear", padding_mode: str = "zeros", align_corners: bool = False
) -> FLOAT:
    """Apply grid sampling to the input tensor `x` using the provided `grid`.

    This function applies grid sampling to the input tensor `x` based on the specified sampling mode and padding mode.
    The main difference between this function and the standard `GridSample` is that it reorders the grid to match the
    expected format for `vidGridSample`.
    Args:
        x (FLOAT): The input tensor to be sampled. Shape: [N, C, H_in, W_in]
        grid (FLOAT): The sampling grid. Shape: [N, 2, H_out, W_out]
        mode (str): The interpolation mode. Options are 'linear', 'nearest', or 'cubic'. Defaults to 'linear'.
        padding_mode (str): The padding mode to use for out-of-boundary grid values. Options are 'zeros', 'border',
            or 'reflection'. Defaults to 'zeros'.
        align_corners (bool): Whether to align the corners of the input and output tensors. Defaults to False.

    Returns:
        FLOAT: The sampled output tensor. Shape: [N, C, H_out, W_out]
    """
    # Reorder grid from [B, 2, H, W] → [B, H, W, 2] to match vidGridSample expectation
    grid = op.Transpose(grid, perm=[0, 2, 3, 1])
    return op.GridSample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def build_retr_transformation(perm: list[int]) -> OnnxFunction:
    """
    Initialize and return RETRTransformation function.

    This function needs to build this way because of the Transpose layer. Transpose expected perm to be fixed
    attribute which cannot be parsed dynamically into the function.
    """

    @script(
        values.Opset("com.videantis.dynamic_functions.RETRTransformation." + "_".join(map(str, perm)), 1),
        default_opset=op,
    )
    def RETRTransformation(
        x: FLOAT, reshape1_shape: INT64, expand_shape: INT64, reshape2_shape: INT64, transpose_perm: INT64
    ) -> FLOAT:
        """Wrap Reshape->Expand->Transpose->Reshape into a single layer.

        Args:
            x (FLOAT): input tensor
            reshape1_shape (INT64): first reshape shape
            expand_shape (INT64): expand shape
            reshape2_shape (INT64): second reshape shape
            transpose_perm (INT64): transpose shape

        Returns:
            FLOAT: result tensor
        """
        x = op.Reshape(x, reshape1_shape)
        x = op.Expand(x, expand_shape)
        x = op.Transpose(x, perm=perm)
        x = op.Reshape(x, reshape2_shape)
        return x

    register_op_schema(RETRTransformation.op_schema)
    return RETRTransformation


@script(values.Opset("com.videantis", 1))
def RERTransformation(x: FLOAT, reshape1_shape: INT64, expand_shape: INT64, reshape2_shape: INT64) -> FLOAT:
    """Wrap Reshape->Expand->Reshape into a single layer.

    Args:
        x (FLOAT): input tensor
        reshape1_shape (INT64): first reshape shape
        expand_shape (INT64): expand shape
        reshape2_shape (INT64): second reshape shape

    Returns:
        FLOAT: result tensor
    """
    x = op.Reshape(x, reshape1_shape)
    x = op.Expand(x, expand_shape)
    x = op.Reshape(x, reshape2_shape)
    return x


def build_rtr_transformation(perm: list[int]) -> OnnxFunction:
    """
    Initialize and return RTRTransformation function.

    This function needs to build this way because of the Transpose layer. Transpose expected perm to be fixed
    attribute which cannot be parsed dynamically into the function.
    """

    @script(
        values.Opset("com.videantis.dynamic_functions.RTRTransformation." + "_".join(map(str, perm)), 1),
        default_opset=op,
    )
    def RTRTransformation(x: FLOAT, reshape1_shape: INT64, reshape2_shape: INT64, transpose_perm: INT64) -> FLOAT:
        """Wrap Reshape->Transpose->Reshape into a single layer.

        Args:
            x (FLOAT): input tensor
            reshape1_shape (INT64): first reshape shape
            reshape2_shape (INT64): second reshape shape
            transpose_perm (INT64): transpose shape

        Returns:
            FLOAT: result tensor
        """
        x = op.Reshape(x, reshape1_shape)
        x = op.Transpose(x, perm=perm)
        x = op.Reshape(x, reshape2_shape)
        return x

    # Check if schema is already registred and if not, do it
    register_op_schema(RTRTransformation.op_schema)
    return RTRTransformation
