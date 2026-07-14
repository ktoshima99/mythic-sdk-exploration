# type: ignore
import inspect
import logging
import math
from typing import List

import numpy as np
from onnx.numpy_helper import from_array
from onnxscript import ir
from onnxscript.rewriter.pattern import RewriteRule, RewriteRuleClassBase

np.random.seed(42)

logger = logging.getLogger(__name__)


def fetch_pattern_rules() -> List[RewriteRule]:
    """
    Fetch and process all subclasses of `PatternReplacementBase`.

    Finds all subclasses of `PatternReplacementBase`, filters and sorts them
    by their `level` attribute in descending order, and returns a list of
    rewrite rule objects created from these classes.

    Returns:
        List[RewriteRule]: A list of rewrite rule objects created from the subclasses
        of `PatternReplacementBase`, excluding `ShortcutPatternLinear`.

    """
    classes = [
        cls
        for cls in RewriteRuleClassBase.__subclasses__()
        if inspect.isclass(cls)
        and issubclass(cls, RewriteRuleClassBase)
        and "vnnort.optimizer.patterns" in cls.__module__  # Make sure to only match vnnort rules
    ]
    for cls in classes:
        if not hasattr(cls, "level") or cls.level is None:
            cls.level = 1

    sorted_classes = sorted(classes, key=lambda x: x.level, reverse=True)
    rules = []
    for cls in sorted_classes:
        if cls not in [ShortcutPatternLinear, vidConvShortcutPreFuse, vidConvShortcutPostFuse]:
            rule = cls.rule()
            rules.append(rule)

    return rules


class PartitionWindowsShifted(RewriteRuleClassBase):
    """Partition and shifting of windows from Swin."""

    @classmethod
    def pattern(
        cls,
        op,
        x,
        reshape1,
        reshape2,
        reshape3,
        pad1,
        pad2,
        starts1,
        ends1,
        axes1,
        steps1,
        starts2,
        ends2,
        axes2,
        steps2,
        starts1_1,
        ends1_1,
        axes1_1,
        starts1_2,
        ends1_2,
        axes1_2,
        starts2_1,
        ends2_1,
        axes2_1,
        starts2_2,
        ends2_2,
        axes2_2,
    ):
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
        return x

    # @classmethod
    # def check(cls, op, inputs, split, z, y):
    #     if len(split.shape) == 1:
    #         if split.shape[0] == 2:
    #             return True
    #     return False

    @classmethod
    def rewrite(
        cls,
        op,
        x,
        reshape1,
        reshape2,
        reshape3,
        pad1,
        pad2,
        starts1,
        ends1,
        axes1,
        steps1,
        starts2,
        ends2,
        axes2,
        steps2,
        starts1_1,
        ends1_1,
        axes1_1,
        starts1_2,
        ends1_2,
        axes1_2,
        starts2_1,
        ends2_1,
        axes2_1,
        starts2_2,
        ends2_2,
        axes2_2,
    ):
        x = op.vidPartitionWindowsShifted(
            x,
            reshape1,
            reshape2,
            reshape3,
            pad1,
            pad2,
            starts1,
            ends1,
            axes1,
            steps1,
            starts2,
            ends2,
            axes2,
            steps2,
            starts1_1,
            ends1_1,
            axes1_1,
            starts1_2,
            ends1_2,
            axes1_2,
            starts2_1,
            ends2_1,
            axes2_1,
            starts2_2,
            ends2_2,
            axes2_2,
            _domain="com.videantis",
            _version=1,
        )
        return x


class PartitionWindowsShiftedReverse(RewriteRuleClassBase):
    """Reverse Partition and shifting of windows from Swin."""

    @classmethod
    def pattern(
        cls,
        op,
        x,
        reshape1,
        reshape2,
        reshape3,
        starts1,
        ends1,
        axes1,
        steps1,
        starts2,
        ends2,
        axes2,
        steps2,
        starts1_1,
        ends1_1,
        axes1_1,
        starts1_2,
        ends1_2,
        axes1_2,
        starts2_1,
        ends2_1,
        axes2_1,
        starts2_2,
        ends2_2,
        axes2_2,
    ):
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
        return x

    # @classmethod
    # def check(cls, op, inputs, split, z, y):
    #     if len(split.shape) == 1:
    #         if split.shape[0] == 2:
    #             return True
    #     return False

    @classmethod
    def rewrite(
        cls,
        op,
        x,
        reshape1,
        reshape2,
        reshape3,
        starts1,
        ends1,
        axes1,
        steps1,
        starts2,
        ends2,
        axes2,
        steps2,
        starts1_1,
        ends1_1,
        axes1_1,
        starts1_2,
        ends1_2,
        axes1_2,
        starts2_1,
        ends2_1,
        axes2_1,
        starts2_2,
        ends2_2,
        axes2_2,
    ):
        x = op.vidPartitionWindowsShiftedReverse(
            x,
            reshape1,
            reshape2,
            reshape3,
            starts1,
            ends1,
            axes1,
            steps1,
            starts2,
            ends2,
            axes2,
            steps2,
            starts1_1,
            ends1_1,
            axes1_1,
            starts1_2,
            ends1_2,
            axes1_2,
            starts2_1,
            ends2_1,
            axes2_1,
            starts2_2,
            ends2_2,
            axes2_2,
            _domain="com.videantis",
            _version=1,
        )
        return x


class PartitionWindows(RewriteRuleClassBase):
    """Partition of windows from Swin."""

    @classmethod
    def pattern(
        cls,
        op,
        x,
        reshape1,
        reshape2,
        reshape3,
        pad1,
        pad2,
        starts1,
        ends1,
        axes1,
        steps1,
        starts2,
        ends2,
        axes2,
        steps2,
    ):
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
        return x

    # @classmethod
    # def check(cls, op, inputs, split, z, y):
    #     if len(split.shape) == 1:
    #         if split.shape[0] == 2:
    #             return True
    #     return False

    @classmethod
    def rewrite(
        cls,
        op,
        x,
        reshape1,
        reshape2,
        reshape3,
        pad1,
        pad2,
        starts1,
        ends1,
        axes1,
        steps1,
        starts2,
        ends2,
        axes2,
        steps2,
    ):
        x = op.vidPartitionWindows(
            x,
            reshape1,
            reshape2,
            reshape3,
            pad1,
            pad2,
            starts1,
            ends1,
            axes1,
            steps1,
            starts2,
            ends2,
            axes2,
            steps2,
            _domain="com.videantis",
            _version=1,
        )
        return x


class PartitionWindowsReverse(RewriteRuleClassBase):
    """Reverse Partition of windows from Swin."""

    @classmethod
    def pattern(
        cls,
        op,
        x,
        reshape1,
        reshape2,
        reshape3,
        starts1,
        ends1,
        axes1,
        steps1,
        starts2,
        ends2,
        axes2,
        steps2,
    ):
        x = op.Transpose(x, perm=[0, 2, 1])
        x = op.Reshape(x, reshape1)
        x = op.Transpose(x, perm=[0, 1, 3, 2, 4, 5])
        x = op.Reshape(x, reshape2)
        x = op.Slice(x, starts1, ends1, axes1, steps1)
        x = op.Slice(x, starts2, ends2, axes2, steps2)
        x = op.Reshape(x, reshape3)
        return x

    # @classmethod
    # def check(cls, op, inputs, split, z, y):
    #     if len(split.shape) == 1:
    #         if split.shape[0] == 2:
    #             return True
    #     return False

    @classmethod
    def rewrite(
        cls,
        op,
        x,
        reshape1,
        reshape2,
        reshape3,
        starts1,
        ends1,
        axes1,
        steps1,
        starts2,
        ends2,
        axes2,
        steps2,
    ):
        x = op.vidPartitionWindowsReverse(
            x,
            reshape1,
            reshape2,
            reshape3,
            starts1,
            ends1,
            axes1,
            steps1,
            starts2,
            ends2,
            axes2,
            steps2,
            _domain="com.videantis",
            _version=1,
        )
        return x


class SplitToSlice(RewriteRuleClassBase):
    """Split the onnx operator Split to 2x Slice.

    v-NN Mapper is not really capable of handling operators with multiple outputs right now. For simple cases of the split
    operator, where the input is only split into two parts (e.g. in Yolov8s), this can also be accomplished by
    the Slice operator.
    """

    @classmethod
    def pattern(cls, op, inputs, split):
        z, y = op.Split(inputs, split, axis=1, _outputs=["z", "y"])
        return z, y

    @classmethod
    def check(cls, op, inputs, split, z, y):
        # Only apply this pattern in the 1->2 split case for now
        return len(split.shape) == 1 and split.shape[0] == 2

    @classmethod
    def rewrite(cls, op, inputs: ir.Value, split: ir.Value, z, y):
        def _get_split_values(split):
            if split.const_value is not None:
                return split.const_value.numpy()

            prod = split._producer
            if prod is not None:
                # attribute called "value" that holds a TensorProto
                values = split._producer.attributes["value"]._value.raw.int64_data
                return values

        split_values = _get_split_values(split)
        z_end = int(split_values[0])
        y_end = z_end + int(split_values[1])

        z = op.Slice(inputs, op.Constant(value_ints=[0]), op.Constant(value_ints=[z_end]), op.Constant(value_ints=[1]))
        y = op.Slice(
            inputs, op.Constant(value_ints=[z_end]), op.Constant(value_ints=[y_end]), op.Constant(value_ints=[1])
        )
        return z, y


class VidReduceMeanPattern(RewriteRuleClassBase):
    """Replace ReduceMean with Average Pooling."""

    level = -1

    @classmethod
    def pattern(cls, op, x, axes):
        z = op.ReduceMean(x, axes, _outputs=["red_m_out"])
        return z

    @classmethod
    def check(cls, op, x, axes: ir.Value, red_m_out: ir.Value):
        if len(x.shape) != 4:
            logger.warning("input shape is not 4dim in pattern match for VidReduceMeanPattern")
            return False
        if axes._const_value is None:
            return False
        axes = axes.const_value.numpy().tolist()
        if axes != [2, 3]:
            logger.warning("axes is not [2,3] in match for ReduceMean")
            return False

        return True

    @classmethod
    def rewrite(cls, op, x: ir.Value, axes: ir.Value, red_m_out: ir.Value):
        kernel_shape = x.shape[-2:]
        node = red_m_out.producer()
        keepdims = node.attributes.get("keepdims", 1)
        keepdims._value = keepdims.value * 2 + 2
        z = op.vidAveragePool(
            x,
            kernel_shape=kernel_shape,
            output_dim=keepdims.value,
            _domain="com.videantis",
            _version=1,
        )
        return z


class VidGlobalAveragePoolPattern(RewriteRuleClassBase):
    """Replace Global Averaging Pooling with custom Average Pool."""

    level = 1

    @classmethod
    def pattern(cls, op, x):
        z = op.GlobalAveragePool(x)
        return z

    @classmethod
    def check(cls, op, x):
        if len(x.shape) != 4:
            logger.warning("input shape is not 4dim in pattern match for VidGlobalAveragePoolPattern")
            return False

        return True

    @classmethod
    def rewrite(cls, op, x):
        kernel_shape = x.shape[-2:]
        x = op.vidAveragePool(
            x,
            kernel_shape=kernel_shape,
            output_dim=4,
            _domain="com.videantis",
            _version=1,
        )
        return x


class VidConcatPattern(RewriteRuleClassBase):
    """Custom Vid Concat---Not used atm."""

    level = 1

    @classmethod
    def pattern(cls, op, x, y):
        z = op.Concat(x, y, axis=1, _outputs=["concat_out"])
        return z

    @classmethod
    def check(cls, op, x: ir.Value, y, concat_out: ir.Value):
        # print(concat_out.producer().inputs)
        return False

    @classmethod
    def rewrite(cls, op, x, y, concat_out: ir.Value):
        additional_inputs = concat_out.producer().inputs[2:]  # this is a list
        z = op.vidConcat(x, y, *additional_inputs, axis=1, _domain="com.videantis", _version=1)
        return z


def calculate_additional_padding(
    input_shape: tuple, kernel_shape: tuple, stride: tuple, dilation: tuple, pads: tuple
) -> list:
    """
    Calculate the additional padding needed to simulate ceil_mode=True behavior using floor_mode in a pooling layer.

    Args:
        input_shape (tuple): Tensor shape as (N, C, H, W).
        kernel_shape (tuple): Kernel size as (kernel_h, kernel_w).
        stride (tuple): Stride as (stride_h, stride_w).
        dilation (tuple): Dilation as (dilation_h, dilation_w).
        pads (tuple): Existing padding in the order (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        list: New padding [pad_top_new, pad_left_new, pad_bottom_new, pad_right_new].
    """
    # Unpack existing padding
    pad_left, pad_right, pad_top, pad_bottom = pads

    # Calculate effective kernel sizes for height and width
    effective_kernel_h = dilation[0] * (kernel_shape[0] - 1) + 1
    effective_kernel_w = dilation[1] * (kernel_shape[1] - 1) + 1

    # --- Height dimension ---
    input_h = input_shape[2]
    # Output size with ceil mode
    output_h_ceil = math.ceil((input_h + pad_top + pad_bottom - effective_kernel_h) / stride[0] + 1)
    # Total padding required (to match the ceil mode output) in height
    required_pad_h = max((output_h_ceil - 1) * stride[0] + effective_kernel_h - input_h, 0)
    # Additional padding required on height (assume extra goes to the bottom)
    extra_pad_h = required_pad_h - (pad_top + pad_bottom)
    add_pad_top = 0
    add_pad_bottom = extra_pad_h

    # --- Width dimension ---
    input_w = input_shape[3]
    output_w_ceil = math.ceil((input_w + pad_left + pad_right - effective_kernel_w) / stride[1] + 1)
    required_pad_w = max((output_w_ceil - 1) * stride[1] + effective_kernel_w - input_w, 0)
    extra_pad_w = required_pad_w - (pad_left + pad_right)
    add_pad_left = 0
    add_pad_right = extra_pad_w

    # Compute new padding values
    new_pad_top = pad_top + add_pad_top
    new_pad_bottom = pad_bottom + add_pad_bottom
    new_pad_left = pad_left + add_pad_left
    new_pad_right = pad_right + add_pad_right

    return [new_pad_top, new_pad_left, new_pad_bottom, new_pad_right]


class VidMaxPoolPattern(RewriteRuleClassBase):
    """Replace default Maxpool with custom vidMaxpool."""

    level = 1

    @classmethod
    def pattern(cls, op, x):
        z = op.MaxPool(x, _outputs=["max_out"])
        return z

    @classmethod
    def check(cls, op, x: ir.Value, max_out: ir.Value):
        # print(x)
        return True

    @classmethod
    def rewrite(cls, op, x: ir.Value, max_out: ir.Value):
        max_node = max_out.producer()

        # Retrieve attributes with default values set to None
        auto_pad = max_node.attributes.get("auto_pad", None)  # Default is None
        ceil_mode = max_node.attributes.get("ceil_mode", None)  # Default is None
        kernel_shape = max_node.attributes.get("kernel_shape", None)  # Required attribute
        dilations = max_node.attributes.get("dilations", None)  # Default is None
        pads = max_node.attributes.get("pads", None)  # Default is None
        storage_order = max_node.attributes.get("storage_order", None)  # Default is None
        strides = max_node.attributes.get("strides", None)  # Default is None
        if ceil_mode is not None:
            if ceil_mode.value == 1:
                new_padding = calculate_additional_padding(
                    x.shape, kernel_shape.value, strides.value, dilations.value, pads.value
                )
                pads._value = new_padding
                ceil_mode._value = 0

        # Ensure kernel_shape is provided since it's required
        if kernel_shape is None:
            raise ValueError("The 'kernel_shape' attribute is required but was not provided.")

        # Transform the operation using vidMaxPool
        x = op.vidMaxPool(
            x,
            kernel_shape=kernel_shape,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=dilations,
            pads=pads,
            strides=strides,
            storage_order=storage_order,
            _domain="com.videantis",
            _version=1,
        )
        return x


class VidFlattenPattern(RewriteRuleClassBase):
    """Replace default Flatten with custom vidFlatten."""

    level = 1

    @classmethod
    def pattern(cls, op, x):
        z = op.Flatten(x, axis=1)
        return z

    @classmethod
    def rewrite(cls, op, x: ir.Value):
        z = op.vidFlatten(x, axis=1, _domain="com.videantis", _version=1)
        return z


class VidAveragePoolPattern(RewriteRuleClassBase):
    """Replace default AvgPool with custom vidAvgPool."""

    level = 10

    @classmethod
    def pattern(cls, op, x):
        z = op.AveragePool(x, _outputs=["avg_out"])
        return z

    @classmethod
    def check(cls, context, *_, **__):
        return True

    @classmethod
    def rewrite(cls, op, x: ir.Value, avg_out: ir.Value):
        avg_node = avg_out.producer()
        # Retrieve attributes with default values
        auto_pad = avg_node.attributes.get("auto_pad", None)  # Default is 'NOTSET'
        ceil_mode = avg_node.attributes.get("ceil_mode", None)  # Default is 0
        count_include_pad = avg_node.attributes.get("count_include_pad", None)  # Default is 0
        kernel_shape = avg_node.attributes.get("kernel_shape", None)  # Required attribute, should be checked separately
        dilations = avg_node.attributes.get("dilations", None)  # Defaults to 1 along each spatial axis
        pads = avg_node.attributes.get("pads", None)  # Defaults to 0 along each spatial axis
        strides = avg_node.attributes.get("strides", None)  # Defaults to 1 along each spatial axis

        # Ensure kernel_shape is provided since it's required
        if kernel_shape is None:
            raise ValueError("The 'kernel_shape' attribute is required but was not provided.")
        x = op.vidAveragePool(
            x,
            kernel_shape=kernel_shape,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            dilations=dilations,
            pads=pads,
            strides=strides,
            output_dim=4,
            _domain="com.videantis",
            _version=1,
        )
        return x


class ShortcutMulPatternLinear(RewriteRuleClassBase):
    """Replace Mul operator with multiplicate Shortcut operator."""

    level = 1

    @classmethod
    def pattern(cls, op, x, y):
        z = op.Mul(x, y, _outputs=["shortcut_out"])
        return z

    @classmethod
    def check(cls, op, x, y, shortcut_out: ir.Value):
        if x.producer() is None and y.producer() is None:
            return False
        elif x.shape is None or y.shape is None:
            return False
        elif len(x.shape) != 4 or len(y.shape) != 4:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x: ir.Value, y: ir.Value, shortcut_out: ir.Value):
        z = op.Shortcut(
            x,
            y,
            mode="multiplication",
            reshape_mode="None",
            _domain="com.videantis",
            _version=1,
            _outputs=["shortcut_linear" + str(np.random.randint(1e10))],
        )
        return z


class ShortcutPatternLinear(RewriteRuleClassBase):
    """Replace Add with Shortcut."""

    level = 1

    @classmethod
    def pattern(cls, op, x, y):
        z = op.Add(x, y, _outputs=["shortcut_out"])
        return z

    @classmethod
    def check(cls, op, x, y, shortcut_out: ir.Value):
        if x.producer() is None and y.producer() is None:
            logger.debug(f"ShortcutPatternLinear: producer is none {x.producer()} , {y.producer()}")
            return False
        elif x.shape is None:
            logger.debug("ShortcutPatternLinear: shape is None")
            return False
        elif len(x.shape) != 4:
            logger.debug(" ShortcutPatternLinear: shape is not 4dim")
            return False
        return True

    @classmethod
    def rewrite(cls, op, x: ir.Value, y: ir.Value, shortcut_out: ir.Value):
        z = op.Shortcut(
            x,
            y,
            reshape_mode="None",
            _domain="com.videantis",
            _version=1,
            _outputs=[
                "shortcut_linear" + str(np.random.randint(1e10)),
            ],
        )
        return z


class ConvBatchNormFuse(RewriteRuleClassBase):
    """Fuse BatchNorm into Conv."""

    level = 2

    @classmethod
    def pattern(cls, op, x, w, b, mul, dilations, group, kernel_shape, pads, strides):
        x = op.Conv(x, w, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides)
        x = op.Mul(x, mul)
        x = op.Add(x, b)
        return x

    @classmethod
    def rewrite(cls, op, x, w, b, mul, dilations, group, kernel_shape, pads, strides):
        mul = op.Transpose(mul, perm=[1, 0, 2, 3])
        w = op.Mul(w, mul)
        b = op.Squeeze(b)
        x = op.Conv(x, w, b, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides)
        return x


class ConvParallelBatchNorm(RewriteRuleClassBase):
    """Fuse BatchNorm into Conv for y = BN(x) + Conv(x)."""

    level = 3

    @classmethod
    def pattern(cls, op, x, w, b, scale, B, input_mean, input_var):
        x1 = op.BatchNormalization(x, scale, B, input_mean, input_var)
        x2 = op.Conv(x, w, b)
        y = op.Add(x1, x2)
        return y

    @classmethod
    def check(cls, context, x, w, b, scale, B, input_mean, input_var):
        # Extract relevant conv node from context
        conv_node = [c for c in context.nodes if c.op_type == "Conv"][0]

        # Extract relevant attributes
        kh, kw = conv_node.attributes.get("kernel_shape", [1, 1]).value
        dilations = conv_node.attributes.get("dilations", [1, 1]).value
        group = conv_node.attributes.get("group", 1).value
        batch_norm_channels = scale.shape[0]
        weight_channels = w.shape[1]

        # This pattern is only applicable for:
        # 1. uneven kernel sizes
        # 2. undilated convolutions
        # 3. #input_channels = #output_channels (our grouped input_channels)
        if (
            kh % 2 == 0
            or kw % 2 == 0
            or dilations[0] != 1
            or dilations[1] != 1
            or batch_norm_channels != weight_channels * group
        ):
            return False
        return True

    @classmethod
    def rewrite(cls, op, x, w, b, scale, B, input_mean, input_var):
        # Extract numpy values
        s = scale.const_value.numpy()
        rv = input_var.const_value.numpy()
        rm = input_mean.const_value.numpy()
        beta = B.const_value.numpy()
        weights = w.const_value.numpy().copy()

        # Handle optional bias
        bias = b.const_value.numpy() if b is not None else np.zeros(weights.shape[0], dtype=weights.dtype)

        eps = 1e-5

        # Calculate BN transformation components
        alpha = s / np.sqrt(rv + eps)
        offset = beta - (rm * alpha)

        # Fuse Identity into Weight
        # Add alpha to the center pixel of the "identity" channel
        out_chs, in_chs, kh, kw = weights.shape
        mid_h, mid_w = kh // 2, kw // 2

        for i in range(out_chs):
            weights[i, i % in_chs, mid_h, mid_w] += alpha[i]

        # Fuse Bias
        new_bias = bias + offset

        # Extract attributes from the matched node
        conv_node = w.uses()[0].node

        return op.Conv(
            x,
            op.Constant(value=from_array(weights)),
            op.Constant(value=from_array(new_bias)),
            **conv_node.attributes,
        )


class ParallelConvFuse(RewriteRuleClassBase):
    """Fuses parallel Conv2D layers whose outputs are combined by elementwise addition into a single, mathematically equivalent Conv2D.

    The fusion is applied only if all convolutions:
    - are 2D convolutions
    - use odd-valued kernel sizes
    - have identical strides
    - have identical dilations
    - have identical group counts (in case of grouped convolutions)
    """

    level = 4

    @classmethod
    def pattern(cls, op, x, w1, b1, w2, b2):
        y1 = op.Conv(x, w1, b1)
        y2 = op.Conv(x, w2, b2)
        result = op.Add(y1, y2)
        return result

    @classmethod
    def check(cls, context, x, w1, b1, w2, b2):
        # Check that constraints defined above are fulfilled
        conv1_node, conv2_node = [c for c in context.nodes if c.op_type == "Conv"]
        if len(w1.shape) != 4 or len(w2.shape) != 4:
            return False

        # Even kernels are not supported
        if any(k % 2 == 0 for k in w1.shape[2:] + w2.shape[2:]):
            return False

        # Strides
        strides1 = conv1_node.attributes.get("strides", [1, 1])
        strides2 = conv2_node.attributes.get("strides", [1, 1])

        if strides1 != strides2:
            return False

        dilations1 = conv1_node.attributes.get("dilations", [1, 1])
        dilations2 = conv2_node.attributes.get("dilations", [1, 1])

        if dilations1 != dilations2:
            return False

        # Check groups (critical for depthwise convs)
        group1 = conv1_node.attributes.get("group", 1)
        group2 = conv2_node.attributes.get("group", 1)

        if group1 != group2:
            return False

        return True

    @classmethod
    def rewrite(cls, op, x, w1, b1, w2, b2):
        # Extract constant values
        v1 = w1.const_value
        v2 = w2.const_value

        # Determine largest spatial dimensions
        h1, w1_size = v1.shape[2:]
        h2, w2_size = v2.shape[2:]

        target_h = max(h1, h2)
        target_w = max(w1_size, w2_size)

        def pad_kernel(kernel, target_h, target_w):
            kh, kw = kernel.shape[2:]
            if kh == target_h and kw == target_w:
                return kernel
            # Calculate padding for the center
            pad_h = (target_h - kh) // 2
            pad_w = (target_w - kw) // 2
            return np.pad(kernel, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")

        # Align spatial dimensions
        v1_padded = pad_kernel(v1, target_h, target_w)
        v2_padded = pad_kernel(v2, target_h, target_w)

        # Fuse weights and biases
        w_fused = from_array((v1_padded + v2_padded).astype(np.float32))
        b_fused = from_array((b1.const_value.numpy() + b2.const_value.numpy()).astype(np.float32))

        # Extract target attributes
        dilations = w1.uses()[0].node.attributes["dilations"]
        group = w1.uses()[0].node.attributes["group"]
        strides = w1.uses()[0].node.attributes["strides"]
        if h1 >= h2 and w1_size >= w2_size:
            pads = w1.uses()[0].node.attributes["pads"]
        else:
            pads = w2.uses()[0].node.attributes["pads"]

        return op.Conv(
            x,
            op.Constant(value=w_fused),
            op.Constant(value=b_fused),
            dilations=dilations,
            group=group,
            strides=strides,
            kernel_shape=(target_h, target_w),
            pads=pads,
        )


class SwishPattern(RewriteRuleClassBase):
    """Matches Swish Pattern (sigmoid(x)*x) to Swish operator."""

    level = 12

    @classmethod
    def pattern(cls, op, x):
        x1 = op.Sigmoid(x)
        x = op.Mul(x, x1)
        return x

    @classmethod
    def rewrite(cls, op, x):
        x = op.Swish(x, _domain="com.videantis", _version=1)
        return x


class MishPattern(RewriteRuleClassBase):
    """Matches Mish Pattern Mish operator."""

    level = 12

    @classmethod
    def pattern(cls, op, x, one):
        x1 = op.Exp(x)
        x1 = op.Add(one, x1)
        x1 = op.Log(x1)
        x1 = op.Tanh(x1)
        x = op.Mul(x, x1)
        return x

    @classmethod
    def rewrite(cls, op, x, one):
        x = op.Mish(x, _domain="com.videantis", _version=1)
        return x


class MishPattern2(RewriteRuleClassBase):
    """Matches Mish Pattern Mish operator."""

    level = 12

    @classmethod
    def pattern(cls, op, x):
        x1 = op.Softplus(x)
        x1 = op.Tanh(x1)
        x = op.Mul(x, x1)
        return x

    @classmethod
    def rewrite(cls, op, x):
        x = op.Mish(x, _domain="com.videantis", _version=1)
        return x


class MultiQueryExpand(RewriteRuleClassBase):
    """Multi Query Expand for Transformers."""

    level = 8

    @classmethod
    def pattern(cls, op, x, unsqueeze_axes, expand_shape, reshape_shape):
        x = op.Unsqueeze(x, unsqueeze_axes)
        x = op.Expand(x, expand_shape)
        x = op.Reshape(x, reshape_shape, allowzero=0)
        return x

    @classmethod
    def check(cls, op, x, unsqueeze_axes, expand_shape, reshape_shape):
        # print("found multiqueryexpand")
        return False

    @classmethod
    def rewrite(cls, op, x, unsqueeze_axes, expand_shape, reshape_shape):
        x = op.vidMultiQueryExpand(x, unsqueeze_axes, expand_shape, reshape_shape, _domain="com.videantis", _version=1)
        return x


# class MulConvReplace(RewriteRuleClassBase):
#     level = 0

#     @classmethod
#     def pattern(cls, op, x, y):
#         x = op.Mul(x, y)
#         return x

#     @classmethod
#     def check(cls, op, x, y):
#         if x is not None and y is not None:
#             if x.shape is not None and y.shape is not None:
#                 if len(x.shape) == 4 and len(y.shape) == 4:
#                     if x.shape[0] == x.shape[2] == x.shape[3] == 1:
#                         if x.shape[1] == y.shape[1]:
#                             if x.producer() is not None and y.producer() is not None:
#                                 return True
#         return False

#     @classmethod
#     def rewrite(cls, op, x, y):
#         x = op.reshapeToWgtsMulExpand(x, _domain="com.videantis", _version=1)
#         x = op.Conv(y, x)
#         return x


class GemmConvReplace2D(RewriteRuleClassBase):
    """Replaces Gemm(=Matmul) with vidConv operator."""

    level = 0

    @classmethod
    def pattern(cls, op, x, w, b):
        x = op.Gemm(x, w, b, transB=1)
        return x

    @classmethod
    def check(cls, op, x, w, b):
        if x.shape is not None:
            if len(x.shape) == 2:
                return True
        return False

    @classmethod
    def rewrite(cls, op, x, w, b):
        x = op.vidConv(x, w, b, dim=2, _domain="com.videantis", _version=1)
        return x


class ConvToVidConvNoBias(RewriteRuleClassBase):
    """Replace Conv without bias with vidConv."""

    level = -1

    @classmethod
    def pattern(cls, op, x, w):
        x = op.Conv(x, w)
        return x

    @classmethod
    def check(cls, op, x, w, b=None):
        # conv_node: ir.Node = list(w.uses())[0][0]
        # group = conv_node.attributes.get("group", None)
        if x.shape is not None:
            if len(x.shape) == 4:
                return True
        return False

    @classmethod
    def rewrite(cls, op, x, w, b=None):
        conv_node = list(w.uses())[0][0]
        dilations = conv_node.attributes.get("dilations", None)
        group = conv_node.attributes.get("group", None)
        pads = conv_node.attributes.get("pads", None)
        strides = conv_node.attributes.get("strides", None)

        # Extract kernel shape from weight to make sure it matches the actual kernel size. (If it is an initializer)
        if w.shape is not None:
            kernel_shape = w.shape[2:]
        else:
            kernel_shape = conv_node.attributes.get("kernel_shape", None)

        p = op.vidConv(
            x,
            w,
            b,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            dim=4,
            _domain="com.videantis",
            _version=1,
            _outputs=["x" + str(np.random.randint(1e10))],
        )
        return p


class ConvToVidConv(RewriteRuleClassBase):
    """Replaces Conv operator with vidConv operator."""

    level = 0

    @classmethod
    def pattern(cls, op, x, w, b):
        x = op.Conv(x, w, b)
        return x

    @classmethod
    def check(cls, op, x, w, b=None):
        # conv_node: ir.Node = list(w.uses())[0][0]
        # group = conv_node.attributes.get("group", None)
        if x.shape is not None:
            if len(x.shape) == 4:
                return True
        return False

    @classmethod
    def rewrite(cls, op, x, w, b=None):
        conv_node = list(w.uses())[0][0]
        auto_pad = conv_node.attributes.get("auto_pad", None)
        dilations = conv_node.attributes.get("dilations", None)
        group = conv_node.attributes.get("group", None)
        pads = conv_node.attributes.get("pads", None)
        strides = conv_node.attributes.get("strides", None)

        # Extract kernel shape from weight to make sure it matches the actual kernel size. (If it is an initializer)
        if w.shape is not None:
            kernel_shape = w.shape[2:]
        else:
            kernel_shape = conv_node.attributes.get("kernel_shape", None)

        p = op.vidConv(
            x,
            w,
            b,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            auto_pad=auto_pad,
            dim=4,
            _domain="com.videantis",
            _version=1,
            # _outputs=["x" + str(np.random.randint(1e10))],
        )
        return p


class ConvTransposePadFuse(RewriteRuleClassBase):
    """Replaces ConvTranspose+Pad with ConvTransposeOutputPadding."""

    # ! TODO: Rename ConvTransposeOutputPadding to vidConvTranspose with attributes for padding

    @classmethod
    def pattern(cls, op, x, w, b, p, constant_value=None):
        x = op.ConvTranspose(x, w, b)
        x = op.Pad(x, p, constant_value, _outputs=["result"])
        return x

    @classmethod
    def check(cls, context, *_, **__):
        return super().check(context, *_, **__)

    @classmethod
    def rewrite(cls, op, x, w, b, p: ir.Value, constant_value, result: ir.Value):
        pad_node = result.producer()
        conv_node = pad_node.inputs[0].producer()
        dilations = conv_node.attributes.get("dilations", None)
        group = conv_node.attributes.get("group", None)
        kernel_shape = conv_node.attributes.get("kernel_shape", None)
        pads = conv_node.attributes.get("pads", None)
        # output_padding = [int(x) for x in p.const_value.numpy()]
        strides = conv_node.attributes.get("strides", None)
        p = p.const_value.numpy().astype(np.int64).tolist()
        x = op.vidDeconv(
            x,
            w,
            b,
            output_pads=p,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            _domain="com.videantis",
            _version=1,
        )
        return x


class FFNTransformer(RewriteRuleClassBase):
    """Matches Relu FFN to Conv schema."""

    level = 6

    @classmethod
    def pattern(cls, op, x, W1, B1, W2, B2):
        x = op.MatMul(x, W1)
        x = op.Add(B1, x)
        x = op.Relu(x)
        x = op.MatMul(x, W2)
        x = op.Add(B2, x)
        return x

    @classmethod
    def check(cls, op, x, W1, B1, W2, B2):
        if len(x.shape) == 3:
            return True
        else:
            return False

    @classmethod
    def rewrite(cls, op, x, W1, B1, W2, B2):
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        W1 = op.Transpose(W1, perm=[1, 0])
        W1 = op.Unsqueeze(W1, wgt_unsqueeze)
        W1 = op.Identity(W1)

        W2 = op.Transpose(W2, perm=[1, 0])
        W2 = op.Unsqueeze(W2, wgt_unsqueeze)
        W2 = op.Identity(W2)

        unsqueeze = op.Constant(value_ints=[0])
        x = op.Transpose(x, perm=[2, 1, 0])
        x = op.Unsqueeze(x, unsqueeze)

        x = op.Conv(x, W1, B1)
        x = op.Relu(x)
        x = op.Conv(x, W2, B2)
        x = op.Transpose(x, perm=[0, 3, 2, 1])
        x = op.Squeeze(x, unsqueeze)
        return x


class FFNTransformerGELU(RewriteRuleClassBase):
    """Matches Gelu FFN to Conv schema."""

    level = 6

    @classmethod
    def pattern(cls, op, x, W1, B1, W2, B2):
        x = op.MatMul(x, W1)
        x = op.Add(x, B1)
        x = op.Gelu(x)
        x = op.MatMul(x, W2)
        x = op.Add(x, B2)
        return x

    @classmethod
    def check(cls, op, x, W1, B1, W2, B2):
        if len(x.shape) == 3:
            return True
        else:
            return False

    @classmethod
    def rewrite(cls, op, x, W1, B1, W2, B2):
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        W1 = op.Transpose(W1, perm=[1, 0])
        W1 = op.Unsqueeze(W1, wgt_unsqueeze)
        W1 = op.Identity(W1)

        W2 = op.Transpose(W2, perm=[1, 0])
        W2 = op.Unsqueeze(W2, wgt_unsqueeze)
        W2 = op.Identity(W2)

        unsqueeze = op.Constant(value_ints=[0])
        x = op.Transpose(x, perm=[2, 1, 0])
        x = op.Unsqueeze(x, unsqueeze)

        x = op.Conv(x, W1, B1)
        x = op.Gelu(x)
        x = op.Conv(x, W2, B2)
        x = op.Transpose(x, perm=[0, 3, 2, 1])
        x = op.Squeeze(x, unsqueeze)
        return x


class FFNTransformerGELUConvNext(RewriteRuleClassBase):
    """Matches LN and GELU FFN to Conv schema."""

    level = 6

    @classmethod
    def pattern(cls, op, x, W1, B1, W2, B2, scale, b):
        x = op.Transpose(x, perm=[0, 2, 3, 1])
        x = op.LayerNormalization(x, scale, b)
        x = op.MatMul(x, W1)
        x = op.Add(x, B1)
        x = op.Gelu(x)
        x = op.MatMul(x, W2)
        x = op.Add(x, B2)
        x = op.Transpose(x, perm=[0, 3, 1, 2])
        return x

    @classmethod
    def check(cls, op, x, W1, B1, W2, B2, scale, b):
        if len(x.shape) == 4:
            return True
        else:
            return False

    @classmethod
    def rewrite(cls, op, x, W1, B1, W2, B2, scale, b):
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        W1 = op.Transpose(W1, perm=[1, 0])
        W1 = op.Unsqueeze(W1, wgt_unsqueeze)
        W1 = op.Identity(W1)

        W2 = op.Transpose(W2, perm=[1, 0])
        W2 = op.Unsqueeze(W2, wgt_unsqueeze)
        W2 = op.Identity(W2)

        x = op.vidLayerNorm(x, scale, b, _domain="com.videantis", _version=1)
        x = op.Conv(x, W1, B1)
        x = op.Gelu(x)
        x = op.Conv(x, W2, B2)
        return x


class LlamaRotaryPositionalEmbeddings(RewriteRuleClassBase):
    """Rotary Positional Embeddings for Llama."""

    level = 3

    @classmethod
    def pattern(cls, op, x, start1, end1, axes1, steps1, start2, end2, axes2, steps2, mul1, mul2):
        x_half1 = op.Slice(x, start1, end1, axes1, steps1)
        x_half1 = op.Neg(x_half1)

        x_half2 = op.Slice(x, start2, end2, axes2, steps2)
        x1 = op.Concat(x_half1, x_half2, axis=-1)
        x1 = op.Mul(x1, mul1)

        x2 = op.Mul(x, mul2)

        x = op.Add(x2, x1)
        return x

    @classmethod
    def check(cls, op, x, start1, end1, axes1, steps1, start2, end2, axes2, steps2, mul1, mul2):
        return True

    @classmethod
    def rewrite(cls, op, x, start1, end1, axes1, steps1, start2, end2, axes2, steps2, mul1, mul2):
        mul_half1 = op.Slice(mul1, start1, end1, axes1, steps1)
        mul_half1 = op.Neg(mul_half1)

        mul_half2 = op.Slice(mul1, start2, end2, axes2, steps2)
        mul1 = op.Concat(mul_half1, mul_half2, axis=-1)
        x1 = op.vidFlip(
            x, start1, end1, axes1, steps1, start2, end2, axes2, steps2, _domain="com.videantis", _version=1
        )
        x1 = op.Mul(x1, mul1)

        x2 = op.Mul(x, mul2)

        x = op.Add(x2, x1)
        return x


class ElemLayerNormPattern(RewriteRuleClassBase):
    """Matches elementary LN to LN operator."""

    level = 2

    @classmethod
    def pattern(cls, op, x, axes1, axes2, epsilon, pow, w, b):
        x1 = op.ReduceMean(x, axes1)
        x = op.Sub(x, x1)
        x1 = op.Pow(x, pow)
        x1 = op.ReduceMean(x1, axes2)
        x1 = op.Add(x1, epsilon)
        x1 = op.Sqrt(x1)
        x = op.Div(x, x1)
        x = op.Mul(x, w)
        x = op.Add(x, b)
        return x

    @classmethod
    def rewrite(cls, op, x, axes1, axes2, epsilon, pow, w, b):
        x = op.LayerNormalization(x, w, b, epsilon=1e-6)
        return x


class ElemLayerNormPatternNoEpsilon(RewriteRuleClassBase):
    """Matches elementary pattern without epsilon to LN operator with very low epsilon (1e-12)."""

    level = 2

    @classmethod
    def pattern(cls, op, x, axes1, axes2, pow, w, b):
        x1 = op.ReduceMean(x, axes1)
        x = op.Sub(x, x1)
        x1 = op.Pow(x, pow)
        x1 = op.ReduceMean(x1, axes2)
        # x1 = op.Add(x1,epsilon)
        x1 = op.Sqrt(x1)
        x = op.Div(x, x1)
        x = op.Mul(x, w)
        x = op.Add(x, b)
        return x

    @classmethod
    def rewrite(cls, op, x, axes1, axes2, pow, w, b):
        x = op.LayerNormalization(x, w, b, epsilon=1e-12)
        return x


class ElemLayerNormNoWeightsPattern(RewriteRuleClassBase):
    """Matches elementary pattern without weights to LN operator with weights set to 1."""

    @classmethod
    def pattern(cls, op, x, axes1, axes2, epsilon, pow):
        x1 = op.ReduceMean(x, axes1)
        x = op.Sub(x, x1)
        x1 = op.Pow(x, pow)
        x1 = op.ReduceMean(x1, axes2)
        x1 = op.Add(x1, epsilon)
        x1 = op.Sqrt(x1)
        x = op.Div(x, x1)
        return x

    @classmethod
    def rewrite(cls, op, x, axes1, axes2, epsilon, pow):
        x_orig = x
        c = op.Shape(x_orig)
        neg_one = op.Constant(value_ints=[-1])
        c = op.Gather(c, neg_one)
        # zero = op.Constant(value_ints=[0])
        # c = op.Unsqueeze(c,zero)

        one = op.Constant(value_floats=[1.0])
        ones = op.Expand(one, c)
        zero = op.Constant(value_floats=[0.0])
        zeros = op.Mul(ones, zero)
        x3 = op.LayerNormalization(x_orig, ones, zeros, epsilon=1e-6)
        # x3 = op.Identity(x3)
        return x3


class ConvNoNormFuse(RewriteRuleClassBase):
    """Fuses vidNoNorm into vidConv operator."""

    level = 2  # Set priority level for pattern matching

    @classmethod
    def pattern(cls, op, x, w, b, axes, nonorm_weights, nonorm_biases):
        # Match the pattern: vidConv->Squeeze->Flatten->vidNoNorm
        x = op.vidConv(x, w, b, _domain="com.videantis", _outputs=["conv_out"])
        x = op.Squeeze(x, axes)
        x = op.Transpose(x, perm=[0, 2, 1])
        x = op.vidNoNorm(x, nonorm_weights, nonorm_biases, _domain="com.videantis")
        return x

    @classmethod
    def check(cls, op, x, w, b, axes, nonorm_weights, nonorm_biases, conv_out: ir.Value):

        # Ensure shapes are compatible for fusion
        if w.shape is None or nonorm_weights.shape is None:
            return False

        # Check if the NoNorm weights can be incorporated into Conv weights
        if len(nonorm_weights.shape) != 1 or len(nonorm_biases.shape) != 1:
            return False

        # Verify the Conv output channels match NoNorm input size
        conv_node = conv_out.producer()
        if conv_node is None:
            return False

        # Check channel dimensions match
        if conv_out.shape.dims[1] != nonorm_weights.shape[0] or conv_out.shape.dims[1] != nonorm_biases.shape[0]:
            return False

        return True

    @classmethod
    def rewrite(cls, op, x, w, b, axes, nonorm_weights, nonorm_biases, conv_out: ir.Value):
        # Get original conv attributes
        conv_node = conv_out.producer()
        dilations = conv_node.attributes.get("dilations", None)
        group = conv_node.attributes.get("group", None)
        kernel_shape = conv_node.attributes.get("kernel_shape", None)
        pads = conv_node.attributes.get("pads", None)
        strides = conv_node.attributes.get("strides", None)

        # Reshape NoNorm weights to match Conv weights shape
        unsqueeze = op.Constant(value_ints=(-3, -2, -1))
        nonorm_weights_expanded = op.Unsqueeze(nonorm_weights, unsqueeze)

        # Fuse weights: Multiply Conv weights with NoNorm weights
        new_w = op.Mul(w, nonorm_weights_expanded)

        # Fuse biases: Multiply Conv bias with NoNorm weights and add NoNorm bias
        if b is not None:
            new_b = op.Add(op.Mul(b, nonorm_weights), nonorm_biases)
        else:
            new_b = nonorm_biases

        # Create new fused vidConv with updated weights and biases
        x = op.vidConv(
            x,
            new_w,
            new_b,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            dim=4,
            _domain="com.videantis",
            _version=1,
            _outputs=["x" + str(np.random.randint(1e10))],
        )

        # Apply the same reshape operations as in the original pattern
        x = op.Squeeze(x, axes)
        x = op.Transpose(x, perm=[0, 2, 1])

        return x


class IdentityPattern(RewriteRuleClassBase):
    """Removes Identity operator from graph."""

    @classmethod
    def pattern(cls, op, x):
        x = op.Identity(x)
        return x

    @classmethod
    def check(cls, context, *_, **__) -> bool:
        return False

    @classmethod
    def rewrite(cls, op, x):
        return x


class TransposeSplit(RewriteRuleClassBase):
    """Fuses double transpose layer from two pathes into single transpose layer in one path."""

    # This pattern seems to be only used for Bosch Convnext.
    # It is really brittle, but it needs to be applied AFTER other patterns involving transposes
    # Therefore: Give it low priority.
    # TODO: Think of better concept to do this
    level = 15

    @classmethod
    def pattern(cls, op, x):
        x1 = op.Transpose(x, perm=[0, 3, 1, 2])
        x2 = op.Transpose(x, perm=[0, 3, 1, 2])
        return x1, x2

    @classmethod
    def check(cls, context, x, *args, **kwargs) -> bool:
        if len(context.nodes) != 2:
            return False

        # Prevent this from going into an infinite loop
        if context.nodes[0] == context.nodes[1]:
            return False
        return True

    @classmethod
    def rewrite(cls, op, x):
        x = op.Transpose(x, perm=[0, 3, 1, 2])
        return x, x


class MatMulConvPattern4Dim(RewriteRuleClassBase):
    """Matches Matmul + Add to Conv with bias if Add is 1D and input is 4D."""

    @classmethod
    def pattern(cls, op, x, w, b):
        y = op.MatMul(x, w)
        out = op.Add(b, y)
        return out

    @classmethod
    def check(cls, context, x, w, b):
        if len(x.shape) == 4 and len(b.shape) == 1:
            return True
        else:
            return False

    @classmethod
    def rewrite(cls, op, x, w, b):
        x = op.Transpose(x, perm=(0, 3, 1, 2))
        w = op.Transpose(w, perm=(1, 0))
        unsqueeze = op.Constant(value_ints=(-1, -2))
        w = op.Unsqueeze(w, unsqueeze)
        x = op.Conv(x, w, b)
        x = op.Transpose(x, perm=(0, 2, 3, 1))
        return x


class MatMulConvPattern3Dim(RewriteRuleClassBase):
    """Matches Matmul + Add to Conv with bias if Add is 1D and input is 3D."""

    level = 0

    @classmethod
    def pattern(cls, op, x, w, b):
        y = op.MatMul(x, w)
        out = op.Add(b, y)
        return out

    @classmethod
    def check(cls, context, x, w, b):
        if len(x.shape) == 3 and x.producer() is not None and len(b.shape) == 1:
            return True
        else:
            return False

    @classmethod
    def rewrite(cls, op, x, w, b):
        x = op.Transpose(x, perm=(0, 2, 1))
        unsqueeze = op.Constant(value_ints=[-2])
        x = op.Unsqueeze(x, unsqueeze)

        w = op.Transpose(w, perm=(1, 0))
        unsqueeze = op.Constant(value_ints=[-1, -2])
        w = op.Unsqueeze(w, unsqueeze)
        x = op.Conv(x, w, b)
        squeeze = op.Constant(value_ints=[-2])
        x = op.Squeeze(x, squeeze)
        x = op.Transpose(x, perm=(0, 2, 1))

        return x


class MatMulConvPattern3DimNoBias(RewriteRuleClassBase):
    """Matches Matmul + Add to Conv without bias if Add is 1D and input is 3D."""

    level = 0

    @classmethod
    def pattern(cls, op, x, w):
        y = op.MatMul(x, w)
        return y

    @classmethod
    def check(cls, context, x, w):
        return False

    @classmethod
    def rewrite(cls, op, x, w):
        x = op.Transpose(x, perm=(0, 2, 1))
        unsqueeze = op.Constant(value_ints=[-1])
        x = op.Unsqueeze(x, unsqueeze)

        w = op.Transpose(w, perm=(1, 0))
        unsqueeze = op.Constant(value_ints=[-1, -2])
        w = op.Unsqueeze(w, unsqueeze)
        x = op.Conv(x, w)
        squeeze = op.Constant(value_ints=[3])
        x = op.Squeeze(x, squeeze)
        x = op.Transpose(x, perm=(0, 2, 1))

        return x


class MatMulConvPattern2Dim(RewriteRuleClassBase):
    """Matches Matmul + Add to Conv with bias if Add is 1D and input is 2D."""

    @classmethod
    def pattern(cls, op, x, w, b):
        y = op.MatMul(x, w)
        out = op.Add(b, y)
        return out

    @classmethod
    def check(cls, context, x, w, b):
        if len(x.shape) == 2:
            return True
        else:
            return False

    @classmethod
    def rewrite(cls, op, x, w, b):
        w = op.Transpose(w, perm=(1, 0))
        unsqueeze = op.Constant(value_ints=(-1, -2))
        w = op.Unsqueeze(w, unsqueeze)
        x = op.Unsqueeze(x, unsqueeze)
        x = op.Conv(x, w, b)
        x = op.Squeeze(x, unsqueeze)
        return x


class VidConvwReshape(RewriteRuleClassBase):
    """Matches Conv+Reshape to Conv with specific reshape_mode."""

    @classmethod
    def pattern(cls, op, x, weights, biases, reshape_shape):
        x = op.vidConv(x, weights, biases, _domain="com.videantis")
        x = op.Reshape(x, reshape_shape)
        return x

    @classmethod
    def check(cls, op, x, weights, biases, reshape_shape):
        lst = reshape_shape.const_value.numpy().tolist()
        return len(lst) == 4 and lst[0] == 1 and lst[2] == 1 and lst[3] == -1

    @classmethod
    def rewrite(cls, op, x, weights, biases, reshape_shape):
        conv_node = list(weights.uses())[0][0]
        dilations = conv_node.attributes.get("dilations", None)
        group = conv_node.attributes.get("group", None)
        kernel_shape = conv_node.attributes.get("kernel_shape", None)
        pads = conv_node.attributes.get("pads", None)
        strides = conv_node.attributes.get("strides", None)

        x = op.vidConv(
            x,
            weights,
            biases,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            dim=4,
            reshape_mode="FLATTEN_W",
            _domain="com.videantis",
            _version=1,
            _outputs=["x" + str(np.random.randint(1e10))],
        )
        return x


class TruncatedAttentionDetr(RewriteRuleClassBase):
    """Matches truncated (first layer of decoder block) Detr block to vidConv schema."""

    level = 6

    @classmethod
    def pattern(cls, op, x, pos_enc_bias_k, q, K_W, K_B, k_reshape, V_W, V_B, v_reshape, out_reshape, O_W, O_B, bias):
        x1 = op.Add(x, pos_enc_bias_k)
        k = op.MatMul(x1, K_W)
        k = op.Add(K_B, k)
        k = op.Reshape(k, k_reshape)
        k = op.Transpose(k, perm=[1, 2, 0])

        qk = op.MatMul(q, k)
        qk = op.Softmax(qk, axis=-1)

        v = op.MatMul(x, V_W)
        v = op.Add(V_B, v)
        v = op.Reshape(v, v_reshape)
        v = op.Transpose(v, perm=[1, 0, 2])

        qkv = op.MatMul(qk, v)
        qkv = op.Transpose(qkv, perm=[1, 0, 2])
        qkv = op.Reshape(qkv, out_reshape)
        out = op.MatMul(qkv, O_W)
        out = op.Add(O_B, out)
        out = op.Add(bias, out)
        return out

    @classmethod
    def rewrite(cls, op, x, pos_enc_bias_k, q, K_W, K_B, k_reshape, V_W, V_B, v_reshape, out_reshape, O_W, O_B, bias):
        shape = op.Shape(pos_enc_bias_k)
        k_new_bias = op.Expand(K_B, shape)
        pos_enc_new_bias = op.MatMul(pos_enc_bias_k, K_W)
        k_new_bias = op.Add(k_new_bias, pos_enc_new_bias)

        unsqueeze = op.Constant(value_ints=[0])
        k_new_bias2 = op.Transpose(k_new_bias, perm=[2, 1, 0])
        k_new_bias2 = op.Unsqueeze(k_new_bias2, unsqueeze)

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        K_W2 = op.Transpose(K_W, perm=[1, 0])
        K_W2 = op.Unsqueeze(K_W2, wgt_unsqueeze)

        k = op.Transpose(x, perm=[2, 1, 0])

        k = op.Unsqueeze(k, unsqueeze)

        indices = op.Constant(value_ints=[2])

        k = op.Conv(k, K_W2)
        k = op.Shortcut(
            k,
            k_new_bias2,
            reshape_mode="TRANSFORMER_QK",
            group=[8],
            _domain="com.videantis",
            _version=1,
            _outputs=["p" + str(np.random.randint(1e10))],
        )

        q = op.Transpose(q, perm=[0, 2, 1])
        indices = op.Constant(value_ints=[1, 2, 1, 0])
        q_wgt_reshape = op.Gather(out_reshape, indices)
        q = op.Reshape(q, q_wgt_reshape)

        qk = op.Conv(q, k, group=8)
        indices = op.Constant(value_ints=[1, 0])

        qk = op.vidSoftmax(qk, group=[8], _domain="com.videantis", _version=1)

        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        V_W2 = op.Transpose(V_W, perm=[1, 0])
        V_W2 = op.Unsqueeze(V_W2, wgt_unsqueeze)

        v_new = op.Transpose(x, perm=[2, 1, 0])

        v_new = op.Unsqueeze(v_new, unsqueeze)

        conv_node = list(V_W.uses())[0][0]
        v_new = op.vidConv(
            v_new,
            V_W2,
            V_B,
            reshape_mode="TRANSFORMER_V",
            _domain="com.videantis",
            _version=1,
            _outputs=["v_new" + str(conv_node.name)],
        )
        qkv = op.Conv(qk, v_new, group=8)

        O_W2 = op.Transpose(O_W, perm=[1, 0])
        O_W2 = op.Unsqueeze(O_W2, wgt_unsqueeze)
        # O_B = op.Add(O_B, bias)
        out = op.Conv(qkv, O_W2, O_B)

        bias = op.Unsqueeze(bias, unsqueeze)
        bias = op.Transpose(
            bias,
            perm=[
                0,
                3,
                2,
                1,
            ],
        )
        out = op.Add(bias, out)

        out = op.Transpose(out, perm=[0, 3, 2, 1])
        out = op.Squeeze(out, unsqueeze)

        return out


class TruncatedAttentionDetr2(RewriteRuleClassBase):
    """Matches truncated (first layer of decoder block) Detr block to vidConv schema without positional encoding bias."""

    level = 6

    @classmethod
    def pattern(
        cls,
        op,
        x,
        y,
        q,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
        bias,
        out_reshape2,
    ):
        k = op.MatMul(y, K_W)
        k = op.Add(K_B, k)
        k = op.Reshape(k, k_reshape)
        k = op.Transpose(k, perm=[1, 2, 0])

        qk = op.MatMul(q, k)
        qk = op.Softmax(qk, axis=-1)

        v = op.MatMul(x, V_W)
        v = op.Add(V_B, v)
        v = op.Reshape(v, v_reshape)
        v = op.Transpose(v, perm=[1, 0, 2])

        qkv = op.MatMul(qk, v)
        qkv = op.Transpose(qkv, perm=[1, 0, 2])
        qkv = op.Reshape(qkv, out_reshape)
        out = op.Gemm(qkv, O_W, O_B)
        out = op.Reshape(out, out_reshape2)
        out = op.Add(bias, out)
        return out

    @classmethod
    def check(
        cls,
        op,
        x,
        y,
        q,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
        bias,
        out_reshape2,
    ):
        if q.producer() is None:
            return True
        return False

    @classmethod
    def rewrite(cls, op, x, y, q, K_W, K_B, k_reshape, V_W, V_B, v_reshape, out_reshape, O_W, O_B, bias, out_reshape2):
        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        K_W2 = op.Transpose(K_W, perm=[1, 0])
        K_W2 = op.Unsqueeze(K_W2, wgt_unsqueeze)

        k = op.Transpose(y, perm=[2, 1, 0])
        k = op.Unsqueeze(k, unsqueeze)

        indices = op.Constant(value_ints=[2])

        k = op.vidConv(
            k,
            K_W2,
            K_B,
            reshape_mode="TRANSFORMER_QK",
            reshape_mode_groups=[8],
            _domain="com.videantis",
            _version=1,
            _outputs=["p" + str(np.random.randint(1e10))],
        )
        q = op.Transpose(q, perm=[0, 2, 1])
        indices = op.Constant(value_ints=[1, 2, 1, 0])
        q_wgt_reshape = op.Gather(out_reshape2, indices)
        q = op.Reshape(q, q_wgt_reshape)

        qk = op.Conv(q, k, group=8)
        indices = op.Constant(value_ints=[1, 0])

        qk = op.vidSoftmax(qk, group=[8], _domain="com.videantis", _version=1)

        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        V_W2 = op.Transpose(V_W, perm=[1, 0])
        V_W2 = op.Unsqueeze(V_W2, wgt_unsqueeze)

        v_new = op.Transpose(x, perm=[2, 1, 0])

        v_new = op.Unsqueeze(v_new, unsqueeze)

        conv_node = list(V_W.uses())[0][0]
        v_new = op.vidConv(
            v_new,
            V_W2,
            V_B,
            reshape_mode="TRANSFORMER_V",
            _domain="com.videantis",
            _version=1,
            _outputs=["v_new" + str(conv_node.name)],
        )
        qkv = op.Conv(qk, v_new, group=8)

        O_W2 = op.Transpose(O_W, perm=[0, 1])
        O_W2 = op.Unsqueeze(O_W2, wgt_unsqueeze)
        O_B = op.Add(O_B, bias)
        out = op.Conv(qkv, O_W2, O_B)

        out = op.Transpose(out, perm=[0, 3, 2, 1])
        out = op.Squeeze(out, unsqueeze)

        return out


class AttentionDetr(RewriteRuleClassBase):
    """Matches standard Detr block to vidConv schema."""

    level = 8

    @classmethod
    def pattern(
        cls,
        op,
        x,
        y,
        z,
        pos_enc_bias_q,
        pos_enc_bias_k,
        Q_W,
        Q_B,
        q_reshape,
        div,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
    ):
        x1 = op.Add(x, pos_enc_bias_q)

        q = op.MatMul(x1, Q_W)
        q = op.Add(Q_B, q)
        q = op.Reshape(q, q_reshape)
        q = op.Transpose(q, perm=[1, 0, 2])
        q = op.Div(q, div)

        y1 = op.Add(y, pos_enc_bias_k)
        k = op.MatMul(y1, K_W)
        k = op.Add(K_B, k)
        k = op.Reshape(k, k_reshape)
        k = op.Transpose(k, perm=[1, 2, 0])

        qk = op.MatMul(q, k, _outputs=["matmul_out"])
        qk = op.Softmax(qk, axis=-1)

        v = op.MatMul(z, V_W)
        v = op.Add(V_B, v)
        v = op.Reshape(v, v_reshape)
        v = op.Transpose(v, perm=[1, 0, 2])

        qkv = op.MatMul(qk, v)
        qkv = op.Transpose(qkv, perm=[1, 0, 2])
        qkv = op.Reshape(qkv, out_reshape)
        out = op.MatMul(qkv, O_W)
        out = op.Add(O_B, out)
        return out

    @classmethod
    def rewrite(
        cls,
        op,
        x,
        y,
        z,
        pos_enc_bias_q,
        pos_enc_bias_k,
        Q_W,
        Q_B,
        q_reshape,
        div,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
        matmul_out: ir.Value,
    ):
        shape = op.Shape(pos_enc_bias_q)
        q_new_bias = op.Expand(Q_B, shape)
        pos_enc_new_bias = op.MatMul(pos_enc_bias_q, Q_W)
        q_new_bias = op.Add(q_new_bias, pos_enc_new_bias)

        Q_W = op.Div(Q_W, div)
        q_new_bias = op.Div(q_new_bias, div)

        unsqueeze = op.Constant(value_ints=[0])
        q_new_bias2 = op.Transpose(q_new_bias, perm=[2, 1, 0])
        q_new_bias2 = op.Unsqueeze(q_new_bias2, unsqueeze)

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        Q_W2 = op.Transpose(Q_W, perm=[1, 0])
        Q_W2 = op.Unsqueeze(Q_W2, wgt_unsqueeze)
        # Q_W2 = op.Identity(Q_W2)

        q = op.Transpose(x, perm=[2, 1, 0])

        q = op.Unsqueeze(q, unsqueeze)

        q = op.Conv(q, Q_W2)
        q = op.Shortcut(
            q,
            q_new_bias2,
            _domain="com.videantis",
            _version=1,
            _outputs=[
                "q" + str(matmul_out.producer()),
            ],
        )

        shape = op.Shape(pos_enc_bias_k)
        k_new_bias = op.Expand(K_B, shape)
        pos_enc_new_bias = op.MatMul(pos_enc_bias_k, K_W)
        k_new_bias = op.Add(k_new_bias, pos_enc_new_bias)

        unsqueeze = op.Constant(value_ints=[0])
        k_new_bias2 = op.Transpose(k_new_bias, perm=[2, 1, 0])
        k_new_bias2 = op.Unsqueeze(k_new_bias2, unsqueeze)

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        K_W2 = op.Transpose(K_W, perm=[1, 0])
        K_W2 = op.Unsqueeze(K_W2, wgt_unsqueeze)

        k = op.Transpose(y, perm=[2, 1, 0])

        k = op.Unsqueeze(k, unsqueeze)

        k = op.Conv(k, K_W2)
        k = op.Shortcut(
            k,
            k_new_bias2,
            reshape_mode="TRANSFORMER_QK",
            group=[8],
            _domain="com.videantis",
            _version=1,
            _outputs=["k" + str(matmul_out.producer())],
        )

        qk = op.Conv(q, k, group=8)
        qk = op.vidSoftmax(qk, group=[8], _domain="com.videantis", _version=1)

        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        V_W2 = op.Transpose(V_W, perm=[1, 0])
        V_W2 = op.Unsqueeze(V_W2, wgt_unsqueeze)

        v_new = op.Transpose(z, perm=[2, 1, 0])

        v_new = op.Unsqueeze(v_new, unsqueeze)
        conv_node = list(V_W.uses())[0][0]
        v_new3 = op.vidConv(
            v_new,
            V_W2,
            V_B,
            reshape_mode="TRANSFORMER_V",
            _domain="com.videantis",
            _version=1,
            _outputs=["v_new3" + str(conv_node.name)],
        )

        qkv = op.Conv(qk, v_new3, group=8)

        O_W2 = op.Transpose(O_W, perm=[1, 0])
        O_W2 = op.Unsqueeze(O_W2, wgt_unsqueeze)
        out = op.Conv(qkv, O_W2, O_B)

        out = op.Transpose(out, perm=[0, 3, 2, 1])
        out = op.Squeeze(out, unsqueeze)

        return out


class AttentionDetr2(RewriteRuleClassBase):
    """Matches standard Detr block with last layer GEMM to vidConv schema."""

    level = 8

    @classmethod
    def pattern(
        cls,
        op,
        x,
        y,
        z,
        pos_enc_bias_q,
        pos_enc_bias_k,
        Q_W,
        Q_B,
        q_reshape,
        mul,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
        out_reshape2,
    ):
        x1 = op.Add(x, pos_enc_bias_q)

        q = op.MatMul(x1, Q_W)
        q = op.Add(Q_B, q)
        q = op.Reshape(q, q_reshape)
        q = op.Transpose(q, perm=[1, 0, 2])
        q = op.Mul(q, mul)

        y1 = op.Add(y, pos_enc_bias_k)
        k = op.MatMul(y1, K_W)
        k = op.Add(K_B, k)
        k = op.Reshape(k, k_reshape)
        k = op.Transpose(k, perm=[1, 2, 0])

        qk = op.MatMul(q, k, _outputs=["matmul_out"])
        qk = op.Softmax(qk, axis=-1)

        v = op.MatMul(z, V_W)
        v = op.Add(V_B, v)
        v = op.Reshape(v, v_reshape)
        v = op.Transpose(v, perm=[1, 0, 2])

        qkv = op.MatMul(qk, v)
        qkv = op.Transpose(qkv, perm=[1, 0, 2])
        qkv = op.Reshape(qkv, out_reshape)
        out = op.Gemm(qkv, O_W, O_B)
        out = op.Reshape(out, out_reshape2)
        return out

    @classmethod
    def rewrite(
        cls,
        op,
        x,
        y,
        z,
        pos_enc_bias_q,
        pos_enc_bias_k,
        Q_W,
        Q_B,
        q_reshape,
        mul,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
        out_reshape2,
        matmul_out: ir.Value,
    ):
        shape = op.Shape(pos_enc_bias_q)
        q_new_bias = op.Expand(Q_B, shape)
        pos_enc_new_bias = op.MatMul(pos_enc_bias_q, Q_W)
        q_new_bias = op.Add(q_new_bias, pos_enc_new_bias)

        Q_W = op.Mul(Q_W, mul)
        q_new_bias = op.Mul(q_new_bias, mul)

        unsqueeze = op.Constant(value_ints=[0])
        q_new_bias2 = op.Transpose(q_new_bias, perm=[2, 1, 0])
        q_new_bias2 = op.Unsqueeze(q_new_bias2, unsqueeze)

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        Q_W2 = op.Transpose(Q_W, perm=[1, 0])
        Q_W2 = op.Unsqueeze(Q_W2, wgt_unsqueeze)

        q = op.Transpose(x, perm=[2, 1, 0])

        q = op.Unsqueeze(q, unsqueeze)

        q = op.Conv(q, Q_W2)
        q = op.Shortcut(
            q,
            q_new_bias2,
            _domain="com.videantis",
            _version=1,
            _outputs=["q" + str(matmul_out.producer())],
        )

        shape = op.Shape(pos_enc_bias_k)
        k_new_bias = op.Expand(K_B, shape)
        pos_enc_new_bias = op.MatMul(pos_enc_bias_k, K_W)
        k_new_bias = op.Add(k_new_bias, pos_enc_new_bias)

        unsqueeze = op.Constant(value_ints=[0])
        k_new_bias2 = op.Transpose(k_new_bias, perm=[2, 1, 0])
        k_new_bias2 = op.Unsqueeze(k_new_bias2, unsqueeze)

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        K_W2 = op.Transpose(K_W, perm=[1, 0])
        K_W2 = op.Unsqueeze(K_W2, wgt_unsqueeze)

        k = op.Transpose(y, perm=[2, 1, 0])

        k = op.Unsqueeze(k, unsqueeze)

        k = op.Conv(k, K_W2)
        k = op.Shortcut(
            k,
            k_new_bias2,
            reshape_mode="TRANSFORMER_QK",
            group=[8],
            _domain="com.videantis",
            _version=1,
            _outputs=["k" + str(matmul_out.producer())],
        )

        qk = op.Conv(q, k, group=8)
        qk = op.vidSoftmax(qk, group=[8], _domain="com.videantis", _version=1)

        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        V_W2 = op.Transpose(V_W, perm=[1, 0])
        V_W2 = op.Unsqueeze(V_W2, wgt_unsqueeze)

        v_new = op.Transpose(z, perm=[2, 1, 0])

        v_new = op.Unsqueeze(v_new, unsqueeze)

        conv_node = list(V_W.uses())[0][0]
        v_new3 = op.vidConv(
            v_new,
            V_W2,
            V_B,
            reshape_mode="TRANSFORMER_V",
            _domain="com.videantis",
            _version=1,
            _outputs=["v_new3" + str(conv_node.name)],
        )

        qkv = op.Conv(qk, v_new3, group=8)

        O_W2 = op.Transpose(O_W, perm=[0, 1])
        O_W2 = op.Unsqueeze(O_W2, wgt_unsqueeze)
        out = op.Conv(qkv, O_W2, O_B)

        out = op.Transpose(out, perm=[0, 3, 2, 1])
        out = op.Squeeze(out, unsqueeze)

        return out


class AttentionDetr3(RewriteRuleClassBase):
    """Matches standard Detr block with last layer GEMM and encoding bias only in q path to vidConv schema."""

    level = 8

    @classmethod
    def pattern(
        cls,
        op,
        x,
        y,
        z,
        pos_enc_bias_q,
        Q_W,
        Q_B,
        q_reshape,
        mul,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
        out_reshape2,
    ):
        x1 = op.Add(x, pos_enc_bias_q)

        q = op.MatMul(x1, Q_W)
        q = op.Add(Q_B, q)
        q = op.Reshape(q, q_reshape)
        q = op.Transpose(q, perm=[1, 0, 2])
        q = op.Mul(q, mul)

        k = op.MatMul(y, K_W)
        k = op.Add(K_B, k)
        k = op.Reshape(k, k_reshape)
        k = op.Transpose(k, perm=[1, 2, 0])

        qk = op.MatMul(q, k, _outputs=["matmul_out"])
        qk = op.Softmax(qk, axis=-1)

        v = op.MatMul(z, V_W)
        v = op.Add(V_B, v)
        v = op.Reshape(v, v_reshape)
        v = op.Transpose(v, perm=[1, 0, 2])

        qkv = op.MatMul(qk, v)
        qkv = op.Transpose(qkv, perm=[1, 0, 2])
        qkv = op.Reshape(qkv, out_reshape)
        out = op.Gemm(qkv, O_W, O_B)
        out = op.Reshape(out, out_reshape2)
        return out

    @classmethod
    def rewrite(
        cls,
        op,
        x,
        y,
        z,
        pos_enc_bias_q,
        Q_W,
        Q_B,
        q_reshape,
        mul,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
        out_reshape2,
        matmul_out: ir.Value,
    ):
        shape = op.Shape(pos_enc_bias_q)
        q_new_bias = op.Expand(Q_B, shape)
        pos_enc_new_bias = op.MatMul(pos_enc_bias_q, Q_W)
        q_new_bias = op.Add(q_new_bias, pos_enc_new_bias)

        Q_W = op.Mul(Q_W, mul)
        q_new_bias = op.Mul(q_new_bias, mul)

        unsqueeze = op.Constant(value_ints=[0])
        q_new_bias2 = op.Transpose(q_new_bias, perm=[2, 1, 0])
        q_new_bias2 = op.Unsqueeze(q_new_bias2, unsqueeze)

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        Q_W2 = op.Transpose(Q_W, perm=[1, 0])
        Q_W2 = op.Unsqueeze(Q_W2, wgt_unsqueeze)

        q = op.Transpose(x, perm=[2, 1, 0])

        q = op.Unsqueeze(q, unsqueeze)

        q = op.Conv(q, Q_W2)
        q = op.Shortcut(
            q,
            q_new_bias2,
            _domain="com.videantis",
            _version=1,
            _outputs=["q" + str(matmul_out.producer())],
        )

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        K_W2 = op.Transpose(K_W, perm=[1, 0])
        K_W2 = op.Unsqueeze(K_W2, wgt_unsqueeze)

        k = op.Transpose(y, perm=[2, 1, 0])

        k = op.Unsqueeze(k, unsqueeze)

        k = op.vidConv(
            k,
            K_W2,
            K_B,
            reshape_mode="TRANSFORMER_QK",
            reshape_mode_groups=[8],
            _domain="com.videantis",
            _version=1,
            _outputs=["k" + str(np.random.randint(1e10))],
        )

        qk = op.Conv(q, k, group=8)
        qk = op.vidSoftmax(qk, group=[8], _domain="com.videantis", _version=1)

        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        V_W2 = op.Transpose(V_W, perm=[1, 0])
        V_W2 = op.Unsqueeze(V_W2, wgt_unsqueeze)

        v_new = op.Transpose(z, perm=[2, 1, 0])

        v_new = op.Unsqueeze(v_new, unsqueeze)

        conv_node = list(V_W.uses())[0][0]
        v_new3 = op.vidConv(
            v_new,
            V_W2,
            V_B,
            reshape_mode="TRANSFORMER_V",
            _domain="com.videantis",
            _version=1,
            _outputs=["v_new3" + str(conv_node.name)],
        )

        qkv = op.Conv(qk, v_new3, group=8)

        O_W2 = op.Transpose(O_W, perm=[0, 1])
        O_W2 = op.Unsqueeze(O_W2, wgt_unsqueeze)
        # O_W2 = op.Identity(O_W2)
        out = op.Conv(qkv, O_W2, O_B)

        out = op.Transpose(out, perm=[0, 3, 2, 1])
        out = op.Squeeze(out, unsqueeze)

        return out


class AttentionBERT(RewriteRuleClassBase):
    """Matches standard Bert block to vidConv schema."""

    level = 8

    @classmethod
    def pattern(
        cls, op, x, mask, Q_W, Q_B, q_reshape, div, K_W, K_B, k_reshape, V_W, V_B, v_reshape, out_reshape, O_W, O_B
    ):

        q = op.MatMul(x, Q_W)
        q = op.Add(q, Q_B)
        q = op.Reshape(q, q_reshape)
        q = op.Transpose(q, perm=[0, 2, 1, 3])

        k = op.MatMul(x, K_W)
        k = op.Add(k, K_B)
        k = op.Reshape(k, k_reshape)
        k = op.Transpose(k, perm=[0, 2, 3, 1])

        qk = op.MatMul(q, k)
        qk = op.Div(qk, div)
        qk = op.Add(qk, mask)
        qk = op.Softmax(qk, axis=-1)

        v = op.MatMul(x, V_W)
        v = op.Add(v, V_B)
        v = op.Reshape(v, v_reshape)
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        qkv = op.MatMul(qk, v)
        qkv = op.Transpose(qkv, perm=[0, 2, 1, 3])
        qkv = op.Reshape(qkv, out_reshape)
        out = op.MatMul(qkv, O_W)
        out = op.Add(out, O_B)
        return out

    @classmethod
    def rewrite(
        cls, op, x, mask, Q_W, Q_B, q_reshape, div, K_W, K_B, k_reshape, V_W, V_B, v_reshape, out_reshape, O_W, O_B
    ):
        q_new_bias = Q_B

        Q_W = op.Div(Q_W, div)
        q_new_bias = op.Div(q_new_bias, div)

        unsqueeze = op.Constant(value_ints=[0])
        q_new_bias2 = q_new_bias

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        Q_W2 = op.Transpose(Q_W, perm=[1, 0])
        Q_W2 = op.Unsqueeze(Q_W2, wgt_unsqueeze)

        q = op.Transpose(x, perm=[2, 0, 1])

        q = op.Unsqueeze(q, unsqueeze)

        q = op.Conv(q, Q_W2, q_new_bias2)

        unsqueeze = op.Constant(value_ints=[0])

        k_new_bias2 = K_B

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        K_W2 = op.Transpose(K_W, perm=[1, 0])
        K_W2 = op.Unsqueeze(K_W2, wgt_unsqueeze)

        k = op.Transpose(x, perm=[2, 0, 1])

        k = op.Unsqueeze(k, unsqueeze)
        k1 = k

        indices = op.Constant(value_ints=[2])
        heads = op.Gather(k_reshape, indices)

        indices = op.Constant(value_ints=[1])
        k_seq_length = op.Gather(k_reshape, indices)

        one = op.Constant(value_ints=[1])

        conv_node = list(K_B.uses())[0][0]
        k = op.vidConv(
            k1,
            K_W2,
            k_new_bias2,
            reshape_mode_groups=[16],
            reshape_mode="TRANSFORMER_QK",
            _domain="com.videantis",
            _version=1,
            _outputs=["k" + str(conv_node.name) + str(np.random.randint(1e10))],
        )

        groups = k_reshape.const_value.numpy()[2].item()
        qk = op.Conv(q, k, group=groups)

        mask_expand = op.Concat(one, one, one, heads, k_seq_length, axis=0)
        mask2 = op.Expand(mask, mask_expand)
        mask2 = op.Flatten(mask2)
        full_val = op.Mul(heads, k_seq_length)
        mask_expand = op.Concat(one, full_val, one, one, axis=0)
        mask2 = op.Reshape(mask2, mask_expand)

        qk = op.Add(qk, mask2)
        qk = op.vidSoftmax(qk, group=[16], _domain="com.videantis", _version=1)

        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        V_W2 = op.Transpose(V_W, perm=[1, 0])
        V_W2 = op.Unsqueeze(V_W2, wgt_unsqueeze)

        v_new = op.Transpose(x, perm=[2, 0, 1])

        v_new = op.Unsqueeze(v_new, unsqueeze)
        conv_node = list(V_B.uses())[0][0]
        v_new = op.vidConv(
            v_new,
            V_W2,
            V_B,
            reshape_mode="TRANSFORMER_V",
            _domain="com.videantis",
            _version=1,
            _outputs=["v_new3" + str(conv_node.name) + str(np.random.randint(1e10))],
        )

        qkv = op.Conv(qk, v_new, group=groups)

        O_W2 = op.Transpose(O_W, perm=[1, 0])
        O_W2 = op.Unsqueeze(O_W2, wgt_unsqueeze)

        out = op.Conv(qkv, O_W2, O_B)

        out = op.Transpose(out, perm=[0, 3, 2, 1])
        unsqueeze = op.Constant(value_ints=[2])
        out = op.Squeeze(out, unsqueeze)
        return out


class AttentionBERT2(RewriteRuleClassBase):
    """Matches standard BERT block without pos enc bias to vidConv schema."""

    level = 8

    @classmethod
    def pattern(
        cls,
        op,
        x,
        mask,
        Q_W,
        Q_B,
        q_reshape,
        q_mul,
        K_W,
        K_B,
        k_reshape,
        k_mul,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
    ):
        # x1=op.Add(x,pos_enc_bias_q)

        q = op.MatMul(x, Q_W)
        q = op.Add(q, Q_B)
        q = op.Reshape(q, q_reshape)
        q = op.Transpose(q, perm=[0, 2, 1, 3])
        q = op.Mul(q, q_mul)
        # q=op.Div(q,div)

        # y1 = op.Add(y,pos_enc_bias_k)
        k = op.MatMul(x, K_W)
        k = op.Add(k, K_B)
        k = op.Reshape(k, k_reshape)
        k = op.Transpose(k, perm=[0, 2, 3, 1])
        k = op.Mul(k, k_mul)

        qk = op.MatMul(q, k)
        # qk = op.Div(qk, div)
        qk = op.Add(qk, mask)
        qk = op.Softmax(qk, axis=-1)

        v = op.MatMul(x, V_W)
        v = op.Add(v, V_B)
        v = op.Reshape(v, v_reshape)
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        qkv = op.MatMul(qk, v)
        qkv = op.Transpose(qkv, perm=[0, 2, 1, 3])
        qkv = op.Reshape(qkv, out_reshape)
        out = op.MatMul(qkv, O_W)
        out = op.Add(out, O_B)
        return out

    @classmethod
    def rewrite(
        cls,
        op,
        x,
        mask,
        Q_W,
        Q_B,
        q_reshape,
        q_mul,
        K_W,
        K_B,
        k_reshape,
        k_mul,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
    ):
        q_new_bias = Q_B

        Q_W = op.Mul(Q_W, q_mul)
        q_new_bias = op.Mul(q_new_bias, q_mul)

        unsqueeze = op.Constant(value_ints=[0])
        q_new_bias2 = q_new_bias

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        Q_W2 = op.Transpose(Q_W, perm=[1, 0])
        Q_W2 = op.Unsqueeze(Q_W2, wgt_unsqueeze)

        q = op.Transpose(x, perm=[2, 0, 1])

        q = op.Unsqueeze(q, unsqueeze)

        q = op.Conv(q, Q_W2, q_new_bias2)

        unsqueeze = op.Constant(value_ints=[0])

        k_new_bias2 = K_B

        K_W = op.Mul(K_W, k_mul)
        k_new_bias2 = op.Mul(k_new_bias2, k_mul)

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        K_W2 = op.Transpose(K_W, perm=[1, 0])
        K_W2 = op.Unsqueeze(K_W2, wgt_unsqueeze)

        k = op.Transpose(x, perm=[2, 0, 1])

        k = op.Unsqueeze(k, unsqueeze)
        k1 = k

        indices = op.Constant(value_ints=[2])
        heads = op.Gather(k_reshape, indices)

        indices = op.Constant(value_ints=[1])
        k_seq_length = op.Gather(k_reshape, indices)

        indices = op.Constant(value_ints=[3])

        one = op.Constant(value_ints=[1])

        conv_node = list(K_B.uses())[0][0]
        k = op.vidConv(
            k1,
            K_W2,
            k_new_bias2,
            reshape_mode_groups=[12],
            reshape_mode="TRANSFORMER_QK",
            _domain="com.videantis",
            _version=1,
            _outputs=["k" + str(conv_node.name) + str(np.random.randint(1e10))],
        )

        groups = k_reshape.const_value.numpy()[2].item()
        qk = op.Conv(q, k, group=groups)

        mask_expand = op.Concat(one, one, one, heads, k_seq_length, axis=0)

        slice_new_mask_end = op.Constant(value_ints=[1, 1, 1, 2147483647])

        slice_new_mask_start = op.Constant(value_ints=[0, 0, 0, 0])

        mask = op.Slice(mask, slice_new_mask_start, slice_new_mask_end)

        mask2 = op.Expand(mask, mask_expand)
        mask2 = op.Flatten(mask2)
        full_val = op.Mul(heads, k_seq_length)
        mask_expand = op.Concat(one, full_val, one, one, axis=0)
        mask2 = op.Reshape(mask2, mask_expand)

        qk = op.Add(qk, mask2)
        qk = op.vidSoftmax(qk, group=[12], _domain="com.videantis", _version=1)

        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        V_W2 = op.Transpose(V_W, perm=[1, 0])
        V_W2 = op.Unsqueeze(V_W2, wgt_unsqueeze)

        v_new = op.Transpose(x, perm=[2, 0, 1])

        v_new = op.Unsqueeze(v_new, unsqueeze)
        conv_node = list(V_B.uses())[0][0]
        v_new = op.vidConv(
            v_new,
            V_W2,
            V_B,
            reshape_mode="TRANSFORMER_V",
            _domain="com.videantis",
            _version=1,
            _outputs=["v_new3" + str(conv_node.name) + str(np.random.randint(1e10))],
        )

        qkv = op.Conv(qk, v_new, group=groups)

        O_W2 = op.Transpose(O_W, perm=[1, 0])
        O_W2 = op.Unsqueeze(O_W2, wgt_unsqueeze)

        out = op.Conv(qkv, O_W2, O_B)

        out = op.Transpose(out, perm=[0, 3, 2, 1])
        unsqueeze = op.Constant(value_ints=[2])
        out = op.Squeeze(out, unsqueeze)
        return out


class AttentionBERT3(RewriteRuleClassBase):
    """Matches standard Bert block to vidConv schema."""

    level = 8

    @classmethod
    def pattern(
        cls, op, x, y, mask, Q_W, Q_B, q_reshape, div, K_W, K_B, k_reshape, V_W, V_B, v_reshape, out_reshape, O_W, O_B
    ):

        q = op.MatMul(x, Q_W)
        q = op.Add(Q_B, q)
        q = op.Reshape(q, q_reshape, allowzero=0)
        q = op.Transpose(q, perm=[0, 2, 1, 3])

        k = op.MatMul(x, K_W)
        k = op.Add(K_B, k)
        k = op.Reshape(k, k_reshape, allowzero=0)
        k = op.Transpose(k, perm=[0, 2, 3, 1])

        qk = op.MatMul(q, k)
        qk = op.Div(qk, div)
        qk = op.Add(qk, mask)
        qk = op.Softmax(qk, axis=-1)

        v = op.MatMul(y, V_W)
        v = op.Add(V_B, v)
        v = op.Reshape(v, v_reshape, allowzero=0)
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        qkv = op.MatMul(qk, v)
        qkv = op.Transpose(qkv, perm=[0, 2, 1, 3])
        qkv = op.Reshape(qkv, out_reshape, allowzero=0)
        out = op.MatMul(qkv, O_W)
        out = op.Add(O_B, out)
        return out

    @classmethod
    def rewrite(
        cls, op, x, y, mask, Q_W, Q_B, q_reshape, div, K_W, K_B, k_reshape, V_W, V_B, v_reshape, out_reshape, O_W, O_B
    ):
        unsqueeze = op.Constant(value_ints=[0])
        q_new_bias = Q_B

        Q_W = op.Div(Q_W, div)
        q_new_bias = op.Div(q_new_bias, div)

        q_new_bias2 = q_new_bias

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        Q_W2 = op.Transpose(Q_W, perm=[1, 0])
        Q_W2 = op.Unsqueeze(Q_W2, wgt_unsqueeze)

        q = op.Transpose(x, perm=[2, 0, 1])

        q = op.Unsqueeze(q, unsqueeze)

        q = op.Conv(q, Q_W2, q_new_bias2)

        k_new_bias2 = K_B

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        K_W2 = op.Transpose(K_W, perm=[1, 0])
        K_W2 = op.Unsqueeze(K_W2, wgt_unsqueeze)

        k = op.Transpose(x, perm=[2, 0, 1])

        k = op.Unsqueeze(k, unsqueeze)
        k1 = k

        indices = op.Constant(value_ints=[2])
        heads = op.Gather(k_reshape, indices)

        indices = op.Constant(value_ints=[1])
        k_seq_length = op.Gather(k_reshape, indices)

        one = op.Constant(value_ints=[1])

        conv_node = list(K_B.uses())[0][0]
        k = op.vidConv(
            k1,
            K_W2,
            k_new_bias2,
            reshape_mode_groups=[4],
            reshape_mode="TRANSFORMER_QK",
            _domain="com.videantis",
            _version=1,
            _outputs=["k" + str(conv_node.name) + str(np.random.randint(1e10))],
        )

        groups = k_reshape.const_value.numpy()[2].item()
        qk = op.Conv(q, k, group=groups)

        mask_expand = op.Concat(one, one, one, heads, k_seq_length, axis=0)
        mask2 = op.Expand(mask, mask_expand)
        mask2 = op.Flatten(mask2)
        full_val = op.Mul(heads, k_seq_length)
        mask_expand = op.Concat(one, full_val, one, one, axis=0)
        mask2 = op.Reshape(mask2, mask_expand)

        qk = op.Add(qk, mask2)
        qk = op.vidSoftmax(qk, group=[4], _domain="com.videantis", _version=1)

        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        V_W2 = op.Transpose(V_W, perm=[1, 0])
        V_W2 = op.Unsqueeze(V_W2, wgt_unsqueeze)

        v_new = op.Transpose(y, perm=[2, 0, 1])

        v_new = op.Unsqueeze(v_new, unsqueeze)
        conv_node = list(V_B.uses())[0][0]
        v_new = op.vidConv(
            v_new,
            V_W2,
            V_B,
            reshape_mode="TRANSFORMER_V",
            _domain="com.videantis",
            _version=1,
            _outputs=["v_new3" + str(conv_node.name) + str(np.random.randint(1e10))],
        )

        qkv = op.Conv(qk, v_new, group=groups)

        O_W2 = op.Transpose(O_W, perm=[1, 0])
        O_W2 = op.Unsqueeze(O_W2, wgt_unsqueeze)

        out = op.Conv(qkv, O_W2, O_B)

        out = op.Transpose(out, perm=[0, 3, 2, 1])
        unsqueeze = op.Constant(value_ints=[2])
        out = op.Squeeze(out, unsqueeze)
        return out


class GeluPattern(RewriteRuleClassBase):
    """Matches Gelu Pattern to Gelu Operator."""

    level = 10

    @classmethod
    def pattern(cls, op, x, sqrt2, add1, mul05):
        x1 = op.Div(x, sqrt2)
        x1 = op.Erf(x1)
        x1 = op.Add(x1, add1)
        x = op.Mul(x, x1)
        x = op.Mul(x, mul05)
        return x

    @classmethod
    def rewrite(cls, op, x, sqrt2, add1, mul05):
        x = op.Gelu(x)
        return x


class GeluPattern_2(RewriteRuleClassBase):
    """Matches Gelu Pattern to Gelu Operator."""

    level = 10

    @classmethod
    def pattern(cls, op, x, sqrt2, add1, mul05):
        x1 = op.Div(x, sqrt2)
        x1 = op.Erf(x1)
        x1 = op.Add(x1, add1)
        x = op.Mul(x, mul05)
        x = op.Mul(x, x1)

        return x

    @classmethod
    def rewrite(cls, op, x, sqrt2, add1, mul05):
        x = op.Gelu(x)
        return x


class doubleTranspose(RewriteRuleClassBase):
    """Removes double transpose from graph."""

    level = 1

    @classmethod
    def pattern(cls, op, x):
        x = op.Transpose(x, perm=[0, 2, 3, 1])
        x = op.Transpose(x, perm=[0, 3, 1, 2])
        return x

    @classmethod
    def rewrite(cls, op, x):
        return x


class doubleTransposeGelu(RewriteRuleClassBase):
    """Removes double Transpose with Gelu inbetween."""

    level = 1

    @classmethod
    def pattern(cls, op, x):
        x = op.Transpose(x, perm=[0, 2, 3, 1])
        x = op.Gelu(x)
        x = op.Transpose(x, perm=[0, 3, 1, 2])
        return x

    @classmethod
    def rewrite(cls, op, x):
        x = op.Gelu(x)
        return x


class doubleTransposeLayerNorm(RewriteRuleClassBase):
    """Fuse Transposed LN to vidLayerNorm."""

    @classmethod
    def pattern(cls, op, x, scale, b):
        x = op.Transpose(x, perm=[0, 2, 3, 1])
        x = op.LayerNormalization(x, scale, b)
        x = op.Transpose(x, perm=[0, 3, 1, 2])
        return x

    @classmethod
    def rewrite(cls, op, x, scale, b):
        x = op.vidLayerNorm(x, scale, b, _domain="com.videantis", _version=1)
        return x


class ConvMulPostFuse(RewriteRuleClassBase):
    """Fuse Mul after Conv into bias and weights."""

    @classmethod
    def pattern(cls, op, x, w, b, mul):
        x = op.Conv(x, w, b)
        x = op.Mul(x, mul)
        return x

    @classmethod
    def check(cls, op, x, w, b, mul):
        if mul.producer() is None:
            if mul.shape[0] == 1:
                if mul.shape[2] == 1:
                    if mul.shape[3] == 1:
                        return True
        return False

    @classmethod
    def rewrite(cls, op, x, w, b, mul):
        mul = op.Transpose(mul, perm=[1, 0, 2, 3])
        w = op.Mul(w, mul)
        b = op.Mul(b, op.Squeeze(mul))
        x = op.Conv(x, w, b)
        return x


class vidConvShortcutPreFuse(RewriteRuleClassBase):
    """Fuse Shortcut before vidConv by incorporating the multiplication into the input data."""

    @classmethod
    def pattern(cls, op, x, w, b, mul):
        x = op.Shortcut(mul, x, mode="multiplication", _domain="com.videantis")
        x = op.vidConv(x, w, b, _domain="com.videantis")
        return x

    @classmethod
    def check(cls, op, x, w, b, mul):
        if mul.producer() is None:
            if mul.shape[0] == 1 and mul.shape[2] == 1 and mul.shape[3] == 1:
                return True
        return False

    @classmethod
    def rewrite(cls, op, x, w, b, mul):
        conv_node = list(w.uses())[0][0]
        dilations = conv_node.attributes.get("dilations", None)
        group = conv_node.attributes.get("group", None)
        kernel_shape = conv_node.attributes.get("kernel_shape", None)
        pads = conv_node.attributes.get("pads", None)
        strides = conv_node.attributes.get("strides", None)

        new_w = op.Mul(w, mul)

        new_x = op.vidConv(
            x,
            new_w,
            b,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            dim=4,
            _domain="com.videantis",
            _version=1,
            _outputs=["x" + str(np.random.randint(1e10))],
        )
        return new_x


class vidConvShortcutPostFuse(RewriteRuleClassBase):
    """Fuse Shortcut after vidConv into bias and weights."""

    @classmethod
    def pattern(cls, op, x, w, b, mul):
        x = op.vidConv(
            x,
            w,
            b,
            _domain="com.videantis",
        )
        x = op.Shortcut(mul, x, mode="multiplication", _domain="com.videantis")
        return x

    @classmethod
    def check(cls, op, x, w, b, mul):
        if mul.producer() is None:
            if mul.shape[0] == 1:
                if mul.shape[2] == 1:
                    if mul.shape[3] == 1:
                        return True
        return False

    @classmethod
    def rewrite(cls, op, x, w, b, mul):
        conv_node = list(w.uses())[0][0]
        dilations = conv_node.attributes.get("dilations", None)
        group = conv_node.attributes.get("group", None)
        kernel_shape = conv_node.attributes.get("kernel_shape", None)
        pads = conv_node.attributes.get("pads", None)
        strides = conv_node.attributes.get("strides", None)

        mul = op.Transpose(mul, perm=[1, 0, 2, 3])
        w = op.Mul(w, mul)
        b = op.Mul(b, op.Squeeze(mul))
        x = op.vidConv(
            x,
            w,
            b,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            dim=4,
            _domain="com.videantis",
            _version=1,
            _outputs=["x" + str(np.random.randint(1e10))],
        )
        return x


class ConvMulPostFuse3DWGT(RewriteRuleClassBase):
    """Fuse 3D Mul into Conv bias and weights."""

    @classmethod
    def pattern(cls, op, x, w, b, mul):
        x = op.Conv(x, w, b)
        x = op.Mul(x, mul)
        return x

    @classmethod
    def check(cls, op, x, w, b, mul):
        if mul.producer() is None:
            if mul.shape[1] == 1:
                if mul.shape[2] == 1:
                    return True
        return False

    @classmethod
    def rewrite(cls, op, x, w, b, mul):
        unsqueeze = op.Constant(value_ints=[0])
        mul = op.Unsqueeze(mul, unsqueeze)
        mul = op.Transpose(mul, perm=[1, 0, 2, 3])
        w = op.Mul(w, mul)
        b = op.Mul(b, op.Squeeze(mul))
        x = op.Conv(x, w, b)
        return x


class ConvMulAddPreFuse(RewriteRuleClassBase):
    """Fuse Mul before Conv into bias and weights."""

    level = 2

    @classmethod
    def pattern(cls, op, x, w, b, mul, add):
        x = op.Mul(x, mul)
        x = op.Add(x, add)
        x = op.Transpose(x, perm=[0, 3, 1, 2])
        x = op.Conv(x, w, b, _outputs=["result"])

        return x

    @classmethod
    def check(cls, op, x, w, b, mul, add, result: ir.Value):
        if len(mul.shape) == 1 and len(add.shape) == 1:
            return True
        return False

    @classmethod
    def rewrite(cls, op, x, w, b, mul, add, result: ir.Value):
        node = result.producer()
        dilations = node.attributes.get("dilations", None)
        group = node.attributes.get("group", None)
        kernel_shape = node.attributes.get("kernel_shape", None)
        pads = node.attributes.get("pads", None)
        strides = node.attributes.get("strides", None)

        unsqueeze_op = op.Constant(value_ints=[0, 2, 3])

        new_b = op.Expand(add, op.Shape(x))
        new_b = op.Transpose(new_b, perm=[0, 3, 1, 2])
        new_b = op.Conv(
            new_b, w, b, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides
        )

        new_b_shape = op.Shape(new_b)
        c_ind = op.Constant(value_int=1)
        new_b_c = op.Gather(new_b_shape, c_ind)
        one = op.Constant(value_int=1)
        zero = op.Constant(value_int=0)
        new_b_c = op.Add(new_b_c, one)
        new_b_c = op.Unsqueeze(new_b_c, zero)
        # new_b_c = op.Identity(new_b_c)

        one_t = op.Constant(value_ints=[1])

        slice_new_b_end = op.Concat(one_t, new_b_c, one_t, one_t, axis=0)

        slice_new_b_start = op.Constant(value_ints=[0, 0, 0, 0])

        new_b = op.Slice(new_b, slice_new_b_start, slice_new_b_end)
        new_b = op.Squeeze(new_b)
        # new_b = op.Identity(new_b)

        new_w = op.Unsqueeze(mul, unsqueeze_op)
        new_w = op.Mul(new_w, w)

        x = op.Transpose(x, perm=[0, 3, 1, 2])
        x = op.Conv(
            x, new_w, new_b, dilations=dilations, group=group, kernel_shape=kernel_shape, pads=pads, strides=strides
        )
        return x


class AttentionViT(RewriteRuleClassBase):
    """VIT Attention Layer pattern."""

    level = 8

    @classmethod
    def pattern(
        cls,
        op,
        x,
        z,
        scale,
        bias,
        Q_W,
        Q_B,
        q_reshape,
        q_mul,
        k_mul,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
    ):
        x1 = op.LayerNormalization(x, scale, bias)
        q = op.MatMul(x1, Q_W)
        q = op.Add(Q_B, q)
        q = op.Reshape(q, q_reshape)
        q = op.Transpose(q, perm=[0, 2, 1, 3])
        q = op.Mul(q, q_mul)

        k = op.MatMul(x1, K_W)
        k = op.Add(K_B, k)
        k = op.Reshape(k, k_reshape)
        k = op.Transpose(k, perm=[0, 2, 3, 1])
        k = op.Mul(k, k_mul)

        qk = op.MatMul(q, k, _outputs=["matmul_out"])
        qk = op.Softmax(qk, axis=-1)

        v = op.MatMul(z, V_W)
        v = op.Add(V_B, v)
        v = op.Reshape(v, v_reshape)
        v = op.Transpose(v, perm=[0, 2, 1, 3])

        qkv = op.MatMul(qk, v)
        qkv = op.Transpose(qkv, perm=[0, 2, 1, 3])
        qkv = op.Reshape(qkv, out_reshape)
        out = op.MatMul(qkv, O_W)
        out = op.Add(O_B, out)
        return out

    @classmethod
    def rewrite(
        cls,
        op,
        x,
        z,
        scale,
        bias,
        Q_W,
        Q_B,
        q_reshape,
        q_mul,
        k_mul,
        K_W,
        K_B,
        k_reshape,
        V_W,
        V_B,
        v_reshape,
        out_reshape,
        O_W,
        O_B,
        matmul_out: ir.Value,
    ):
        x1 = op.LayerNormalization(x, scale, bias)

        Q_W = op.Mul(Q_W, q_mul)
        Q_B = op.Mul(Q_B, q_mul)

        unsqueeze = op.Constant(value_ints=[0])

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        Q_W2 = op.Transpose(Q_W, perm=[1, 0])
        Q_W2 = op.Unsqueeze(Q_W2, wgt_unsqueeze)
        # Q_W2 = op.Identity(Q_W2)

        q = op.Transpose(x1, perm=[2, 0, 1])

        q = op.Unsqueeze(q, unsqueeze)

        q = op.Conv(q, Q_W2, Q_B)

        K_W = op.Mul(K_W, k_mul)
        K_B2 = op.Mul(K_B, k_mul)
        unsqueeze = op.Constant(value_ints=[0])

        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        K_W2 = op.Transpose(K_W, perm=[1, 0])
        K_W2 = op.Unsqueeze(K_W2, wgt_unsqueeze)

        k = op.Transpose(x1, perm=[2, 0, 1])

        k = op.Unsqueeze(k, unsqueeze)

        conv_node = list(K_B.uses())[0][0]
        k = op.vidConv(
            k,
            K_W2,
            K_B2,
            reshape_mode_groups=[12],
            reshape_mode="TRANSFORMER_QK",
            _domain="com.videantis",
            _version=1,
            _outputs=["k" + str(conv_node.name) + str(np.random.randint(1e10))],
        )

        qk = op.Conv(q, k, group=12)

        qk = op.vidSoftmax(qk, group=[12], _domain="com.videantis", _version=1)

        unsqueeze = op.Constant(value_ints=[0])
        wgt_unsqueeze = op.Constant(value_ints=[2, 3])
        V_W2 = op.Transpose(V_W, perm=[1, 0])
        V_W2 = op.Unsqueeze(V_W2, wgt_unsqueeze)

        v_new = op.Transpose(x1, perm=[2, 0, 1])

        v_new = op.Unsqueeze(v_new, unsqueeze)
        conv_node = list(V_W.uses())[0][0]
        v_new3 = op.vidConv(
            v_new,
            V_W2,
            V_B,
            reshape_mode="TRANSFORMER_V",
            _domain="com.videantis",
            _version=1,
            _outputs=["v_new3" + str(conv_node.name)],
        )

        qkv = op.Conv(qk, v_new3, group=12)

        O_W2 = op.Transpose(O_W, perm=[1, 0])
        O_W2 = op.Unsqueeze(O_W2, wgt_unsqueeze)
        out = op.Conv(qkv, O_W2, O_B)

        out = op.Transpose(out, perm=[0, 3, 2, 1])
        unsqueeze = op.Constant(value_ints=[2])
        out = op.Squeeze(out, unsqueeze)

        return out


class ElemRMSNormPattern(RewriteRuleClassBase):
    """Matches elementary RMSNorm to RMSNorm operator."""

    level = 2

    @classmethod
    def pattern(cls, op, x, axes2, epsilon, pow, one, w):
        x1 = op.Pow(x, pow)
        x1 = op.ReduceMean(x1, axes2)
        x1 = op.Add(x1, epsilon)
        x1 = op.Sqrt(x1)
        x1 = op.Div(one, x1)
        x = op.Mul(x, x1)
        x = op.Mul(x, w)
        return x

    @classmethod
    def rewrite(cls, op, x, axes2, epsilon, pow, one, w):
        x = op.RMSNormalization(x, w, epsilon, _domain="com.videantis", _version=1)
        return x


class ClipToRelu6(RewriteRuleClassBase):
    """Replace a Clip operator with Relu6, if applicable."""

    level = 2

    @classmethod
    def pattern(cls, op, x, min_value, max_value):
        x = op.Clip(x, min_value, max_value)
        return x

    @classmethod
    def rewrite(cls, op, x, min_value, max_value):
        x = op.Relu6(x, _domain="com.videantis", _version=1)
        return x

    @classmethod
    def check(cls, op, x, min_value, max_value):
        # Both values must be constants
        if not (hasattr(min_value, "const_value") and hasattr(max_value, "const_value")):
            return False

        # Extract the constants
        min_const = float(min_value.const_value.numpy())
        max_const = float(max_value.const_value.numpy())

        # Validate numeric values
        return min_const == 0.0 and max_const == 6.0


class SoftmaxToVidSoftmax(RewriteRuleClassBase):
    """Replace Softmax with our own grouped vidSoftmax."""

    level = 2

    @classmethod
    def pattern(cls, op, x):
        x = op.Softmax(x, _outputs=["softmax_out"])
        return x

    @classmethod
    def rewrite(cls, op, x, softmax_out: ir.Value):
        x = op.vidSoftmax(x, group=[1], _domain="com.videantis", _version=1)
        return x

    @classmethod
    def check(cls, op, x, softmax_out: ir.Value):
        softmax_node = softmax_out.producer()
        axis = softmax_node.attributes.get("axis", None)

        # Parse axis and set to -1 for default case
        if axis is None:
            axis = -1
        else:
            axis = axis.value

        # Check that rank is at least 2 and that softmax is applied on the second dimension (after batch)
        rank = len(x.shape)
        if rank < 2:
            return False

        axis = axis % rank  # remap negative axis-index back to [0, rank-1]
        return axis == 1


class MergeScalarIntoConvTranspose:
    """Base class for merging scalar multiplication into ConvTranspose weights and bias.

    This class is not a rewrite rule itself, but provides the implementation for the two cases of ConvTranspose with
    and without bias.
    """

    level = 2

    @classmethod
    def _rewrite_impl(cls, op, x, w, scalar, b=None):
        np_w = w.const_value.numpy()
        np_scalar = scalar.const_value.numpy()
        new_w = np_w * np_scalar
        new_w_const = op.Constant(value=from_array(new_w))

        conv_node = list(w.uses())[0][0]
        dilations = conv_node.attributes.get("dilations", None)
        group = conv_node.attributes.get("group", None)
        kernel_shape = conv_node.attributes.get("kernel_shape", None)
        output_padding = conv_node.attributes.get("output_padding", None)
        output_shape = conv_node.attributes.get("output_shape", None)
        pads = conv_node.attributes.get("pads", None)
        strides = conv_node.attributes.get("strides", None)

        if b is None:
            return op.ConvTranspose(
                x,
                new_w_const,
                dilations=dilations,
                group=group,
                kernel_shape=kernel_shape,
                output_padding=output_padding,
                output_shape=output_shape,
                pads=pads,
                strides=strides,
            )
        return op.ConvTranspose(
            x,
            new_w_const,
            b,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            output_padding=output_padding,
            output_shape=output_shape,
            pads=pads,
            strides=strides,
        )

    @classmethod
    def check(cls, op, x, w, scalar, b=None):
        # Check that scalar is a constant and has only one element
        if scalar.producer() is None and scalar.const_value is not None and scalar.const_value.numpy().size == 1:
            return True
        return False


class MergeScalarIntoConvTransposeWithBias(MergeScalarIntoConvTranspose, RewriteRuleClassBase):
    """Merge a post-ConvTranspose scalar Mul into the weights when the ConvTranspose has a bias input."""

    @classmethod
    def pattern(cls, op, x, w, b, scalar):
        conv = op.ConvTranspose(x, w, b, _allow_other_attributes=True)
        return op.Mul(conv, scalar)

    @classmethod
    def rewrite(cls, op, x, w, b, scalar):
        return cls._rewrite_impl(op, x, w, scalar, b)


class MergeScalarIntoConvTransposeNoBias(MergeScalarIntoConvTranspose, RewriteRuleClassBase):
    """Merge a post-ConvTranspose scalar Mul into the weights when the ConvTranspose has no bias input."""

    @classmethod
    def pattern(cls, op, x, w, scalar):
        conv = op.ConvTranspose(x, w, _allow_other_attributes=True)
        return op.Mul(conv, scalar)

    @classmethod
    def rewrite(cls, op, x, w, scalar):
        return cls._rewrite_impl(op, x, w, scalar)


class RemoveIdentityLayer(RewriteRuleClassBase):
    """Removes Identity layer that is not contributing to the graph in any way."""

    level = 2

    @classmethod
    def pattern(cls, op, x):
        x = op.Identity(x)
        return x

    @classmethod
    def rewrite(cls, op, x):
        return x
