"""This module contains all layer handlers for quantization.

In our quantization implementation tensors that need to quantized need an associated TensorQuantInfo object, which
contains information about the desired number of bits, axis over which to quantize over and so on. These TensorQuantInfo
objects primarily come from LayerHandler objects. All "layers" (corresponding to ONNX operators) have an associated
layer handler, which defines how its inputs and outputs should be quantized. All layer handlers need to derive from
LayerHandlerBase and implement its tensor_quant_infos() method. Optionally they can implement tensor_range_postprocess()
method, which will be called after the tensor ranges have been collected and can be used to postprocess them.

This module contains the QUANT_HANDLER_OP_REGISTRY dictionary, which maps op_types to layer handlers and can be used
to load the correct handler for a given op_type.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vnnort.quantizer.quant_utils import TensorQuantInfo
from vnnort.utils.onnx_utils.graph_helper import Node, TensorType


class LayerHandlerBase(ABC):
    """Base class for all layer handlers."""

    def __init__(self, node: Node):
        """Initialize a new LayerHandlerBase."""
        self._node = node

    @abstractmethod
    def tensor_quant_infos(self) -> dict[str, TensorQuantInfo]:
        """Return a dict mapping tensor names to TensorQuantInfo objects.

        Raises:
            NotImplementedError: if not implemented

        Returns:
            dict[str, TensorQuantInfo]: mapping between tensor names and TensorQuantInfo objects
        """
        raise NotImplementedError

    def tensor_range_postprocess(self, tensor_quant_infos: dict[str, TensorQuantInfo]) -> None:
        """Postprocess tensor ranges after they have been collected by the calibrator.

        Args:
            tensor_quant_infos (dict[str, TensorQuantInfo]): Dict mapping tensor names to TensorQuantInfo objects, which
                should contain TensorQuantInfos for all tensors in the model.

        Returns:
            None: TBD(?)
        """
        pass


class DefaultLayerHandler(LayerHandlerBase):
    """Default layer handler, which is used, when no other handler is specified."""

    def __init__(self, node: Node):
        """Initialize a new DefaultLayerHandler.

        Args:
            node (Node): Node to quantize
        """
        super().__init__(node)
        result = {}

        # By default we quantize each output tensor of a layer
        for tensor in node.outputs:
            tensor_name = tensor.name
            quant_info = TensorQuantInfo(
                tensor=tensor,
                n_bits=8,  # By default we quantize with 8bit (wordwidth = 1)
                n_fraction_bits=7,  # By default we quantize with 7 fraction bits
                axis=1,  # In almost all cases, we have shapes of the form [N, C, H, W] and quantize over C
            )
            result[tensor_name] = quant_info
        self.result = result

    def tensor_quant_infos(self) -> dict[str, TensorQuantInfo]:
        """Return a dict mapping tensor names to TensorQuantInfo objects."""
        return self.result


class VidConvHandler(LayerHandlerBase):
    """Handler for VidConv layers."""

    def __init__(self, node: Node):
        """Initialize a new VidConvHandler.

        Args:
            node (Node): Node to quantize
        """
        super().__init__(node)

        node = self._node

        # Main Output
        main_output = node.outputs[0]
        self.output = TensorQuantInfo(
            tensor=main_output,
            n_bits=8,
            n_fraction_bits=7,
            axis=1,
        )

        # Weights are handled by this handler ONLY if they are static
        # Dynamic weights, which are the output of another layer are handled by that layer
        weight_tensor = node.inputs[1]
        self.weights = None
        if weight_tensor.tensor_type == TensorType.INITIALIZER:
            # For weights 2D max exponents are required
            self.weights = TensorQuantInfo(tensor=weight_tensor, n_bits=8, n_fraction_bits=7, axis=0)

        # Bias (is optional)
        self.bias = None
        if len(node.inputs) == 3 and node.inputs[2] is not None:
            bias_tensor = node.inputs[2]
            # Bias tensors have the fixed point format 2s14
            self.bias = TensorQuantInfo(
                tensor=bias_tensor,
                n_bits=16,
                n_fraction_bits=14,
                axis=0,
            )

        # Number of groups in convolution
        n_groups = node.attributes.get("group")
        self.n_groups = 1 if n_groups is None else int(n_groups)

    def tensor_quant_infos(self) -> dict[str, TensorQuantInfo]:
        """Return a dict mapping tensor names to TensorQuantInfo objects.

        Returns:
            dict[str, TensorQuantInfo]: Dict mapping tensor names to TensorQuantInfo objects
        """
        result = {self.output.tensor.name: self.output}
        if self.weights is not None:
            result[self.weights.tensor.name] = self.weights
        if self.bias is not None:
            result[self.bias.tensor.name] = self.bias

        # In case the input is also an initializer add it to tensor quant infos
        input_tensor = self._node.inputs[0]
        if input_tensor.tensor_type is TensorType.INITIALIZER:
            result[input_tensor.name] = TensorQuantInfo(tensor=input_tensor, n_bits=8, n_fraction_bits=7, axis=1)

        # Also add the pre activation tensor as output
        # preactivation_tensor = self._node.outputs[1]
        # result[preactivation_tensor.name] = TensorQuantInfo(
        #     tensor=preactivation_tensor, n_bits=8, n_fraction_bits=7, axis=1
        # )

        return result

    def tensor_range_postprocess(self, tensor_quant_infos: dict[str, TensorQuantInfo]) -> None:  # noqa
        """Postprocess tensor ranges after they have been collected by the calibrator.

        Args:
            tensor_quant_infos (dict[str, TensorQuantInfo]): Dict mapping tensor names to TensorQuantInfo objects, which
                should contain TensorQuantInfos for all tensors in the model.

        Raises:
            RuntimeError: If the ranges for the weight tensor should already be set at this point

        Returns:
            None: postprocessing is done internally.
        """
        # For dynamic weights, there is nothing to do
        if self.weights is None:
            return

        # We do not need to do anything if we dont do power of two scaling
        if not self.weights.power_of_two_scaling_only:
            return

        input_tensor_name = self._node.inputs[0].name
        # Make sure that input tensors share the same max exponents between groups
        input_max_exponents = tensor_quant_infos[input_tensor_name].max_exponents
        if input_max_exponents is None:
            raise RuntimeError("The max exponents for the input tensor should be set at this point")
        n_groups = self.n_groups

        weight_max_exponents = self.weights.max_exponents
        if weight_max_exponents is None:
            raise RuntimeError("The max exponents for the weight tensor should be set at this point")
        if weight_max_exponents.ndim != 1:
            raise RuntimeError("The max exponents for the weight tensor should be 1D at this point")

        bias_max_exponents = None
        if self.bias is not None:
            bias_max_exponents = self.bias.max_exponents

        weight_max_exponents, bias_max_exponents = self._adjust_max_exponents(
            input_max_exponents, weight_max_exponents, bias_max_exponents, self.n_groups
        )
        self.weights.adjusted_max_exponents = weight_max_exponents
        if self.bias is not None:
            self.bias.adjusted_max_exponents = bias_max_exponents

        # Change the axis to be 2D, because we actually have 2D max exponents
        self.weights.axis = (0, 1)

        # Final sanity check that max exponents match
        out_group_channels = weight_max_exponents.shape[0] // n_groups
        in_channels = weight_max_exponents.shape[1]
        weight_input_mul_exponents = (
            weight_max_exponents.reshape([n_groups, out_group_channels, in_channels])
            + input_max_exponents.reshape([n_groups, 1, in_channels])
        ).reshape([-1, in_channels])[:, 0]
        # FIXME for Grouped Convolutions>
        if bias_max_exponents is not None and not np.allclose(weight_input_mul_exponents, bias_max_exponents):
            raise RuntimeError("Bias and WeightMultInput max exponents should be equal")

        # Manually saturate too large pre- and postactivation max exponents
        if "reshape_mode" in self._node.attributes and self._node.attributes["reshape_mode"] != "None":
            return
        # FIXME Make this more general and investigate method to regulate exponents between output and preactivation
        output = tensor_quant_infos[self._node.outputs[0].name]
        output_max_exponents = output.adjusted_max_exponents

        diff = output_max_exponents - weight_input_mul_exponents
        max_dev = 2
        mask = (diff > max_dev) & (output_max_exponents > 5)
        output_max_exponents[mask] = weight_input_mul_exponents[mask] + max_dev

        if len(output.tensor.consumers) > 0 and output.tensor.consumers[0].op_type == "Swish":
            activation_tensor = output.tensor.consumers[0].outputs[0]
            tensor_quant_infos[activation_tensor.name].adjusted_max_exponents[mask] = (
                weight_input_mul_exponents[mask] + max_dev
            )

    @staticmethod
    def _adjust_max_exponents(
        input_max_exponents: NDArray[Any],
        weight_max_exponents: NDArray[Any],
        bias_max_exponents: NDArray[Any],
        n_groups: int,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Adjust weight max exponents to input max exponents.

        Args:
            input_max_exponents (NDArray[Any]): Input max exponents of input tensor
            weight_max_exponents (NDArray[Any]): Weight max exponents
            bias_max_exponents (NDArray[Any]): Bias max exponents
            n_groups (int): number of groups used in convolution

        Returns:
            tuple[NDArray[Any], NDArray[Any]]: Adjusted weight, bias max exponents

        Raises:
            RuntimeError: If input channels does not match groups parameter
        """
        if input_max_exponents.shape[0] % n_groups != 0:
            raise RuntimeError(f"Input chanels should be divisble into {n_groups} groups")
        in_channels = input_max_exponents.shape[0] // n_groups

        # Aggregate tensor ranges in groups of input

        # Make sure that the inner sum between weights and inputs is always the same
        # In the actual implementation this is implemented by shifting the mantissa of the weights
        # weight_ranges: [Cout, Cin]
        # input_ranges: [Cin]
        # This needs to be calculated groupwise
        weight_max_exponents = np.repeat(weight_max_exponents[:, np.newaxis], repeats=in_channels, axis=1)
        weight_input_add = weight_max_exponents.reshape([n_groups, -1, in_channels]) + input_max_exponents.reshape(
            [n_groups, 1, in_channels]
        )
        weight_input_add = weight_input_add.reshape([-1, in_channels])

        max_max_exponent = weight_input_add.max(axis=-1)
        adjustment = max_max_exponent[..., None] - weight_input_add
        adjusted_weight_max_exponents = weight_max_exponents + adjustment

        # Ensure that max exponents for inner sums and bias match for each output channel
        # only take first channel and group of Cout because all input channels are the same
        if bias_max_exponents is not None:
            bias_exponents_larger = max_max_exponent + 1 < bias_max_exponents
            # Either adjust bias or weight ranges, according to which is greater
            # bias_range_higher = bias_weight_input_ratio > 1.0
            weight_exponents_larger = ~bias_exponents_larger
            adjusted_weight_max_exponents[bias_exponents_larger] += (
                bias_max_exponents[bias_exponents_larger] - max_max_exponent[bias_exponents_larger]
            )[:, None]

            adjusted_bias_max_exponents = bias_max_exponents.copy()
            adjusted_bias_max_exponents[weight_exponents_larger] = max_max_exponent[weight_exponents_larger]
        else:
            adjusted_bias_max_exponents = None

        return adjusted_weight_max_exponents, adjusted_bias_max_exponents


class ShortcutHandler(LayerHandlerBase):
    """Handler for shortcut layers."""

    def __init__(self, node: Node):
        """Initialize a new DefaultLayerHandler.

        Args:
            node (Node): Node to quantize
        """
        super().__init__(node)

        # Only main output and not debug outputs
        main_output = node.outputs[0]

        output = TensorQuantInfo(
            tensor=main_output,
            n_bits=8,  # By default we quantize with 8bit
            n_fraction_bits=7,
            axis=1,  # In almost all cases, we have shapes of the form [N, C, H, W]
        )
        self.quant_infos = {output.tensor.name: output}

        # If any of the inputs are INITIALIZER tensors, add them as well
        input0, input1 = node.inputs
        if input0.tensor_type is TensorType.INITIALIZER:
            self.quant_infos[input0.name] = TensorQuantInfo(tensor=input0, n_bits=8, n_fraction_bits=7, axis=1)
        if input1.tensor_type is TensorType.INITIALIZER:
            self.quant_infos[input1.name] = TensorQuantInfo(tensor=input1, n_bits=8, n_fraction_bits=7, axis=1)

        self.input0_name = input0.name
        self.input1_name = input1.name

    def tensor_quant_infos(self) -> dict[str, TensorQuantInfo]:
        """Return a dict mapping tensor names to TensorQuantInfo objects.

        Returns:
            dict[str, TensorQuantInfo]: Dict mapping tensor names to TensorQuantInfo objects
        """
        return self.quant_infos

    # def tensor_range_postprocess(self, tensor_quant_infos: dict[str, TensorQuantInfo]) -> None:
    #     """Postprocess tensor ranges after they have been collected by the calibrator.

    #     For Short layers, the max exponents between input0 and input1 should be the same
    #     This may occur, when outlier filtering is applied.

    #     Args:
    #         tensor_quant_infos (dict[str, TensorQuantInfo]): Dict mapping tensor names to TensorQuantInfo objects, which
    #             should contain TensorQuantInfos for all tensors in the model.

    #     Returns:
    #         None: postprocessing is done internally.
    #     """
    #     input0_quant_info = tensor_quant_infos[self.input0_name]
    #     input1_quant_info = tensor_quant_infos[self.input1_name]
    #     max_exponents0 = input0_quant_info.max_exponents
    #     max_exponents1 = input1_quant_info.max_exponents
    #     input0_higher = max_exponents0 > max_exponents1

    #     new_max_exponents = np.where(input0_higher, max_exponents0, max_exponents1)
    #     new_quantization_ranges = np.where(
    #         input0_higher, input0_quant_info.quantization_ranges, input0_quant_info.quantization_ranges
    #     )

    #     input0_quant_info.max_exponents = new_max_exponents.copy()
    #     input1_quant_info.max_exponents = new_max_exponents.copy()

    #     input0_quant_info.quantization_ranges = new_quantization_ranges.copy()
    #     input1_quant_info.quantization_ranges = new_quantization_ranges.copy()


class ResizeHandler(DefaultLayerHandler):
    """Handler for resize layers."""

    def __init__(self, node: Node):
        """Initialize a new ResizeHandler.

        Args:
            node (Node): Node to quantize
        """
        super().__init__(node)
        self.input_name = node.inputs[0].name
        self.output_name = node.outputs[0].name

    def tensor_range_postprocess(self, tensor_quant_infos: dict[str, TensorQuantInfo]) -> None:
        """Postprocess tensor ranges after they have been collected by the calibrator.

        For Resize layers, the max exponents between input and output should not change in most cases.
        The only exception is bilinear filtering

         Returns:
            None: postprocessing is done internally.
        """
        input_quant_info = tensor_quant_infos[self.input_name]
        output_quant_info = tensor_quant_infos[self.output_name]
        output_quant_info.adjusted_max_exponents = input_quant_info.adjusted_max_exponents.copy()


class ConcatHandler(LayerHandlerBase):
    """Handler for concat layers."""

    def __init__(self, node: Node):
        """Initialize a new ConcatHandler.

        Args:
            node (Node): Node to quantize
        """
        super().__init__(node)

        # Only main output and not debug outputs
        main_output = node.outputs[0]

        output = TensorQuantInfo(
            tensor=main_output,
            n_bits=8,  # By default we quantize with 8bit
            n_fraction_bits=7,
            axis=1,  # In almost all cases, we have shapes of the form [N, C, H, W]
        )
        self.quant_infos = {output.tensor.name: output}

        # If any of the inputs are INITIALIZER tensors, add them as well
        for input_tensor in node.inputs:
            if input_tensor.tensor_type is TensorType.INITIALIZER:
                self.quant_infos[input_tensor.name] = TensorQuantInfo(
                    tensor=input_tensor, n_bits=8, n_fraction_bits=7, axis=1
                )

    def tensor_quant_infos(self) -> dict[str, TensorQuantInfo]:
        """Return a dict mapping tensor names to TensorQuantInfo objects.

        Returns:
            dict[str, TensorQuantInfo]: Dict mapping tensor names to TensorQuantInfo objects
        """
        return self.quant_infos


class LayernormHandler(LayerHandlerBase):
    """Handler for layernorm layers."""

    def __init__(self, node: Node):
        """Initialize a new Handler.

        Args:
            node (Node): Layer Norm node to quantize
        """
        super().__init__(node)

        # Main output
        results = {}
        main_output = node.outputs[0]

        results[main_output.name] = TensorQuantInfo(
            tensor=main_output,
            n_bits=8,  # By default we quantize with 8bit
            n_fraction_bits=7,
            axis=1,  # In almost all cases, we have shapes of the form [N, C, H, W]
        )

        # Scale tensor
        if len(node.inputs) >= 2:
            scale_tensor = node.inputs[1]
            results[scale_tensor.name] = TensorQuantInfo(
                tensor=scale_tensor,
                n_bits=8,
                n_fraction_bits=7,
                axis=0,  # 1D tensor
            )
        # Bias tensor
        if len(node.inputs) == 3:
            bias_tensor = node.inputs[2]
            results[bias_tensor.name] = TensorQuantInfo(
                tensor=bias_tensor,
                n_bits=16,  # Biases are quantized to 16 bits
                n_fraction_bits=15,
                axis=0,  # 1D tensor
            )

        self.results = results

    def tensor_quant_infos(self) -> dict[str, TensorQuantInfo]:
        """Return a dict mapping tensor names to TensorQuantInfo objects.

        Returns:
            dict[str, TensorQuantInfo]: Dict mapping tensor names to TensorQuantInfo objects
        """
        return self.results


class MaxPoolHandler(DefaultLayerHandler):
    """Handler for max pool layers."""

    def __init__(self, node: Node):
        """Initialize the MaxPoolHandler by keeping track of input output tensor names."""
        super().__init__(node)
        self.input_name = node.inputs[0].name
        self.output_name = node.outputs[0].name

    def tensor_range_postprocess(self, tensor_quant_infos: dict[str, TensorQuantInfo]) -> None:
        """Postprocess tensor ranges after they have been collected by the calibrator.

        For MaxPool layers, the max exponents between input and output should not change (required by v-NN Mapper).
        This may occur, when outlier filtering is applied.

         Returns:
            None: postprocessing is done internally.
        """
        input_quant_info = tensor_quant_infos[self.input_name]
        output_quant_info = tensor_quant_infos[self.output_name]
        output_quant_info.adjusted_max_exponents = input_quant_info.adjusted_max_exponents.copy()


class SoftmaxHandler(DefaultLayerHandler):
    """Handler for softmax layers."""

    def __init__(self, node: Node):
        """Initialize a new Handler.

        Args:
            node (Node): Softmax node to quantize

        Raises:
            RuntimeError: For invalid group parameter
        """
        super().__init__(node)

        # Main output
        self.input_name = node.inputs[0].name
        self.output_name = node.outputs[0].name
        group = node.attributes.get("group", [1])
        if not len(group) == 1:
            raise RuntimeError("Group parameter should only have one element")

        self.groups = group[0]

    def tensor_range_postprocess(self, tensor_quant_infos: dict[str, TensorQuantInfo]) -> None:
        """Postprocess tensor ranges after they have been collected by the calibrator.

        For (grouped) softmax as used in our attention implementation, it necessary to ensure that
        all max exponents shared inbetween groups are the same

        Args:
            tensor_quant_infos (dict[str, TensorQuantInfo]): Dict mapping tensor names to TensorQuantInfo objects, which
                should contain TensorQuantInfos for all tensors in the model.

        Returns:
            None: postprocessing is done internally.
        """
        input_info = tensor_quant_infos[self.input_name]
        output_info = tensor_quant_infos[self.output_name]

        tensor_shape = input_info.tensor.shape
        n_channels_per_group = tensor_shape[1] // self.groups

        input_max_exponents = input_info.max_exponents
        output_max_exponents = output_info.max_exponents

        input_max_exponents_per_group = input_max_exponents.reshape([self.groups, -1]).max(axis=-1, keepdims=True)
        input_info.adjusted_max_exponents = input_max_exponents_per_group.repeat(
            axis=1, repeats=n_channels_per_group
        ).reshape([-1])

        output_max_exponents_per_group = output_max_exponents.reshape([self.groups, -1]).max(axis=-1, keepdims=True)
        output_info.adjusted_max_exponents = output_max_exponents_per_group.repeat(
            axis=1, repeats=n_channels_per_group
        ).reshape([-1])


class GatherHandler(DefaultLayerHandler):
    """Handler for gather layers."""

    def __init__(self, node: Node):
        """Initialize the GatherHandler."""
        super().__init__(node)

    def tensor_quant_infos(self) -> dict[str, TensorQuantInfo]:
        """Return a dict mapping tensor names to TensorQuantInfo objects.

        Returns:
            dict[str, TensorQuantInfo]: Dict mapping tensor names to TensorQuantInfo objects

        Raises:
            RuntimeError: If the initializer has the wrong shape.
        """
        output = super().tensor_quant_infos()

        # In some cases the gather layer has weights as inputs, which need to be quantized
        # An example of this would be the initial embeddings layer of transformer networks
        input_tensor = self._node.inputs[0]
        if input_tensor.tensor_type is TensorType.INITIALIZER:
            # Make sure this is a weight tensor in the standard videantis format [1, C, 1, W]
            shape = input_tensor.shape
            if shape[0] != 1 or shape[2] != 1:
                raise RuntimeError("Gather initializer input tensor is not in format [1, C, 1, W]")

            output[input_tensor.name] = TensorQuantInfo(
                tensor=input_tensor,
                n_bits=8,
                n_fraction_bits=7,
                axis=1,
            )
        return output


class ReluHandler(DefaultLayerHandler):
    """Handler for relu layers."""

    def __init__(self, node: Node):
        """Initialize the GatherHandler."""
        super().__init__(node)
        self.input_name = node.inputs[0].name
        self.output_name = node.outputs[0].name

    def tensor_range_postprocess(self, tensor_quant_infos: dict[str, TensorQuantInfo]) -> None:
        """Adjust the max exponents of the preactivation tensor for Relu layers.

        For Relu layers we have the strong requirement that the max exponent of the output and input need to be the
        same. Here, we clip the max exponent of the input to the profiled max exponents of the output, because
        large negative values will be clipped to 0 anyway and this way, we may even achieve additional accuracy.
        """
        input_info = tensor_quant_infos[self.input_name]
        output_info = tensor_quant_infos[self.output_name]

        input_info.adjusted_max_exponents[:] = output_info.adjusted_max_exponents


class SigmoidHandler(DefaultLayerHandler):
    """Handler for Sigmoid layers."""

    def tensor_range_postprocess(self, tensor_quant_infos: dict[str, TensorQuantInfo]) -> None:
        """Postprocess tensor ranges after they have been collected by the calibrator.

        For (grouped) softmax as used in our attention implementation, it necessary to ensure that
        all max exponents shared inbetween groups are the same

        Args:
            tensor_quant_infos (dict[str, TensorQuantInfo]): Dict mapping tensor names to TensorQuantInfo objects, which
                should contain TensorQuantInfos for all tensors in the model.

        Returns:
            None: postprocessing is done internally.
        """
        # Sigmoid max exponents are always exptected to be 0

        # In some cases the gather layer has weights as inputs, which need to be quantized
        # An example of this would be the initial embeddings layer of transformer networks
        output_tenor = self._node.outputs[0]
        output_quant_info = tensor_quant_infos[output_tenor.name]
        output_quant_info.adjusted_max_exponents.fill(0)


class GridSampleHandler(DefaultLayerHandler):
    """Handler for GridSample layers."""

    def __init__(self, node: Node):
        """Initialize the GridSampleHandler."""
        super().__init__(node)
        self.input_name = node.inputs[0].name
        self.grid_name = node.inputs[1].name
        self.output_name = node.outputs[0].name

    def tensor_quant_infos(self) -> dict[str, TensorQuantInfo]:
        """Return quant infos for the grid and output tensors, both quantized with 7 fraction bits."""
        # Add grid sample tensor as well
        return {
            self.grid_name: TensorQuantInfo(tensor=self._node.inputs[1], n_bits=8, n_fraction_bits=7, axis=1),
            self.output_name: TensorQuantInfo(tensor=self._node.outputs[0], n_bits=8, n_fraction_bits=7, axis=1),
        }


QUANT_HANDLER_OP_REGISTRY: dict[str, type[LayerHandlerBase]] = {
    "default": DefaultLayerHandler,
    "vidConv": VidConvHandler,
    "Shortcut": ShortcutHandler,
    "vidLayerNorm": LayernormHandler,
    "Resize": ResizeHandler,
    "Concat": ConcatHandler,
    "vidMaxPool": MaxPoolHandler,
    "vidSoftmax": SoftmaxHandler,
    "Gather": GatherHandler,
    "RMSNormalization": LayernormHandler,  # Exactly the same behaviour as layernorm
    "Relu": ReluHandler,
    "Relu6": ReluHandler,  # Exactly the same behaviour as Relu
    "Sigmoid": SigmoidHandler,
    "ConvTranspose": VidConvHandler,  # Exactly the same behaviour as VidConvHandler
    "vidGridSample": GridSampleHandler,
}
