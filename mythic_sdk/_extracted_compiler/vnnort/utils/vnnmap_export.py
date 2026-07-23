import logging
from typing import Any

import numpy as np

from vnnmap.capnproto_interface import (
    ActivationType,
    AveragePoolAttributes,
    ConcatAttributes,
    ConvAttributes,
    ConvTransposeAttributes,
    FlattenAttributes,
    GatherAttributes,
    GridSampleAttributes,
    LayerNormAttributes,
    LayerType,
    MaxPoolAttributes,
    RERTransformationAttributes,
    RETRTransformationAttributes,
    RMSNormalizationAttributes,
    RTRTransformationAttributes,
    ReshapeAttributes,
    ResizeAttributes,
    RopeAttributes,
    ScatterAttributes,
    ShortcutAttributes,
    SliceAttributes,
    SoftmaxAttributes,
    SqueezeAttributes,
)
from vnnmap.capnproto_interface import TensorType as CapnProtoTensorType
from vnnmap.capnproto_interface import (
    TransposeAttributes,
)
from vnnmap.network import CapnprotoNetwork
from vnnort.optimizer.symbolic_shape_inference import _resolve_axis, _resolve_dynamic_shape
from vnnort.quantizer.quant_utils import TensorQuantInfo
from vnnort.utils.onnx_utils.graph_helper import Node, ONNXGraphHelper
from vnnort.utils.onnx_utils.graph_helper import TensorType as ONNXTensorType

logger = logging.getLogger(__name__)


class VNNMapExporter:
    """Export an ONNX graph to an v-NN Mapper format using CapnProto serialization.

    This class handles the conversion of an ONNX graph to v-NN Mapper format by creating
    a CapnProto representation of the network's tensors and layers.
    """

    def __init__(self, name: str, graph_helper: ONNXGraphHelper, tensor_quant_infos: dict[str, TensorQuantInfo]):
        """Initialize the VNNMapExporter.

        Args:
            name (str): The name of the network to be created.
            graph_helper (ONNXGraphHelper): Helper object for accessing ONNX graph information.
            tensor_quant_infos (dict[str, TensorQuantInfo]): Dictionary mapping tensor names to
                their quantization information.
        """
        self._network = CapnprotoNetwork(name, consume_data=False)
        self._graph_helper = graph_helper
        self._tensor_quant_infos = tensor_quant_infos

        # Dictionary mapping onnx node names to their conversion function
        self.onnx_to_capnproto_convert = {
            "vidConv": self._parse_vid_conv,
            "vidMaxPool": self._parse_max_pool,
            "Shortcut": self._parse_shortcut,
            "vidAveragePool": self._parse_average_pool,
            "vidFlatten": self._parse_flatten,
            "Concat": self._parse_concat,
            "Resize": self._parse_resize,
            "vidLayerNorm": self._parse_layer_norm,
            "vidSoftmax": self._parse_softmax,
            "Squeeze": self._parse_squeeze,
            "Reshape": self._parse_reshape,
            "Transpose": self._parse_transpose,
            "Gather": self._parse_gather,
            "Slice": self._parse_slice,
            "RMSNormalization": self._parse_rms_norm,
            "vidRope": self._parse_rope,
            "Expand": self._parse_expand,
            "RETRTransformation": self._parse_retr_transformation,
            "RTRTransformation": self._parse_rtr_transformation,
            "RERTransformation": self._parse_rer_transformation,
            "vidScatter": self._parse_scatter,
            "ConvTranspose": self._parse_conv_transpose,
            "vidGridSample": self._parse_grid_sample,
        }

    def export(self) -> CapnprotoNetwork:
        """Execute the export process.

        Converts the ONNX graph to v-NN Mapper format by first adding tensors, then
        adding layers to the CapnProto network representation.

        Returns:
            CapnprotoNetwork: The constructed network in CapnProto format.
        """
        self._add_tensors()
        self._add_layers()

        return self._network

    def _add_tensors(self) -> None:  # noqa
        """
        Add tensors to the CapnProto network representation.

        This method processes the tensors stored in `_tensor_quant_infos`, applies quantization,
        and adds them to the CapnProto network.
        """
        # Almost all relevant tensors come from the tensor_quant_infos dict as a result of the quantizer
        for tensor_name, quant_info in self._tensor_quant_infos.items():
            tensor = quant_info.tensor
            max_exponents = quant_info.max_exponents

            # For weights, we have 2D max exponents (output AND input channels). This stems from the fact that
            # we want so simulate the weight bit shift in out own onnx implementation.
            # This is not needed here anymore
            quant_axis = quant_info.axis

            # Weights may have a 2D quant axis for vidORT purposes (output-channel, input-channel)
            # vnnmap is only interested in the first
            if isinstance(quant_axis, int):
                pass  # No need to index if it's already an int
            else:
                quant_axis = quant_axis[0]

            if max_exponents.ndim > 2:
                raise ValueError("Max exponents can only be 1D or 2D")

            n_bits = quant_info.n_bits
            # TODO: Remove fixed point calculation, when this is removed from capnproto schema
            # fixed_point_data = quantize_values(floating_point_data, max_exponents, n_bits, quant_axis)  # type: ignore

            # VidORT supports dynamic batch sizes, but vnnmap does not (yet?), set dynamic batch size to 1
            shape = tensor.shape
            if shape is None or len(shape) == 0:
                raise RuntimeError(f"Shapes must not be empty. (Tensor: {tensor.name})")
            if shape[0] == -1:
                shape[0] = 1

            self._network.add_tensor(
                name=tensor_name,
                tensor_type=self._parse_tensor_type(tensor.tensor_type),
                data=tensor.data,
                fixed_point_data=None,
                max_exponents=max_exponents,
                adjusted_max_exponents=quant_info.adjusted_max_exponents,
                shape=shape,
                n_bits=n_bits,
                quant_axis=quant_axis,
            )

        # Sometimes tjere are integer tensors (e.g. position ids in transformers) which are not quantized, but still required
        for tensor in self._graph_helper.tensors.values():
            if np.issubdtype(tensor.dtype, np.integer):
                # VidORT supports dynamic batch sizes, but vnnmap does not (yet?), set dynamic batch size to 1
                shape = tensor.shape
                if shape == [] and tensor.data is not None and tensor.data.size == 1:
                    shape = [1]
                if shape is None or len(shape) == 0:
                    raise RuntimeError(f"Shapes must not be empty. (Tensor: {tensor.name})")
                if shape[0] == -1:
                    shape[0] = 1
                self._network.add_tensor(
                    name=tensor.name,
                    tensor_type=self._parse_tensor_type(tensor.tensor_type),
                    data=tensor.data,
                    fixed_point_data=None,
                    max_exponents=None,
                    adjusted_max_exponents=None,
                    shape=shape,
                    n_bits=None,
                    quant_axis=None,
                )

    def _add_layers(self) -> None:
        """
        Add different layer types from the optimized ONNX model to the CapnProto network.

        This method iterates over the ONNX graph layers and maps each layer type
        to its corresponding CapnProto representation using predefined converters.
        The converted layers are then added to the network.
        """
        for layer_name, layer in self._graph_helper.nodes.items():
            # Skip activations. These are merged in the corresponding parse functions of conv and shortcut
            if self._is_activation_function(layer):
                continue

            converter_func = self.onnx_to_capnproto_convert.get(layer.op_type)
            if converter_func is None:
                msg = f"Layer type {layer.op_type} is not supported."
                raise ValueError(msg)

            layer_type, inputs, outputs, attributes = converter_func(layer)
            self._network.add_layer(layer_name, layer_type, inputs, outputs, attributes)

    def _parse_tensor_type(self, tensor_type: ONNXTensorType) -> Any:
        """Convert ONNX Tensor type to CapnProto tensor type.

        Args:
            tensor_type (ONNXTensorType): The ONNX tensor type to convert.

        Returns:
            Any: The corresponding CapnProto tensor type.

        Raises:
            ValueError: If the ONNX tensor type is invalid.
        """
        match tensor_type:
            case ONNXTensorType.GRAPH_INPUT:
                return CapnProtoTensorType.graphInput
            case ONNXTensorType.GRAPH_OUTPUT:
                return CapnProtoTensorType.graphOutput
            case ONNXTensorType.NODE_OUTPUT:
                return CapnProtoTensorType.dynamic
            case ONNXTensorType.INITIALIZER:
                return CapnProtoTensorType.static
            case _:
                raise ValueError("Invalid ONNX tensor type")

    def _parse_activation_type(self, activation_type: str) -> Any:
        """Convert activation type string to CapnProto ActivationType enum.

        Args:
            activation_type (str): The activation type string to convert.

        Returns:
            Any: The corresponding CapnProto ActivationType enum.

        Raises:
            ValueError: If the activation type is not supported.
        """
        match str(activation_type):
            case "Relu":
                return ActivationType.relu
            case "Relu6":
                return ActivationType.relu6
            case "Swish":
                return ActivationType.swish
            case "HardSwish":
                return ActivationType.hardswish
            case "Clip":
                return ActivationType.clip
            case "HardSigmoid":
                return ActivationType.hardsigmoid
            case "Sigmoid":
                return ActivationType.sigmoid
            case "Gelu":
                return ActivationType.gelu
            case "Mish":
                return ActivationType.mish
            case "LeakyRelu":
                return ActivationType.leakyrelu
            case _:
                raise ValueError(f"Activation type {activation_type} not supported.")

    def _parse_vid_conv(self, node: Node) -> tuple[Any, list[str], list[str], Any]:  # noqa: C901
        """Convert vid convolution attributes to CapnProto ConvAttributes message.

        Args:
            node (Node): The ONNX node containing the vid convolution attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto conv attributes object.

        Raises:
            ValueError: If the reshape mode is not supported.
        """
        ReshapeMode = ConvAttributes.ReshapeMode
        attributes = node.attributes
        reshape_mode_str = attributes["reshape_mode"]
        match reshape_mode_str:
            case "None" | None:
                reshape_mode = ReshapeMode.none
            case "MUL_EXPAND":
                reshape_mode = ReshapeMode.mulExpand
            case "TRANSFORMER_QK":
                reshape_mode = ReshapeMode.transformerQK
            case "TRANSFORMER_V":
                reshape_mode = ReshapeMode.transformerV
            case "FLATTEN_W":
                reshape_mode = ReshapeMode.flattenW
            case _:
                msg = f"Reshape mode {reshape_mode_str} not supported."
                raise ValueError(msg)

        attributes_message = ConvAttributes.new_message(
            activation=ActivationType.linear,
            reshapeMode=reshape_mode,
        )
        if attributes.get("dilations") is not None:
            attributes_message.init("dilations", len(attributes["dilations"]))
            attributes_message.dilations = attributes["dilations"]
        if attributes.get("group") is not None:
            attributes_message.group = attributes["group"]

        if attributes.get("pads") is not None:
            attributes_message.init("pads", len(attributes["pads"]))
            attributes_message.pads = attributes["pads"]
        if attributes.get("strides") is not None:
            attributes_message.init("strides", len(attributes["strides"]))
            attributes_message.strides = attributes["strides"]
        if attributes.get("dim") is not None:
            attributes_message.dim = attributes["dim"]
        if attributes.get("reshapeModeGroups") is not None:
            attributes_message.init("reshapeModeGroups", len(attributes["reshapeModeGroups"]))
            attributes_message.reshapeModeGroups = attributes["reshapeModeGroups"]

        # Ignore ONNX kernel shape attribute and directly infer it from weight tensor
        attributes_message.init("kernelShape", 2)
        weight_tensor = node.inputs[1]
        if len(weight_tensor.shape) == 2:
            kernel_shape = [1, 1]
            attributes_message.kernelShape = kernel_shape
        elif len(weight_tensor.shape) == 4:
            attributes_message.kernelShape = weight_tensor.shape[2:]
        else:
            raise ValueError(f"Cannot handle kernel of shape {weight_tensor.shape}")

        inputs = [tensor.name for tensor in node.inputs if tensor is not None]

        if len(inputs) < 2 or len(inputs) > 3:
            msg = f"Invalid number of inputs conv node {len(inputs)}."
            raise ValueError(msg)
        attributes_message.input = self._network.get_tensor(inputs[0]).idx
        attributes_message.weight = self._network.get_tensor(inputs[1]).idx
        if len(inputs) == 3:
            attributes_message.bias = self._network.get_tensor(inputs[2]).idx

        outputs = [node.outputs[0].name]
        # Check if this vidConv is followed by an activation layer and merge it by rewiring its outputs
        if len(node.outputs[0].consumers) == 1 and self._is_activation_function(node.outputs[0].consumers[0]):
            activation_layer = node.outputs[0].consumers[0]
            activation_output_tensor = activation_layer.outputs[0]
            outputs = [activation_output_tensor.name]
            attributes_message.output = self._network.get_tensor(activation_output_tensor.name).idx
            attributes_message.preActivationOutput = self._network.get_tensor(node.outputs[0].name).idx

            # Overwrite the activation attribute
            attributes_message.activation = self._parse_activation_type(activation_layer.op_type)

        else:
            # If there is not activation, these fields are the same
            attributes_message.output = self._network.get_tensor(node.outputs[0].name).idx
            attributes_message.preActivationOutput = self._network.get_tensor(node.outputs[0].name).idx

        layer_type = LayerType.conv
        return layer_type, inputs, outputs, attributes_message

    def _parse_max_pool(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert max pooling attributes to CapnProto MaxPoolAttributes message.

        Args:
            node (Node): The ONNX node containing the max pooling attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto max pool attributes object.

        Raises:
            ValueError: If auto pad mode or ceil mode is not implemented.
        """
        auto_pad = None
        attributes = node.attributes
        match (attributes["auto_pad"]):
            case "NOTSET" | None:
                auto_pad = MaxPoolAttributes.AutoPadMode.notSet
            case "SAME_UPPER":
                auto_pad = MaxPoolAttributes.AutoPadMode.sameUpper
            case "SAME_LOWER":
                auto_pad = MaxPoolAttributes.AutoPadMode.sameLower
            case "VALID":
                auto_pad = MaxPoolAttributes.AutoPadMode.valid
            case _:
                raise ValueError(f"Auto pad mode {attributes['auto_pad']} not implemented.")

        ceil_mode = None
        match (attributes["ceil_mode"]):
            case 0 | None:  # Default in onnx
                ceil_mode = MaxPoolAttributes.CeilMode.floor
            case 1:
                ceil_mode = MaxPoolAttributes.CeilModel.ceil
            case _:
                raise ValueError(f"Ceil mode {attributes['ceil_mode']} not implemented.")

        attributes_message = MaxPoolAttributes.new_message(
            autoPad=auto_pad,
            ceilMode=ceil_mode,
        )
        if attributes.get("kernel_shape") is not None:
            attributes_message.init("kernelShape", len(attributes["kernel_shape"]))
            attributes_message.kernelShape = attributes["kernel_shape"]
        if attributes.get("dilations") is not None:
            attributes_message.init("dilations", len(attributes["dilations"]))
            attributes_message.dilations = attributes["dilations"]
        if attributes.get("pads") is not None:
            attributes_message.init("pads", len(attributes["pads"]))
            attributes_message.pads = attributes["pads"]
        if attributes.get("strides") is not None:
            attributes_message.init("strides", len(attributes["strides"]))
            attributes_message.strides = attributes["strides"]
        inputs = [tensor.name for tensor in node.inputs]
        outputs = [node.outputs[0].name]
        attributes_message.input = self._network.get_tensor(inputs[0]).idx
        attributes_message.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.maxPool
        return layer_type, inputs, outputs, attributes_message

    def _parse_shortcut(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert shortcut attributes to CapnProto ShortcutAttributes message.

        Args:
            node (Node): The ONNX node containing the shortcut attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto shortcut attributes object.

        Raises:
            ValueError: If the mode or reshape mode is not supported.
        """
        Mode = ShortcutAttributes.Mode
        mode = None
        attributes = node.attributes
        match attributes["mode"]:
            case "addition":
                mode = Mode.addition
            case "multiplication":
                mode = Mode.multiplication
            case "division":
                mode = Mode.division
            case _:
                raise ValueError(f"Mode {attributes['mode']} not supported.")

        ReshapeMode = ShortcutAttributes.ReshapeMode
        reshape_mode = None
        match attributes["reshape_mode"]:
            case None | "None":
                reshape_mode = ReshapeMode.none
            case "TRANSFORMER_QK":
                reshape_mode = ReshapeMode.transformerQK
            case _:
                raise ValueError(f"Reshape mode {reshape_mode} not supported.")

        attributes_message = ShortcutAttributes.new_message(
            mode=mode,
            reshapeMode=reshape_mode,
            activation=ActivationType.linear,
        )
        # Check if this Shortcut is followed by an activation layer
        if len(node.outputs[0].consumers) == 1 and self._is_activation_function(node.outputs[0].consumers[0]):
            activation_layer = node.outputs[0].consumers[0]
            activation_output_tensor = activation_layer.outputs[0]
            outputs = [activation_output_tensor.name]
            attributes_message.output = self._network.get_tensor(activation_output_tensor.name).idx
            attributes_message.preActivationOutput = self._network.get_tensor(node.outputs[0].name).idx

            # Overwrite the activation attribute
            attributes_message.activation = self._parse_activation_type(activation_layer.op_type)

        else:
            # If there is not activation, these fields are the same
            attributes_message.output = self._network.get_tensor(node.outputs[0].name).idx
            attributes_message.preActivationOutput = self._network.get_tensor(node.outputs[0].name).idx
            outputs = [node.outputs[0].name]

        if attributes.get("group") is not None:
            attributes_message.init("reshapeModeGroups", len(attributes["group"]))
            attributes_message.reshapeModeGroups = attributes["group"]

        inputs = [tensor.name for tensor in node.inputs]
        attributes_message.input0 = self._network.get_tensor(inputs[0]).idx
        attributes_message.input1 = self._network.get_tensor(inputs[1]).idx

        layer_type = LayerType.shortcut
        return layer_type, inputs, outputs, attributes_message

    def _parse_average_pool(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert average pooling attributes to CapnProto AveragePoolAttributes message.

        Args:
            node (Node): The ONNX node containing the average pooling attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto average pool attributes object.

        Raises:
            ValueError: If auto pad mode, ceil mode, or count include pad is not implemented.
        """
        auto_pad = None
        attributes = node.attributes
        match (attributes["auto_pad"]):
            case "NOTSET" | None:
                auto_pad = MaxPoolAttributes.AutoPadMode.notSet
            case "SAME_UPPER":
                auto_pad = MaxPoolAttributes.AutoPadMode.sameUpper
            case "SAME_LOWER":
                auto_pad = MaxPoolAttributes.AutoPadMode.sameLower
            case "VALID":
                auto_pad = MaxPoolAttributes.AutoPadMode.valid
            case _:
                raise ValueError(f"Auto pad mode {attributes['auto_pad']} not implemented.")

        ceil_mode = None
        match (attributes["ceil_mode"]):
            case 0 | None:
                ceil_mode = MaxPoolAttributes.CeilMode.floor
            case 1:
                ceil_mode = MaxPoolAttributes.CeilModel.ceil
            case _:
                raise ValueError(f"Ceil mode {attributes['ceil_mode']} not implemented.")

        count_include_pad = False
        match (attributes["count_include_pad"]):
            case 0 | None:
                count_include_pad = False
            case 1:
                count_include_pad = True
            case _:
                raise ValueError(f"Value {attributes['count_include_pad']} for countIncludePad not implemented.")

        attributes_message = AveragePoolAttributes.new_message(
            autoPad=auto_pad,
            ceilMode=ceil_mode,
            countIncludePad=count_include_pad,
        )
        if attributes.get("kernel_shape") is not None:
            attributes_message.init("kernelShape", len(attributes["kernel_shape"]))
            attributes_message.kernelShape = attributes["kernel_shape"]
        if attributes.get("dilations") is not None:
            attributes_message.init("dilations", len(attributes["dilations"]))
            attributes_message.dilations = attributes["dilations"]
        if attributes.get("pads") is not None:
            attributes_message.init("pads", len(attributes["pads"]))
            attributes_message.pads = attributes["pads"]
        if attributes.get("strides") is not None:
            attributes_message.init("strides", len(attributes["strides"]))
            attributes_message.strides = attributes["strides"]
        if attributes.get("output_dim") is not None:
            attributes_message.outputDim = attributes["output_dim"]

        inputs = [tensor.name for tensor in node.inputs]
        outputs = [node.outputs[0].name]
        attributes_message.input = self._network.get_tensor(inputs[0]).idx
        attributes_message.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.averagePool
        return layer_type, inputs, outputs, attributes_message

    def _parse_flatten(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert flatten attributes to CapnProto FlattenAttributes message.

        Args:
            node (Node): The ONNX node containing the flatten attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto flatten attributes object.
        """
        attributes = node.attributes
        attributes = FlattenAttributes.new_message(axis=attributes["axis"])
        inputs = [tensor.name for tensor in node.inputs]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.flatten
        return layer_type, inputs, outputs, attributes

    def _parse_concat(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert concatenation attributes to CapnProto ConcatAttributes message.

        Args:
            node (Node): The ONNX node containing the concatenation attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto concat attributes object.
        """
        attributes = node.attributes
        axis = _resolve_axis(attributes["axis"], len(node.outputs[0].shape))

        attributes = ConcatAttributes.new_message(axis=axis)
        inputs = [tensor.name for tensor in node.inputs]
        outputs = [node.outputs[0].name]
        attributes.init("inputs", len(inputs))
        attributes.inputs = [self._network.get_tensor(name).idx for name in inputs]
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.concat
        return layer_type, inputs, outputs, attributes

    def _parse_resize(self, node: Node) -> tuple[Any, list[str], list[str], Any]:  # noqa
        """Convert resize layer input attributes to CapnProto ResizeAttributes message.

        Args:
            node(Node): The ONNX node containing the resize layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto resize attributes object.

        Raises:
            ValueError: If the number of inputs is not 3 or 4
        """
        # TODO: How to handle other resize attributes?
        node_inputs = node.inputs
        if len(node_inputs) < 3 or len(node_inputs) > 4:
            msg = f"Resize requires 3 or 4 inputs. Got {len(node_inputs)}"
            raise ValueError(msg)
        roi = node_inputs[1]
        if roi is not None and roi.data.size != 0:  # For some reason an unset ROI is encoded as an empty tensor
            msg = "ROI input in tf_crop_and_resize mode is currently not supported"
            raise ValueError(msg)
        scales = node_inputs[2]
        if scales is not None and scales.shape != [0]:
            # Transform scales into sizes
            tensor_shape = node_inputs[0].shape
            if len(tensor_shape) != len(scales.data):
                msg = f"Scales must match tensor shape. Got {scales.data} and {tensor_shape}"
            sizes = [int(scale * dim) for scale, dim in zip(scales.data, tensor_shape)]
        else:
            sizes = node_inputs[3].data.tolist()

        attributes = ResizeAttributes.new_message()

        # Check that sizes information is viable for us
        if not len(sizes) == 4:
            raise ValueError("Resize: sizes need to be 4d")
        input_shape = node_inputs[0].shape
        if input_shape[0] != sizes[0] and input_shape[1] != sizes[1]:
            raise ValueError(f"Only H and W may change. Input tensor: {input_shape}. Request sizes: {sizes}")
        attributes.init("sizes", 2)
        attributes.sizes = sizes[2:]

        inputs = [node.inputs[0].name]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx

        # antialias attribute
        if node.attributes.get("antialias") != 0:
            raise ValueError(f"antialias attribute {node.attributes["antialias"]} is not supported for Resize.")

        # axes attribute
        if node.attributes.get("axes") is not None:
            raise ValueError(f"axes attribute {node.attributes["axes"]} is not supported for Resize.")

        # coordinate_transformation_mode
        if node.attributes["coordinate_transformation_mode"] == "half_pixel":
            attributes.coordinateTransformationMode = ResizeAttributes.CoordinateTransformationMode.halfPixel
        elif node.attributes["coordinate_transformation_mode"] == "asymmetric":
            attributes.coordinateTransformationMode = ResizeAttributes.CoordinateTransformationMode.asymmetric
        elif node.attributes["coordinate_transformation_mode"] == "align_corners":
            attributes.coordinateTransformationMode = ResizeAttributes.CoordinateTransformationMode.alignCorners
        else:
            raise ValueError(
                f"coordinate_transformation_mode {node.attributes["coordinate_transformation_mode"]} is not supported for Resize."
            )

        # cubic_coeff_a attribute does not yet nned to be considered, since we do not allow it.

        # exclude_outside attribute
        if node.attributes.get("exclude_outside") != 0:
            raise ValueError(
                f"exclude_outside attribute {node.attributes["exclude_outside"]} is not supported for Resize."
            )

        # extrapolation_value attribute does not yet nned to be considered, since we do not allow tf_crop_and_resize.

        # keep_aspect_ratio_policy attribute
        if node.attributes.get("keep_aspect_ratio_policy") != "stretch":
            raise ValueError(
                f"keep_aspect_ratio_policy attribute {node.attributes["keep_aspect_ratio_policy"]} is not supported for Resize."
            )

        # mode attribute
        if node.attributes["mode"] == "nearest":
            attributes.mode = ResizeAttributes.InterpolationMode.nearest
        elif node.attributes["mode"] == "linear":
            attributes.mode = ResizeAttributes.InterpolationMode.linear
        else:
            raise ValueError(f"Interpolation mode {node.attributes["mode"]} is not supported for Resize.")

        # nearest_mode attribute
        if node.attributes["mode"] == "nearest":
            if node.attributes["nearest_mode"] != "floor":
                raise ValueError(f"Nearest mode {node.attributes["nearest_mode"]} is not supported for Resize.")

        layer_type = LayerType.resize
        return layer_type, inputs, outputs, attributes

    def _parse_layer_norm(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert layer norm layer input attributes to CapnProto LayerNormAttributes message.

        Args:
            node (Node): The ONNX node containing the layer norm layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto layer norm attributes object.

        Raises:
            ValueError: If the number of inputs is not 2 or 3
        """
        attributes = LayerNormAttributes.new_message()
        inputs = [tensor.name for tensor in node.inputs]
        if len(inputs) < 2 or len(inputs) > 3:
            msg = f"Invalid number of inputs encountered for LayerNorm node {node.name}. Expected 2 or 3 but got {len(inputs)}"
            raise ValueError(msg)
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.scale = self._network.get_tensor(inputs[1]).idx
        if len(inputs) == 3:
            attributes.bias = self._network.get_tensor(inputs[2]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.layerNorm
        return layer_type, inputs, outputs, attributes

    def _parse_softmax(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert softmax layer input attributes to CapnProto SoftmaxAttributes message.

        Args:
            node (Node): The ONNX node containing the softmax layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto softmax attributes object.
        """
        attributes = node.attributes
        group = attributes.get("group")
        if group is None:
            group = [1]  # FIXME vfor onnx.Softmax vs vidSoftmax
        attributes = SoftmaxAttributes.new_message(reshapeModeGroups=group)
        inputs = [tensor.name for tensor in node.inputs]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.softmax
        return layer_type, inputs, outputs, attributes

    def _parse_squeeze(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert squeeze layer input attributes to CapnProto SqueezeAttributes message.

        Args:
            node (Node): The ONNX node containing the squeeze layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto squeeze attributes object.
        """
        axes = node.inputs[1].data.tolist()
        # Convert negative axis indices to positive
        ndims = len(node.inputs[0].shape)
        for list_index, axis in enumerate(axes):
            if axis < 0:
                axes[list_index] = ndims + axis
        attributes = SqueezeAttributes.new_message()
        attributes.init("axes", len(axes))
        attributes.axes = axes

        inputs = [node.inputs[0].name]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.squeeze
        return layer_type, inputs, outputs, attributes

    def _parse_reshape(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert reshape layer input attributes to CapnProto ReshapeAttributes message.

        Args:
            node (Node): The ONNX node containing the reshape layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto reshape attributes object.
        """
        # ONNX defines reshape shape as input -> move it to attributes
        shape = node.outputs[0].shape
        # Instead of using the shape input attribute, use the output tensor shape to avoid -1 entries
        attributes = ReshapeAttributes.new_message()
        attributes.init("shape", len(shape))
        attributes.shape = shape

        inputs = [node.inputs[0].name]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.reshape
        return layer_type, inputs, outputs, attributes

    def _parse_transpose(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert transpose layer input attributes to CapnProto TransposeAttributes message.

        Args:
            node (Node): The ONNX node containing the transpose layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto transpose attributes object.
        """
        attributes = TransposeAttributes.new_message()
        perm = node.attributes["perm"]
        attributes.init("perm", len(perm))
        attributes.perm = perm
        inputs = [node.inputs[0].name]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.transpose
        return layer_type, inputs, outputs, attributes

    def _parse_gather(self, node: Node) -> Any:
        """Convert gather layer input attributes to CapnProto GatherAttributes message.

        Args:
            node (Node): The ONNX node containing the gather layer attributes.

        Returns:
            Any: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto gather attributes object.

        """
        attributes = GatherAttributes.new_message(axis=node.attributes["axis"])
        inputs = [node.inputs[0].name]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.indexTensor = self._network.get_tensor(node.inputs[1].name).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx

        layer_type = LayerType.gather
        return layer_type, inputs, outputs, attributes

    def _parse_slice(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert slice layer input attributes to CapnProto SliceAttributes message.

        Args:
            node (Node): The ONNX node containing the slice layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto slice attributes object.

        Raises:
            ValueError: If the number of inputs is not 4 or 5
        """
        if len(node.inputs) < 4 or len(node.inputs) > 5:
            msg = f"Invalid number of inputs encountered for Slice node {node.name}. Expected 4 or 5 but got {len(node.inputs)}"
            raise ValueError(msg)

        inputs = [node.inputs[0].name]  # Remove all but the input data tensor from op inputs
        outputs = [node.outputs[0].name]
        layer_type = LayerType.slice

        # Move starts, ends, axes and steps from inputs to attributes
        attributes = SliceAttributes.new_message()

        starts = node.inputs[1].data.tolist()
        ends = node.inputs[2].data.tolist()
        axes = node.inputs[3].data.tolist()

        # Clamp ends to input shape to avoid issues with indexing in v-NN Mapper
        # Sometimes ends contains very large numbers (e.g. 2**63-1) to indicate slicing until the end of the dimension
        input_shape = node.inputs[0].shape
        for i in range(len(ends)):
            if ends[i] > input_shape[axes[i]]:
                ends[i] = input_shape[axes[i]]
        attributes.init("starts", len(starts))
        attributes.init("ends", len(ends))
        attributes.init("axes", len(axes))
        attributes.starts = starts
        attributes.ends = ends
        attributes.axes = axes

        if len(node.inputs) == 5:
            steps = node.inputs[4].data.tolist()
            attributes.init("steps", len(steps))
            attributes.steps = steps

        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        return layer_type, inputs, outputs, attributes

    def _parse_rms_norm(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert layer norm layer input attributes to CapnProto RMSNormalizationAttribute message.

        Args:
            node (Node): The ONNX node containing the RMS Norm layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto layer norm attributes object.

        Raises:
            ValueError: If the number of inputs is not 2
        """
        attributes = RMSNormalizationAttributes.new_message()
        inputs = [tensor.name for tensor in node.inputs]
        if len(inputs) != 2:
            msg = f"Invalid number of inputs encountered for RMSNormalization node {node.name}. Expected 2, got {len(inputs)}"
            raise ValueError(msg)
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.scale = self._network.get_tensor(inputs[1]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx

        # attributes.axis = node.attributes["axis"]
        layer_type = LayerType.rmsNormalization

        return layer_type, inputs, outputs, attributes

    def _parse_rope(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert layer norm layer input attributes to CapnProto RMSNormalizationAttribute message.

        Args:
            node (Node): The ONNX node containing the RMS Norm layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto layer norm attributes object.

        Raises:
            ValueError: If the number of inputs is not 2
        """
        attributes = RopeAttributes.new_message()
        inputs = [tensor.name for tensor in node.inputs]
        if len(inputs) != 3:
            msg = f"Invalid number of inputs encountered for RMSNormalization node {node.name}. Expected 2, got {len(inputs)}"
            raise ValueError(msg)
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.cos = self._network.get_tensor(inputs[1]).idx
        attributes.sin = self._network.get_tensor(inputs[2]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx

        # attributes.axis = node.attributes["axis"]
        layer_type = LayerType.rope

        return layer_type, inputs, outputs, attributes

    def _parse_expand(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert expand layer input attributes to CapnProto ExpandAttributes message.

        Args:
            node (Node): The ONNX node containing the expand layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto expand attributes object.
        """
        # ONNX defines reshape shape as input -> move it to attributes
        shape = node.outputs[0].shape
        # Instead of using the shape input attribute, use the output tensor shape to avoid -1 entries
        attributes = ReshapeAttributes.new_message()
        attributes.init("shape", len(shape))
        attributes.shape = shape

        inputs = [node.inputs[0].name]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.expand
        return layer_type, inputs, outputs, attributes

    def _parse_retr_transformation(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert expand layer input attributes to CapnProto RETRTransformationAttributes message.

        Args:
            node (Node): The ONNX node containing the expand layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto expand attributes object.
        """
        # ONNX defines reshape shape as input -> move it to attributes
        # Instead of using the shape input attribute, use the output tensor shape to avoid -1 entries
        input_shape = node.inputs[0].shape
        attributes = RETRTransformationAttributes.new_message()
        reshape_1_shape = _resolve_dynamic_shape([int(s) for s in node.inputs[1].data], input_shape)
        expand_shape = [int(entry) for entry in node.inputs[2].data]
        transpose_perm = [int(entry) for entry in node.inputs[4].data]
        reshape_2_shape = _resolve_dynamic_shape([int(s) for s in node.inputs[3].data], input_shape)
        attributes.init("reshape1Shape", len(reshape_1_shape))
        attributes.init("expandShape", len(expand_shape))
        attributes.init("transposePerm", len(transpose_perm))
        attributes.init("reshape2Shape", len(reshape_2_shape))
        attributes.reshape1Shape = reshape_1_shape
        attributes.expandShape = expand_shape
        attributes.transposePerm = transpose_perm
        attributes.reshape2Shape = reshape_2_shape

        inputs = [node.inputs[0].name]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.retr
        return layer_type, inputs, outputs, attributes

    def _parse_rer_transformation(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert expand layer input attributes to CapnProto RETRTransformationAttributes message.

        Args:
            node (Node): The ONNX node containing the expand layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto expand attributes object.
        """
        # ONNX defines reshape shape as input -> move it to attributes
        # Instead of using the shape input attribute, use the output tensor shape to avoid -1 entries
        input_shape = node.inputs[0].shape
        attributes = RERTransformationAttributes.new_message()

        reshape_1_shape = _resolve_dynamic_shape([int(s) for s in node.inputs[1].data], input_shape)
        expand_shape = [int(entry) for entry in node.inputs[2].data]
        reshape_2_shape = _resolve_dynamic_shape([int(s) for s in node.inputs[3].data], input_shape)
        attributes.init("reshape1Shape", len(reshape_1_shape))
        attributes.init("expandShape", len(expand_shape))
        attributes.init("reshape2Shape", len(reshape_2_shape))
        attributes.reshape1Shape = reshape_1_shape
        attributes.expandShape = expand_shape
        attributes.reshape2Shape = reshape_2_shape

        inputs = [node.inputs[0].name]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.rer
        return layer_type, inputs, outputs, attributes

    def _parse_rtr_transformation(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert expand layer input attributes to CapnProto RETRTransformationAttributes message.

        Args:
            node (Node): The ONNX node containing the expand layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto expand attributes object.
        """
        input_shape = node.inputs[0].shape
        attributes = RTRTransformationAttributes.new_message()
        reshape_1_shape = _resolve_dynamic_shape([int(s) for s in node.inputs[1].data], input_shape)
        transpose_perm = [int(entry) for entry in node.inputs[3].data]
        reshape_2_shape = _resolve_dynamic_shape([int(s) for s in node.inputs[2].data], input_shape)
        attributes.init("reshape1Shape", len(reshape_1_shape))
        attributes.init("transposePerm", len(transpose_perm))
        attributes.init("reshape2Shape", len(reshape_2_shape))
        attributes.reshape1Shape = reshape_1_shape
        attributes.transposePerm = transpose_perm
        attributes.reshape2Shape = reshape_2_shape
        inputs = [node.inputs[0].name]
        outputs = [node.outputs[0].name]
        attributes.input = self._network.get_tensor(inputs[0]).idx
        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.rtr
        return layer_type, inputs, outputs, attributes

    def _parse_scatter(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert vidScatter layer input attributes to CapnProto ScatterAttributes message.

        Args:
            node (Node): The ONNX node containing the scatter layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto scatter attributes object.

        Raises:
            ValueError: If the number of inputs is not 2 or 3
        """
        attributes = ScatterAttributes.new_message()
        inputs = [tensor.name for tensor in node.inputs]
        if len(inputs) != 3:
            msg = f"Invalid number of inputs encountered for Scatter node {node.name}. Expected 3 but got {len(inputs)}"
            raise ValueError(msg)
        outputs = [node.outputs[0].name]
        attributes.data = self._network.get_tensor(inputs[0]).idx
        attributes.update = self._network.get_tensor(inputs[1]).idx
        attributes.index = self._network.get_tensor(inputs[2]).idx

        attributes.output = self._network.get_tensor(outputs[0]).idx
        layer_type = LayerType.scatter
        return layer_type, inputs, outputs, attributes

    def _parse_conv_transpose(self, node: Node) -> tuple[Any, list[str], list[str], Any]:  # noqa
        """Convert conv transpose layer input attributes to CapnProto ConvTransposeAttributes message.

        Args:
            node (Node): The ONNX node containing the conv transpose layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto conv transpose attributes object.

        Raises:
            ValueError: If the weight tensor has an unsupported number of dimensions or the node has an invalid number of inputs.
        """
        attributes = node.attributes

        output_padding = attributes.get("output_padding")
        if output_padding is not None and any(v != 0 for v in output_padding):
            raise ValueError(
                f"ConvTranspose node has non-zero output_padding={output_padding}, "
                "which is not supported by the ConvTransposeAttributes schema."
            )
        if attributes.get("output_shape") is not None:
            raise ValueError(
                "ConvTranspose node has output_shape attribute, "
                "which is not supported by the ConvTransposeAttributes schema."
            )

        attributes_message = ConvTransposeAttributes.new_message(
            activation=ActivationType.linear,
        )
        if attributes.get("dilations") is not None:
            attributes_message.init("dilations", len(attributes["dilations"]))
            attributes_message.dilations = attributes["dilations"]
        if attributes.get("group") is not None:
            attributes_message.group = attributes["group"]

        if attributes.get("pads") is not None:
            attributes_message.init("pads", len(attributes["pads"]))
            attributes_message.pads = attributes["pads"]
        if attributes.get("strides") is not None:
            attributes_message.init("strides", len(attributes["strides"]))
            attributes_message.strides = attributes["strides"]

        # Ignore ONNX kernel shape attribute and directly infer it from weight tensor
        attributes_message.init("kernelShape", 2)
        weight_tensor = node.inputs[1]
        if len(weight_tensor.shape) == 2:
            kernel_shape = [1, 1]
            attributes_message.kernelShape = kernel_shape
        elif len(weight_tensor.shape) == 4:
            attributes_message.kernelShape = weight_tensor.shape[2:]
        else:
            raise ValueError(f"Cannot handle kernel of shape {weight_tensor.shape}")

        inputs = [tensor.name for tensor in node.inputs if tensor is not None]

        if len(inputs) < 2 or len(inputs) > 3:
            msg = f"Invalid number of inputs conv node {len(inputs)}."
            raise ValueError(msg)
        attributes_message.input = self._network.get_tensor(inputs[0]).idx
        attributes_message.weight = self._network.get_tensor(inputs[1]).idx
        if len(inputs) == 3:
            attributes_message.bias = self._network.get_tensor(inputs[2]).idx

        outputs = [node.outputs[0].name]
        # Check if this conv is followed by an activation layer and merge it by rewiring its outputs
        if len(node.outputs[0].consumers) == 1 and self._is_activation_function(node.outputs[0].consumers[0]):
            activation_layer = node.outputs[0].consumers[0]
            activation_output_tensor = activation_layer.outputs[0]
            outputs = [activation_output_tensor.name]
            attributes_message.output = self._network.get_tensor(activation_output_tensor.name).idx
            attributes_message.preActivationOutput = self._network.get_tensor(node.outputs[0].name).idx

            # Overwrite the activation attribute
            attributes_message.activation = self._parse_activation_type(activation_layer.op_type)

        else:
            # If there is not activation, these fields are the same
            attributes_message.output = self._network.get_tensor(node.outputs[0].name).idx
            attributes_message.preActivationOutput = self._network.get_tensor(node.outputs[0].name).idx

        layer_type = LayerType.convTranspose
        return layer_type, inputs, outputs, attributes_message

    def _parse_grid_sample(self, node: Node) -> tuple[Any, list[str], list[str], Any]:
        """Convert grid sample layer input attributes to CapnProto GridSampleAttributes message.

        Args:
            node (Node): The ONNX node containing the grid sample layer attributes.

        Returns:
            tuple[Any, list[str], list[str], Any]: A tuple containing the layer capnproto layer type,
                a list of input tensor names, output tensor names , and the capnproto grid sample attributes object.

        Raises:
            ValueError: If the padding mode or interpolation mode is not supported.
        """
        attributes = node.attributes

        padding_mode = attributes["padding_mode"]
        if padding_mode != "zeros":
            raise ValueError(f"Padding mode {padding_mode} is not supported for GridSample.")

        align_corners = attributes.get("align_corners", 0)

        attributes_message = GridSampleAttributes.new_message(
            alignCorners=align_corners,
        )

        if attributes["mode"] == "linear":
            attributes_message.mode = GridSampleAttributes.InterpolationMode.linear
        elif attributes["mode"] == "nearest":
            attributes_message.mode = GridSampleAttributes.InterpolationMode.nearest
        else:
            raise ValueError(f"Interpolation mode {attributes['mode']} is not supported for GridSample.")

        inputs = [tensor.name for tensor in node.inputs]
        outputs = [node.outputs[0].name]
        attributes_message.input = self._network.get_tensor(inputs[0]).idx
        attributes_message.grid = self._network.get_tensor(inputs[1]).idx
        attributes_message.output = self._network.get_tensor(outputs[0]).idx

        layer_type = LayerType.gridSample
        return layer_type, inputs, outputs, attributes_message

    @staticmethod
    def _is_activation_function(node: Node) -> bool:
        op_type = node.op_type
        if op_type in ["Gelu", "Sigmoid", "Swish", "HardSigmoid", "Relu", "Relu6", "Mish", "LeakyRelu", "HardSwish"]:
            # Some checks for default values
            if op_type == "LeakyRelu":
                if not bool(np.isclose(node.attributes["alpha"], 0.1, atol=0.0001)):
                    raise RuntimeError("Leaky Relu is only supported for alpha=0.1")
            if op_type == "HardSigmoid":
                if (
                    not bool(np.isclose(node.attributes["alpha"], 0.166666, atol=0.0001))
                    or node.attributes["beta"] != 0.5
                ):
                    raise RuntimeError("Hardsigmoid is only supported for alpha=0.166666 and beta=0.5")
            return True
        else:
            return False
